from causvid.models.model_interface import (
    DiffusionModelInterface,
    TextEncoderInterface,
    VAEInterface
)
from causvid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
from causvid.models.wan.wan_base.modules.model import WanModel
from causvid.models.wan.wan_base.modules.vae import _video_vae
from causvid.models.wan.wan_base.modules.t5 import umt5_xxl
from causvid.models.wan.flow_match import FlowMatchScheduler
from causvid.models.wan.causal_model import CausalWanModel
from typing import List, Dict, Optional
import torch
import os
import torch.distributed as dist
import time

_STARTUP_TS = time.perf_counter()


def _log_startup(message: str) -> None:
    elapsed = time.perf_counter() - _STARTUP_TS
    print(f"[SingleGPUInference][startup {elapsed:0.2f}s] {message}")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _human_bytes(byte_count: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(byte_count)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _normalize_checkpoint_payload(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint file payload is not a dict; expected state_dict payload.")

    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]

    return payload


def _log_weight_load_step(label: str, checkpoint_path: str, target: torch.nn.Module) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    file_size = os.path.getsize(checkpoint_path)
    _log_startup(f"Inspecting {label} checkpoint file")
    _log_startup(f"{label} checkpoint file size: {_human_bytes(file_size)} ({checkpoint_path})")

    _log_startup(f"Reading {label} checkpoint payload into memory")
    payload_start = time.perf_counter()
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    payload_elapsed = time.perf_counter() - payload_start
    _log_startup(
        f"{label} checkpoint payload loaded in {payload_elapsed:0.2f}s"
    )

    state_dict = _normalize_checkpoint_payload(payload)
    _log_startup(f"{label} checkpoint tensors: {len(state_dict)} entries")

    _log_startup(f"Applying {label} state dict")
    apply_start = time.perf_counter()
    target_state = target.state_dict()
    total_tensors = len(state_dict)
    applied_tensors = 0
    missing_tensors = 0
    unexpected_tensors = 0
    tensor_bytes = 0
    report_every = max(1, total_tensors // 12)

    for idx, (name, tensor) in enumerate(state_dict.items(), 1):
        if not isinstance(tensor, torch.Tensor):
            unexpected_tensors += 1
            continue

        destination = target_state.get(name)
        if destination is None:
            unexpected_tensors += 1
            if unexpected_tensors <= 4:
                _log_startup(f"{label} checkpoint contains unexpected key: {name}")
            continue

        if destination.shape != tensor.shape:
            _log_startup(
                f"{label} tensor shape mismatch for {name}: "
                f"checkpoint={tuple(tensor.shape)} model={tuple(destination.shape)}"
            )
            raise RuntimeError(f"{label} checkpoint tensor shape mismatch for {name}")

        if idx == 1:
            _log_startup(
                f"{label} loading tensor '{name}' ({tuple(tensor.shape)} -> {tuple(destination.shape)})"
            )
        destination.data.copy_(
            tensor.to(dtype=destination.dtype, device=destination.device, non_blocking=True)
        )
        applied_tensors += 1
        tensor_bytes += tensor.element_size() * tensor.numel()

        if idx % report_every == 0 or idx == total_tensors:
            percent = int((idx / total_tensors) * 100)
            elapsed = time.perf_counter() - apply_start
            bytes_per_second = tensor_bytes / elapsed if elapsed > 0 else 0.0
            _log_startup(
                f"{label} state dict progress: {idx}/{total_tensors} tensors applied "
                f"({applied_tensors} copied, {percent}%, { _human_bytes(bytes_per_second) }/s)"
            )

    missing_tensors = len(target_state) - applied_tensors
    apply_elapsed = time.perf_counter() - apply_start
    _log_startup(f"{label} state dict bytes: {_human_bytes(tensor_bytes)}")
    _log_startup(
        f"{label} state dict applied in {apply_elapsed:0.2f}s "
        f"(applied={applied_tensors}, missing={missing_tensors}, unexpected={unexpected_tensors})"
    )


class WanTextEncoder(TextEncoderInterface):
    def __init__(self, model_type="T2V-1.3B") -> None:
        super().__init__()
        _log_startup(f"Initializing text encoder for {model_type}")

        _log_startup("Building text encoder backbone module graph")
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device("cpu"),
            startup_log_fn=_log_startup,
        ).eval().requires_grad_(False)
        _log_startup("Text encoder backbone module graph ready")
        _log_weight_load_step(
            "text encoder",
            os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/models_t5_umt5-xxl-enc-bf16.pth"),
            self.text_encoder
        )
        _log_startup("Preparing text encoder tokenizer")
        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/google/umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(VAEInterface):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()
        _log_startup(f"Initializing VAE for {model_type}")
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        _log_startup("Loading VAE model")
        _log_startup("Building VAE module graph")
        self.model = _video_vae(
            pretrained_path=os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)
        _log_startup("VAE model initialized")

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.decode(u.unsqueeze(0),
                              scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = self.model.decode(zs, scale).clamp_(-1, 1)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        # output = output.permute(0, 2, 1, 3, 4)
        return output
    
    def stream_encode(self, video: torch.Tensor, is_scale=False) -> torch.Tensor:
        if is_scale:
            device, dtype = video.device, video.dtype
            scale = [self.mean.to(device=device, dtype=dtype),
                    1.0 / self.std.to(device=device, dtype=dtype)]
        else:
            scale = None
        return self.model.stream_encode(video, scale)
    
    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        zs = zs.to(torch.bfloat16).to('cuda')
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = self.model.stream_decode(zs, scale).float().clamp_(-1, 1)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(DiffusionModelInterface):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()
        _log_startup(f"Initializing diffusion wrapper for {model_type}")

        _log_startup("Loading Wan diffusion architecture")
        self.model = WanModel.from_pretrained(os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/"))
        _log_startup("Wan model loaded")
        self.model.eval()

        _log_startup("Initializing diffusion scheduler")
        self.uniform_timestep = True

        self.scheduler = FlowMatchScheduler(
            shift=8.0, sigma_min=0.0, extra_one_step=True
        )
        _log_startup("Diffusion scheduler initialized")
        _log_startup("Configuring diffusion timesteps")
        self.scheduler.set_timesteps(1000, training=True)
        _log_startup("Diffusion wrapper ready")

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        super().post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0

    def forward_input(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor,block_mode: str='input', block_num = None, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        patched_x_shape: torch.Tensor = None,
        block_x: torch.Tensor = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache must be provided"

        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        
        if block_x is not None and block_mode == 'middle':
            noisy_image_or_video = block_x
        else:
            noisy_image_or_video = noisy_image_or_video.permute(0, 2, 1, 3, 4)

        output, patched_x_shape = self.model(
            noisy_image_or_video,
            t=input_timestep, context=prompt_embeds,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            block_mode=block_mode,
            block_num=block_num,
            patched_x_shape=patched_x_shape,
        )

        return output, patched_x_shape

    def forward_output(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, block_mode: str='output', block_num = None, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        patched_x_shape: torch.Tensor = None,
        block_x: torch.Tensor = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache must be provided"

        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        flow_pred = self.model(
            block_x,
            t=input_timestep, context=prompt_embeds,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            block_mode=block_mode,
            block_num=block_num,
            patched_x_shape=patched_x_shape,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0


class CausalWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()

        self.model = CausalWanModel.from_pretrained(
            os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/"))
        self.model.eval()

        self.uniform_timestep = False
