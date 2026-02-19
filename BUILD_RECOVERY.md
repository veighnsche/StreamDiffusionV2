# Build Recovery Notes (flash_attn + pycuda)

## What "wheel" means here
`pip` installs modern Python packages by building a **wheel** (`.whl`) first.
If a package cannot compile, wheel creation fails, and install fails.

So this message:
- `Failed building wheel for pycuda`
- `failed-wheel-build-for-install`

means **`pycuda` compile failed**. Wheel is not the root cause; the compile error is.

## Current root cause (pycuda)
Build log shows:
- `fatal error: cudaProfiler.h: No such file or directory`

That means your active CUDA dev headers are incomplete for `pycuda` (specifically missing profiler API headers).

## How to resume `flash_attn` from the existing temp build
Current temp tree:
`/home/vince/.cache/pip-tmp/pip-install-otm4wd0t/flash-attn_d8bcf93248074bb9b460917639e0d6b3`

1. Resume incremental compile (ninja state already exists):

```bash
conda run -n stream bash -c '
cd /home/vince/.cache/pip-tmp/pip-install-otm4wd0t/flash-attn_d8bcf93248074bb9b460917639e0d6b3/build/temp.linux-x86_64-cpython-310
ninja -j"$(nproc)"
'
```

2. Then finish install from source tree (so extension gets installed into site-packages):

```bash
conda run -n stream bash -c '
cd /home/vince/.cache/pip-tmp/pip-install-otm4wd0t/flash-attn_d8bcf93248074bb9b460917639e0d6b3
python -m pip install --no-build-isolation .
'
```

3. Verify install:

```bash
conda run -n stream python - <<'PY'
import flash_attn
print('flash_attn OK:', flash_attn.__file__)
PY
```

## `pycuda` root-cause fix (no shim)
Install the CUDA dev packages that provide required headers/stubs:

```bash
conda install -n stream -c nvidia cuda-profiler-api
conda install -n stream -c nvidia libcurand-dev
conda install -n stream -c nvidia cuda-driver-dev=12.4.127
```

Then ensure required headers/stubs are present:

```bash
ls /home/vince/.conda/envs/stream/include/cudaProfiler.h
ls /home/vince/.conda/envs/stream/targets/x86_64-linux/include/curand.h
ls /home/vince/.conda/envs/stream/lib/stubs/libcuda.so
```

Then build/install `pycuda` with include path pointed at Conda CUDA target includes:

```bash
conda run -n stream env \
  CUDA_INC_DIR=/home/vince/.conda/envs/stream/targets/x86_64-linux/include \
  C_INCLUDE_PATH=/home/vince/.conda/envs/stream/targets/x86_64-linux/include \
  CPLUS_INCLUDE_PATH=/home/vince/.conda/envs/stream/targets/x86_64-linux/include \
  python -m pip install --no-build-isolation pycuda==2025.1.1
```

## Important note
`flash_attn` and `pycuda` are independent failures.
- `flash_attn`: partial build existed and can be resumed.
- `pycuda`: failed due to missing CUDA header during wheel build.
