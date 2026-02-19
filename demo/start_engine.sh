#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

PYTHON_BIN=""

if [[ -n "${STREAMDIFFUSION_PYTHON:-}" ]]; then
  PYTHON_BIN="${STREAMDIFFUSION_PYTHON}"
elif [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif command -v conda >/dev/null 2>&1; then
  if stream_python="$(conda run -n stream which python 2>/dev/null | tail -n 1)"; then
    if [[ -x "${stream_python}" ]]; then
      PYTHON_BIN="${stream_python}"
    fi
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys
import importlib.util

if importlib.util.find_spec("fastapi") is None:
    raise RuntimeError("missing fastapi module")
print(f"[streamdiffusion] using python: {sys.executable}")
PY
then
  echo -e "\033[1;31m\nselected python interpreter does not have fastapi installed: ${PYTHON_BIN}\n\033[0m" >&2
  echo "Tip: install deps with:"
  echo "  conda activate stream"
  echo "  pip install -r StreamDiffusionV2/requirements.txt"
  exit 1
fi

STREAMDIFFUSION_DISABLE_FRONTEND_MOUNT=1 \
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" sidecar/run_with_logs.py -- \
  "$PYTHON_BIN" main.py --port 7860 --host 0.0.0.0 --num_gpus 1 --step 1 --model_type T2V-1.3B --enable-metrics
