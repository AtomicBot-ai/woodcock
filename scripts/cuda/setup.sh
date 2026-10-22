#!/bin/bash
# One-time setup on a rented Linux CUDA machine, run from the repo root after
# scripts/cuda/push.sh. Two environments, because vLLM pins its own torch/CUDA stack:
#   .venv-vllm  vLLM, for target-regenerated training data (scripts/cuda/regen_vllm.py)
#   .venv       Woodcock on MLX's CUDA backend (mlx[cuda13]), for `woodcock adapt`
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v uv >/dev/null || python3 -m pip install --quiet --user --break-system-packages uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv-vllm
VIRTUAL_ENV=.venv-vllm uv pip install vllm
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install "mlx[cuda13]==0.32.4" "mlx-lm==0.32.2" -e ".[test]"
.venv/bin/hf download Qwen/Qwen3-1.7B >/dev/null
.venv/bin/hf download Qwen/Qwen3-0.6B >/dev/null   # test target
.venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name())"
.venv/bin/python -c "import mlx.core as mx; print('mlx', mx.__version__, mx.default_device()); print(mx.ones(4).sum())"
