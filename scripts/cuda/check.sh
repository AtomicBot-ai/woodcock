#!/bin/bash
# Does Woodcock's MLX code run correctly and fast on MLX's CUDA backend?
# 1) the whole test suite, including fp32 greedy equality on Qwen3-0.6B and the
#    TTT-vs-inference equivalence; 2) where an adaptation step spends its time.
set -euo pipefail
source .venv/bin/activate
python -m pytest -q 2>&1 | tail -5
python scripts/profile_train_step.py --target Qwen/Qwen3-1.7B 2>&1 | tail -20
