#!/usr/bin/env bash
# Builds models/qwen_vl/.venv: PyTorch cu128 wheels (RTX 50-series), transformers, and
# the benchmark package itself (for the shared prompt / parsing in physics_arcade.vlm).
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv "transformers>=4.57" accelerate pillow "kernels>=0.17.2,<0.18" -e ../..  # kernels: FP8 checkpoints
