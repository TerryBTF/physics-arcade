#!/usr/bin/env bash
# Builds models/cosmos/.venv (PyTorch cu128 wheels, needed for RTX 50-series / sm_120).
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements.lock.txt
