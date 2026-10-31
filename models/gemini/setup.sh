#!/usr/bin/env bash
# Builds models/gemini/.venv: the Google Gen AI SDK plus the benchmark package (shared prompt/parsing).
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv "google-genai>=2.34" -e ../..
