#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/models}"
if [[ -f uv.lock ]]; then uv sync --locked --python 3.11; else uv sync --python 3.11; fi
set -- --model "${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
if [[ -n "${MODEL_REVISION:-}" ]]; then set -- "$@" --model-revision "$MODEL_REVISION"; fi
uv run python -m toolrl.resources "$@"
