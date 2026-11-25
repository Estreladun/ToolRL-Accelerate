#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/models}"
if [[ -f uv.lock ]]; then uv sync --locked --python 3.11; else uv sync --python 3.11; fi
uv run python -m toolrl.data.reconstruct "$@"
