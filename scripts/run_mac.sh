#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/models}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
command -v uv >/dev/null || { echo 'Please install uv first.'; exit 1; }
STAGE="${1:-summary}"
if [[ -f uv.lock ]]; then uv sync --locked --python 3.11; else uv sync --python 3.11; fi
uv run python -m pytest -q tests/
if [[ ! -f "${DATA_DIR:-data/processed}/manifest.json" ]]; then
    bash scripts/prepare_data.sh --out "${DATA_DIR:-data/processed}"
fi
set --
if [[ "${SKIP_DPO:-0}" == 1 ]]; then set -- "$@" --skip-dpo; fi
if [[ -n "${MODEL_REVISION:-}" ]]; then set -- "$@" --model-revision "$MODEL_REVISION"; fi
if [[ -n "${DATA_DIR:-}" ]]; then set -- "$@" --data-dir "$DATA_DIR"; fi
uv run python -m toolrl.pipeline \
    --until "$STAGE" --run "${RUN:-runs/smoke}" \
    --model "${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
    --device "${DEVICE:-mps}" --dtype "${DTYPE:-float32}" \
    --sft "${SFT:-8}" --dpo "${DPO:-4}" --grpo "${GRPO:-4}" \
    --grpo-pool "${GRPO_POOL:-$(( ${GRPO:-4} * 3 ))}" \
    --val "${VAL:-4}" --test "${TEST:-4}" --group "${GROUP:-4}" \
    --max-prompt "${MAX_PROMPT:-1536}" --max-response "${MAX_RESPONSE:-256}" \
    --sft-response "${SFT_RESPONSE:-1024}" --rollout-batch "${ROLLOUT_BATCH:-2}" \
    --rank "${RANK:-8}" --seed "${SEED:-42}" "$@"
echo "Pipeline completed successfully. Results: ${RUN:-runs/smoke}"
