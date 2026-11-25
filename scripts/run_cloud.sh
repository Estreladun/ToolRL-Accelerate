#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/models}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
PROFILE="${1:-cloud_smoke}"
STAGE="${2:-summary}"
case "$PROFILE" in cloud_smoke|cloud3b) ;; *) echo 'Profile must be cloud_smoke or cloud3b'; exit 2;; esac
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[A-Z_]+$ ]] || continue
  if [[ -z "${!key+x}" ]]; then export "$key=$value"; fi
done < "configs/$PROFILE.env"
# Use a separate CUDA environment on Linux; uv chooses a compatible CUDA wheel.
if [[ -f uv.lock ]]; then uv sync --locked --python 3.11; else uv sync --python 3.11; fi
uv run --no-sync python - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable. Check CUDA PyTorch and NVIDIA driver.')
print('torch',torch.__version__,'CUDA runtime',torch.version.cuda)
for i in range(torch.cuda.device_count()): print(i,torch.cuda.get_device_name(i))
PY
if [[ ! -f "${DATA_DIR:-data/processed}/manifest.json" ]]; then
    bash scripts/prepare_data.sh --out "${DATA_DIR:-data/processed}"
fi
set --
if [[ "${SKIP_DPO:-0}" == 1 ]]; then set -- "$@" --skip-dpo; fi
if [[ -n "${MODEL_REVISION:-}" ]]; then set -- "$@" --model-revision "$MODEL_REVISION"; fi
if [[ "${GRADIENT_CHECKPOINTING:-1}" == 0 ]]; then set -- "$@" --no-gradient-checkpointing; fi
uv run --no-sync python -m toolrl.pipeline \
    --until "$STAGE" --run "$RUN" --model "$MODEL" --data-dir "${DATA_DIR:-data/processed}" \
    --backend ddp --device cuda --dtype bfloat16 --nproc "$NPROC" --attention "$ATTENTION" \
    --sft "$SFT" --dpo "$DPO" --grpo "$GRPO" --grpo-pool "$GRPO_POOL" --val "$VAL" --test "$TEST" \
    --max-prompt "$MAX_PROMPT" --max-response "$MAX_RESPONSE" --sft-response "$SFT_RESPONSE" \
    --group "$GROUP" --rollout-batch "$ROLLOUT_BATCH" --rank "$RANK" --seed "$SEED" \
    --train-batch "$TRAIN_BATCH" --dpo-batch "$DPO_BATCH" --rl-prompts "$RL_PROMPTS" \
    --loss-batch "$LOSS_BATCH" --accum "$ACCUM" --rl-accum "$RL_ACCUM" --eval-batch "$EVAL_BATCH" \
    --sft-epochs "$SFT_EPOCHS" --dpo-epochs "$DPO_EPOCHS" --rl-epochs "$RL_EPOCHS" \
    --sft-lr "$SFT_LR" --dpo-lr "$DPO_LR" --rl-lr "$RL_LR" --beta "$BETA" --temperature "$TEMPERATURE" --dpo-beta "$DPO_BETA" "$@"
