#!/usr/bin/env bash
# Train ACT (Action Chunking Transformer) on demos recorded by scripts/08_record.sh.
# Run on the Mac (MPS). Do not train on the 4 GB Omen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

HF_USER="${HF_USER:-local}"
DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/train/$JOB_NAME}"
STEPS="${STEPS:-40000}"
WANDB="${WANDB:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train missing. Re-run: bash env/setup_env.sh" >&2
  exit 1
fi

require_local_dataset

DEVICE="${DEVICE:-$(detect_torch_device)}"
if [[ "$DEVICE" == "cuda" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-8}"
  NUM_WORKERS="${NUM_WORKERS:-4}"
elif [[ "$DEVICE" == "mps" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-2}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
else
  BATCH_SIZE="${BATCH_SIZE:-1}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  echo "Warning: CPU training is very slow. Prefer Mac MPS or a CUDA box / HF Jobs."
fi

echo "============================================================"
echo " ACT TRAINING (Mac) — not the Omen"
echo " Dataset:  $(dataset_repo_id)"
echo " Path:     $(dataset_local_root)"
echo " Device:   $DEVICE  batch=$BATCH_SIZE  steps=$STEPS"
echo " Output:   $OUTPUT_DIR"
echo "============================================================"
echo "Keep cameras/task identical between record and eval."
confirm_continue

# Do not mkdir the output dir first: lerobot-train creates it and
# raises FileExistsError if the path already exists with resume=false.
run_logged train_act lerobot-train \
  --dataset.repo_id="$(dataset_repo_id)" \
  --policy.type=act \
  --policy.device="$DEVICE" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers="$NUM_WORKERS" \
  --wandb.enable="$WANDB" \
  --policy.push_to_hub="$PUSH_TO_HUB"

echo
echo "Checkpoints: $OUTPUT_DIR/checkpoints/last/pretrained_model"
echo "Next: bash scripts/11_eval_act.sh"
