#!/usr/bin/env bash
# Train ACT on demos in data/datasets/<name>/. Run on the Mac (MPS). Not the 4 GB Omen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
ALGO="${ALGO:-act}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/train/$JOB_NAME}"
STEPS="${STEPS:-40000}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-1000}"

require_local_dataset

DEVICE="${DEVICE:-$(detect_torch_device)}"
if [[ "$DEVICE" == "cuda" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-8}"
elif [[ "$DEVICE" == "mps" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-2}"
else
  BATCH_SIZE="${BATCH_SIZE:-1}"
  echo "Warning: CPU training is very slow. Prefer Mac MPS or a CUDA box."
fi

echo "============================================================"
echo " ACT TRAINING (Mac) — not the Omen"
echo " Dataset:  $(dataset_local_root)"
echo " Algo:     $ALGO"
echo " Device:   $DEVICE  batch=$BATCH_SIZE  steps=$STEPS"
echo " Split:    val_frac=$VAL_FRAC  seed=$SEED  eval_every=$EVAL_EVERY"
echo " Output:   $OUTPUT_DIR"
echo "============================================================"
echo "Holds out episodes; pick best_val for protocol eval. Train loss is not the score."
echo "Keep cameras/task identical between record and eval."
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_logged train_act "$(conda_lerobot_python)" -m so101 train \
  --algo="$ALGO" \
  --dataset="$DATASET_NAME" \
  --steps="$STEPS" \
  --batch-size="$BATCH_SIZE" \
  --output-dir="$OUTPUT_DIR" \
  --val-frac="$VAL_FRAC" \
  --seed="$SEED" \
  --eval-every="$EVAL_EVERY"

echo
echo "Checkpoints: $OUTPUT_DIR/last/model.pt and $OUTPUT_DIR/best_val/model.pt"
echo "Eval uses best_val when present. Next: bash scripts/11_eval_act.sh"
