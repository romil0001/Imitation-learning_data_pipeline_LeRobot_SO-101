#!/usr/bin/env bash
# Fine-tune ACT on demos plus DAgger corrections.
# Corrections are forced into the train split. Val stays a holdout of the original demos.
# Writes a new checkpoint directory so the baseline eval is not overwritten.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
CORRECTIONS_NAME="${CORRECTIONS_NAME:-${DATASET_NAME}_dagger}"
BASE_JOB="${BASE_JOB:-act_${DATASET_NAME}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT/outputs/train/$BASE_JOB}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}_dagger}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/train/$JOB_NAME}"
STEPS="${STEPS:-8000}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-500}"

if [[ ! -f "$ROOT/data/datasets/$CORRECTIONS_NAME/meta.json" ]]; then
  echo "No corrections at data/datasets/$CORRECTIONS_NAME — run bash scripts/18_dagger.sh" >&2
  exit 1
fi

DEVICE="${DEVICE:-$(detect_torch_device)}"
if [[ "$DEVICE" == "cuda" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-8}"
elif [[ "$DEVICE" == "mps" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-2}"
else
  BATCH_SIZE="${BATCH_SIZE:-1}"
fi

echo "============================================================"
echo " ACT + DAgger fine-tune (Mac)"
echo " Demos:       data/datasets/$DATASET_NAME"
echo " Corrections: data/datasets/$CORRECTIONS_NAME"
echo " Init:        $INIT_CHECKPOINT"
echo " Output:      $OUTPUT_DIR"
echo " Steps:       $STEPS  (shorter than the original 40k train)"
echo "============================================================"
echo "Baseline checkpoint is left in place. Eval this output with POLICY_PATH=$OUTPUT_DIR bash scripts/11_eval_act.sh"
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_logged train_dagger "$(conda_lerobot_python)" -m so101 train \
  --dataset="$DATASET_NAME" \
  --corrections="$CORRECTIONS_NAME" \
  --init-checkpoint="$INIT_CHECKPOINT" \
  --steps="$STEPS" \
  --batch-size="$BATCH_SIZE" \
  --output-dir="$OUTPUT_DIR" \
  --val-frac="$VAL_FRAC" \
  --seed="$SEED" \
  --eval-every="$EVAL_EVERY" \
  --backend=torch

echo
echo "Checkpoints: $OUTPUT_DIR/best_val/model.pt"
echo "Re-score (act25 default, saves y→ ${DATASET_NAME}_eval):"
echo "  POLICY_PATH=$OUTPUT_DIR CONTROL_MODE=act25 bash scripts/11_eval_act.sh"
echo "Compare pc_success vs baseline eval run:"
echo "  python -m so101 compare-eval \\"
echo "    --baseline outputs/eval/<baseline_run>/summary.json \\"
echo "    --candidate outputs/eval/<dagger_run>/summary.json"
echo "If grasp still weak: POLICY_BASE=act bash scripts/15_eval_hybrid.sh  (depth assist check)"
echo "Helper: bash scripts/20_rescore_compare.sh"
