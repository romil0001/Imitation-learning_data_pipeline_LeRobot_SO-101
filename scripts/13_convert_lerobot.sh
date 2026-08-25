#!/usr/bin/env bash
# Convert local SO101 episodes → LeRobot dataset (RGB + task text, no depth).
# Depth stays in the original dataset for hybrid grasp at eval time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data/lerobot/$DATASET_NAME}"
REPO_ID="${REPO_ID:-local/$DATASET_NAME}"
MAX_EPISODES="${MAX_EPISODES:-}"

require_local_dataset

echo "============================================================"
echo " CONVERT → LeRobot (for SmolVLA)"
echo " Source:  $(dataset_local_root)"
echo " Output:  $OUTPUT_DIR"
echo " repo_id: $REPO_ID"
echo " Depth:   dropped (use hybrid grasp at eval)"
echo "============================================================"
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  "$(conda_lerobot_python)" -m so101 convert
  --dataset="$DATASET_NAME"
  --output-dir="$OUTPUT_DIR"
  --repo-id="$REPO_ID"
)
if [[ -n "$MAX_EPISODES" ]]; then
  ARGS+=(--max-episodes="$MAX_EPISODES")
fi
run_logged convert_lerobot "${ARGS[@]}"

echo
echo "Next (CUDA recommended): bash scripts/14_train_smolvla.sh"
echo "Then hybrid eval: bash scripts/15_eval_hybrid.sh"
