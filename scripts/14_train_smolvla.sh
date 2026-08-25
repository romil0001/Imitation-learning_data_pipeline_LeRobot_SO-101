#!/usr/bin/env bash
# Fine-tune SmolVLA on the converted LeRobot dataset via lerobot-train.
# Prefer a NVIDIA GPU. Mac MPS may work at batch_size=1 but will be very slow.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$ROOT/data/lerobot/$DATASET_NAME}"
REPO_ID="${REPO_ID:-local/$DATASET_NAME}"
JOB_NAME="${JOB_NAME:-smolvla_${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/train/$JOB_NAME}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-}"
BASE_POLICY="${BASE_POLICY:-lerobot/smolvla_base}"
DEVICE="${DEVICE:-$(detect_torch_device)}"

if [[ ! -d "$LEROBOT_ROOT/meta" ]]; then
  echo "Missing converted dataset at $LEROBOT_ROOT" >&2
  echo "Run first: bash scripts/13_convert_lerobot.sh" >&2
  exit 1
fi

if [[ -z "$BATCH_SIZE" ]]; then
  if [[ "$DEVICE" == "cuda" ]]; then
    BATCH_SIZE=8
  elif [[ "$DEVICE" == "mps" ]]; then
    BATCH_SIZE=1
  else
    BATCH_SIZE=1
  fi
fi

echo "============================================================"
echo " SmolVLA FINE-TUNE"
echo " Dataset:  $LEROBOT_ROOT  (repo_id=$REPO_ID)"
echo " Base:     $BASE_POLICY"
echo " Device:   $DEVICE  batch=$BATCH_SIZE  steps=$STEPS"
echo " Output:   $OUTPUT_DIR"
echo "============================================================"
if [[ "$DEVICE" != "cuda" ]]; then
  echo "Warning: SmolVLA training is intended for CUDA. MPS/CPU will be slow."
fi
confirm_continue

if ! command -v lerobot-train >/dev/null 2>&1; then
  LERO_TRAIN="$(conda_lerobot_python)"
  LERO_TRAIN="${LERO_TRAIN%/python}/lerobot-train"
  if [[ ! -x "$LERO_TRAIN" ]]; then
    echo "lerobot-train not found. Re-run env/setup_env.sh with a LeRobot that includes SmolVLA." >&2
    exit 1
  fi
else
  LERO_TRAIN="lerobot-train"
fi

mkdir -p "$OUTPUT_DIR"
# SmolVLA base expects camera1/camera2/camera3; map our wrist+overhead and pad one empty cam.
RENAME_MAP='{"observation.images.wrist":"observation.images.camera1","observation.images.overhead":"observation.images.camera2"}'
run_logged train_smolvla "$LERO_TRAIN" \
  --policy.path="$BASE_POLICY" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$LEROBOT_ROOT" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --rename_map="$RENAME_MAP" \
  --wandb.enable=false \
  --num_workers=0 \
  --save_freq=5000

echo
echo "Checkpoints under $OUTPUT_DIR/checkpoints/*/pretrained_model"
echo "Hybrid eval: bash scripts/15_eval_hybrid.sh"
