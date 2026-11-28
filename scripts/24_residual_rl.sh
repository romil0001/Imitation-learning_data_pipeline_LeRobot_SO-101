#!/usr/bin/env bash
# Phase-mix / residual RL: ACT + dense depth reward + online SAC residual.
# Default config is phase-mix (far=ACT, mid=ACT+δ, near=depth-grasp).
# Plain residual: CONFIG=config/residual_rl.yaml OUTPUT_DIR=.../residual_rl_act
# Follower + Omen RGB-D required. Leader OFF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
if [[ -z "${POLICY_PATH:-}" ]]; then
  if [[ -d "$ROOT/outputs/train/act_${DATASET_NAME}_v2" ]]; then
    POLICY_PATH="$ROOT/outputs/train/act_${DATASET_NAME}_v2"
  else
    POLICY_PATH="$ROOT/outputs/train/act_${DATASET_NAME}"
  fi
fi
CONFIG="${CONFIG:-$ROOT/config/phase_mix.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/train/phase_mix_act}"
NUM_EPISODES="${NUM_EPISODES:-20}"
DURATION="${DURATION:-45}"
RESIDUAL_CKPT="${RESIDUAL_CKPT:-}"

if [[ ! -d "$POLICY_PATH" ]]; then
  echo "No ACT checkpoint at $POLICY_PATH" >&2
  exit 1
fi

echo "============================================================"
echo " PHASE-MIX / RESIDUAL RL (dense depth reward + SAC)"
echo " Base ACT:  $POLICY_PATH"
echo " Residual:  ${RESIDUAL_CKPT:-'(train from scratch)'}"
echo " Output:    $OUTPUT_DIR"
echo " Episodes:  $NUM_EPISODES × ${DURATION}s"
echo " Config:    $CONFIG"
echo "============================================================"
echo "Keys: e/space = end episode   q = quit"
echo "Omen must stream wrist RGB+depth. Leader OFF."
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  "$(conda_lerobot_python)" -m so101 residual-rl
  --dataset="$DATASET_NAME"
  --checkpoint="$POLICY_PATH"
  --output-dir="$OUTPUT_DIR"
  --config="$CONFIG"
  --num-episodes="$NUM_EPISODES"
  --duration="$DURATION"
  --yes
)
if [[ -n "$RESIDUAL_CKPT" ]]; then
  ARGS+=(--residual-checkpoint="$RESIDUAL_CKPT")
fi
run_logged residual_rl "${ARGS[@]}"

echo
echo "Eval phase-mix (deterministic):"
echo "  python -m so101 eval --policy phase-mix --checkpoint=$POLICY_PATH \\"
echo "    --residual-checkpoint=$OUTPUT_DIR --grasp-config=$ROOT/config/hybrid_grasp.yaml \\"
echo "    --duration=60 -y"
echo "(or keep collecting: RESIDUAL_CKPT=$OUTPUT_DIR bash scripts/24_residual_rl.sh)"
