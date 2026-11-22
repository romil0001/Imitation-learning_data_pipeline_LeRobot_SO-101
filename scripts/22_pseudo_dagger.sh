#!/usr/bin/env bash
# Depth pseudo-DAgger — no leader. Follower + Omen RGB-D required.
# Wrist-depth blob servo nudges ACT actions; large nudges are saved as corrections.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
CORRECTIONS_NAME="${CORRECTIONS_NAME:-${DATASET_NAME}_pseudo_dagger}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}_v2}"
# Prefer v2 if present, else classic act_<dataset>
if [[ -z "${POLICY_PATH:-}" ]]; then
  if [[ -d "$ROOT/outputs/train/act_${DATASET_NAME}_v2" ]]; then
    POLICY_PATH="$ROOT/outputs/train/act_${DATASET_NAME}_v2"
  else
    POLICY_PATH="$ROOT/outputs/train/act_${DATASET_NAME}"
  fi
fi
NUM_EPISODES="${NUM_EPISODES:-10}"
POLICY="${POLICY:-act}"
CONFIG="${CONFIG:-$ROOT/config/pseudo_dagger.yaml}"
DURATION="${DURATION:-45}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"

if [[ ! -d "$POLICY_PATH" ]]; then
  echo "No policy at $POLICY_PATH — train first: bash scripts/10_train_act.sh" >&2
  exit 1
fi

echo "============================================================"
echo " PSEUDO-DAGGER (depth, no leader)"
echo " Policy:      $POLICY_PATH  kind=$POLICY  n_action_steps=$N_ACTION_STEPS"
echo " Demos:       $(dataset_local_root)"
echo " Corrections: $ROOT/data/datasets/$CORRECTIONS_NAME"
echo " Config:      $CONFIG"
echo " Episodes:    ~$NUM_EPISODES  (${DURATION}s each)"
echo "============================================================"
echo "Keys: e/space = end episode (auto-save if enough corrected frames)"
echo "      n = force save    r = discard    q = quit"
echo "Omen must stream wrist RGB+depth. Leader can stay OFF."
echo "After saves:"
echo "  CORRECTIONS_NAME=$CORRECTIONS_NAME \\"
echo "  INIT_CHECKPOINT=$POLICY_PATH \\"
echo "  JOB_NAME=act_${DATASET_NAME}_pseudo \\"
echo "  bash scripts/19_train_dagger.sh"
echo "============================================================"
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  "$(conda_lerobot_python)" -m so101 pseudo-dagger
  --dataset="$DATASET_NAME"
  --corrections="$CORRECTIONS_NAME"
  --checkpoint="$POLICY_PATH"
  --num-episodes="$NUM_EPISODES"
  --policy="$POLICY"
  --config="$CONFIG"
  --duration="$DURATION"
  --n-action-steps="$N_ACTION_STEPS"
  --yes
)
if [[ "${SAVE_ALL:-0}" == "1" ]]; then
  ARGS+=(--save-all)
fi
run_logged pseudo_dagger "${ARGS[@]}"

echo
echo "Fine-tune:"
echo "  CORRECTIONS_NAME=$CORRECTIONS_NAME INIT_CHECKPOINT=$POLICY_PATH JOB_NAME=act_${DATASET_NAME}_pseudo bash scripts/19_train_dagger.sh"
