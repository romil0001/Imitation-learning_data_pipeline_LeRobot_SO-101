#!/usr/bin/env bash
# Human-gated DAgger. Both arms powered. Policy runs until you press p.
# p moves the leader to the follower, then torque turns off so you can grab it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
require_leader_port_exists
load_camera

DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
CORRECTIONS_NAME="${CORRECTIONS_NAME:-${DATASET_NAME}_dagger}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}}"
POLICY_PATH="${POLICY_PATH:-$ROOT/outputs/train/$JOB_NAME}"
NUM_EPISODES="${NUM_EPISODES:-10}"
POLICY="${POLICY:-act}"

if [[ ! -d "$POLICY_PATH" ]]; then
  echo "No policy at $POLICY_PATH — train first: bash scripts/10_train_act.sh" >&2
  exit 1
fi

banner_both_arms
echo "Policy:      $POLICY_PATH  kind=$POLICY"
echo "Demos:       $(dataset_local_root)"
echo "Corrections: $ROOT/data/datasets/$CORRECTIONS_NAME"
echo "Target:      ~${NUM_EPISODES} grasp-focused corrections (missed clip approaches)"
echo "Keys: p = pause and move leader to follower"
echo "      c = record the correction (only these frames are saved)"
echo "      p = hand control back to the policy"
echo "      n = save    r = discard    q = quit"
echo "The leader stays still until you press p."
echo "--------------------------------------------------------------"
echo "Bag-clip focus (missed grasp):"
echo "  Let the policy approach. When it misses or stalls near the clip,"
echo "  press p → c, then guide approach + close gripper, then n."
echo "  Prefer correcting the grasp phase, not the whole episode."
echo "  After ~${NUM_EPISODES} saves: bash scripts/19_train_dagger.sh"
echo "Jerky eval? Use CONTROL_MODE=act25 on scripts/11_eval_act.sh (not reactive)."
echo "--------------------------------------------------------------"
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_logged dagger "$(conda_lerobot_python)" -m so101 dagger \
  --dataset="$DATASET_NAME" \
  --corrections="$CORRECTIONS_NAME" \
  --checkpoint="$POLICY_PATH" \
  --num-episodes="$NUM_EPISODES" \
  --policy="$POLICY" \
  --yes

echo
echo "Next: bash scripts/19_train_dagger.sh"
echo "Then:  bash scripts/20_rescore_compare.sh"
