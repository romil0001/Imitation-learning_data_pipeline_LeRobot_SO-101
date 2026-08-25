#!/usr/bin/env bash
# Run a trained ACT policy on the follower. Omen D405 publisher must be running.
# Leader can be unplugged. Follower 12V + USB ON.
# Default: scored protocol (conditions × trials, y/n per trial). Unscored timed run: PROTOCOL= bash scripts/11_eval_act.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_comm
# on.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}}"
POLICY_PATH="${POLICY_PATH:-$ROOT/outputs/train/$JOB_NAME}"
DURATION="${DURATION:-60}"
PROTOCOL="${PROTOCOL-$ROOT/config/eval/pen_to_keys.yaml}"
N_TRIALS="${N_TRIALS:-}"

if [[ -f "$POLICY_PATH/best_val/model.pt" ]]; then
  echo "Checkpoint: $POLICY_PATH/best_val/model.pt"
elif [[ -f "$POLICY_PATH/last/model.pt" ]]; then
  echo "Checkpoint: $POLICY_PATH/last/model.pt (no best_val)"
elif [[ -f "$POLICY_PATH/model.pt" ]]; then
  echo "Checkpoint: $POLICY_PATH/model.pt (legacy)"
else
  echo "No ACT checkpoint at $POLICY_PATH (expected best_val/model.pt or last/model.pt)" >&2
  echo "Train first: bash scripts/10_train_act.sh" >&2
  exit 1
fi

echo "============================================================"
echo " ACT EVAL — FOLLOWER + D405 (Omen publisher)"
echo " - Power ON:  FOLLOWER bus + 12V + USB"
echo " - Leader can be OFF"
echo " - Omen: omen\\run_publish.bat must be streaming"
echo "============================================================"
echo "Policy:   $POLICY_PATH"
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "Camera:   ${CAMERA_NAMES:-$CAMERA_NAME} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-}"
if [[ -n "$PROTOCOL" ]]; then
  echo "Protocol: $PROTOCOL"
  echo "Score:    y=success  n=fail after each trial (or during the run to stop early)"
  echo "Reset:    set the pen on the taped mark, then Enter — still prompted per trial"
  if [[ -n "$N_TRIALS" ]]; then
    echo "Trials:   $N_TRIALS per condition (override)"
  fi
else
  echo "Duration: ${DURATION}s (unscored)"
fi
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EVAL_ARGS=(
  "$(conda_lerobot_python)" -m so101 eval
  --dataset="$DATASET_NAME"
  --checkpoint="$POLICY_PATH"
  --yes
)
if [[ -n "$PROTOCOL" ]]; then
  EVAL_ARGS+=(--protocol="$PROTOCOL")
  if [[ -n "$N_TRIALS" ]]; then
    EVAL_ARGS+=(--n-trials="$N_TRIALS")
  fi
else
  EVAL_ARGS+=(--duration="$DURATION")
fi
run_logged eval_act "${EVAL_ARGS[@]}"
