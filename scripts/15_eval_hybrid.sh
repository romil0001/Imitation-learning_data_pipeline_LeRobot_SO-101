#!/usr/bin/env bash
# Hybrid eval: SmolVLA (or ACT) for overall motion + wrist-depth grasp assist.
# Omen publisher must stream RGB+depth. Follower ON; leader can be OFF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
POLICY_BASE="${POLICY_BASE:-smolvla}"
if [[ "$POLICY_BASE" == "act" || "$POLICY_BASE" == "act-jax" ]]; then
  POLICY_PATH="${POLICY_PATH:-$ROOT/outputs/train/act_${DATASET_NAME}}"
elif [[ "$POLICY_BASE" == "smolvla" ]]; then
  JOB_NAME="${JOB_NAME:-smolvla_${DATASET_NAME}}"
  POLICY_PATH="${POLICY_PATH:-$ROOT/outputs/train/$JOB_NAME}"
else
  POLICY_PATH="${POLICY_PATH:-$ROOT/outputs/train/${POLICY_BASE}_${DATASET_NAME}}"
fi
DURATION="${DURATION:-60}"
PROTOCOL="${PROTOCOL-$ROOT/config/eval/pen_to_keys.yaml}"
N_TRIALS="${N_TRIALS:-}"
GRASP_CONFIG="${GRASP_CONFIG:-$ROOT/config/hybrid_grasp.yaml}"
TASK="${TASK:-}"

echo "============================================================"
echo " HYBRID EVAL — SmolVLA/ACT + depth grasp"
echo " - Power ON:  FOLLOWER bus + 12V + USB"
echo " - Leader can be OFF"
echo " - Omen: omen\\run_publish.bat must stream RGB+depth"
echo "============================================================"
echo "Base:     $POLICY_BASE"
echo "Policy:   $POLICY_PATH"
echo "Grasp:    $GRASP_CONFIG"
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "Camera:   ${CAMERA_NAMES:-$CAMERA_NAME} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-}"
if [[ -n "$PROTOCOL" ]]; then
  echo "Protocol: $PROTOCOL"
else
  echo "Duration: ${DURATION}s (unscored)"
fi
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EVAL_ARGS=(
  "$(conda_lerobot_python)" -m so101 eval
  --dataset="$DATASET_NAME"
  --checkpoint="$POLICY_PATH"
  --policy=hybrid
  --policy-base="$POLICY_BASE"
  --grasp-config="$GRASP_CONFIG"
  --yes
)
if [[ -n "$TASK" ]]; then
  EVAL_ARGS+=(--task="$TASK")
fi
if [[ -n "$PROTOCOL" ]]; then
  EVAL_ARGS+=(--protocol="$PROTOCOL")
  if [[ -n "$N_TRIALS" ]]; then
    EVAL_ARGS+=(--n-trials="$N_TRIALS")
  fi
else
  EVAL_ARGS+=(--duration="$DURATION")
fi
run_logged eval_hybrid "${EVAL_ARGS[@]}"
