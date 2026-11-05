#!/usr/bin/env bash
# Interactive eval launcher: pick policy, checkpoint, protocol or time limit.
# Non-interactive: INTERACTIVE=0 POLICY=act POLICY_PATH=... DURATION=60 PROTOCOL= bash scripts/17_eval.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
# shellcheck source=scripts/_eval_prompt.sh
source "$ROOT/scripts/_eval_prompt.sh"

require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

prompt_eval_options

GRASP_CONFIG="${GRASP_CONFIG:-$ROOT/config/hybrid_grasp.yaml}"
TASK="${TASK:-}"
N_TRIALS="${N_TRIALS:-}"

echo "============================================================"
echo " EVAL — FOLLOWER + cameras"
echo " - Power ON:  FOLLOWER bus + 12V + USB"
echo " - Leader can be OFF"
echo " - Omen publisher must be streaming"
echo "============================================================"
echo "Policy:   $POLICY  path=$POLICY_PATH"
if [[ "$POLICY" == "hybrid" ]]; then
  echo "Base:     $POLICY_BASE  grasp=$GRASP_CONFIG"
fi
echo "Dataset:  $DATASET_NAME"
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "Camera:   ${CAMERA_NAMES:-$CAMERA_NAME} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-}"
if [[ "${EVAL_MODE}" == "protocol" && -n "${PROTOCOL:-}" ]]; then
  echo "Protocol: $PROTOCOL"
  echo "Score:    y=success  n=fail"
else
  echo "Duration: ${DURATION}s (unscored timed run)"
fi
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EVAL_ARGS=(
  "$(conda_lerobot_python)" -m so101 eval
  --dataset="$DATASET_NAME"
  --checkpoint="$POLICY_PATH"
  --policy="$POLICY"
  --yes
)
if [[ "$POLICY" == "hybrid" ]]; then
  EVAL_ARGS+=(--policy-base="$POLICY_BASE" --grasp-config="$GRASP_CONFIG")
fi
if [[ -n "${TASK:-}" ]]; then
  EVAL_ARGS+=(--task="$TASK")
fi
if [[ "${EVAL_MODE}" == "protocol" && -n "${PROTOCOL:-}" ]]; then
  EVAL_ARGS+=(--protocol="$PROTOCOL")
  if [[ -n "${N_TRIALS:-}" ]]; then
    EVAL_ARGS+=(--n-trials="$N_TRIALS")
  fi
else
  EVAL_ARGS+=(--duration="$DURATION")
fi
_eval_append_control_flags
if ((${#EXTRA_EVAL_FLAGS[@]})); then
  EVAL_ARGS+=("${EXTRA_EVAL_FLAGS[@]}")
fi
run_logged eval "${EVAL_ARGS[@]}"
