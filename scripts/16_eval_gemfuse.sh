#!/usr/bin/env bash
# Gemfuse recipe eval (ACT checkpoint + async/act5/low-pass/freespace2x).
# Interactive menu by default. Skip: INTERACTIVE=0 CHECKPOINT=... DURATION=120
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_common.sh"
# shellcheck source=scripts/_eval_prompt.sh
source "$ROOT/scripts/_eval_prompt.sh"

require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
POLICY="${POLICY:-gemfuse}"
DURATION="${DURATION:-120}"
# Gemfuse already uses act5+async via its default recipe; don't force reactive_act1.
CONTROL_MODE="${CONTROL_MODE:-act5}"
if [[ -z "${PROTOCOL+x}" ]]; then
  PROTOCOL=""
  EVAL_MODE="${EVAL_MODE:-timed}"
fi
POLICY_PATH="${POLICY_PATH:-${CHECKPOINT:-$ROOT/outputs/train/act_${DATASET_NAME}}}"

prompt_eval_options
POLICY=gemfuse
CHECKPOINT="${POLICY_PATH}"
# Prefer gemfuse built-in recipe unless user picked reactive/smooth overrides
if [[ "${CONTROL_MODE}" == "act5" ]]; then
  RECIPE=""
  N_ACTION_STEPS=""
fi

echo "============================================================"
echo " GEMFUSE EVAL"
echo " Dataset: $DATASET_NAME"
echo " Checkpoint: $CHECKPOINT"
echo " Follower: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo " Camera: ${CAMERA_NAMES:-$CAMERA_NAME} zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-}"
if [[ "${EVAL_MODE}" == "protocol" && -n "${PROTOCOL:-}" ]]; then
  echo " Protocol: $PROTOCOL"
else
  echo " Duration: ${DURATION}s (timed, no protocol)"
fi
echo "============================================================"
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  -m so101 eval
  --dataset="$DATASET_NAME"
  --policy=gemfuse
  --yes
)
if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint="$CHECKPOINT")
fi
if [[ "${EVAL_MODE}" == "protocol" && -n "${PROTOCOL:-}" ]]; then
  ARGS+=(--protocol="$PROTOCOL")
else
  ARGS+=(--duration="$DURATION")
fi
_eval_append_control_flags
if ((${#EXTRA_EVAL_FLAGS[@]})); then
  ARGS+=("${EXTRA_EVAL_FLAGS[@]}")
fi

run_logged eval_gemfuse "$(conda_lerobot_python)" "${ARGS[@]}"
