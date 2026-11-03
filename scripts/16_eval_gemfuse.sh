#!/usr/bin/env bash
# Gemfuse recipe eval (ACT checkpoint + async/act5/low-pass/freespace2x).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_common.sh"

confirm_continue() { echo "(auto-confirmed shell banner)"; }

require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
DURATION="${DURATION:-120}"
PROTOCOL="${PROTOCOL-}"
CHECKPOINT="${CHECKPOINT:-}"

echo "============================================================"
echo " GEMFUSE EVAL"
echo " Dataset: $DATASET_NAME"
echo " Follower: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo " Camera: ${CAMERA_NAMES:-$CAMERA_NAME} zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-}"
if [[ -n "${PROTOCOL}" ]]; then
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
if [[ -n "${PROTOCOL}" ]]; then
  ARGS+=(--protocol="$PROTOCOL")
else
  ARGS+=(--duration="$DURATION")
fi

run_logged eval_gemfuse "$(conda_lerobot_python)" "${ARGS[@]}"
