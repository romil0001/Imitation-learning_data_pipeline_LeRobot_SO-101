#!/usr/bin/env bash
# Prove the Omen ZMQ publisher is transmitting wrist RGB, overhead RGB, and depth.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

if ! load_camera; then
  echo "Fix config/camera.env first (copy from camera.env.example, set ZMQ_HOST)." >&2
  exit 1
fi

if [[ "${CAMERA_BACKEND}" != "zmq" ]]; then
  echo "CAMERA_BACKEND=${CAMERA_BACKEND} — link-check expects zmq from the Omen." >&2
  exit 1
fi

echo "============================================================"
echo " OMEN → MAC LINK CHECK"
echo "============================================================"
echo " ZMQ: ${ZMQ_HOST}:${ZMQ_PORT:-5555}"
echo " Names: ${CAMERA_NAMES:-$CAMERA_NAME}"
echo " Depth: ${CAMERA_USE_DEPTH}"
echo " Omen must be running: omen\\run_publish.bat"
echo "============================================================"

SECONDS_SAMPLE="${LINK_CHECK_SECONDS:-3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$(conda_lerobot_python)" -m so101 link-check --seconds "$SECONDS_SAMPLE"
