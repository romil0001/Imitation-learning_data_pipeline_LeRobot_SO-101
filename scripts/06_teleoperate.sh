#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
require_leader_port_exists

banner_both_arms
echo "Follower port: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "Leader port:   $LEADER_PORT ($LEADER_ID)"

CAMERA_ARGS=()
if load_camera; then
  echo "Camera:        $CAMERA_NAME birdseye=${BIRDSEYE_NAME:-off} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-} index=${CAMERA_INDEX:-} ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${CAMERA_FPS}"
  CAMERA_ARGS=(--robot.cameras="$(robot_cameras_json)" --display_data=true)
else
  echo "No camera.env — teleop without camera (run scripts/07_find_cameras.sh to add D405)."
fi
confirm_continue

run_logged teleoperate lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  "${CAMERA_ARGS[@]}"
