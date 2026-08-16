#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Intentionally fail until _common.sh exists and exports helpers
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# equal ports must fail
cat > "$TMP/ports.env" <<EOF
FOLLOWER_PORT=/dev/tty.usbmodemA
LEADER_PORT=/dev/tty.usbmodemA
FOLLOWER_ID=my_follower_arm
LEADER_ID=my_leader_arm
EOF
PORTS_FILE="$TMP/ports.env"
if load_ports; then
  echo "FAIL: load_ports should reject equal ports"
  exit 1
fi

# placeholders must fail
cat > "$TMP/ports.env" <<EOF
FOLLOWER_PORT=CHANGE_ME_FOLLOWER
LEADER_PORT=/dev/tty.usbmodemB
FOLLOWER_ID=my_follower_arm
LEADER_ID=my_leader_arm
EOF
if load_ports; then
  echo "FAIL: load_ports should reject placeholders"
  exit 1
fi

# distinct real-looking ports must pass
cat > "$TMP/ports.env" <<EOF
FOLLOWER_PORT=/dev/tty.usbmodemA
LEADER_PORT=/dev/tty.usbmodemB
FOLLOWER_ID=my_follower_arm
LEADER_ID=my_leader_arm
EOF
load_ports
test "$FOLLOWER_PORT" = "/dev/tty.usbmodemA"
test "$LEADER_PORT" = "/dev/tty.usbmodemB"

# zmq camera json
cat > "$TMP/camera.env" <<EOF
CAMERA_BACKEND=zmq
ZMQ_HOST=192.168.1.50
ZMQ_PORT=5555
CAMERA_NAME=front
EOF
CAMERA_FILE="$TMP/camera.env"
load_camera
json="$(robot_cameras_json)"
echo "$json" | grep -q 'type: zmq'
echo "$json" | grep -q '192.168.1.50'
echo "PASS: test_common.sh"
