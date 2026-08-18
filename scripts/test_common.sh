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

# zmq camera json without BIRDSEYE_NAME must stay single-camera
echo "$json" | grep -q 'camera_name: "front"'
if echo "$json" | grep -q birdseye; then
  echo "FAIL: birdseye must be absent when BIRDSEYE_NAME is unset"
  exit 1
fi

# zmq camera json with BIRDSEYE_NAME must list both cameras on the same host/port
cat > "$TMP/camera.env" <<EOF
CAMERA_BACKEND=zmq
ZMQ_HOST=192.168.1.50
ZMQ_PORT=5555
CAMERA_NAME=front
BIRDSEYE_NAME=birdseye
EOF
CAMERA_FILE="$TMP/camera.env"
load_camera
test "$BIRDSEYE_NAME" = "birdseye"
json="$(robot_cameras_json)"
echo "$json" | grep -q 'front: {type: zmq'
echo "$json" | grep -q 'birdseye: {type: zmq'
echo "$json" | grep -q 'camera_name: "front"'
echo "$json" | grep -q 'camera_name: "birdseye"'
echo "$json" | grep -q 'server_address: "192.168.1.50"'
echo "$json" | grep -q 'port: 5555'

echo "PASS: test_common.sh"
