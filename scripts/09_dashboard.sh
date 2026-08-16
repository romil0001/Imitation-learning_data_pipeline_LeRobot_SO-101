#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

echo "Local site: live camera + session log (outputs/session.log)"
echo "Keep this running while you teleop/record in another terminal."
if load_camera; then
  echo "Camera backend: $CAMERA_BACKEND"
  if [[ "$CAMERA_BACKEND" == "zmq" ]]; then
    echo "ZMQ: $ZMQ_HOST:${ZMQ_PORT:-5555} name=$CAMERA_NAME"
  fi
else
  echo "No camera.env yet — dashboard will show a blank feed until you set ZMQ_HOST."
fi

HOST="${DASHBOARD_HOST:-127.0.0.1}"
PORT="${DASHBOARD_PORT:-8080}"
echo "Open http://${HOST}:${PORT}"
python "$ROOT/dashboard/app.py" --host "$HOST" --port "$PORT"
