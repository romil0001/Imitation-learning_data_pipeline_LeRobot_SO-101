#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"

echo "============================================================"
echo " Cameras view setup"
echo " - Installs dashboard Python deps"
echo " - Writes config/camera.env if missing"
echo " - Optional: start mock ZMQ publisher for local testing"
echo "============================================================"

REQ="$ROOT/dashboard/requirements.txt"
if [[ ! -f "$REQ" ]]; then
  echo "Missing $REQ" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx lerobot; then
    conda activate lerobot
    echo "Installing dashboard deps into conda env 'lerobot' …"
    pip install -q -r "$REQ"
  else
    echo "Conda env 'lerobot' not found — installing into current Python …"
    python3 -m pip install -q -r "$REQ"
  fi
else
  echo "Installing dashboard deps with python3 -m pip …"
  python3 -m pip install -q -r "$REQ"
fi

CAM="$ROOT/config/camera.env"
if [[ ! -f "$CAM" ]]; then
  cat > "$CAM" <<'EOF'
# Local dev: mock publisher on this machine (scripts/mock_zmq_camera.py)
# Production: set ZMQ_HOST to the HP Omen LAN IP and run omen/run_publish.bat
CAMERA_BACKEND=zmq
ZMQ_HOST=127.0.0.1
ZMQ_PORT=5555
CAMERA_NAME=front
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=15
CAMERA_USE_DEPTH=false
EOF
  echo "Wrote $CAM (ZMQ localhost — good for mock testing)."
else
  echo "Using existing $CAM"
fi

echo
echo "Next steps:"
echo "  1. Mock feed (no hardware):"
echo "       python3 scripts/mock_zmq_camera.py"
echo "     then in another terminal:"
echo "       bash scripts/09_dashboard.sh"
echo "     Open http://127.0.0.1:8080"
echo
echo "  2. Real D405 on HP Omen:"
echo "       omen/run_publish.bat on Windows"
echo "       Set ZMQ_HOST=<Omen LAN IP> in config/camera.env"
echo "       bash scripts/09_dashboard.sh"
echo
echo "  3. Mac-local OpenCV (D405 RGB via AVFoundation):"
echo "       bash scripts/07_find_cameras.sh"
echo "       bash scripts/09_dashboard.sh"
