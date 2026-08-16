#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

echo "============================================================"
echo " D405 / cameras"
echo " - USB 3 SuperSpeed for the D405 (not a USB 2 hub)"
echo " - On macOS, Librealsense often fails (UVC driver owns the USB interface)"
echo " - Fallback: OpenCV / AVFoundation webcam index"
echo "============================================================"
confirm_continue

echo
echo "=== AVFoundation devices (ffmpeg) ==="
ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | sed -n '/AVFoundation video devices:/,/AVFoundation audio devices:/p' || true

echo
echo "=== Librealsense (may fail or segfault on macOS) ==="
set +e
lerobot-find-cameras realsense
rs_status=$?
set -e
echo "lerobot-find-cameras exit: $rs_status"

OUT="$ROOT/config/camera.env"
INDEX="${CAMERA_INDEX:-0}"
cat > "$OUT" <<EOF
CAMERA_BACKEND=opencv
CAMERA_INDEX=$INDEX
CAMERA_NAME=front
CAMERA_SERIAL=219323071036
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=15
CAMERA_USE_DEPTH=false
EOF

echo
echo "Wrote $OUT using opencv index $INDEX (on this Mac, D405 RGB is index 0)."
echo "If preview is the laptop webcam, try CAMERA_INDEX=0 or 1 in camera.env."
echo
echo "Preferred for real D405 (RGB + optional depth dump): plug the camera into the HP Omen,"
echo "run omen/setup_omen.bat then omen/run_publish.bat, copy config/camera.env.example,"
echo "and set CAMERA_BACKEND=zmq plus ZMQ_HOST=<Omen LAN IP>."
echo
echo "To retry true RealSense depth on this Mac (often needs sudo, still flaky):"
echo "  sudo conda run -n lerobot lerobot-find-cameras realsense"
echo
echo "Next: bash scripts/06_teleoperate.sh  or  bash scripts/09_dashboard.sh"
