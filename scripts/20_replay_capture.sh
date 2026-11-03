#!/usr/bin/env bash
# Replay teleop demos while capturing live wrist+overhead+base (+depths) into so101_bag_clip_bowl_capture.
# Omen must publish all cams. Follower ON; leader OFF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

SOURCE="${SOURCE:-so101_bag_clip_bowl}"
DEST="${DEST:-so101_bag_clip_bowl_capture}"
TASK="${TASK:-Place bag clip in bowl}"
EPISODE="${EPISODE:-}"
MAX_EPISODES="${MAX_EPISODES:-}"

echo "============================================================"
echo " REPLAY-CAPTURE — actions from $SOURCE → cameras into $DEST"
echo " - Power ON:  FOLLOWER bus + 12V + USB"
echo " - Leader can be OFF"
echo " - Omen: dual D405 + overhead publishing (wrist,overhead,base)"
echo "============================================================"
echo "Source: $SOURCE"
echo "Dest:   $DEST  task=$TASK"
echo "Cams:   ${CAMERA_NAMES:-$CAMERA_NAME}"
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  "$(conda_lerobot_python)" -m so101 replay-capture
  --source="$SOURCE"
  --dest="$DEST"
  --task="$TASK"
  --yes
)
if [[ -n "$EPISODE" ]]; then
  ARGS+=(--episode="$EPISODE")
fi
if [[ -n "$MAX_EPISODES" ]]; then
  ARGS+=(--max-episodes="$MAX_EPISODES")
fi
run_logged replay_capture "${ARGS[@]}"
