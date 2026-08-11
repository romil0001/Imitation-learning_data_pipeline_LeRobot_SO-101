#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports

banner_follower_only
echo "Move follower to mid-range pose, then sweep each joint when prompted."
confirm_continue

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID"

echo "Follower calibrated as id=$FOLLOWER_ID. Next: bash scripts/05_calibrate_leader.sh"
