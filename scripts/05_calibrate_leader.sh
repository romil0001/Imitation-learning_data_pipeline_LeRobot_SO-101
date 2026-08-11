#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports

banner_leader_only
echo "Move leader to mid-range pose, then sweep each joint when prompted."
confirm_continue

lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID"

echo "Leader calibrated as id=$LEADER_ID. Next: bash scripts/06_teleoperate.sh"
