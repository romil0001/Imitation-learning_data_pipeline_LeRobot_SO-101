#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_leader_port_exists

banner_leader_only
echo "Connect ONE motor at a time as prompted (gripper → ... → shoulder_pan)."
confirm_continue

lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT"

echo "Leader motors configured. Next: bash scripts/04_calibrate_follower.sh"
