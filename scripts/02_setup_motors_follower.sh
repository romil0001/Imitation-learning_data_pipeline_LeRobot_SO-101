#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports

banner_follower_only
echo "Connect ONE motor at a time as prompted (gripper → ... → shoulder_pan)."
confirm_continue

lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT"

echo "Follower motors configured. Next: bash scripts/03_setup_motors_leader.sh"
