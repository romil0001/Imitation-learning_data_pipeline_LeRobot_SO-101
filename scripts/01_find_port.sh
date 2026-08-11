#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot

PORTS_OUT="$ROOT/config/ports.env"

echo "=== Discover FOLLOWER port ==="
banner_follower_only
confirm_continue
echo "Running lerobot-find-port — disconnect FOLLOWER when prompted."
FOLLOWER_PORT="$(lerobot-find-port | tee /dev/stderr | awk '/The port of this MotorsBus is/{print $NF}')"
if [[ -z "$FOLLOWER_PORT" ]]; then
  read -r -p "Paste FOLLOWER port path (e.g. /dev/tty.usbmodem...): " FOLLOWER_PORT
fi

echo
echo "=== Discover LEADER port ==="
banner_leader_only
confirm_continue
echo "Running lerobot-find-port — disconnect LEADER when prompted."
LEADER_PORT="$(lerobot-find-port | tee /dev/stderr | awk '/The port of this MotorsBus is/{print $NF}')"
if [[ -z "$LEADER_PORT" ]]; then
  read -r -p "Paste LEADER port path (e.g. /dev/tty.usbmodem...): " LEADER_PORT
fi

if [[ "$FOLLOWER_PORT" == "$LEADER_PORT" ]]; then
  echo "Ports must differ. Got follower=$FOLLOWER_PORT leader=$LEADER_PORT" >&2
  exit 1
fi

cat > "$PORTS_OUT" <<EOF
FOLLOWER_PORT=$FOLLOWER_PORT
LEADER_PORT=$LEADER_PORT
FOLLOWER_ID=my_follower_arm
LEADER_ID=my_leader_arm
EOF

echo
echo "Wrote $PORTS_OUT"
cat "$PORTS_OUT"
echo "Next: bash scripts/02_setup_motors_follower.sh (FOLLOWER only powered)"
