#!/usr/bin/env bash
# Point Mac ZMQ client at a direct USB-Ethernet link to the Omen.
# Pair with omen\setup_ethernet_link.bat on Windows (Omen = 192.168.2.2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"

MAC_IP="${MAC_ETH_IP:-192.168.2.1}"
OMEN_IP="${OMEN_ETH_IP:-192.168.2.2}"
NETMASK="${ETH_NETMASK:-255.255.255.0}"
PORT_NAME="${MAC_ETH_PORT_NAME:-USB 10/100 LAN}"
CAM_ENV="$ROOT/config/camera.env"

echo "Ethernet link defaults: Mac $MAC_IP  ↔  Omen $OMEN_IP  (mask $NETMASK)"
echo "Hardware port: $PORT_NAME"

if ! networksetup -listallhardwareports | grep -Fq "$PORT_NAME"; then
  echo "Port '$PORT_NAME' not found. Available:" >&2
  networksetup -listallhardwareports >&2
  echo "Re-run with MAC_ETH_PORT_NAME='…' matching your adapter." >&2
  exit 1
fi

echo "Setting $PORT_NAME → $MAC_IP (may prompt for admin)…"
if ! networksetup -setmanual "$PORT_NAME" "$MAC_IP" "$NETMASK"; then
  echo "networksetup failed. Run manually:" >&2
  echo "  sudo networksetup -setmanual \"$PORT_NAME\" $MAC_IP $NETMASK" >&2
  exit 1
fi

if [[ ! -f "$CAM_ENV" ]]; then
  cp "$ROOT/config/camera.env.example" "$CAM_ENV"
fi
WIFI_IP="${OMEN_WIFI_IP:-192.168.0.59}"
HOSTS_LINE="ZMQ_HOSTS=$OMEN_IP,$WIFI_IP"
if grep -q '^ZMQ_HOSTS=' "$CAM_ENV"; then
  sed -i.bak "s/^ZMQ_HOSTS=.*/$HOSTS_LINE/" "$CAM_ENV"
else
  printf '\n%s\n' "$HOSTS_LINE" >>"$CAM_ENV"
fi
if grep -q '^ZMQ_HOST=' "$CAM_ENV"; then
  sed -i.bak "s/^ZMQ_HOST=.*/ZMQ_HOST=$OMEN_IP/" "$CAM_ENV"
else
  printf 'ZMQ_HOST=%s\n' "$OMEN_IP" >>"$CAM_ENV"
fi
rm -f "$CAM_ENV.bak"
echo "Updated $CAM_ENV → $HOSTS_LINE (Ethernet first, Wi-Fi fallback)"

echo
echo "Next on the Omen: run omen\\setup_ethernet_link.bat then omen\\run_publish.bat"
echo "Then on Mac: restart dashboard (bash scripts/09_dashboard.sh) and record."
echo "Probe: nc -z -G 2 $OMEN_IP 5555 && echo zmq_ok"
