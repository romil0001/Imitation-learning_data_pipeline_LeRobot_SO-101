#!/usr/bin/env bash
# Replay a recorded episode on the FOLLOWER only (no leader, no camera).
# Usage: EPISODE=0 bash scripts/12_replay.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists

DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
EPISODE="${EPISODE:-0}"
require_local_dataset
require_saved_episode "$EPISODE"

echo "============================================================"
echo " REPLAY episode $EPISODE on FOLLOWER"
echo " - Power ON: FOLLOWER USB + 12V"
echo " - Leader can be OFF"
echo " - Clear the workspace — the arm will move"
echo "============================================================"
echo "Dataset:  $(dataset_local_root)  ($(dataset_episode_count) episodes)"
echo "Follower: $FOLLOWER_PORT  PID P=$FOLLOWER_P I=$FOLLOWER_I D=$FOLLOWER_D"
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_logged replay "$(conda_lerobot_python)" -m so101 replay \
  --dataset="$DATASET_NAME" \
  --episode="$EPISODE" \
  --yes
