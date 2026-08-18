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

HF_USER="${HF_USER:-local}"
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
echo "Dataset:  $(dataset_repo_id)  ($(dataset_episode_count) episodes)"
echo "Follower: $FOLLOWER_PORT  PID P=$FOLLOWER_P I=$FOLLOWER_I D=$FOLLOWER_D"
confirm_continue

run_logged replay lerobot-replay \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  "${FOLLOWER_PID_ARGS[@]}" \
  --dataset.repo_id="$(dataset_repo_id)" \
  --dataset.root="$(dataset_local_root)" \
  --dataset.episode="$EPISODE" \
  --play_sounds=false
