#!/usr/bin/env bash
# Collect teleop episodes into data/datasets/<name>/ (Mac). Omen publisher must be running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
require_leader_port_exists
load_camera

DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
NUM_EPISODES="${NUM_EPISODES:-50}"
TASK="${TASK:-Pick pen and place it next to the keys}"

EXISTING="$(dataset_episode_count)"
banner_both_arms
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID)  PID P=$FOLLOWER_P I=$FOLLOWER_I D=$FOLLOWER_D"
echo "Leader:   $LEADER_PORT ($LEADER_ID)"
echo "Camera:   ${CAMERA_NAMES:-$CAMERA_NAME} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-}"
echo "Dataset:  $(dataset_local_root)  existing=$EXISTING  target=$NUM_EPISODES"
echo "Task:     $TASK"
echo "Episode:  n = save, r = redo, q = quit  — keep this terminal focused"
if [[ "$NUM_EPISODES" -lt 30 ]]; then
  echo "Note: ACT usually needs ~50 consistent episodes. $NUM_EPISODES is a smoke test."
fi
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
run_logged record "$(conda_lerobot_python)" -m so101 record \
  --dataset="$DATASET_NAME" \
  --num-episodes="$NUM_EPISODES" \
  --task="$TASK" \
  --yes
