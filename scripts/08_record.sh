#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
require_leader_port_exists
load_camera

# Defaults — override via env: HF_USER, DATASET_NAME, NUM_EPISODES, TASK
HF_USER="${HF_USER:-local}"
DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
NUM_EPISODES="${NUM_EPISODES:-50}"
TASK="${TASK:-Pick pen and place it next to the keys}"
FPS="${CAMERA_FPS:-30}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
# Stay in the episode (and reset) until n / r / q. 24h is only a safety cap.
EPISODE_TIME_S="${EPISODE_TIME_S:-86400}"
RESET_TIME_S="${RESET_TIME_S:-86400}"

EXISTING="$(dataset_episode_count)"
RESUME_ARGS=(--resume=false)
RECORD_N="$NUM_EPISODES"
DS_ROOT="$(dataset_local_root)"
if [[ -f "$DS_ROOT/meta/info.json" ]]; then
  RECORD_N=$((NUM_EPISODES - EXISTING))
  if [[ "$RECORD_N" -lt 1 ]]; then
    echo "Dataset $(dataset_repo_id) already has $EXISTING episodes (target $NUM_EPISODES)."
    echo "Raise NUM_EPISODES to add more, e.g. NUM_EPISODES=60 bash scripts/08_record.sh"
    exit 0
  fi
  RESUME_ARGS=(--resume=true --dataset.root="$DS_ROOT")
  echo "Resuming $(dataset_repo_id) — $EXISTING saved, recording $RECORD_N more (target $NUM_EPISODES)."
fi

banner_both_arms
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID)  PID P=$FOLLOWER_P I=$FOLLOWER_I D=$FOLLOWER_D"
echo "Leader:   $LEADER_PORT ($LEADER_ID)"
echo "Camera:   $CAMERA_NAME backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-} index=${CAMERA_INDEX:-} depth=$CAMERA_USE_DEPTH"
echo "Dataset:  $(dataset_repo_id)  existing=$EXISTING  this_run=$RECORD_N  target=$NUM_EPISODES  push_to_hub=$PUSH_TO_HUB"
echo "Task:     $TASK"
echo "Episode:  until you press n (save) or r (redo)  — keep this terminal focused"
echo "Reset:    same; press n when the scene is ready for the next episode"
if [[ "$NUM_EPISODES" -lt 30 ]]; then
  echo "Note: ACT usually needs ~50 consistent episodes (same cameras, same task). $NUM_EPISODES is a smoke test."
fi
confirm_continue

CAMERAS_JSON="$(robot_cameras_json)"

run_logged record lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  "${FOLLOWER_PID_ARGS[@]}" \
  --robot.cameras="$CAMERAS_JSON" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --dataset.repo_id="$(dataset_repo_id)" \
  --dataset.num_episodes="$RECORD_N" \
  --dataset.single_task="$TASK" \
  --dataset.fps="$FPS" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.push_to_hub="$PUSH_TO_HUB" \
  --dataset.no_stamp=true \
  "${RESUME_ARGS[@]}" \
  --display_data=true
