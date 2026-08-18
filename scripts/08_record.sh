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
NUM_EPISODES="${NUM_EPISODES:-5}"
TASK="${TASK:-Pick up the object and place it in the bin}"
FPS="${CAMERA_FPS:-30}"

banner_both_arms
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "Leader:   $LEADER_PORT ($LEADER_ID)"
echo "Camera:   $CAMERA_NAME birdseye=${BIRDSEYE_NAME:-off} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-} index=${CAMERA_INDEX:-} depth=$CAMERA_USE_DEPTH"
echo "Dataset:  ${HF_USER}/${DATASET_NAME}  episodes=$NUM_EPISODES"
echo "Task:     $TASK"
confirm_continue

CAMERAS_JSON="$(robot_cameras_json)"

run_logged record lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS_JSON" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.single_task="$TASK" \
  --dataset.fps="$FPS" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true
