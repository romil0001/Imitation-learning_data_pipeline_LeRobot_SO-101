#!/usr/bin/env bash
# Run a trained ACT policy on the follower. Omen D405 publisher must be running.
# Leader can be unplugged. Follower 12V + USB ON.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

HF_USER="${HF_USER:-local}"
DATASET_NAME="${DATASET_NAME:-so101_d405_demo}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}}"
POLICY_PATH="${POLICY_PATH:-$ROOT/outputs/train/$JOB_NAME/checkpoints/last/pretrained_model}"
TASK="${TASK:-Pick pen and place it next to the keys}"
DURATION="${DURATION:-60}"

if [[ ! -d "$POLICY_PATH" && ! -f "$POLICY_PATH/config.json" ]]; then
  echo "No ACT checkpoint at $POLICY_PATH" >&2
  echo "Train first: bash scripts/10_train_act.sh" >&2
  exit 1
fi

if ! command -v lerobot-rollout >/dev/null 2>&1; then
  echo "lerobot-rollout missing. Re-run: bash env/setup_env.sh" >&2
  exit 1
fi

CAMERAS_JSON="$(robot_cameras_json)"

echo "============================================================"
echo " ACT EVAL — FOLLOWER + D405 (Omen publisher)"
echo " - Power ON:  FOLLOWER bus + 12V + USB"
echo " - Leader can be OFF"
echo " - Omen: omen\\run_publish.bat must be streaming"
echo "============================================================"
echo "Policy:   $POLICY_PATH"
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID)  PID P=$FOLLOWER_P I=$FOLLOWER_I D=$FOLLOWER_D"
echo "Camera:   $CAMERA_NAME backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-}"
echo "Task:     $TASK"
echo "Duration: ${DURATION}s"
confirm_continue

run_logged eval_act lerobot-rollout \
  --strategy.type=base \
  --policy.path="$POLICY_PATH" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  "${FOLLOWER_PID_ARGS[@]}" \
  --robot.cameras="$CAMERAS_JSON" \
  --task="$TASK" \
  --duration="$DURATION" \
  --display_data=true
