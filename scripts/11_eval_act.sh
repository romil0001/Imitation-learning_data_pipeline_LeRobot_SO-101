#!/usr/bin/env bash
# Run a trained policy on the follower. Interactive menu by default (TTY).
# Skip menu: INTERACTIVE=0 POLICY=act POLICY_PATH=... bash scripts/11_eval_act.sh
# Unscored timed: EVAL_MODE=timed DURATION=60  or PROTOCOL=
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"
# shellcheck source=scripts/_eval_prompt.sh
source "$ROOT/scripts/_eval_prompt.sh"
require_conda_lerobot
load_ports
require_follower_port_exists
load_camera

# Legacy env defaults before menu (menu overwrites when interactive)
DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
POLICY="${POLICY:-act}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}}"
POLICY_PATH="${POLICY_PATH:-$ROOT/outputs/train/$JOB_NAME}"
DURATION="${DURATION:-60}"
# PROTOCOL default only if unset; empty string means timed
if [[ -z "${PROTOCOL+x}" ]]; then
  PROTOCOL="$ROOT/config/eval/bag_clip_in_bowl.yaml"
fi
N_TRIALS="${N_TRIALS:-}"

prompt_eval_options

if [[ "$POLICY" == "act" || "$POLICY" == "act-jax" ]]; then
  ck="$POLICY_PATH"
  if [[ -f "$ck/best_val/model.pt" || -f "$ck/best_val/model.msgpack" ]]; then
    echo "Checkpoint: $ck/best_val"
  elif [[ -f "$ck/last/model.pt" || -f "$ck/last/model.msgpack" ]]; then
    echo "Checkpoint: $ck/last"
  elif [[ -f "$ck/model.pt" || -f "$ck/model.msgpack" ]]; then
    echo "Checkpoint: $ck"
  elif [[ -f "$ck" && "$ck" == *.pt ]]; then
    echo "Checkpoint: $ck"
  else
    # POLICY_PATH may already be .../last
    if [[ -f "$ck/model.pt" || -f "$ck/model.msgpack" ]]; then
      echo "Checkpoint: $ck"
    else
      echo "No ACT checkpoint at $POLICY_PATH (expected best_val/model.pt or last/model.pt)" >&2
      echo "Train first: bash scripts/10_train_act.sh" >&2
      exit 1
    fi
  fi
fi

echo "============================================================"
echo " $POLICY EVAL — FOLLOWER + D405 (Omen publisher)"
echo " - Power ON:  FOLLOWER bus + 12V + USB"
echo " - Leader can be OFF"
echo " - Omen: omen\\run_publish.bat must be streaming"
echo "============================================================"
echo "Policy:   $POLICY  ($POLICY_PATH)"
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "Camera:   ${CAMERA_NAMES:-$CAMERA_NAME} backend=$CAMERA_BACKEND zmq=${ZMQ_HOST:-}:${ZMQ_PORT:-}"
echo "Control:  ${CONTROL_MODE:-act25}  (act25=less jerk; reactive=twitchy; miss-grasp → bash scripts/18_dagger.sh)"
if [[ "${EVAL_MODE}" == "protocol" && -n "${PROTOCOL:-}" ]]; then
  echo "Protocol: $PROTOCOL"
  echo "Score:    y=success  n=fail after each trial (or during the run to stop early)"
  echo "Reset:    set the scene, then Enter — still prompted per trial"
  echo "Save:     y trials → data/datasets/${DATASET_NAME}_eval (disable: --no-save-successes)"
  if [[ -n "${N_TRIALS:-}" ]]; then
    echo "Trials:   $N_TRIALS per condition (override)"
  fi
else
  echo "Duration: ${DURATION}s (unscored)"
fi
confirm_continue

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
EVAL_ARGS=(
  "$(conda_lerobot_python)" -m so101 eval
  --dataset="$DATASET_NAME"
  --checkpoint="$POLICY_PATH"
  --policy="$POLICY"
  --yes
)
if [[ "$POLICY" == "hybrid" ]]; then
  EVAL_ARGS+=(--policy-base="${POLICY_BASE:-smolvla}")
  EVAL_ARGS+=(--grasp-config="${GRASP_CONFIG:-$ROOT/config/hybrid_grasp.yaml}")
fi
if [[ "${EVAL_MODE}" == "protocol" && -n "${PROTOCOL:-}" ]]; then
  EVAL_ARGS+=(--protocol="$PROTOCOL")
  if [[ -n "${N_TRIALS:-}" ]]; then
    EVAL_ARGS+=(--n-trials="$N_TRIALS")
  fi
else
  EVAL_ARGS+=(--duration="$DURATION")
fi
_eval_append_control_flags
if ((${#EXTRA_EVAL_FLAGS[@]})); then
  EVAL_ARGS+=("${EXTRA_EVAL_FLAGS[@]}")
fi
run_logged eval_act "${EVAL_ARGS[@]}"
