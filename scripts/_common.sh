#!/usr/bin/env bash
# Shared helpers for SO-101 bring-up scripts. Source me; do not execute.

project_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  printf '%s' "$here"
}

banner_follower_only() {
  cat <<'EOF'
============================================================
 FOLLOWER ONLY
 - Power ON:  FOLLOWER bus + 12V PSU + USB
 - Power OFF: LEADER bus (unplug USB and/or 12V)
============================================================
EOF
}

banner_leader_only() {
  cat <<'EOF'
============================================================
 LEADER ONLY
 - Power ON:  LEADER bus + 12V PSU + USB
 - Power OFF: FOLLOWER bus (unplug USB and/or 12V)
============================================================
EOF
}

banner_both_arms() {
  cat <<'EOF'
============================================================
 BOTH ARMS (teleop)
 - Power ON: FOLLOWER bus + 12V + USB
 - Power ON: LEADER bus + 12V + USB
============================================================
EOF
}

load_ports() {
  local root ports
  root="$(project_root)"
  ports="${PORTS_FILE:-$root/config/ports.env}"
  if [[ ! -f "$ports" ]]; then
    echo "Missing $ports — run scripts/01_find_port.sh first (or copy ports.env.example)." >&2
    return 1
  fi
  # shellcheck disable=SC1090
  set -a
  source "$ports"
  set +a
  FOLLOWER_ID="${FOLLOWER_ID:-my_follower_arm}"
  LEADER_ID="${LEADER_ID:-my_leader_arm}"
  if [[ -z "${FOLLOWER_PORT:-}" || -z "${LEADER_PORT:-}" ]]; then
    echo "FOLLOWER_PORT and LEADER_PORT must both be set in $ports" >&2
    return 1
  fi
  if [[ "$FOLLOWER_PORT" == CHANGE_ME* || "$LEADER_PORT" == CHANGE_ME* ]]; then
    echo "Replace CHANGE_ME_* placeholders in $ports" >&2
    return 1
  fi
  if [[ "$FOLLOWER_PORT" == "$LEADER_PORT" ]]; then
    echo "FOLLOWER_PORT and LEADER_PORT must be different devices (got $FOLLOWER_PORT)" >&2
    return 1
  fi
  return 0
}

require_follower_port_exists() {
  if [[ ! -e "$FOLLOWER_PORT" ]]; then
    echo "FOLLOWER_PORT '$FOLLOWER_PORT' does not exist — is the follower connected? Re-run scripts/01_find_port.sh if the device path changed." >&2
    return 1
  fi
  return 0
}

require_leader_port_exists() {
  if [[ ! -e "$LEADER_PORT" ]]; then
    echo "LEADER_PORT '$LEADER_PORT' does not exist — is the leader connected? Re-run scripts/01_find_port.sh if the device path changed." >&2
    return 1
  fi
  return 0
}

require_conda_lerobot() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda, then re-run env/setup_env.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  if ! conda activate lerobot; then
    echo "Failed to activate conda env 'lerobot'. Run bash env/setup_env.sh" >&2
    return 1
  fi
  if ! command -v lerobot-calibrate >/dev/null 2>&1 && ! python -c "import lerobot" >/dev/null 2>&1; then
    echo "lerobot not installed in conda env 'lerobot'. Run bash env/setup_env.sh" >&2
    return 1
  fi
  return 0
}

confirm_continue() {
  read -r -p "Confirm hardware matches the banner above, then press Enter (Ctrl-C to abort) "
}

load_camera() {
  local root cam
  root="$(project_root)"
  cam="${CAMERA_FILE:-$root/config/camera.env}"
  if [[ ! -f "$cam" ]]; then
    echo "Missing $cam — plug in the D405 and run scripts/07_find_cameras.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  set -a
  source "$cam"
  set +a
  CAMERA_NAME="${CAMERA_NAME:-front}"
  CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
  CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
  CAMERA_FPS="${CAMERA_FPS:-30}"
  CAMERA_USE_DEPTH="${CAMERA_USE_DEPTH:-false}"
  CAMERA_BACKEND="${CAMERA_BACKEND:-opencv}"
  CAMERA_INDEX="${CAMERA_INDEX:-0}"
  ZMQ_HOST="${ZMQ_HOST:-}"
  ZMQ_PORT="${ZMQ_PORT:-5555}"
  BIRDSEYE_NAME="${BIRDSEYE_NAME:-}"
  if [[ "$CAMERA_BACKEND" == "intelrealsense" ]]; then
    if [[ -z "${CAMERA_SERIAL:-}" || "$CAMERA_SERIAL" == CHANGE_ME* ]]; then
      echo "Set CAMERA_SERIAL in $cam (run scripts/07_find_cameras.sh)" >&2
      return 1
    fi
  elif [[ "$CAMERA_BACKEND" == "zmq" ]]; then
    if [[ -z "$ZMQ_HOST" || "$ZMQ_HOST" == CHANGE_ME* ]]; then
      echo "Set ZMQ_HOST in $cam to the HP Omen LAN IP (ipconfig on Windows)." >&2
      return 1
    fi
  elif [[ -z "${CAMERA_INDEX:-}" ]]; then
    echo "Set CAMERA_INDEX in $cam for opencv backend" >&2
    return 1
  fi
  return 0
}

# JSON fragment for --robot.cameras=...
robot_cameras_json() {
  local backend="${CAMERA_BACKEND:-opencv}"
  if [[ "$backend" == "intelrealsense" ]]; then
    local depth="${CAMERA_USE_DEPTH:-true}"
    printf '{ %s: {type: intelrealsense, serial_number_or_name: "%s", width: %s, height: %s, fps: %s, use_depth: %s, use_rgb: true} }' \
      "${CAMERA_NAME:-front}" \
      "${CAMERA_SERIAL}" \
      "${CAMERA_WIDTH:-640}" \
      "${CAMERA_HEIGHT:-480}" \
      "${CAMERA_FPS:-30}" \
      "$depth"
  elif [[ "$backend" == "zmq" ]]; then
    local json
    json="$(printf '{ %s: {type: zmq, server_address: "%s", port: %s, camera_name: "%s"}' \
      "${CAMERA_NAME:-front}" \
      "${ZMQ_HOST}" \
      "${ZMQ_PORT:-5555}" \
      "${CAMERA_NAME:-front}")"
    if [[ -n "${BIRDSEYE_NAME:-}" ]]; then
      json+="$(printf ', %s: {type: zmq, server_address: "%s", port: %s, camera_name: "%s"}' \
        "${BIRDSEYE_NAME}" \
        "${ZMQ_HOST}" \
        "${ZMQ_PORT:-5555}" \
        "${BIRDSEYE_NAME}")"
    fi
    printf '%s }' "$json"
  else
    printf '{ %s: {type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s} }' \
      "${CAMERA_NAME:-front}" \
      "${CAMERA_INDEX:-0}" \
      "${CAMERA_WIDTH:-1280}" \
      "${CAMERA_HEIGHT:-720}" \
      "${CAMERA_FPS:-15}"
  fi
}

session_log() {
  local root
  root="$(project_root)"
  mkdir -p "$root/outputs"
  printf '%s' "$root/outputs/session.log"
}

# Usage: run_logged label command [args...]
run_logged() {
  local label="$1"
  shift
  local log
  log="$(session_log)"
  {
    echo
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') $label ====="
    PYTHONUNBUFFERED=1 "$@"
  } 2>&1 | tee -a "$log"
}
