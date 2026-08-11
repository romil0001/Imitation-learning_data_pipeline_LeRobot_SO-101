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

require_conda_lerobot() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda, then re-run env/setup_env.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate lerobot
  if ! command -v lerobot-calibrate >/dev/null 2>&1 && ! python -c "import lerobot" >/dev/null 2>&1; then
    echo "lerobot not installed in conda env 'lerobot'. Run bash env/setup_env.sh" >&2
    return 1
  fi
  return 0
}

confirm_continue() {
  read -r -p "Confirm hardware matches the banner above, then press Enter (Ctrl-C to abort) "
}
