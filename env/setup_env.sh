#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEROBOT_DIR="$ROOT/third_party/lerobot"
REPO_URL="https://github.com/huggingface/lerobot.git"
LEROBOT_REF="v0.6.1"

if ! command -v conda >/dev/null 2>&1; then
  echo "Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx lerobot; then
  echo "Conda env 'lerobot' already exists — reusing."
else
  conda create -y -n lerobot python=3.12
fi
conda activate lerobot

mkdir -p "$ROOT/third_party"
if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
  git clone --branch "$LEROBOT_REF" --depth 1 "$REPO_URL" "$LEROBOT_DIR"
else
  echo "Found existing clone at $LEROBOT_DIR — pinning to $LEROBOT_REF"
  git -C "$LEROBOT_DIR" fetch --depth 1 origin tag "$LEROBOT_REF"
  git -C "$LEROBOT_DIR" checkout "$LEROBOT_REF"
fi

cd "$LEROBOT_DIR"
pip install -e ".[feetech,hardware,intelrealsense,core_scripts,training,smolvla]"
pip install pyzmq fastapi uvicorn pytest
pip install -e "$ROOT"

# Apply local patches (leader IDs, ZMQ RGB). Safe to re-run.
for PATCH in "$ROOT/patches/so_leader_ids_7_12.patch" "$ROOT/patches/zmq_bgr_to_rgb.patch" "$ROOT/patches/so_follower_configure_retries.patch" "$ROOT/patches/record_save_before_reset.patch"; do
  if [[ -f "$PATCH" ]]; then
    if git -C "$LEROBOT_DIR" apply --check "$PATCH" 2>/dev/null; then
      git -C "$LEROBOT_DIR" apply "$PATCH"
      echo "Applied $(basename "$PATCH")."
    else
      echo "$(basename "$PATCH") already applied or not needed."
    fi
  fi
done

echo
echo "Setup complete. Next:"
echo "  conda activate lerobot"
echo "  bash scripts/01_find_port.sh"
echo "  python -m so101 --help"
