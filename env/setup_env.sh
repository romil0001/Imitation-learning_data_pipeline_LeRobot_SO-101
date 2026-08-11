#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEROBOT_DIR="$ROOT/third_party/lerobot"
REPO_URL="https://github.com/huggingface/lerobot.git"

if ! command -v conda >/dev/null 2>&1; then
  echo "Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx lerobot; then
  echo "Conda env 'lerobot' already exists — reusing."
else
  conda create -y -n lerobot python=3.10
fi
conda activate lerobot

mkdir -p "$ROOT/third_party"
if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
  git clone "$REPO_URL" "$LEROBOT_DIR"
else
  echo "Found existing clone at $LEROBOT_DIR"
fi

cd "$LEROBOT_DIR"
pip install -e ".[feetech]"

echo
echo "Setup complete. Next:"
echo "  conda activate lerobot"
echo "  bash scripts/01_find_port.sh"
