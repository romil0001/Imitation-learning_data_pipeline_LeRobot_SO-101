#!/usr/bin/env bash
# Watch/control the lab cameras off the home network.
# Does not start or modify the Omen publisher. See docs/remote_camera.md.
#
# `serve` alone is stdlib and can run on a public relay.
# `publish`, `--local`, and `--zmq-host` need the Mac lab env (OpenCV + pyzmq).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

needs_lab_python=0
if [[ "${1:-}" == "publish" ]]; then
  needs_lab_python=1
fi
for arg in "$@"; do
  case "$arg" in
    --local|--zmq-host|--zmq-host=*) needs_lab_python=1 ;;
  esac
done

if [[ -n "${PYTHON:-}" ]]; then
  exec "$PYTHON" -m so101 remote-cam "$@"
fi

if [[ "$needs_lab_python" -eq 1 ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck source=scripts/_common.sh
  source "$ROOT/scripts/_common.sh"
  require_conda_lerobot
  exec "$(conda_lerobot_python)" -m so101 remote-cam "$@"
fi

exec python3 -m so101 remote-cam "$@"
