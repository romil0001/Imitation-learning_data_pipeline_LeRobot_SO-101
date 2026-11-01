#!/usr/bin/env bash
# Convenience wrapper: ssh omen [command...]
set -euo pipefail
exec ssh omen "$@"
