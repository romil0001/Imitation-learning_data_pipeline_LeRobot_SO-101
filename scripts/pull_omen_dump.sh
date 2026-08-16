#!/usr/bin/env bash
# Copy optional RGB/depth dump from the Omen (if you used --dump-dir there).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OMEN_USER="${OMEN_USER:-romil}"
OMEN_HOST="${OMEN_HOST:?Set OMEN_HOST to the Windows PC LAN IP}"
OMEN_DUMP="${OMEN_DUMP:-C:/Users/${OMEN_USER}/Developer/Lerobot setup/outputs/d405_dump}"
DEST="${DEST:-$ROOT/outputs/d405_dump}"
mkdir -p "$DEST"
# Windows OpenSSH: use a POSIX-ish path if you installed Git/OpenSSH.
# Override OMEN_DUMP if scp cannot see the Windows path.
scp -r "${OMEN_USER}@${OMEN_HOST}:${OMEN_DUMP}/." "$DEST/"
echo "Copied into $DEST"
