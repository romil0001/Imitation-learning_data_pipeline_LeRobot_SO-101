"""Allow `python -m so101.remote_cam serve|publish`."""

from __future__ import annotations

import sys

from so101.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["remote-cam", *sys.argv[1:]]))
