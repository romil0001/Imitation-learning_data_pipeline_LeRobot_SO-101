"""Publish live joint angles for the operator dashboard (no Rerun)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from so101.config import project_root
from so101.dataset.schema import JOINT_NAMES

LIVE_JOINTS_PATH = project_root() / "outputs" / "live_joints.json"


def publish_live_joints(
    state: Sequence[float] | np.ndarray,
    action: Sequence[float] | np.ndarray,
    *,
    path: Path | None = None,
) -> None:
    """Atomically write latest follower state + leader action for the dashboard."""
    out = path or LIVE_JOINTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "names": list(JOINT_NAMES),
        "state": [float(x) for x in np.asarray(state, dtype=np.float64).reshape(-1)],
        "action": [float(x) for x in np.asarray(action, dtype=np.float64).reshape(-1)],
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(out)


def read_live_joints(path: Path | None = None, max_age_s: float = 2.0) -> dict | None:
    p = path or LIVE_JOINTS_PATH
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ts = float(data.get("ts") or 0.0)
    age = time.time() - ts
    data["age_s"] = round(age, 3)
    data["fresh"] = age <= max_age_s
    return data
