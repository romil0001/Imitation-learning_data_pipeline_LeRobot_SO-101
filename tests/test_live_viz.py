"""Live joint publish/read for the operator dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101.live_viz import publish_live_joints, read_live_joints


def test_publish_and_read_live_joints(tmp_path: Path) -> None:
    path = tmp_path / "live_joints.json"
    state = np.arange(6, dtype=np.float32)
    action = state + 10
    publish_live_joints(state, action, path=path)
    data = read_live_joints(path=path, max_age_s=5.0)
    assert data is not None
    assert data["fresh"] is True
    assert data["names"][0] == "shoulder_pan"
    assert data["state"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert data["action"][0] == 10.0
