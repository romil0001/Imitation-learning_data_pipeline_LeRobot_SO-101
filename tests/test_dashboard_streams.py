"""Dashboard stream catalog covers every RGB + depth view from camera.env."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import app as dash  # noqa: E402


def test_stream_catalog_includes_rgb_and_depth_for_triple_cam(monkeypatch) -> None:
    monkeypatch.setitem(dash.CFG, "camera_names", "wrist,overhead,base")
    monkeypatch.setitem(dash.CFG, "camera_name", "wrist")
    monkeypatch.setitem(dash.CFG, "use_depth", True)
    streams = dash.stream_catalog(live_ids=["wrist", "base_depth"])
    ids = [s["id"] for s in streams]
    assert ids[:5] == ["wrist", "overhead", "base", "wrist_depth", "base_depth"]
    by_id = {s["id"]: s for s in streams}
    assert by_id["wrist"]["live"] is True
    assert by_id["overhead"]["live"] is False
    assert by_id["base_depth"]["kind"] == "depth"
    assert by_id["wrist"]["url"] == "/mjpeg/wrist"


def test_depth_camera_names_prefers_wrist_and_base(monkeypatch) -> None:
    monkeypatch.setitem(dash.CFG, "use_depth", True)
    assert dash.depth_camera_names(["wrist", "overhead", "base"]) == ["wrist", "base"]
    monkeypatch.setitem(dash.CFG, "use_depth", False)
    assert dash.depth_camera_names(["wrist", "base"]) == []


def test_fresh_stream_ids_drops_stale_cached_frames(monkeypatch) -> None:
    """Once Omen drops, tiles must stop reporting live from leftover JPEGs."""
    now = time.time()
    monkeypatch.setattr(
        dash,
        "_frame_ts",
        {"wrist": now, "overhead": now - 10.0, "base": now - 0.2},
    )
    assert dash.fresh_stream_ids(max_age_s=2.0) == ["base", "wrist"]


def test_frame_is_fresh_false_when_missing_or_stale(monkeypatch) -> None:
    now = time.time()
    monkeypatch.setattr(dash, "_frame_ts", {"wrist": now - 5.0})
    assert dash.frame_is_fresh("wrist", max_age_s=2.0) is False
    assert dash.frame_is_fresh("missing", max_age_s=2.0) is False
    monkeypatch.setattr(dash, "_frame_ts", {"wrist": now})
    assert dash.frame_is_fresh("wrist", max_age_s=2.0) is True
