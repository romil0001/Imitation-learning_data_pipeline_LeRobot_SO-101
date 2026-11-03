"""Unit tests for resilient camera slots in omen/publish_d405 (no hardware)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def publish_mod(monkeypatch: pytest.MonkeyPatch):
    """Import publish_d405 with stubbed cv2 / pyrealsense2 / zmq (Omen-only deps)."""
    stub_cv2 = types.ModuleType("cv2")
    stub_cv2.CAP_DSHOW = 700
    stub_cv2.CAP_ANY = 0
    stub_cv2.CAP_PROP_FRAME_WIDTH = 3
    stub_cv2.CAP_PROP_FRAME_HEIGHT = 4
    stub_cv2.CAP_PROP_FPS = 5
    stub_cv2.IMWRITE_JPEG_QUALITY = 1
    stub_cv2.IMREAD_UNCHANGED = -1

    class _Cap:
        def __init__(self, *a, **k):
            self._open = True

        def set(self, *a, **k):
            return True

        def isOpened(self):
            return self._open

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self._open = False

    stub_cv2.VideoCapture = _Cap
    stub_cv2.imencode = lambda *a, **k: (True, np.array([1, 2, 3], dtype=np.uint8))
    stub_cv2.imdecode = lambda *a, **k: None
    stub_cv2.resize = lambda img, wh: img
    stub_cv2.imwrite = lambda *a, **k: True

    stub_rs = types.ModuleType("pyrealsense2")
    stub_zmq = types.ModuleType("zmq")
    monkeypatch.setitem(sys.modules, "cv2", stub_cv2)
    monkeypatch.setitem(sys.modules, "pyrealsense2", stub_rs)
    monkeypatch.setitem(sys.modules, "zmq", stub_zmq)

    # Fresh import each time
    sys.modules.pop("omen.publish_d405", None)
    # Ensure repo root on path
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import omen.publish_d405 as mod

    return mod


def test_mark_offline_schedules_retry(publish_mod) -> None:
    mod = publish_mod
    slot = mod.CamSlot(kind="webcam", name="overhead", webcam_index=2, online=True)
    slot.cap = object()  # type: ignore[assignment]
    # release path expects .release
    class _C:
        def release(self):
            return None

    slot.cap = _C()  # type: ignore[assignment]
    before = 0.0
    mod.mark_offline(slot, "unplugged", retry_s=2.0)
    assert slot.online is False
    assert slot.cap is None
    assert slot.last_error == "unplugged"
    assert slot.next_retry_t > before


def test_grab_slot_reconnects_webcam(publish_mod, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = publish_mod
    slot = mod.CamSlot(
        kind="webcam",
        name="overhead",
        webcam_index=2,
        width=640,
        height=480,
        fps=15,
        online=False,
        next_retry_t=0.0,
    )
    bgr, depth = mod.grab_slot(slot, retry_s=0.1)
    assert depth is None
    assert bgr is not None
    assert slot.online is True
    assert bgr.shape == (480, 640, 3)
