"""Unit tests for Omen publisher helpers (no RealSense hardware required)."""

from __future__ import annotations

from unittest import mock

import pytest

# Import module under test; pyrealsense2 is listed in omen/requirements but may be
# absent on the Mac/CI image. Stub it so helpers still import.
if "pyrealsense2" not in __import__("sys").modules:
    import sys
    import types

    rs_stub = types.ModuleType("pyrealsense2")
    rs_stub.pipeline = object
    rs_stub.config = object
    rs_stub.align = object
    rs_stub.stream = types.SimpleNamespace(color=0, depth=1)
    rs_stub.format = types.SimpleNamespace(bgr8=0, z16=1)
    rs_stub.camera_info = types.SimpleNamespace(name="name", serial_number="serial_number")
    rs_stub.context = object
    sys.modules["pyrealsense2"] = rs_stub

import omen.publish_d405 as pub  # noqa: E402


def test_webcam_hint_labels_known_resolutions() -> None:
    assert "laptop" in pub._webcam_hint(1280, 720).lower()
    assert "xwf" in pub._webcam_hint(1920, 1080).lower() or "usb" in pub._webcam_hint(1920, 1080).lower()
    assert "d405" in pub._webcam_hint(640, 480).lower()
    assert pub._webcam_hint(800, 600) == ""


def test_resolve_webcam_index_prefers_usb_over_laptop() -> None:
    fake = [(0, 1280, 720), (1, 640, 480), (2, 1920, 1080)]
    with mock.patch.object(pub, "list_webcams", return_value=fake):
        assert pub.resolve_webcam_index(-1) == 2
        assert pub.resolve_webcam_index(1) == 1


def test_resolve_webcam_index_errors_when_empty() -> None:
    with mock.patch.object(pub, "list_webcams", return_value=[]):
        with pytest.raises(RuntimeError, match="No OpenCV"):
            pub.resolve_webcam_index(-1)


def test_lan_ipv4_hints_filters_loopback() -> None:
    with mock.patch.object(pub.socket, "gethostname", return_value="omen"):
        with mock.patch.object(
            pub.socket,
            "getaddrinfo",
            return_value=[
                (None, None, None, None, ("127.0.0.1", 0)),
                (None, None, None, None, ("192.168.1.42", 0)),
            ],
        ):
            with mock.patch("omen.publish_d405.socket.socket") as sock_cls:
                sock = sock_cls.return_value
                sock.getsockname.return_value = ("192.168.1.42", 0)
                ips = pub.lan_ipv4_hints()
    assert "127.0.0.1" not in ips
    assert "192.168.1.42" in ips


def test_run_self_check_ok_when_devices_present() -> None:
    with mock.patch.object(pub, "list_realsense_devices", return_value=[("Intel D405", "123")]):
        with mock.patch.object(pub, "list_webcams", return_value=[(2, 1920, 1080)]):
            with mock.patch.object(pub, "probe_zmq_bind"):
                with mock.patch.object(pub, "lan_ipv4_hints", return_value=["192.168.1.42"]):
                    assert pub.run_self_check("0.0.0.0", 5555) == 0


def test_run_self_check_fails_without_realsense() -> None:
    with mock.patch.object(pub, "list_realsense_devices", return_value=[]):
        with mock.patch.object(pub, "list_webcams", return_value=[(2, 1920, 1080)]):
            with mock.patch.object(pub, "probe_zmq_bind"):
                with mock.patch.object(pub, "lan_ipv4_hints", return_value=[]):
                    assert pub.run_self_check("0.0.0.0", 5555) == 1
