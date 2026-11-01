"""Link-check proves ZMQ RGB+depth transmission (local fixture publisher)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import zmq

from so101.cli import build_parser, main
from so101.link_check import encode_fixture_payload, format_report, probe_zmq_stream


def _serve_fixture(port: int, stop: threading.Event, with_depth: bool = True) -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://127.0.0.1:{port}")
    # PUB sockets drop early messages before subscribers connect.
    time.sleep(0.2)
    payload = encode_fixture_payload(with_depth=with_depth)
    while not stop.is_set():
        sock.send_string(payload)
        time.sleep(0.05)
    sock.close(0)


def test_probe_pass_with_local_publisher(tmp_path: Path) -> None:
    port = 19555
    stop = threading.Event()
    thread = threading.Thread(target=_serve_fixture, args=(port, stop), daemon=True)
    thread.start()
    try:
        time.sleep(0.3)
        report = probe_zmq_stream(
            "127.0.0.1",
            port,
            seconds=1.0,
            expected_images=["wrist", "overhead"],
            expect_depth=True,
            depth_name="wrist",
            dump_dir=tmp_path / "dump",
            min_frames=2,
        )
        assert report.ok, format_report(report)
        assert "wrist" in report.image_keys
        assert "overhead" in report.image_keys
        assert "wrist" in report.depth_keys
        assert report.depth_stats["wrist"].valid_frac > 0.5
        assert (tmp_path / "dump" / "wrist.jpg").is_file()
        assert (tmp_path / "dump" / "overhead.jpg").is_file()
        assert (tmp_path / "dump" / "wrist_depth.png").is_file()
        text = format_report(report)
        assert "PASS" in text
    finally:
        stop.set()
        thread.join(timeout=2.0)


def test_probe_fails_when_depth_missing() -> None:
    port = 19556
    stop = threading.Event()
    thread = threading.Thread(
        target=_serve_fixture, args=(port, stop), kwargs={"with_depth": False}, daemon=True
    )
    thread.start()
    try:
        time.sleep(0.3)
        report = probe_zmq_stream(
            "127.0.0.1",
            port,
            seconds=0.8,
            expected_images=["wrist", "overhead"],
            expect_depth=True,
            min_frames=2,
        )
        assert not report.ok
        assert any("depth" in e.lower() for e in report.errors)
    finally:
        stop.set()
        thread.join(timeout=2.0)


def test_probe_fails_when_host_down() -> None:
    report = probe_zmq_stream("127.0.0.1", 19999, seconds=0.2, min_frames=1)
    assert not report.ok
    assert not report.tcp_ok
    assert report.frames == 0


def test_link_check_cli_parser() -> None:
    args = build_parser().parse_args(
        ["link-check", "--host", "192.168.1.10", "--seconds", "2", "--no-depth"]
    )
    assert args.host == "192.168.1.10"
    assert args.seconds == 2.0
    assert args.no_depth is True


def test_link_check_cli_against_fixture(capsys) -> None:
    port = 19557
    stop = threading.Event()
    thread = threading.Thread(target=_serve_fixture, args=(port, stop), daemon=True)
    thread.start()
    try:
        time.sleep(0.3)
        code = main(
            [
                "link-check",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--seconds",
                "1",
                "--min-frames",
                "2",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "PASS" in out
    finally:
        stop.set()
        thread.join(timeout=2.0)
