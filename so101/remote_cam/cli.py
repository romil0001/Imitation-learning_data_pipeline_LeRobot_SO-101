"""`python -m so101 remote-cam serve|publish`."""

from __future__ import annotations

import os
import sys
import threading

from so101.remote_cam.publish import http_fetch_control, http_post_frame, make_zmq_source, publish_loop
from so101.remote_cam.relay import enable_tls, make_server, resolve_token


def _zmq_from_env() -> tuple[str, int]:
    from so101.config import load_cameras

    try:
        cameras = load_cameras()
    except (FileNotFoundError, ValueError, OSError):
        return "", 5555
    camera = cameras[0]
    if camera.backend != "zmq":
        return "", camera.zmq_port
    return camera.zmq_host, camera.zmq_port


def _resolve_zmq(host: str, port: int, use_env: bool) -> tuple[str, int]:
    env_host, env_port = _zmq_from_env() if use_env or not host or port <= 0 else ("", 5555)
    resolved_host = host or env_host
    resolved_port = port if port > 0 else env_port or 5555
    return resolved_host, resolved_port


def serve_command(args) -> int:
    supplied = (args.token or os.environ.get("CAMERA_RELAY_TOKEN", "")).strip()
    try:
        token = resolve_token(args.host, supplied)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    httpd = make_server(args.host, args.port, token)
    if args.tls_cert or args.tls_key:
        if not args.tls_cert or not args.tls_key:
            print("Pass both --tls-cert and --tls-key", file=sys.stderr)
            return 2
        enable_tls(httpd, args.tls_cert, args.tls_key)
    scheme = "https" if httpd.tls else "http"
    shown_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Remote camera  {scheme}://{shown_host}:{args.port}", flush=True)
    if args.host in {"0.0.0.0", "::"}:
        print("From another network, open that URL with this machine's public or Tailscale address.", flush=True)
    if supplied:
        print("Using the token from --token or CAMERA_RELAY_TOKEN.", flush=True)
    elif token:
        print(f"Generated token (viewers and the publisher need it): {token}", flush=True)
    else:
        print("No token — reachable only on localhost.", flush=True)

    stop = threading.Event()
    source = None
    publisher: threading.Thread | None = None
    zmq_host = (args.zmq_host or "").strip()
    if args.local or zmq_host:
        zmq_host, zmq_port = _resolve_zmq(zmq_host, args.zmq_port, use_env=True)
        if not zmq_host:
            print("Pass --zmq-host or set ZMQ_HOST in config/camera.env", file=sys.stderr)
            httpd.server_close()
            return 2
        print(f"Subscribing to Omen ZMQ at {zmq_host}:{zmq_port} (publisher code is unchanged).", flush=True)
        try:
            source = make_zmq_source(zmq_host, zmq_port, stop)
        except Exception as exc:
            print(f"Could not subscribe to {zmq_host}:{zmq_port}: {exc}", file=sys.stderr)
            httpd.server_close()
            return 2

        def fetch():
            return httpd.relay.control_copy()

        def post(camera, jpeg, known):
            return httpd.relay.ingest(camera, jpeg, known)

        publisher = threading.Thread(
            target=_run_publisher,
            args=(source, fetch, post, stop),
            name="remote-cam-zmq",
            daemon=True,
        )
        publisher.start()
    else:
        print("Waiting for a publisher on this relay.", flush=True)
        print(
            "On the Mac (same LAN as the Omen): python -m so101 remote-cam publish "
            f"--relay {scheme}://<this-host>:{args.port}",
            flush=True,
        )
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        stop.set()
        httpd.shutdown()
        if publisher is not None:
            publisher.join(timeout=2)
        httpd.server_close()
    return 0


def publish_command(args) -> int:
    token = (args.token or os.environ.get("CAMERA_RELAY_TOKEN", "")).strip()
    try:
        if token:
            resolve_token("127.0.0.1", token)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    host, port = _resolve_zmq((args.zmq_host or "").strip(), args.zmq_port, use_env=True)
    if not host:
        print("Pass --zmq-host or set ZMQ_HOST in config/camera.env", file=sys.stderr)
        return 2
    relay = args.relay.rstrip("/")
    print(f"Publishing Omen ZMQ {host}:{port} -> {relay}", flush=True)
    print("The Omen publisher is not modified. This process only subscribes.", flush=True)
    stop = threading.Event()
    source = make_zmq_source(host, port, stop)

    def fetch():
        return http_fetch_control(relay, token)

    def post(camera, jpeg, known):
        return http_post_frame(relay, token, camera, jpeg, known)

    try:
        publish_loop(source, fetch, post, stop)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        stop.set()
        source.close()
    return 0


def _run_publisher(source, fetch, post, stop) -> None:
    try:
        publish_loop(source, fetch, post, stop)
    finally:
        source.close()


def dispatch(args) -> int:
    if args.remote_cmd == "serve":
        return serve_command(args)
    if args.remote_cmd == "publish":
        return publish_command(args)
    print("Use: python -m so101 remote-cam serve|publish", file=sys.stderr)
    return 2
