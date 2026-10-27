"""HTTP relay: one publisher uplink, many viewers, shared camera control.

Viewers log in with a shared token (cookie). The home publisher presents the
same token as ``Authorization: Bearer``. The server stores JPEG bytes as the
publisher sent them; it does not open a route to the Omen.
"""

from __future__ import annotations

import base64
import hmac
import json
import secrets
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from so101.remote_cam.control import CameraControl, merge_control, sanitize_view

# Gray stand-in used when OpenCV is not installed on the relay machine.
_FALLBACK_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABQODxIPDRQSEBIXFRQYHjIhHhwcHj0sLiQySUBMS0dARkVQWnNiUFVtVkVGZIhlbXd7gYKBTmCNl4x9lnN+gXz/2wBDARUXFx4aHjshITt8U0ZTfHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHx8fHz/wAARCAAgADADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDmaKKKACiiigAooooAKKKKACiiigAooooA/9k="
)

_MAX_BODY = 2_000_000

_VIEWER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>SO-101 camera</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 16px/1.4 ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; }
  main { max-width: 760px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 1.25rem; margin: 0 0 4px; }
  p { margin: 0 0 12px; color: #bbb; }
  img { width: 100%; background: #000; border-radius: 8px; min-height: 180px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
  button, a.btn { background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 8px; padding: 10px 14px; font: inherit; text-decoration: none; }
  button.on { background: #1d4ed8; border-color: #1d4ed8; }
  label { display: grid; gap: 4px; margin: 8px 0; }
  input[type=range] { width: 100%; }
  .pad { display: grid; grid-template-columns: repeat(3, 64px); gap: 6px; justify-content: start; }
  #state { font-weight: 600; }
</style>
</head>
<body>
<main>
  <h1>SO-101 camera</h1>
  <p>Digital pan, zoom, and rotation. The cameras on the Omen stay fixed; this page only changes the stream you receive.</p>
  <img id="live" alt="Live camera stream" src="/mjpeg">
  <p><span id="state">connecting</span> · <span id="meta"></span></p>
  <div class="row" id="views"></div>
  <label>Zoom <input id="zoom" type="range" min="1" max="4" step="0.1" value="1"></label>
  <label>Quality <input id="quality" type="range" min="20" max="95" step="1" value="70"></label>
  <label>Max frames per second <input id="fps" type="range" min="0.5" max="30" step="0.5" value="10"></label>
  <div class="pad">
    <span></span><button type="button" id="up">Up</button><span></span>
    <button type="button" id="left">Left</button><button type="button" id="reset">Reset</button><button type="button" id="right">Right</button>
    <span></span><button type="button" id="down">Down</button><span></span>
  </div>
  <div class="row">
    <button type="button" id="rotate">Rotate</button>
    <button type="button" id="pause">Pause</button>
    <a class="btn" href="/snapshot.jpg" target="_blank" rel="noopener">Snapshot</a>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let control = {view: "wrist", zoom: 1, pan_x: 0, pan_y: 0, quality: 70, max_fps: 10, rotate: 0, paused: false};

async function send(patch) {
  const res = await fetch("/api/control", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(patch),
  });
  if (res.status === 401) location.reload();
  if (res.ok) {
    const data = await res.json();
    control = data.control;
    paint();
  }
}

function paint() {
  $("zoom").value = control.zoom;
  $("quality").value = control.quality;
  $("fps").value = control.max_fps;
  $("pause").textContent = control.paused ? "Resume" : "Pause";
  $("rotate").textContent = "Rotate " + control.rotate + "°";
  for (const button of document.querySelectorAll("#views button")) {
    button.className = button.dataset.view === control.view ? "on" : "";
  }
}

function views(names) {
  const bar = $("views");
  const wanted = names.length ? names : ["wrist", "overhead", "wrist_depth"];
  const current = [...bar.querySelectorAll("button")].map((b) => b.dataset.view).join(",");
  if (current === wanted.join(",")) return;
  bar.replaceChildren();
  for (const name of wanted) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.view = name;
    button.textContent = name.replaceAll("_", " ");
    button.onclick = () => send({view: name});
    bar.appendChild(button);
  }
}

async function refresh() {
  const res = await fetch("/api/status");
  if (res.status === 401) location.reload();
  if (!res.ok) return;
  const data = await res.json();
  const editing = document.activeElement && document.activeElement.type === "range";
  control = data.control;
  $("state").textContent = data.publisher_online ? "Live" : "Waiting for the camera";
  const age = data.last_frame_age_s == null ? "no frames yet" : data.last_frame_age_s + "s since last frame";
  $("meta").textContent = (data.active_view || control.view) + " · " + data.frames + " frames · " + age;
  views(data.cameras || []);
  if (!editing) paint();
}

function bindRange(id, key) {
  let timer = 0;
  $(id).addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => send({[key]: Number($(id).value)}), 120);
  });
}
bindRange("zoom", "zoom");
bindRange("quality", "quality");
bindRange("fps", "max_fps");

function nudge(dx, dy) {
  send({
    pan_x: Math.max(-1, Math.min(1, Number(control.pan_x || 0) + dx)),
    pan_y: Math.max(-1, Math.min(1, Number(control.pan_y || 0) + dy)),
  });
}
$("up").onclick = () => nudge(0, -0.2);
$("down").onclick = () => nudge(0, 0.2);
$("left").onclick = () => nudge(-0.2, 0);
$("right").onclick = () => nudge(0.2, 0);
$("reset").onclick = () => send({zoom: 1, pan_x: 0, pan_y: 0, rotate: 0});
$("rotate").onclick = () => {
  const order = [0, 90, 180, 270];
  const next = order[(order.indexOf(Number(control.rotate)) + 1) % order.length];
  send({rotate: next});
};
$("pause").onclick = () => send({paused: !control.paused});
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


def placeholder_jpeg() -> bytes:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _FALLBACK_JPEG
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:] = (24, 24, 24)
    cv2.putText(
        image,
        "waiting for camera",
        (150, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 220, 220),
        2,
    )
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return _FALLBACK_JPEG
    return buf.tobytes()


def resolve_token(host: str, token: str) -> str:
    """Use the given token. Public binds with an empty token get a generated one."""
    cleaned = token.strip()
    if any(char in cleaned for char in "\r\n; "):
        raise ValueError("token must not contain spaces, semicolons, or newlines")
    if cleaned:
        return cleaned
    if host in {"127.0.0.1", "localhost", "::1"}:
        return ""
    return secrets.token_urlsafe(24)


def _token_matches(expected: str, presented: str) -> bool:
    if not expected:
        return True
    if not presented:
        return False
    return hmac.compare_digest(presented.encode(), expected.encode())


class RelayState:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frames: dict[str, bytes] = {}
        self._known: list[str] = []
        self._control = CameraControl()
        self._last = 0.0
        self._count = 0
        self._seq = 0

    def ingest(self, camera: str, jpeg: bytes, known: list[str] | None = None) -> CameraControl:
        camera = sanitize_view(camera)
        if not jpeg.startswith(b"\xff\xd8") or len(jpeg) > _MAX_BODY:
            raise ValueError("frame must be a JPEG under 2 MB")
        cleaned: list[str] = []
        for name in known or []:
            try:
                item = sanitize_view(name)
            except ValueError:
                continue
            if item not in cleaned:
                cleaned.append(item)
        with self._cond:
            self._frames[camera] = jpeg
            self._last = time.monotonic()
            self._count += 1
            self._seq += 1
            if cleaned:
                self._known = cleaned
            elif camera not in self._known:
                self._known.append(camera)
            self._cond.notify_all()
            return self._control

    def update_control(self, patch: dict) -> CameraControl:
        with self._cond:
            self._control = merge_control(self._control, patch)
            self._cond.notify_all()
            return self._control

    def control_copy(self) -> CameraControl:
        with self._cond:
            return self._control

    def latest_jpeg(self) -> bytes:
        with self._cond:
            return self._frames.get(self._control.view, b"")

    def wait_jpeg(self, last_seq: int, timeout: float) -> tuple[int, bytes]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout=timeout)
            return self._seq, self._frames.get(self._control.view, b"")

    def status(self) -> dict:
        with self._cond:
            age = None if self._last <= 0 else round(time.monotonic() - self._last, 3)
            online = self._last > 0 and (time.monotonic() - self._last) < 3.0
            active = self._control.view if self._control.view in self._frames else None
            return {
                "publisher_online": online,
                "last_frame_age_s": age,
                "frames": self._count,
                "cameras": list(self._known),
                "active_view": active,
                "control": self._control.to_dict(),
            }


class CameraRelayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], token: str) -> None:
        self.token = token
        self.tls = False
        self.relay = RelayState()
        self.placeholder = placeholder_jpeg()
        super().__init__(server_address, RelayHandler)


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server: CameraRelayServer

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def _route(self, method: str) -> None:
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/health" and method == "GET":
                self._json(200, {"ok": True})
                return
            if path == "/login" and method == "POST":
                self._login()
                return
            if path == "/" and method == "GET":
                if not self._authorized():
                    self._html(200, _login_page(""))
                else:
                    self._html(200, _VIEWER)
                return
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            if path == "/api/status" and method == "GET":
                self._json(200, self.server.relay.status())
                return
            if path == "/api/control" and method == "GET":
                self._json(200, {"control": self.server.relay.control_copy().to_dict()})
                return
            if path == "/api/control" and method == "POST":
                self._json(200, {"control": self.server.relay.update_control(self._json_body()).to_dict()})
                return
            if path == "/api/frame" and method == "POST":
                self._ingest()
                return
            if path == "/snapshot.jpg" and method == "GET":
                self._snapshot()
                return
            if path == "/mjpeg" and method == "GET":
                self._mjpeg()
                return
            self._json(404, {"error": "not found"})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return

    def _authorized(self) -> bool:
        expected = self.server.token
        if not expected:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and _token_matches(expected, header[7:].strip()):
            return True
        for part in self.headers.get("Cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == "cam_token" and _token_matches(expected, value):
                return True
        return False

    def _body(self) -> bytes:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length < 0 or length > _MAX_BODY:
            raise ValueError("body too large")
        if length == 0:
            return b""
        return self.rfile.read(length)

    def _json_body(self) -> dict:
        raw = self._body()
        if not raw:
            raise ValueError("expected a JSON object")
        try:
            data = json.loads(raw.decode())
        except json.JSONDecodeError as exc:
            raise ValueError("expected a JSON object") from exc
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _login(self) -> None:
        form = urllib.parse.parse_qs(self._body().decode(errors="replace"))
        presented = (form.get("token") or [""])[0]
        if not _token_matches(self.server.token, presented):
            self._html(401, _login_page("Wrong token."))
            return
        self.send_response(302)
        self.send_header("Location", "/")
        if self.server.token:
            flags = "HttpOnly; Path=/; SameSite=Strict"
            if self.server.tls:
                flags += "; Secure"
            self.send_header("Set-Cookie", f"cam_token={self.server.token}; {flags}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _ingest(self) -> None:
        camera = self.headers.get("X-Camera", "")
        known_raw = self.headers.get("X-Known-Cameras", "")
        known = [part.strip() for part in known_raw.split(",") if part.strip()] if known_raw else None
        control = self.server.relay.ingest(camera, self._body(), known)
        self._json(200, {"ok": True, "control": control.to_dict()})

    def _snapshot(self) -> None:
        frame = self.server.relay.latest_jpeg() or self.server.placeholder
        self._bytes(200, "image/jpeg", frame)

    def _mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                last, frame = self.server.relay.wait_jpeg(last, timeout=1.0)
                payload = frame or self.server.placeholder
                chunk = (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(payload)).encode()
                    + b"\r\n\r\n"
                    + payload
                    + b"\r\n"
                )
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def _json(self, code: int, payload: dict) -> None:
        self._bytes(code, "application/json", json.dumps(payload).encode())

    def _html(self, code: int, page: str) -> None:
        self._bytes(code, "text/html; charset=utf-8", page.encode())

    def _bytes(self, code: int, content_type: str, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(payload)


def _login_page(message: str) -> str:
    note = f"<p>{message}</p>" if message else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>SO-101 camera</title>
<style>
  body {{ margin: 0; font: 16px/1.4 ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; }}
  main {{ max-width: 420px; margin: 10vh auto; padding: 16px; }}
  input, button {{ font: inherit; padding: 10px; width: 100%; box-sizing: border-box; }}
  button {{ margin-top: 8px; }}
</style>
</head>
<body>
<main>
  <h1>SO-101 camera</h1>
  {note}
  <form method="post" action="/login">
    <label>Token <input type="password" name="token" autocomplete="current-password" autofocus></label>
    <button type="submit">Open stream</button>
  </form>
</main>
</body>
</html>
"""


def make_server(host: str, port: int, token: str) -> CameraRelayServer:
    return CameraRelayServer((host, port), resolve_token(host, token))


def serve_background(host: str, port: int, token: str) -> tuple[CameraRelayServer, threading.Thread]:
    httpd = make_server(host, port, token)
    thread = threading.Thread(
        target=httpd.serve_forever,
        kwargs={"poll_interval": 0.1},
        name="remote-cam",
        daemon=True,
    )
    thread.start()
    return httpd, thread


def enable_tls(httpd: CameraRelayServer, certfile: str, keyfile: str) -> None:
    import ssl

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    httpd.tls = True
