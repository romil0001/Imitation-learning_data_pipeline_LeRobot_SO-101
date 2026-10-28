# Remote camera

Watch the wrist and overhead cameras from a phone or laptop that is **not** on the home Wi-Fi, and control what the stream shows. The HP Omen publisher (`omen/publish_d405.py`) is not modified and does not need a public port. It keeps publishing ZMQ on the LAN exactly as it does today.

The cameras are fixed (no pan-tilt motors). Control is applied to the picture after it leaves the Omen:

- switch view: wrist, overhead, or wrist depth (when the Omen is publishing depth)
- digital pan, zoom (1×–4×), and rotation (0/90/180/270)
- JPEG quality and max frame rate, so a cellular link stays usable
- pause, and save a snapshot

## Why two processes

Home routers block inbound connections. A viewer on cellular data cannot open `tcp://<omen-lan-ip>:5555`. The Mac, which is already on that LAN, subscribes to the Omen and makes an **outbound** connection to a relay. Viewers talk only to the relay.

```text
D405 + overhead USB
        │
        ▼
Omen  publish_d405.py  :5555     (unchanged, LAN only)
        │  ZMQ
        ▼
Mac   remote-cam publish          outbound HTTP(S)
        │
        ▼
Relay remote-cam serve            public address or Tailscale
        │
        ▼
Phone / laptop anywhere           browser, token login
```

You can also run `serve` and the ZMQ subscription in one process on the Mac, and reach that process over [Tailscale](https://tailscale.com/) (or a router port forward). Split mode is what you want when the relay is a small VPS and the Mac stays behind the home router.

## One process on the Mac (Tailscale or port forward)

On the Omen, leave `omen\run_publish.bat` running.

On the Mac:

```bash
# token is a password. Prefer CAMERA_RELAY_TOKEN over putting it in the shell history.
export CAMERA_RELAY_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
bash scripts/16_remote_cam.sh serve --host 0.0.0.0 --port 8090 --local
```

`--local` reads `ZMQ_HOST` from `config/camera.env`. Or pass `--zmq-host <omen-lan-ip>` yourself.

Open `http://<mac-tailscale-ip>:8090` from the phone (Wi-Fi or cellular, as long as Tailscale is connected). Log in with the token. The stream is an MJPEG image; the buttons under it are the camera controls.

## Split mode (home Mac behind NAT, relay on a public machine)

On the public machine (no Omen, no cameras, no PyTorch):

```bash
export CAMERA_RELAY_TOKEN="<same token as the Mac>"
python3 -m so101 remote-cam serve --host 0.0.0.0 --port 8090 \
  --tls-cert /path/fullchain.pem --tls-key /path/privkey.pem
```

TLS is optional and strongly recommended on a public address. Without the cert flags the relay speaks plain HTTP. A missing token on a public bind is generated and printed once; localhost with no token is left open for tests only.

On the Mac (same LAN as the Omen):

```bash
export CAMERA_RELAY_TOKEN="<same token>"
bash scripts/16_remote_cam.sh publish \
  --relay https://<relay-host>:8090
```

`publish` reads `ZMQ_HOST` / `ZMQ_PORT` from `config/camera.env` unless you pass `--zmq-host`. It subscribes as a normal ZMQ client. It does not bind a port on the Omen and it does not change exposure, laser, or USB settings.

Then open `https://<relay-host>:8090` from anywhere and log in.

## What stays on the LAN

| Piece | Where it runs | Reaches the internet? |
|---|---|---|
| `omen/publish_d405.py` | Omen | No. LAN ZMQ only |
| Arm USB, record, train, eval | Mac | No |
| Dashboard `:8080` | Mac | No, still `127.0.0.1` |
| `remote-cam publish` | Mac | Outbound to the relay only |
| `remote-cam serve` | Mac or a VPS | Yes, this is the URL you open |

The dashboard is unchanged. Recording and eval keep using ZMQ directly and do not go through the relay.

## Commands

```bash
python -m so101 remote-cam serve --help
python -m so101 remote-cam publish --help
```

Environment:

| Variable | Role |
|---|---|
| `CAMERA_RELAY_TOKEN` | Shared secret for the publisher and the login page |
| `ZMQ_HOST`, `ZMQ_PORT` | In `config/camera.env`, used when `--zmq-host` is omitted |

API (same token as `Authorization: Bearer`, or the `cam_token` cookie set by the login form):

| Request | Role |
|---|---|
| `POST /api/frame` | Publisher uploads one JPEG (`X-Camera`, `X-Known-Cameras`) |
| `GET /api/control` | Publisher reads the operator's current control |
| `POST /api/control` | JSON patch: `view`, `zoom`, `pan_x`, `pan_y`, `rotate`, `quality`, `max_fps`, `paused` |
| `GET /mjpeg` | Live stream of the active view |
| `GET /snapshot.jpg` | Latest JPEG |
| `GET /api/status` | Publisher online, camera list, control, frame count |
