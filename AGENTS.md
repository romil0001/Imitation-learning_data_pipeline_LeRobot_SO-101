# AGENTS.md

## Cursor Cloud specific instructions

This repo (`LeRobot SO-101 Bring-up`) is an operations toolkit for a two‑machine
robotics rig: a Mac drives two SO‑101 arms over USB and runs LeRobot
(`lerobot-teleoperate` / `lerobot-record`), while an HP Omen (Windows) hosts an
Intel RealSense D405 and publishes JPEG frames over ZMQ. See `README.md` and
`docs/checklist.md` for the full hardware workflow.

### What can and cannot run in the cloud VM

- Runnable without hardware: the **cameras view** flow — the FastAPI dashboard
  (`dashboard/app.py`) fed by the mock ZMQ publisher (`scripts/mock_zmq_camera.py`),
  plus the bash unit test `scripts/test_common.sh`.
- NOT runnable here: everything that needs physical hardware or conda — the full
  LeRobot install (`env/setup_env.sh`), the numbered bring‑up scripts
  (`scripts/0[1-8]_*.sh`), teleop/record, and the real D405 publisher
  (`omen/publish_d405.py`, needs `pyrealsense2` + the camera on Windows USB3).

### Gotchas (non-obvious)

- There is **no conda** in the cloud VM. `scripts/09_dashboard.sh` and all the
  numbered hardware scripts call `require_conda_lerobot` and will fail here.
  Run the dashboard directly instead: `python3 dashboard/app.py --host 127.0.0.1 --port 8080`.
- Dashboard deps install into the user site (`~/.local`); the `uvicorn`/`fastapi`
  console scripts land in `~/.local/bin` (not on PATH). This is harmless because
  the app is launched via `python3 dashboard/app.py`, not the `uvicorn` CLI.
- `config/camera.env` is **gitignored** and required. The update script
  (`scripts/setup_cameras_view.sh`) recreates it defaulting to `ZMQ_HOST=127.0.0.1`
  (the mock). `dashboard/app.py` exits immediately if `ZMQ_HOST` is unset.
- The dashboard only shows a live feed when a publisher is running. Start
  `python3 scripts/mock_zmq_camera.py` (binds `tcp://0.0.0.0:5555`) before/while
  the dashboard runs; the status pill turns green ("live") once frames arrive.

### Run the cameras view end-to-end (no hardware)

```bash
python3 scripts/mock_zmq_camera.py                        # terminal A (mock D405)
python3 dashboard/app.py --host 127.0.0.1 --port 8080     # terminal B
# open http://127.0.0.1:8080  -> status "live", animated mock frames
```

### Test

```bash
bash scripts/test_common.sh   # prints two expected rejection messages, then "PASS"
```
