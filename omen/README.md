# HP Omen — sensing node

Windows-only camera publisher for the SO-101 cell. The Omen owns the wrist
**D405** (USB 3) and the overhead **USB webcam**, and streams JPEG (+ optional
metric depth) to the Mac over ZMQ port **5555**.

Do **not** install PyTorch / LeRobot or train here (4 GB GPU). Train on the Mac.

```text
D405 (wrist) USB 3  ─┐
USB webcam (overhead)─┴→  publish_d405.py :5555  →  Wi-Fi  →  Mac
```

## One-time setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda).
2. Install [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/releases)
   (`Intel.RealSense.SDK-*.exe`) so Windows has D405 USB drivers. Reboot if prompted.
3. Clone this repo (or copy the `omen/` folder) onto the Omen.
4. From the repo root in **cmd** or **Anaconda Prompt**:

   ```bat
   omen\setup_omen.bat
   ```

   That creates the `omen-d405` conda env, installs `omen\requirements.txt`,
   verifies imports, and prints this PC’s LAN IPv4 addresses.

5. Allow inbound **TCP 5555** (run as Administrator once):

   ```bat
   omen\open_firewall.bat
   ```

6. Plug **D405 → USB 3** (wrist) and the **USB webcam** (overhead). Close
   RealSense Viewer if it is open.

## Every session

```bat
python omen\publish_d405.py --list-webcams
omen\run_publish.bat
```

Leave the publisher running. On the Mac, set `ZMQ_HOST` in `config/camera.env`
to this PC’s IPv4 (`omen\show_lan_ip.bat` reprints it).

| Index (typical on this Omen) | Camera |
|------------------------------|--------|
| 0 | Laptop built-in (HP Wide Vision) — ignore |
| 1 | Often empty |
| 2 | XWF-1080P overhead USB — use this |

Override with `omen\run_publish.bat --webcam-index N` if `--list-webcams` disagrees.

## Self-check

Without starting the long-running stream:

```bat
conda run -n omen-d405 --no-capture-output python omen\publish_d405.py --self-check
```

Confirms RealSense devices, OpenCV webcams, and that port 5555 can bind.

## Mac side — prove data is transmitting

With `omen\\run_publish.bat` still running on the Omen:

```bash
cp config/camera.env.example config/camera.env
# set ZMQ_HOST=<Omen IPv4 from show_lan_ip.bat>
bash scripts/16_test_link.sh
# or:  python -m so101 link-check
```

PASS means wrist RGB, overhead RGB, and wrist depth (if `CAMERA_USE_DEPTH=true`)
are arriving over ZMQ. Sample frames land in `outputs/link_check/<timestamp>/`.

Visual check:

```bash
bash scripts/09_dashboard.sh   # http://127.0.0.1:8080 — wrist / overhead / depth
```

Same Wi-Fi/LAN as the Omen. Dashboard **D** key refreshes status.
