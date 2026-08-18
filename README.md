# LeRobot SO-101 Bring-up

Leader + follower SO-101 with **12V STS3215** servos (6 each), 2 bus drivers, and an Intel RealSense **D405**.

Official arm guide: https://huggingface.co/docs/lerobot/en/so101

## Machines

| Machine | Role |
|---------|------|
| **Mac** | Leader + follower USB, `lerobot-teleoperate` / `lerobot-record`, dataset, local site |
| **HP Omen (Windows)** | D405 USB 3 host only. Tiny RealSense → JPEG → ZMQ publisher. No PyTorch, no full LeRobot |

Librealsense on macOS usually cannot claim the D405. Put the camera on the Omen (USB 3), keep the arms on the Mac.

```text
D405 USB 3  →  Omen (publish_d405.py :5555)  →  Wi-Fi  →  Mac (LeRobot + dashboard)
```

## Motor IDs
- **Follower:** 1–6 (shoulder_pan…gripper)
- **Leader (patched):** 7–12 — gripper=12 … shoulder_pan=7

See `patches/so_leader_ids_7_12.patch` (re-applied by `env/setup_env.sh`).

## Danger
- Use **12V** power supplies only for these servos.
- During motor ID and calibration, power **one arm at a time**.
- Teleop is the only step that needs **both** arms powered.
- Do not run training or full `lerobot-record` on the 4 GB Omen.

## Clone (both machines)

```bash
git clone git@github.com:OWNER/REPO.git
# or: git clone https://github.com/OWNER/REPO.git
```

On the Omen you only need the `omen/` folder after clone.

## Mac setup
1. `bash env/setup_env.sh`
2. Label boards/PSUs FOLLOWER vs LEADER
3. `bash scripts/01_find_port.sh` (saves `config/ports.env`)
4. Follower motors only → `bash scripts/02_setup_motors_follower.sh`
5. Leader motors only → `bash scripts/03_setup_motors_leader.sh`
6. Calibrate each arm (`04` then `05`)
7. `bash scripts/setup_cameras_view.sh` then set `ZMQ_HOST` in `config/camera.env` to the Omen LAN IP

## Omen setup (Windows)
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Plug D405 into a **USB 3** port (not a USB 2 hub)
3. Double-click `omen/setup_omen.bat` (or run it from Anaconda Prompt)
4. `omen/run_publish.bat`
5. Note IPv4 from `ipconfig` (e.g. `192.168.1.50`)
6. Allow **TCP 5555 inbound** in Windows Firewall if the Mac cannot connect

Optional depth dump (copy to Mac later):

```text
python omen\publish_d405.py --dump-dir outputs\d405_dump
```

Then on the Mac: `OMEN_HOST=192.168.1.50 bash scripts/pull_omen_dump.sh`

## Teleop, record, site (Mac)
1. Omen publisher running
2. `bash scripts/09_dashboard.sh` → http://127.0.0.1:8080
3. Other terminal: `bash scripts/06_teleoperate.sh`
4. Record demos: `bash scripts/08_record.sh`

LeRobot’s ZMQ camera stores **RGB JPEGs**. 16-bit depth is only in the optional Omen dump, not in the dataset unless you add a custom camera type later.

## Quick start
See `docs/checklist.md`.
