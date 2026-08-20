# LeRobot SO-101 Bring-up

Leader + follower SO-101 with **12V STS3215** servos (6 each), 2 bus drivers, and an Intel RealSense **D405**.

Official arm guide: https://huggingface.co/docs/lerobot/en/so101

## Machines

| Machine | Role |
|---------|------|
| **Mac** | Leader + follower USB, teleop, local `so101` record/train/eval, dashboard |
| **HP Omen (Windows)** | D405 USB 3 host only. Tiny RealSense → JPEG → ZMQ publisher. No PyTorch |

Librealsense on macOS usually cannot claim the D405. Put the **wrist D405** and **overhead USB webcam** on the Omen (USB 3 for the D405), keep the arms on the Mac.

```text
D405 (wrist) USB 3  ─┐
USB webcam (overhead)─┴→  Omen (publish_d405.py :5555)  →  Wi-Fi  →  Mac (so101 + dashboard)
```

## Motor IDs
- **Follower:** 1–6 (shoulder_pan…gripper)
- **Leader (patched):** 7–12 — gripper=12 … shoulder_pan=7

See `patches/so_leader_ids_7_12.patch` (re-applied by `env/setup_env.sh`).

## Danger
- Use **12V** power supplies only for these servos.
- During motor ID and calibration, power **one arm at a time**.
- Teleop and record need **both** arms powered.
- Do not train on the 4 GB Omen. Train ACT on the Mac (MPS).

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
7. Copy `config/camera.env.example` → `config/camera.env` and set `ZMQ_HOST` to the Omen LAN IP (`CAMERA_NAMES=wrist,overhead`)

## Omen setup (Windows)
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Plug **D405** into a **USB 3** port (wrist) and the **USB webcam** (overhead)
3. Double-click `omen/setup_omen.bat` (or run it from Anaconda Prompt)
4. `python omen\publish_d405.py --list-webcams` — pick the USB camera index, not the HP laptop camera and not the D405 (often 640x480)
5. `omen/run_publish.bat` (XWF-1080P is usually `--webcam-index 2`; `0` is the laptop cam)
6. Note IPv4 from `ipconfig` (e.g. `192.168.1.50`)
7. Allow **TCP 5555 inbound** in Windows Firewall if the Mac cannot connect

Optional depth dump (copy to Mac later):

```text
python omen\publish_d405.py --dump-dir outputs\d405_dump
```

Then on the Mac: `OMEN_HOST=192.168.1.50 bash scripts/pull_omen_dump.sh`

## Teleop, record, train, eval (Mac)

Record / train / eval are the local `so101` package (`python -m so101`), not Hugging Face Hub and not `lerobot-train`. LeRobot is used only to talk to the arms and ZMQ camera. Bring-up scripts `01`–`07` still wrap LeRobot CLIs.

1. Omen publisher running
2. `bash scripts/09_dashboard.sh` → http://127.0.0.1:8080 (camera, Omen/arm status, live log; **D** refreshes status)
3. Other terminal: `bash scripts/06_teleoperate.sh`
4. Record demos: `bash scripts/08_record.sh` (~50 episodes; **n** save, **r** redo, **q** quit). Data lands in `data/datasets/so101_d405_demo/`
5. Train ACT on the **Mac**: `bash scripts/10_train_act.sh` (Apple MPS; never the 4 GB Omen)
6. Autonomous eval: Omen publisher + follower only → `bash scripts/11_eval_act.sh` (scored grid; **y**/**n** after each trial)

Same commands without the shell wrappers:

```bash
conda activate lerobot
python -m so101 record --dataset so101_d405_demo --num-episodes 50
python -m so101 inspect
python -m so101 train --dataset so101_d405_demo --val-frac 0.2 --seed 0 --eval-every 1000
python -m so101 eval --protocol config/eval/pen_to_keys.yaml --n-trials 5
python -m so101 eval --duration 60
python -m so101 replay --episode 0
```

Default eval is a Google DeepMind-style protocol: binary success, independent trials, small condition grid (`config/eval/pen_to_keys.yaml` — pen left/center/right). After each rollout (or during it to stop early) score **y** success / **n** fail (`s`/`f` aliases). Logs: `outputs/eval/<checkpoint_name>/<run_id>/` (`protocol.json`, `summary.json` with `pc_success`, `trials/0001/`). Per-trial reset prompts still wait for Enter; `--yes` only skips the initial hardware confirm. Unscored timed run: `python -m so101 eval --duration 60` or `PROTOCOL= bash scripts/11_eval_act.sh`.

RGB from the Omen ZMQ stream is stored per episode as `wrist.mp4` (D405) and `overhead.mp4` (USB webcam). With `CAMERA_USE_DEPTH=true`, metric wrist depth is streamed over ZMQ and saved as `wrist_depth.npy` (uint16 mm, aligned to color).

ACT (Action Chunking Transformer) copies your teleop demos. Same scene, same task string, same camera config at record and eval. Training holds out episodes (`val_frac=0.2`, `--seed`) and logs offline `val_l1`; **train loss is not the score**. Checkpoints: `outputs/train/act_so101_d405_demo/` (`config.json`, `split.json`, `stats.json`, `metrics.jsonl`, `last/model.pt`, `best_val/model.pt`). Protocol eval loads `best_val` when present (`python -m so101 eval --protocol ...`).

## Quick start
See `docs/checklist.md`.
