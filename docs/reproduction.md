# Reproduction guide

Hardware and software setup for the SO-101 Visuomotor Manipulation Platform.
For research framing see [`methods.md`](methods.md) and [`outcomes.md`](outcomes.md).

Official arm reference: https://huggingface.co/docs/lerobot/en/so101

## Clone

```bash
git clone git@github.com:romil0001/SO101_Visuomotor_Manipulation.git
cd SO101_Visuomotor_Manipulation
```

On the Windows sensing node, only `omen/` is required after clone.

## Machines

| Machine | Role |
|---------|------|
| **Mac** | Leader + follower USB, teleop, record/train/eval, dashboard |
| **HP Omen (Windows)** | D405 + overhead webcam → ZMQ publisher (no PyTorch) |

```text
D405 (wrist) USB 3  ─┐
USB webcam (overhead)─┴→  Omen (publish_d405.py :5555)  →  Wi-Fi  →  Mac
```

Librealsense on macOS usually cannot claim the D405 — keep the D405 on the Omen.

## Safety and power

- **12V** STS3215 servos only — never 5V
- Motor ID and calibration: **one arm powered at a time**
- Teleop and record: **both** arms powered
- Train ACT on the Mac (MPS), not the 4 GB Omen

Full power table: [`checklist.md`](checklist.md).

## Motor IDs

- **Follower:** 1–6 (shoulder_pan … gripper)
- **Leader:** 7–12 (gripper=12 … shoulder_pan=7) — `patches/so_leader_ids_7_12.patch`

## Mac setup

1. `bash env/setup_env.sh`
2. Label boards/PSUs FOLLOWER vs LEADER
3. `bash scripts/01_find_port.sh` → `config/ports.env`
4. Follower only → `bash scripts/02_setup_motors_follower.sh`
5. Leader only → `bash scripts/03_setup_motors_leader.sh`
6. Calibrate → `bash scripts/04_calibrate_follower.sh` then `05_calibrate_leader.sh`
7. `config/camera.env.example` → `config/camera.env`; set `ZMQ_HOST` to Omen LAN IP

## Omen setup (Windows)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. D405 on USB 3 (wrist); USB webcam (overhead)
3. `omen\setup_omen.bat`
4. `python omen\publish_d405.py --list-webcams` — pick USB cam index (not laptop cam)
5. `omen\run_publish.bat` (XWF-1080P often `--webcam-index 2`)
6. Allow **TCP 5555 inbound** in Windows Firewall

## Experiment workflow

```bash
conda activate lerobot
bash scripts/09_dashboard.sh          # optional monitoring
bash scripts/08_record.sh             # ~50 demos
bash scripts/10_train_act.sh          # ACT
bash scripts/11_eval_act.sh           # protocol eval → outcomes
bash scripts/13_convert_lerobot.sh    # for SmolVLA
bash scripts/14_train_smolvla.sh
bash scripts/15_eval_hybrid.sh
```

Direct CLI equivalents:

```bash
python -m so101 algorithms          # setup routines vs learned policies
python -m so101 record --dataset so101_d405_demo --num-episodes 50
python -m so101 train --algo act --dataset so101_d405_demo --val-frac 0.2 --seed 0
python -m so101 eval --policy act --protocol config/eval/pen_to_keys.yaml --n-trials 5
python -m so101 eval --policy hybrid --policy-base act --grasp-config config/hybrid_grasp.yaml --protocol config/eval/pen_to_keys.yaml -y
```

Eval artifacts: `outputs/eval/<checkpoint>/<run_id>/summary.json`.
