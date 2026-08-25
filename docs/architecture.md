# Architecture

**Project:** SO-101 Visuomotor Manipulation Platform  
**Author:** [romil0001](https://github.com/romil0001)  
**Repository:** [SO101_Visuomotor_Manipulation](https://github.com/romil0001/SO101_Visuomotor_Manipulation)

Research docs: [`methods.md`](methods.md) · [`outcomes.md`](outcomes.md) · [`reproduction.md`](reproduction.md)

This stack is a two-machine, no-ROS cell. Hugging Face LeRobot talks to the
arms and (optionally) cameras. Everything else — dataset format, ACT, scored
eval, hybrid depth grasp, and the operator dashboard — is local Python owned
by this project.

## Machines

```text
D405 (wrist) USB 3  ─┐
USB webcam (overhead)─┴→  HP Omen  (omen/publish_d405.py :5555)
                              │  JPEG (+ optional Z16 depth) over ZMQ
                              ▼
                         Mac  (so101 + dashboard + arms)
Leader USB + follower USB ───►
```

| Machine | Owns | Does not own |
|---------|------|----------------|
| **Mac** | Leader + follower USB, `python -m so101`, ACT train/eval, dashboard | D405 USB (Librealsense cannot claim it reliably on macOS) |
| **HP Omen (Windows)** | Wrist D405 + overhead USB webcam, ZMQ publisher | PyTorch, training, arm USB |

Train ACT on the Mac (MPS). Never train on the 4 GB Omen.

## Hardware identity

- Follower motors: IDs **1–6** (shoulder_pan … gripper)
- Leader motors: IDs **7–12** (shoulder_pan=7 … gripper=12), applied by `patches/so_leader_ids_7_12.patch`
- Power: **12V** STS3215 only. Motor ID and calibration: one arm powered at a time
- Hardware configuration scripts `01`–`07` wrap official LeRobot CLIs; they do not reimplement Feetech

Ports live in gitignored `config/ports.env`. Camera/ZMQ live in gitignored `config/camera.env`.

## Control plane

`python -m so101` is the only operator CLI after hardware configuration.

| Command | Role |
|---------|------|
| `record` | Leader → follower teleop; **n** save, **r** redo, **q** quit |
| `train` | ACT (torch/MPS default; optional JAX/Flax `tiny` backbone) |
| `eval` | ACT, SmolVLA, or hybrid + scored protocol |
| `convert` | Local dataset → LeRobot export for SmolVLA |
| `replay` | Play one episode's `action.npy` on the follower |
| `inspect` | Print `meta.json` / episode inventory |

Numbered wrappers in `scripts/` call the same package (`08` record, `10` train, `11` eval, `12` replay, `13` convert, `14` SmolVLA, `15` hybrid).

LeRobot is imported only from `so101.hardware` (and SmolVLA wrap). Dataset I/O, models, and eval scoring do not go through Hub APIs.

## Package map

```text
so101/
  cli.py            argparse entry
  config.py         ports.env + camera.env
  hardware.py       SO-101 follower/leader + ZMQ cameras
  record.py         teleop loop
  replay.py         action replay
  train.py          ACT training
  eval.py           rollout + optional protocol
  eval_protocol.py  scored protocol eval (conditions × trials, y/n success)
  convert_lerobot.py
  hybrid_policy.py  base policy + depth-grasp override
  depth_grasp.py    wrist-depth height assist
  live_viz.py       joint bars for the dashboard
  zmq_rgbd.py       RGB (+ depth) ZMQ client
  models/act.py     ACT (Zhao et al. 2023), not a copy of LeRobot modeling_act
  models/act_jax.py optional Flax backend
  models/smolvla_wrap.py  numpy ↔ SmolVLA
  dataset/          local episode folders
omen/publish_d405.py
dashboard/          FastAPI operator site (cameras, status, logs)
```

## Data

Root: `data/datasets/<name>/` (gitignored). Default name `so101_d405_demo`.

```text
meta.json
stats.json
episodes/000000/
  episode.json
  state.npy          float32 [T, 6] follower qpos
  action.npy         float32 [T, 6] leader command
  timestamps.npy
  wrist.mp4
  overhead.mp4
  wrist_depth.npy    optional uint16 mm, aligned to color
```

Joint order is fixed: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`. RGB is stored native (typically 640×480); ACT resizes
to 224. `CAMERA_USE_DEPTH=true` records metric wrist depth and trains ACT on
RGB + depth. SmolVLA never consumes depth; hybrid eval uses it only in the
grasp controller.

## Policies

1. **ACT** — ResNet18 (or tiny) + CVAE transformer, chunk size 100, L1 + KL.
   Train holds out episodes (`val_frac=0.2`). Checkpoints under
   `outputs/train/act_<dataset>/` (`last/`, `best_val/`). Train loss is not
   the score.
2. **SmolVLA** — convert demos (`scripts/13_convert_lerobot.sh`), fine-tune
   (`14`), eval with language task string matching record.
3. **Hybrid** — ACT or SmolVLA for pick-and-place; `DepthGraspController`
   overrides vertical joints near contact using wrist depth
   (`config/hybrid_grasp.yaml`), then hands control back for lift/place.

Scored eval (`config/eval/pen_to_keys.yaml`): independent trials, binary
success (**y**/**n**), small condition grid. Logs:
`outputs/eval/<checkpoint>/<run_id>/`.

## Operator site

`bash scripts/09_dashboard.sh` → http://127.0.0.1:8080

Live MJPEG (overhead, wrist, wrist depth), Omen/arm status, joint bars while
recording, `outputs/session.log`. The dashboard does not start record or train.

## Boundaries

- No Hugging Face Hub uploads from this repo
- No agent orchestration or worker-plan docs
- Dashboard does not own the robot; terminals running `python -m so101` do
- Omen publisher is capture-only
