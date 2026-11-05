# Architecture

**Project:** SO-101 Visuomotor Manipulation Platform  
**Author:** [romil0001](https://github.com/romil0001)  
**Repository:** [SO101_Visuomotor_Manipulation](https://github.com/romil0001/SO101_Visuomotor_Manipulation)

Research docs: [`methods.md`](methods.md) · [`outcomes.md`](outcomes.md) · [`reproduction.md`](reproduction.md)

This stack is a two-machine, no-ROS cell. Hugging Face LeRobot talks to the
arms and (optionally) cameras. Everything else — dataset format, ACT, scored
eval, hybrid depth grasp, and the operator dashboard — is local Python owned
by this project (including HG-DAgger correction collection and fine-tune).

## Machines

```text
D405 wrist + base (USB 3) ─┐
USB webcam (overhead)     ─┴→  HP Omen  (omen/publish_d405.py :5555)
                              │  JPEG (+ Z16 depth on wrist/base) over ZMQ
                              │  Ethernet preferred, Wi-Fi failover (ZMQ_HOSTS)
                              ▼
                         Mac  (so101 + dashboard + arms)
Leader USB + follower USB ───►
```

| Machine | Owns | Does not own |
|---------|------|----------------|
| **Mac** | Leader + follower USB, `python -m so101`, ACT train/eval, dashboard | D405 USB (Librealsense cannot claim it reliably on macOS) |
| **HP Omen (Windows)** | Wrist + base D405, overhead USB webcam, ZMQ publisher (Ethernet preferred, Wi-Fi backup) | PyTorch, training, arm USB |

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
| `algorithms` | List setup routines and ML policies (`--family setup` or `ml`) |
| `record` | Leader → follower teleop; **n** save, **r** redo, **q** quit |
| `dagger` | HG-DAgger: policy rollout + **p**/**c** corrections only |
| `train` | Train a registered ML policy (default ACT; `--corrections` for DAgger fine-tune) |
| `eval` | Any registered ML policy, or hybrid + scored protocol |
| `compare-eval` | Diff two protocol `summary.json` files (`pc_success` Δ) |
| `convert` | Local dataset → LeRobot export for SmolVLA |
| `replay` | Play one episode's `action.npy` on the follower |
| `inspect` | Print `meta.json` / episode inventory |

Numbered wrappers in `scripts/` call the same package (`08` record, `10` train, `11`/`15`/`16`/`17` eval, `12` replay, `13` convert, `14` SmolVLA, `18` DAgger, `19` DAgger train, `20` rescore/compare hints + replay capture, `21` Ethernet link helper).

LeRobot is imported only from `so101.hardware` (and SmolVLA wrap). Dataset I/O, models, and eval scoring do not go through Hub APIs.

## Package map

```text
so101/
  cli.py            argparse entry
  algorithms/       setup vs ML registry; plugins/ for a new policy
  config.py         ports.env + camera.env
  hardware.py       SO-101 follower/leader + ZMQ cameras
  record.py         teleop loop
  dagger.py         human-gated DAgger correction loop
  leader_sync.py    move leader onto follower pose on pause
  replay.py         action replay
  train.py          ACT training (+ DAgger merge / fine-tune)
  eval.py           rollout + optional protocol
  eval_protocol.py  scored protocol eval (conditions × trials, y/n success)
  compare_eval.py   baseline vs candidate protocol summaries
  convert_lerobot.py
  hybrid_policy.py  base policy + depth-grasp override
  depth_grasp.py    wrist-depth height assist
  live_viz.py       joint bars for the dashboard
  zmq_rgbd.py       RGB (+ depth) ZMQ client
  models/act.py     ACT (Zhao et al. 2023); LeRobot-aligned defaults (1 decoder layer, FrozenBN, temporal ensemble)
  models/act_jax.py optional Flax backend
  models/smolvla_wrap.py  numpy ↔ SmolVLA
  dataset/          local episode folders (+ combine.py for DAgger merge)
omen/publish_d405.py
dashboard/          FastAPI operator site (cameras, status, logs)
```

## Data

Root: `data/datasets/<name>/` (gitignored). Default name `so101_bag_clip_bowl`.

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
  base.mp4             when `CAMERA_NAMES` includes base
  wrist_depth.npy      optional uint16 mm (wrist; base depth when recorded)
```

Joint order is fixed: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`. RGB is stored native (typically 640×480); ACT resizes
to 224. `CAMERA_USE_DEPTH=true` records metric wrist depth and trains ACT on
RGB + depth. SmolVLA never consumes depth; hybrid eval uses it only in the
grasp controller.

## Policies

Setup routines (motors, record, replay, home, depth grasp, protocol) and learned
policies are separate registries. `python -m so101 algorithms` prints both.
A new model is a plugin under `so101/algorithms/plugins/`; eval, record, and
the depth-grasp override stay unchanged. See [`algorithms.md`](algorithms.md).

1. **ACT** — ResNet18 (or tiny) + CVAE transformer, chunk size 100, L1 + KL.
   Train holds out episodes (`val_frac=0.2`). Checkpoints under
   `outputs/train/act_<dataset>/` (`last/`, `best_val/`). Train loss is not
   the score.
2. **SmolVLA** — convert demos (`scripts/13_convert_lerobot.sh`), fine-tune
   (`14`), eval with language task string matching record.
3. **Hybrid** — any registered ML policy for pick-and-place; `DepthGraspController`
   overrides vertical joints near contact using wrist depth
   (`config/hybrid_grasp.yaml`), then hands control back for lift/place.

Scored eval (`config/eval/bag_clip_in_bowl.yaml`): independent trials, binary
success (**y**/**n**), small condition grid. Logs:
`outputs/eval/<checkpoint>/<run_id>/` (see [`eval_logs.md`](eval_logs.md)).

## Operator site

`bash scripts/09_dashboard.sh` → http://127.0.0.1:8080

Live MJPEG (overhead, wrist, base, wrist depth), Omen/arm status, camera
`link_mode` (`ethernet` / `wifi`), joint bars while recording, append-only
`outputs/session.log` (via `run_logged` in numbered scripts). The dashboard does
not start record or train; it can launch eval with the same control modes as
`scripts/11_eval_act.sh`.

## Boundaries

- No Hugging Face Hub uploads from this repo
- No agent orchestration or worker-plan docs
- Dashboard does not own the robot; terminals running `python -m so101` do
- Omen publisher is capture-only
