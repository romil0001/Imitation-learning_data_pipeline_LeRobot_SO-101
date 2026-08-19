# Local Mac + Omen ACT Pipeline Design

**Date:** 2026-08-18  
**Status:** Approved  
**Hardware:** SO-101 leader + follower (12V STS3215), Intel RealSense D405 on HP Omen  
**Hosts:** Mac (arms, record, train, eval, dashboard); Omen (D405 ZMQ publisher only)

## Goal

Own Python on the Mac for local dataset recording, ACT training, evaluation, and replay. Hugging Face LeRobot is the hardware driver only. No `lerobot-record`, `lerobot-train`, `lerobot-rollout`, or Hub uploads.

## Constraints

- Omen has 4 GB RAM: camera publisher only; never train or run full LeRobot there.
- Datasets and checkpoints stay on local disk under the project (`data/datasets/`, `outputs/train/`).
- Record and eval must share the same camera factory so names and resolution cannot drift.
- Bring-up (find port, motor IDs, calibrate) stays on existing numbered scripts wrapping LeRobot CLIs.
- Dashboard stays camera + status + logs; it does not start record or train.
- Implement ACT from Zhao et al. 2023; do not copy LeRobot `modeling_act.py`.

## Machines

```
D405 USB 3 → Omen publish_d405.py :5555 → Wi-Fi → Mac (so101 package + dashboard)
Leader USB + follower USB → Mac
```

## Package

`so101/` in the project root, run as `python -m so101 {record,train,eval,replay,inspect}` inside conda env `lerobot`.

| Module | Responsibility |
|--------|----------------|
| `so101.config` | Load `config/ports.env` and `config/camera.env` |
| `so101.hardware` | Only LeRobot imports: SO101 follower/leader + ZMQ/OpenCV cameras |
| `so101.dataset` | Custom episode folders: `meta.json`, `stats.json`, `state.npy`, `action.npy`, `front.mp4` |
| `so101.record` | Teleop loop: leader action → follower; keys n/r/q |
| `so101.replay` | Play `action.npy` on follower at dataset fps |
| `so101.models.act` | ACT (ResNet18 + CVAE transformer, action chunks) |
| `so101.train` | MPS/CPU train loop, checkpoints, `metrics.jsonl` |
| `so101.eval` | Load checkpoint, temporal ensemble, follower + camera |
| `so101.cli` | argparse entry |

Scripts `08`, `10`, `11`, `12` become thin wrappers around `python -m so101`.

## Dataset format

Root: `data/datasets/<name>/` (gitignored). Default name `so101_d405_demo`.

```
data/datasets/so101_d405_demo/
  meta.json
  stats.json
  episodes/000000/
    episode.json
    state.npy        # float32 [T, 6] follower qpos
    action.npy       # float32 [T, 6] leader qpos (command sent to follower)
    timestamps.npy   # float64 [T]
    front.mp4        # RGB, same fps as record
```

`meta.json` fields: `name`, `fps`, `joint_names`, `camera_name`, `camera_width`, `camera_height`, `task`, `episode_count`.

Joint order (fixed): `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.

Images stored native (typically 640×480). Training resizes to 224. `stats.json` holds mean/std of state and action.

Record UX: **n** save, **r** redo, **q** quit; reset until **n**; resume by appending the next episode index.

## ACT

Paper defaults, Mac MPS-friendly batch:

- ResNet18 ImageNet backbone; 1×1 conv projection of spatial map into transformer tokens
- CVAE latent `z`: train encode (state, action chunk, images) → μ,σ; eval `z = 0`
- Transformer encoder 4 layers, decoder 7 layers, `dim=512`, 8 heads, FFN 3200
- Chunk size 100, L1 action loss + KL (`kl_weight=10`)
- AdamW: backbone `1e-5`, rest `1e-4`; 40k steps; batch 2 on MPS, workers 0
- Checkpoints: `outputs/train/act_<name>/` with `config.json`, `model.pt`, `stats.json`
- Train log: `metrics.jsonl` (`step`, `loss`, `l1`, `kl`)

Eval: same ZMQ camera, follower only, temporal ensemble over overlapping chunks, default duration 60s.

## Testing

No arms required: dataset roundtrip, stats, ACT forward/backward on tiny tensors, two-step train + reload `select_action`, hardware factory parse of example env files.

## Out of scope

- Rewriting Feetech, calibration, or D405 capture
- Training on the Omen
- Hugging Face Hub
- Dashboard control of record/train
