# Setup algorithms and ML algorithms

Two lists. The first is the robot cell. The second is the learned policy that
turns cameras and joint positions into a 6-joint command. Swap only the second
list when you try a new model.

```bash
python -m so101 algorithms
python -m so101 algorithms --family ml
python -m so101 algorithms --family setup
```

## Setup (cell)

These bring the arms up, collect demonstrations, move home, and score trials.
They do not learn a mapping from images to actions.

| Name | What it does | Run |
|------|----------------|-----|
| `find-port` | USB ports → `config/ports.env` | `bash scripts/01_find_port.sh` |
| `setup-motors-follower` | Follower IDs 1–6 | `bash scripts/02_setup_motors_follower.sh` |
| `setup-motors-leader` | Leader IDs 7–12 | `bash scripts/03_setup_motors_leader.sh` |
| `calibrate-follower` | Follower joint ranges | `bash scripts/04_calibrate_follower.sh` |
| `calibrate-leader` | Leader joint ranges | `bash scripts/05_calibrate_leader.sh` |
| `find-cameras` | Camera list for `config/camera.env` | `bash scripts/07_find_cameras.sh` |
| `omen-publish` | Wrist RGB + depth and overhead JPEG over ZMQ | `python omen/publish_d405.py` |
| `teleoperate` | Leader drives follower, no logging | `bash scripts/06_teleoperate.sh` |
| `record` | Demos → `data/datasets/<name>/` | `python -m so101 record` |
| `replay` | Open-loop `action.npy` | `python -m so101 replay` |
| `home` | Joint-space move to the mean demo start | used by eval |
| `depth-grasp` | Wrist-depth override near contact | `config/hybrid_grasp.yaml` |
| `hybrid` | Any ML policy, plus depth-grasp at contact | `--policy hybrid --policy-base <ml>` |
| `protocol` | Conditions × trials, binary y/n | `--protocol config/eval/pen_to_keys.yaml` |
| `convert` | LeRobot export (RGB only) for SmolVLA | `python -m so101 convert` |

`hybrid` is setup: the grasp law is analytic. The base you name with
`--policy-base` is the ML algorithm.

## ML (learned policies)

Each policy implements the same call the eval loop already makes:

```text
policy.reset()
action = policy.select_action(images, state, depth_u16=depth)  # float32 [6]
```

| | `images` | `state` / `action` | `depth_u16` |
|--|----------|--------------------|-------------|
| Shape | uint8 H×W×3, or a list in `camera.env` order (wrist, overhead) | float32 `[6]` | uint16 millimeters, or `None` |
| Order | — | `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` | Set `cfg.use_depth = True` to receive frames |

Units match `state.npy` and `action.npy`.

| Name | Train | Eval on the follower |
|------|--------|----------------------|
| `act` | `python -m so101 train --algo act` | `python -m so101 eval --policy act` |
| `act-jax` | `python -m so101 train --algo act-jax` | `python -m so101 eval --policy act` (checkpoint `config.json` selects Flax) |
| `smolvla` | `bash scripts/13_convert_lerobot.sh` then `bash scripts/14_train_smolvla.sh` | `python -m so101 eval --policy smolvla` |

`--backend jax` on `--algo act` is the same trainer as `--algo act-jax`.
SmolVLA training stays in `lerobot-train`; `python -m so101 train --algo smolvla`
prints that command and stops.

Wrap a learned policy with the contact override:

```bash
python -m so101 eval --policy hybrid --policy-base act \
  --grasp-config config/hybrid_grasp.yaml \
  --protocol config/eval/pen_to_keys.yaml
```

`--policy-base` accepts any registered ML name, including one you added.

## Add a policy

1. Copy `so101/algorithms/plugins/_template.py` to `so101/algorithms/plugins/<name>.py`.
   The leading underscore is what keeps the template off the robot.
2. Implement `select_action`. The template holds the current pose so you can
   check the wiring before the model is ready.
3. Load weights from `ctx.checkpoint` inside `_load`.
4. Uncomment `register(...)`. Import it from `so101.algorithms.registry`.
5. `python -m so101 algorithms` should list the new name under **ML**.
6. `python -m so101 eval --policy <name> --checkpoint <dir>`.

Optional: set `train=` to a function of `TrainContext` and run
`python -m so101 train --algo <name>`. Leave `train` unset to print a custom
train command instead.

Files you do not edit: `so101/eval.py`, `so101/hardware.py`, `so101/record.py`,
`so101/depth_grasp.py`. The eval loop, dataset, and depth-grasp override stay
on the setup side of the split.
