# Methods

**Platform:** SO-101 Visuomotor Manipulation  
**Author:** [romil0001](https://github.com/romil0001)

## Research question

Can a **distributed visuomotor stack** on commodity hardware (Mac + Windows
 sensing node) support reproducible imitation learning, language-conditioned
 manipulation, and depth-assisted contact control—with **protocol-based**
 evaluation rather than proxy train loss?

## Apparatus

- **Plant:** SO-101 leader (teleoperator) + follower (executed plant), 6× STS3215
  @ 12V per arm
- **Sensing:** Intel RealSense D405 (wrist RGB + metric depth), USB overhead RGB
- **Compute split:** Mac — control, learning, eval; HP Omen — camera publisher
  only (ZMQ JPEG + Z16 depth)
- **Software boundary:** Hugging Face LeRobot for hardware I/O only; all dataset,
  training, and eval logic in local `so101` package

See [`architecture.md`](architecture.md) for module map and [`reproduction.md`](reproduction.md)
for installation.

## Data collection

Leader joint commands are recorded as **actions**; follower joint states as
**states**. Cameras are fixed at record and eval time via shared
`config/camera.env` (`CAMERA_NAMES=wrist,overhead`, optional `CAMERA_USE_DEPTH`).

Episode schema (`data/datasets/<name>/`):

| Modality | File | Shape / format |
|----------|------|----------------|
| Proprioception (state) | `state.npy` | float32 [T, 6] |
| Command (action) | `action.npy` | float32 [T, 6] |
| Wrist RGB | `wrist.mp4` | H×W×3, dataset fps |
| Overhead RGB | `overhead.mp4` | H×W×3, dataset fps |
| Wrist depth | `wrist_depth.npy` | uint16 mm, aligned to color |

Joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`.

Default task string: *Place bag clip in bowl* (legacy keys demos remain under
`so101_d405_demo` / `config/eval/pen_to_keys.yaml`).

## Policies

Setup algorithms run the cell (motor IDs, calibration, teleoperation, recording,
replay, homing, analytic depth grasp, protocol scoring). ML algorithms are the
learned map from cameras and joint state to a 6-D action. The lists, the shared
`select_action` contract, and how to add a policy are in
[`algorithms.md`](algorithms.md). `python -m so101 algorithms` prints the split.

### ACT (Action Chunking Transformer)

Implementation follows Zhao et al. (2023) with LeRobot-aligned defaults:
ResNet18 + FrozenBatchNorm, CVAE latent, 1 decoder layer (post-norm), action
chunks of size 100, L1 + KL loss. Training: AdamW `lr=1e-5`,
`weight_decay=1e-4`, episode-level holdout (`val_frac=0.2`, fixed seed),
checkpoint selection on offline `val_l1` (`best_val/`). Eval: temporal
ensemble over overlapping chunks (`ensemble_temp=0.01`; set to `null` for
open-loop action queue).

Optional: RGB + normalized wrist depth in the vision stack; JAX/Flax `tiny`
 backbone via `--backend jax`.

### SmolVLA

Demonstrations converted to LeRobot format (`python -m so101 convert`); fine-tune
with language task conditioning. **Depth is dropped** at export. Eval uses the
same camera names and task string as recording.

### Human-gated DAgger

After a scored ACT baseline, `python -m so101 dagger` (or `bash scripts/18_dagger.sh`)
rolls the policy out on the follower. The leader stays still until **p**. Pause
holds the follower, moves the leader onto that pose (~1.6 s), then disables
torque so the operator can grab it with no jump. **c** records only the
correction. Those frames are expert actions on states the policy actually
visited (Ross et al., 2011; Kelly et al., HG-DAgger, 2019). Autonomous frames
are not saved as demonstrations.

`python -m so101 train --corrections <name>` (or `bash scripts/19_train_dagger.sh`)
merges demos and corrections, forces correction episodes into the train split,
and fine-tunes from the baseline `best_val` into a new checkpoint directory.
Re-score with the same protocol and diff the two `summary.json` files:

```bash
python -m so101 compare-eval --baseline <first>/summary.json --candidate <dagger>/summary.json
```

### Hybrid (depth-assisted grasp)

Base policy (ACT or SmolVLA) controls approach and transport. When wrist depth
 enters a ROI near the target, `DepthGraspController` overrides vertical joints
 using metric depth (`config/hybrid_grasp.yaml`), then releases control after
 contact / lift. Depth is **not** an input to SmolVLA.

## Evaluation protocol

Scored eval uses YAML protocols (default: `config/eval/bag_clip_in_bowl.yaml`). Each
trial runs under a defined scene condition; the workspace is reset between
trials; success is judged with a binary yes/no against a written criterion.
Repeating independent trials per condition yields overall and per-condition
success rates that reflect real task completion rather than training loss,
support fair comparisons across policies, and produce logs suitable for a
thesis or paper.

- **Conditions:** small grid (e.g. key position left / center / right)
- **Trials:** independent repetitions per condition
- **Success:** binary human judgment after each rollout
- **Primary metric:** `pc_success` in `summary.json`

Train loss is explicitly **not** used as a manipulation score.

Outcome graphs: [`outcomes.md`](outcomes.md).

## References

- Zhao, T. et al. (2023). Learning Fine-Grained Bimanual Manipulation with
  Low-Cost Hardware. *CoRL* — ACT.
- Ross, S. et al. (2011). A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning. *AISTATS* — DAgger.
- Kelly, M. et al. (2019). HG-DAgger: Interactive Imitation Learning with
  Human Experts. *ICRA*.
- Hugging Face LeRobot SO-101 documentation:
  https://huggingface.co/docs/lerobot/en/so101
- SmolVLA: LeRobot vision-language-action model (fine-tuned locally via
  `scripts/14_train_smolvla.sh`).
