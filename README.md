# SO-101 Visuomotor Manipulation Platform

**Author:** [romil0001](https://github.com/romil0001) · **License:** [MIT](LICENSE)

> A research platform for bimanual SO-101 manipulation: leader–follower teleoperation,
> multi-view RGB and metric wrist depth, imitation learning (ACT, SmolVLA), depth-assisted
> hybrid grasping, and protocol-based evaluation with per-condition success rates.

## Abstract

We study visuomotor manipulation on a low-cost SO-101 dual-arm cell with a
**distributed sensing architecture** (Mac control + Windows depth/RGB publisher).
Human demonstrations are collected via leader teleoperation and stored in a local
episode format with synchronized proprioception and video. Policies are trained
offline (ACT with optional depth; SmolVLA with language) and evaluated in closed
loop under a **fixed YAML protocol** with binary success labels and condition
strata. A hybrid controller delegates contact-phase height correction to metric
wrist depth while leaving approach and place to the base policy.

## Research contributions

1. **Local visuomotor pipeline** — record, train, eval, and replay without Hub
   upload; LeRobot used as a hardware driver only.
2. **Multi-view + metric depth** — wrist D405 and overhead RGB streamed over ZMQ;
   depth optionally in ACT training and in a grasp-phase controller.
3. **Protocol-based outcomes** — repeated trials per scene condition with binary
   success labels; reports `pc_success` and per-condition breakdown instead of
   train loss.
4. **Hybrid depth grasp** — decouple VLA approach from analytic contact control.

## Visualizations

### Real-robot observations

| Overhead (scene) | Wrist (ego) |
|:---:|:---:|
| ![Overhead view of SO-101 gripper, metal cup, and keys](assets/frames/overhead_real.jpg) | ![Wrist camera view toward keys and cup](assets/frames/wrist_real.jpg) |

Task scene: SO-101 follower gripper, stainless cup, and key ring on the table — the
pen-to-keys / keys-into-cup manipulation setup used for teleop demos and protocol eval.

### Eval success rates (pen-to-keys protocol)

Example outcomes — 5 trials × 3 conditions (15 trials total):

| Overall pc_success | Success by condition |
|:---:|:---:|
| ![overall success](assets/graphs/success_overall.png) | ![by condition](assets/graphs/success_by_condition.png) |

Fixture data: [`assets/eval/`](assets/eval/). Regenerate after real eval runs — see [`assets/README.md`](assets/README.md).

## System overview

```mermaid
flowchart TB
  subgraph mac["Mac — control & learning"]
    L[Leader arm]
    F[Follower arm]
    SO101["python -m so101"]
    DASH[Dashboard]
    L --> SO101
    F --> SO101
    SO101 --> DASH
  end

  subgraph omen["Omen — sensing"]
    D405[D405 wrist]
    CAM[Overhead USB]
    PUB[publish_d405.py]
    D405 --> PUB
    CAM --> PUB
  end

  PUB -->|ZMQ RGB + depth| SO101
```

## Cameras from outside the lab

The Omen publisher stays on the home LAN. A relay on the Mac (or a small public
machine the Mac dials out to) serves the wrist, overhead, and depth views to a
browser on any network, with pan, zoom, rotation, quality, and frame-rate
controls. The dashboard on port 8080 is unchanged.

**→ [`docs/remote_camera.md`](docs/remote_camera.md)**

## Outcome graphs

Full outcome flow, policy comparison, and eval protocol diagrams:

**→ [`docs/outcomes.md`](docs/outcomes.md)**

```mermaid
flowchart LR
  DEMO[Demonstrations] --> TRAIN[Policy training]
  TRAIN --> EVAL[Protocol eval]
  EVAL --> PC[pc_success]
  EVAL --> BC[by_condition]
```

## Quick reproduction

```bash
git clone https://github.com/romil0001/SO101_Visuomotor_Manipulation.git
cd SO101_Visuomotor_Manipulation
bash env/setup_env.sh
# configure config/ports.env and config/camera.env — see docs/reproduction.md
bash scripts/08_record.sh
bash scripts/10_train_act.sh
bash scripts/11_eval_act.sh   # → outputs/eval/.../summary.json
```

Hardware setup, safety, and full command list: [`docs/reproduction.md`](docs/reproduction.md).

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/algorithms.md`](docs/algorithms.md) | Setup vs ML algorithms, and how to add a policy |
| [`docs/methods.md`](docs/methods.md) | Apparatus, data schema, policies, eval protocol |
| [`docs/outcomes.md`](docs/outcomes.md) | Outcome graphs and reported metrics |
| [`docs/literature_review.md`](docs/literature_review.md) | Related work and the ML research gap this platform targets |
| [`research/`](research/README.md) | PhD-style ML research programme: proposal, pre-registration, LaTeX paper |
| [`docs/architecture.md`](docs/architecture.md) | Software module map |
| [`docs/reproduction.md`](docs/reproduction.md) | Lab setup and experiment workflow |
| [`docs/checklist.md`](docs/checklist.md) | Power and labeling checklist |
| [`docs/remote_camera.md`](docs/remote_camera.md) | Watch and control cameras off the home Wi-Fi |

## Citation

If you use this platform, please cite:

```bibtex
@software{so101_visuomotor_manipulation,
  author = {Modi, Romil},
  title = {SO-101 Visuomotor Manipulation Platform},
  year = {2026},
  url = {https://github.com/romil0001/SO101_Visuomotor_Manipulation}
}
```

Or use [`CITATION.cff`](CITATION.cff) for automatic GitHub citation metadata.

## Acknowledgments

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) — SO-101 hardware drivers
- SO-101 arm documentation: https://huggingface.co/docs/lerobot/en/so101
