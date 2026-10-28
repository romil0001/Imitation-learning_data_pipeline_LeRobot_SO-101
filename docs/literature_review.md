# Literature review and ML research gap

**Platform:** SO-101 Visuomotor Manipulation
**Author:** [romil0001](https://github.com/romil0001)
**Scope:** the learned-policy side of the platform (ACT, SmolVLA, hybrid depth
grasp, protocol evaluation). Hardware setup is out of scope.
**Literature cut-off:** October 2026.

Companion docs: [`methods.md`](methods.md) · [`algorithms.md`](algorithms.md) ·
[`outcomes.md`](outcomes.md).

## 1. What this platform does, in the terms the literature uses

| Platform component | Where | Literature category |
|---|---|---|
| ACT with optional depth tokens (`use_depth`) | `so101/models/act.py` | Single-task behaviour cloning with action chunking; RGB-D variant |
| SmolVLA fine-tune, RGB + language, depth dropped at export | `so101/models/smolvla_wrap.py`, `so101/convert_lerobot.py` | Small vision-language-action (VLA) model, fine-tuned per cell |
| `DepthGraspController` overriding `shoulder_lift`/`elbow_flex` from a wrist-depth ROI median | `so101/depth_grasp.py`, `so101/hybrid_policy.py` | Hybrid learned + analytic control; phase-switched visual servoing |
| YAML protocol, 5 trials × 3 conditions, binary human score | `so101/eval_protocol.py`, `config/eval/pen_to_keys.yaml` | Real-robot policy evaluation methodology |
| Leader–follower teleop, local episode store, episode-level holdout | `so101/record.py`, `so101/train.py` | Demonstration collection and data regime |

The embodiment matters for what follows: the SO-101 is a 6-DoF,
position-controlled STS3215 servo arm with **no force/torque sensing**. The
only contact-relevant signal the platform has is metric wrist depth from the
D405.

## 2. Policy classes for low-cost arms

### 2.1 Behaviour cloning with action chunking

ACT (Zhao et al., 2023) introduced chunked action prediction with a CVAE and
temporal ensembling and was itself motivated by low-cost hardware (ALOHA).
Diffusion Policy (Chi et al., 2023) replaced the CVAE with a denoising head and
is the other standard single-task baseline. LeRobot ships both, plus VQ-BeT,
and exposes them to the SO-100/SO-101 community. The project's `act.py` is a
faithful ACT reimplementation (ResNet-18, chunk 100, L1 + KL, queue-then-pop
inference matching LeRobot) with one addition: a from-scratch 3-layer CNN that
turns a single wrist depth map into extra encoder tokens.

### 2.2 Small VLAs

SmolVLA (Shukor et al., 2025) is a 0.45 B VLA pretrained on community LeRobot
data and evaluated on SO-100/SO-101. Reported single-task numbers on SO-101
pick-place-Lego: SmolVLA 90 % in-distribution / 50 % out-of-distribution versus
ACT 70 % / 40 %. On SO-100 multi-task, SmolVLA averaged 78.3 % versus ACT 48.3 %
and π0 61.7 %. Larger VLAs that LeRobot also supports include π0 (Black et al.,
2024), π0.5, OpenVLA (Kim et al., 2024), Octo, GR00T, Wall-X and MolmoAct.

### 2.3 What independent SO-101 benchmarks actually report

Two 2026 benchmarks evaluate several of these policies on the **same** hardware
as this project, which makes them the right external reference points.

- **Yu & Qiu (2026)**, "Benchmarking VLA models on SO-101": 4 tasks, 100 demos
  per task, 20 trials per task. Mean success: π0.5 56.3 %, Wall-X 51.3 %, ACT
  33.8 %, SmolVLA 32.5 %. Pick-and-place-style "Pen Transfer" is 70–95 % for
  every policy; precision and sorting tasks collapse to 0–10 %. Over failed
  trials, **grasp instability** appears in 91–100 % of failures for every
  policy, and recovery rates are low (ACT 6.5 %, SmolVLA 3.2 %, π0.5 30.8 %).
  The authors note that the ACT-vs-SmolVLA ordering is within noise at 20
  trials.
- **ArmnetBench v0.1 (2026)**, 12 tasks on a fleet of SO-101 cells, 50 demos
  per task, ~30 human-scored rollouts per policy-task: π0.5 47.6 %, π0 35.1 %,
  GR00T N1.7 29.4 %, Diffusion Policy 26.7 %, ACT 16.4 %, SmolVLA 13.1 %
  (secondary source; see references).

Two conclusions carry into the gap analysis. First, on this embodiment the
dominant failure is **the last few centimetres before and during the grasp**,
not approach or language grounding. Second, ACT and SmolVLA, the two policies
this platform ships, sit at the bottom of both leaderboards, so the
contact-phase intervention this platform proposes targets the right failure
mode for exactly these policies.

### 2.4 Data regime

Lin et al. (ICLR 2025) show that single-task IL generalisation scales roughly
as a power law in the number of training environments and objects, that
diversity matters far more than demo count per setting, and that performance
per setting plateaus once a threshold of demos is reached. The project's
default protocol has one object set, one scene, three key positions, and every
condition is marked `split: id`; there is no populated out-of-distribution
split.

## 3. Depth and 3D in visuomotor policies

### 3.1 Evidence that raw depth is not a free win

- **3D Diffusion Policy** (Ze et al., 2024): on four real tasks, RGB Diffusion
  Policy averaged 35 %, **depth-image Diffusion Policy 20 %**, and a compact
  point-cloud representation 85 %. Depth helped one task (Roll-Up, 40 % vs 0 %)
  and hurt the other three. Naïvely feeding a depth image to a BC policy can
  reduce success.
- **SPARTN** (Zhou et al., CVPR 2023): for eye-in-hand grasping, geometric
  corrective augmentation of an RGB-only policy closed the RGB-vs-RGB-D gap,
  "eliminating the previous need for depth sensors" and improving real grasp
  success by 22.5 points. Geometry can be *distilled* into an RGB policy
  rather than fed at inference.
- **Spatial RoboGrasp** (2025): monocular depth plus a depth-aware grasp
  prompt raised grasp success by up to 40 points under lighting variation, on
  an industrial arm with a wrist RGB camera.
- Wrist cameras themselves are well established as the most valuable single
  view for BC (Hsu et al., ICLR 2022), which supports the platform's wrist-first
  sensing choice.

### 3.2 Depth inside VLAs (2025–2026)

SpatialVLA (Ego3D position encoding), PointVLA (point-cloud features injected
into a frozen action expert), DepthVLA (mixture-of-transformers with a
pretrained depth transformer; 78.5 % vs 65.0 % real-task progress) and
AugVLA-3D (sensor-free depth from VGGT) all show that adding 3D structure to a
VLA helps spatial precision. PointVLA explicitly demonstrates **height
adaptability**, i.e. grasping at table heights unseen in training, which is
the same failure the platform's depth controller targets.

All of these (a) operate on multi-billion-parameter backbones, (b) use
third-person or multi-view depth / point clouds rather than a single metric
wrist depth map, and (c) are evaluated on Franka/UR/humanoid-class hardware
with precise controllers. None report results on a servo arm in the SO-101
class, and none test a 0.45 B model.

## 4. Hybrid learned + analytic control and residual correction

### 4.1 Phase-switched hybrids

- **Intermittent Visual Servoing** (Paradis et al., ICRA 2021): a classical
  planner handles coarse motion and a learned servoing policy takes over only
  where precision is required; switching is based on the precision demanded by
  the phase.
- **Sphinx** (Sundaresan et al., 2024): a point-cloud waypoint policy for
  long-range motion hands off to a wrist-image dense policy for precise phases;
  the **switch is learned** from demonstrator-provided mode labels.
- **VLA + Diffusion switching** (Pan, Junge & Hughes, 2024): a VLA drives approach, a
  diffusion policy drives a compliant grasp, with an event signal recorded in
  the training data to trigger the transition; success >80 % versus <40 % for
  the VLA alone.

The platform inverts the usual direction: the *learned* policy handles the
coarse phase and the *analytic* law handles the precise phase. That is defensible
for a servo arm without force sensing, but no prior work evaluates this
ordering with a metric wrist-depth law, and no prior work learns the switch for
it.

### 4.2 Residual and corrective learning on top of a frozen base

- **Compliant Residual DAgger** (Xu, Hou et al., NeurIPS 2025): freeze the base
  policy, learn a residual from human delta corrections with force feedback;
  >60 point gains on contact-rich tasks with minimal correction data.
- **ForceRFT** (2026): a **frozen SmolVLA** prior plus a lightweight residual
  actor conditioned on wrist wrench, trained from human corrections and
  autonomous outcomes; plug insertion 7/30 → 25/30.
- **Object-centric residual RL** (2026): residual trained in simulation on
  object poses, deployed zero-shot on a real VLA; 42 % → 76 %; improved
  rollouts are then reused to retrain the base VLA.
- **HIL-SERL** (Luo et al., 2024) reaches near-perfect success in 1–2.5 h of
  real-world RL on Franka-class arms. Community ports to SO-101 (2026) report
  ~70 % grasp success after ~750 episodes / 3 h and note that the policy rarely
  experiences reward without interventions early on.

Every one of the contact-rich residual methods above relies on **force or
wrench** as the correction signal. The SO-101 has none; the platform's bet is
that metric depth can stand in for it. That substitution is untested in the
literature.

## 5. Real-robot evaluation methodology

- Kress-Gazit et al. (2024), "Robot Learning as an Empirical Science":
  report trial *counts* not just percentages, state whether policies were
  evaluated A/B in the same session, describe initial conditions and their
  relation to training, and provide interval estimates rather than point
  estimates.
- TRI "A Careful Examination of Large Behavior Models" (2025): 1 800
  real trials, **blind A/B**, controlled initial conditions, explicit
  statistical analysis.
- Snyder et al. (RSS 2025): sequential tests for comparing two policies in
  the 10–50 trial regime, stopping near the minimal number of trials without
  p-hacking.
- RoboArena (2025): pairwise comparisons converge to stable rankings in ~100
  comparisons, versus unstable rankings from small fixed-task evaluations.
- Yu & Qiu (2026): a four-class failure taxonomy (grasp instability,
  repetition loop, state mismatch, precision misalignment) and a recovery-rate
  metric, arguing binary success alone hides most of what distinguishes
  policies.

What the platform's protocol already does well relative to this bar: a written
success criterion frozen at eval time, scene reset between trials, a
stratified condition grid, per-condition counts stored alongside rates, and
train loss explicitly excluded as a metric. What it does not yet do: confidence
intervals, interleaved A/B ordering across policies, blind scoring, failure
taxonomy, recovery or time-to-success metrics.

The default budget is worth quantifying. With 15 trials, an observed 10/15
(67 %) has a 95 % Wilson interval of [42 %, 85 %]; a per-condition 3/5 is
[23 %, 88 %]. For a two-policy comparison with 15 trials per arm, a two-sided
Fisher test has only ~45 % power to detect 40 % vs 80 % and ~10 % power to
detect 60 % vs 80 %. Thirty trials per arm gives ~85 % power for a 40-point
gap and still only ~30 % for a 20-point gap. Any hybrid-vs-base claim at the
default budget is underpowered unless the effect is very large.

## 6. Research gap

The one-sentence version: **no published work tests whether a single metric
wrist-depth signal, used as an analytic contact-phase override around a frozen
RGB imitation policy, improves grasp reliability on a force-blind, low-cost
servo arm, and nothing compares that analytic override against learning the
same correction.** The gap decomposes as follows.

### G1. Depth-as-contact-proxy on force-blind hardware (primary gap)

Contact-rich residual methods (CR-DAgger, ForceRFT, HIL-SERL) all consume force
or wrench. Independent SO-101 benchmarks find grasp instability in >90 % of
failures for ACT and SmolVLA. The platform has the sensor and the controller to
test whether metric depth can substitute for force in the contact phase on this
class of arm, and nobody has reported that experiment.

### G2. Analytic override vs. learned use of the same depth

Three ways of using the same D405 stream are available in the codebase, and
the literature predicts they will not be equivalent:

| Variant | Platform support | Literature prior |
|---|---|---|
| RGB only | ACT, SmolVLA | Baseline |
| Depth as input tokens | ACT `use_depth=True` | DP3: naïve depth input can *reduce* success |
| Depth as analytic override at contact | `--policy hybrid` | No prior on this hardware |
| Depth distilled into an RGB policy | Not implemented | SPARTN: geometry can be distilled, removing the sensor at test time |

The controlled four-way ablation under one protocol has not been published for
any low-cost arm. It is also the experiment that decides whether depth belongs
in the model, in the controller, or only in training.

### G3. Hand-tuned switching and the chunk-override interaction

`DepthGraspController` switches phase on fixed thresholds (`enter_mm`,
`target_mm`, `tol_mm`, `exit_mm`) applied to a centred-ROI median, and
closes the gripper after `hold_close_frames`. Sphinx and the VLA+diffusion
switch show the phase boundary can be learned from demonstrations. Two
specific unstudied issues follow from the current design:

1. The phase detector is depth-only. Gripper state, approach velocity and the
   base policy's own intent are ignored, so the override can fire during
   transport or place whenever any object passes through the ROI.
2. The override writes joint targets around the *current* state while the
   base policy keeps popping from a chunk planned before the override moved
   the arm (`ACTPolicy` fills a queue of `n_action_steps` and does not
   replan; SmolVLA likewise executes its own queued chunk). How an external
   correction interacts with action chunking and temporal ensembling has no
   treatment in the literature; the obvious mitigations (flush the queue on
   phase exit, blend rather than overwrite, learn a gating signal) are
   untested.

### G4. Depth injection into a small VLA

PointVLA, DepthVLA and AugVLA-3D inject 3D structure into multi-billion-parameter
VLAs. The platform drops depth entirely at SmolVLA export. Whether a 0.45 B VLA
fine-tuned on tens of episodes benefits from a lightweight wrist-depth adapter
(the PointVLA "inject into a frozen expert" recipe) on commodity compute is open.

### G5. Closing the loop: override as teacher

The hybrid rollout produces depth-corrected trajectories that are never fed
back. Object-centric residual RL and CR-DAgger both show that corrected
rollouts can retrain the base policy. On this platform the correction is free
(no human, no force sensor), so a DAgger-style loop where the analytic law
labels contact-phase actions and the RGB policy is retrained on them is
immediately implementable and unexplored.

### G6. Statistical adequacy and failure attribution in low-cost evaluation

Section 5 shows the default 15-trial budget cannot support the comparisons
the platform is designed to make. Sequential testing (Snyder et al.) and a
failure taxonomy with recovery rate (Yu & Qiu) have not been brought to
single-cell, low-cost evaluation, where the human-in-the-loop cost per trial is
the binding constraint.

### G7. Out-of-distribution conditions

Every default condition is `split: id`. SmolVLA's own SO-101 result drops from
90 % to 50 % ID → OOD; PointVLA frames height adaptability as the canonical 3D
benefit. Whether analytic depth helps *more* under OOD (new table height,
unseen object size, distractors) than in distribution is the sharpest
hypothesis the platform can test, and the protocol already has the `split`,
`lighting` and `distractors` fields to express it.

## 7. Recommended research question and experiment matrix

**Research question.** On a force-blind, position-controlled low-cost arm, does
metric wrist depth improve grasp-phase reliability of RGB imitation policies
more effectively as an analytic contact-phase override, as a model input, or as
a training-time teacher, and does the benefit grow out of distribution?

Suggested matrix, all under one frozen protocol YAML, interleaved A/B order,
blind scorer where possible:

| Row | Policy | Depth use | Split |
|---|---|---|---|
| 1 | ACT | none | id, ood |
| 2 | ACT | input tokens | id, ood |
| 3 | ACT + hybrid | analytic override | id, ood |
| 4 | SmolVLA | none | id, ood |
| 5 | SmolVLA + hybrid | analytic override | id, ood |
| 6 | ACT retrained on hybrid rollouts | distilled (G5) | id, ood |

Minimum ~30 trials per row for a 40-point effect at 80 % power; use a
sequential test to stop early. Log per-trial failure class and recovery
events alongside the binary score so the contact-phase hypothesis can be
checked directly rather than inferred from overall success.

## 8. References

Policies and VLAs

- Zhao, T., Kumar, V., Levine, S., Finn, C. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. RSS. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- Chi, C. et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- Shukor, M. et al. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
- Black, K. et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- Kim, M. J. et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- Hsu, K. et al. (2022). Vision-Based Manipulators Need to Also See from Their Hands. ICLR. [arXiv:2203.12677](https://arxiv.org/abs/2203.12677)
- Hugging Face LeRobot. [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

SO-101 benchmarks and data regime

- Yu, Y., Qiu, X. (2026). Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis. [arXiv:2606.08881](https://arxiv.org/abs/2606.08881)
- ArmnetBench v0.1 (2026). SO-101 fleet evaluation dataset; figures above are from the OpenBot.ai dataset summary and should be re-checked against the primary release before citation. [openbot.ai/datasets/armnetbench-v0-1-dataset](https://openbot.ai/datasets/armnetbench-v0-1-dataset)
- Lin, F. et al. (2025). Data Scaling Laws in Imitation Learning for Robotic Manipulation. ICLR. [arXiv:2410.18647](https://arxiv.org/abs/2410.18647)

Depth and 3D

- Ze, Y. et al. (2024). 3D Diffusion Policy. RSS. [arXiv:2403.03954](https://arxiv.org/abs/2403.03954)
- Zhou, A. et al. (2023). NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis (SPARTN). CVPR. [arXiv:2301.08556](https://arxiv.org/abs/2301.08556)
- Huang, Y. et al. (2025). Spatial RoboGrasp: Generalized Robotic Grasping Control Policy. [arXiv:2505.20814](https://arxiv.org/abs/2505.20814)
- Qu, D. et al. (2025). SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Models. [arXiv:2501.15830](https://arxiv.org/abs/2501.15830)
- Li, C. et al. (2025). PointVLA: Injecting the 3D World into Vision-Language-Action Models. [arXiv:2503.07511](https://arxiv.org/abs/2503.07511)
- Yuan, T. et al. (2025). DepthVLA: Enhancing Vision-Language-Action Models with Depth-Aware Spatial Reasoning. [arXiv:2510.13375](https://arxiv.org/abs/2510.13375)
- Rao, Z. et al. (2026). AugVLA-3D: Depth-Driven Feature Augmentation for Vision-Language-Action Models. [arXiv:2602.10698](https://arxiv.org/abs/2602.10698)

Hybrid, residual and online learning

- Paradis, S. et al. (2021). Intermittent Visual Servoing. ICRA. [arXiv:2011.06163](https://arxiv.org/abs/2011.06163)
- Sundaresan, P. et al. (2024). What's the Move? Hybrid Imitation Learning via Salient Points (Sphinx). [arXiv:2412.05426](https://arxiv.org/abs/2412.05426)
- Pan, C., Junge, K., Hughes, J. (2024). Vision-Language-Action Model and Diffusion Policy Switching Enables Dexterous Control of an Anthropomorphic Hand. [arXiv:2410.14022](https://arxiv.org/abs/2410.14022)
- Xu, X., Hou, Y., Liu, Z., Song, S. (2025). Compliant Residual DAgger. NeurIPS. [arXiv:2506.16685](https://arxiv.org/abs/2506.16685)
- ForceRFT (2026). Refining VLA Actions through Force-Guided Residual Reinforcement Learning. [arXiv:2609.22840](https://arxiv.org/abs/2609.22840)
- Kim, K. et al. (2026). Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement. [arXiv:2606.18953](https://arxiv.org/abs/2606.18953)
- Luo, J. et al. (2024). Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning (HIL-SERL). [arXiv:2410.21845](https://arxiv.org/abs/2410.21845)

Evaluation methodology

- Kress-Gazit, H. et al. (2024). Robot Learning as an Empirical Science: Best Practices for Policy Evaluation. [arXiv:2409.09491](https://arxiv.org/abs/2409.09491)
- Toyota Research Institute (2025). A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation. [arXiv:2507.05331](https://arxiv.org/abs/2507.05331)
- Snyder, D. et al. (2025). Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping. RSS. [doi:10.15607/rss.2025.xxi.077](https://doi.org/10.15607/rss.2025.xxi.077)
- Atreya, P. et al. (2025). RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies. [arXiv:2506.18123](https://arxiv.org/abs/2506.18123)
