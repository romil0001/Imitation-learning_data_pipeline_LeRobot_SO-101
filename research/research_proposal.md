# Research proposal

**Title.** Metric wrist depth as a contact proxy for imitation-learned grasping on force-blind low-cost manipulators.

**Author.** Romil Modi

**Form.** PhD-style research programme in machine learning for visuomotor control. The full formulation, method, experimental design and pilot analysis are in [`paper/main.pdf`](paper/main.pdf) (updated November 2026).

## Thesis

On a force-blind, position-controlled low-cost arm, metric wrist depth can stand in for the force feedback the embodiment lacks. Where that depth enters the learning problem — as a model input, as an analytic contact-phase override, as a training-time teacher, or as a learned gate — determines whether it improves grasp reliability, and the benefit is larger out of distribution than in distribution.

## Why this is a learning problem

Independent SO-101 benchmarks find grasp instability in more than 90% of failures of ACT and SmolVLA. The remedies that work on research-grade arms condition a residual or a corrective policy on wrist force, which this arm does not have. The literature also disagrees about depth itself: feeding a depth image into a behaviour-cloning policy can reduce success, while distilling geometry into an RGB policy can remove the sensor at test time. The hybrid controller already in the codebase is the instrument for settling that disagreement, not the contribution.

## Research questions

1. Does a contact-phase override driven only by metric wrist depth improve grasp reliability of frozen RGB policies (ACT, SmolVLA)?
2. Holding the depth stream fixed, which use of depth wins: model input, analytic override, or training-time teacher?
3. Can the hand-off from the learned policy to the contact law be learned from demonstrations, and does a learned gate avoid the false triggers and stale action chunks of a threshold gate?
4. Is the benefit larger out of distribution (table height, object scale, distractors) than in distribution?

## Hypotheses

| ID | Claim | Falsified if |
|---|---|---|
| H1 | The depth-gated hybrid gains at least 20 points over its frozen base, concentrated in grasp-instability failures. | The gain is under 20 points at the pre-registered budget, or it is spread across failure classes. |
| H2 | With tens of demonstrations, depth as an ACT input does not help in distribution and can hurt. | RGB-D ACT beats RGB ACT in distribution by a margin the sequential test accepts. |
| H3 | Retraining RGB-only ACT on override-corrected rollouts recovers most of the in-distribution gain without the sensor, and less of the out-of-distribution gain. | The distilled policy keeps neither, or keeps both equally. |
| H4 | A learned gate cuts false triggers and stale-chunk events at equal or higher success, and degrades less when the entry threshold is mis-set. | It matches the threshold gate on both error types, or only wins by being tuned to one threshold. |

## Contributions

1. A grasp-height error model under which the ROI median depth is a sufficient statistic for the unobserved vertical error, and a switched-policy formulation of the hybrid controller.
2. A four-way controlled comparison of depth usage on one embodiment and pre-registered evaluation protocols.
3. An account of how an external override interacts with open-loop action chunking, with flush and blend hand-back rules.
4. A learned gate whose labels are recovered from gripper closure in existing demonstrations.
5. A powered, pre-registered single-cell evaluation protocol with a failure taxonomy and sequential testing.

## Scope

In scope: ACT and SmolVLA, one SO-101 cell, pick-and-place tasks with metric wrist depth from the D405.

- **Pre-registered powered matrix:** `config/eval/pen_to_keys.yaml` (keys in cup), extended with out-of-distribution conditions when phase 1 completes.
- **Active development (Nov 2026):** bag-clip-in-bowl — dataset `so101_bag_clip_bowl`, protocol `config/eval/bag_clip_in_bowl.yaml`, ACT+ResNet-18+depth (chunk 100, `best_val`). Early protocol eval is logged under `outputs/eval/`; success trajectories can be saved to `so101_bag_clip_bowl_eval`. Interactive eval recipes: `config/recipes/act25.yaml`, `act200.yaml`, `reactive_act1.yaml`. HG-DAgger: `scripts/18_dagger.sh`, `19_train_dagger.sh`.

Out of scope: new robot platforms, force sensing, residual reinforcement learning, and training a VLA from scratch. A lightweight depth adapter inside SmolVLA is a second-phase extension, not part of the first comparison.

## Programme

Five phases, each gated by a deliverable: instrumentation of the logger and protocol; H1 and H2; distillation (H3); the learned gate (H4); the out-of-distribution split and synthesis. Detail, risks and the pilot analysis on fixture data are in the paper.

**Status (November 2026):** instrumentation for protocol eval, ZMQ Ethernet/Wi-Fi failover, dashboard eval control, `*_eval` success-demo export, and bag-clip data/training are in place. Exploratory bag-clip ACT protocol runs show roughly **15% success on incomplete runs** — not evidence for or against H1–H4. The pre-registered keys matrix at 30+ ID trials per row has **not** been completed; do not treat exploratory bag-clip logs as powered results.
