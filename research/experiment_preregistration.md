# Experiment pre-registration

Frozen before any scored trial. Changes after the first scored trial are amendments and are logged at the bottom with a date, not edited in place.

Companion documents: [`research_proposal.md`](research_proposal.md), [`analysis_plan.md`](analysis_plan.md), [`paper/main.pdf`](paper/main.pdf) section 5.

## Task and success criterion

### Primary pre-registration (keys in cup)

Task string: *Place keys into metal cup*. A trial succeeds only if the keys are inside the metal cup, the gripper is clear of the keys, and nothing was knocked over. The criterion is the one committed in `config/eval/pen_to_keys.yaml` and is not reinterpreted per trial. The powered experiment matrix in this document applies to this protocol until a formal amendment moves the primary endpoint (see amendments below).

### Development protocol (bag clip in bowl) — added Nov 2026

Task string: *Place bag clip in bowl*. Success: bag clip inside the bowl, gripper clear of the clip, nothing knocked over. Frozen in `config/eval/bag_clip_in_bowl.yaml`. In-distribution conditions: `clip=left`, `clip=center`, `clip=right` (placement slots `pen: left|center|right`). This protocol is used for ongoing real-robot ACT/hybrid development and exploratory logging; **early ACT-only runs are incomplete and ~15% success overall to date**. They do not satisfy the trial budget or hypothesis decision rules below until explicitly promoted via amendment.

## Conditions

| Split | Condition | Manipulation |
|---|---|---|
| id | `keys=left`, `keys=center`, `keys=right` | Key position on taped marks. Lighting overhead, no distractors. |
| ood | `height+2cm` | Table raised 2 cm with a riser. Keys centred. |
| ood | `ring-large` | A larger key ring than any in the demonstrations. Keys centred. |
| ood | `distractors` | Two extra objects on the table, outside the grasp zone. Keys centred. |

The out-of-distribution rows are added to the protocol YAML in phase 1 and then frozen.

## Arms

Every arm runs on the same cell, cameras, task string and operator script.

| Row | Base policy | Depth use | Tests |
|---|---|---|---|
| A1 | ACT | none | baseline |
| A2 | ACT | depth as input tokens | H2 |
| A3 | ACT + hybrid, threshold gate | analytic override | H1 |
| A4 | ACT + hybrid, learned gate | analytic override | H4 |
| A5 | ACT retrained on corrected rollouts, RGB only | distilled | H3 |
| S1 | SmolVLA | none | baseline |
| S2 | SmolVLA + hybrid, threshold gate | analytic override | H1 |
| S3 | SmolVLA + hybrid, learned gate | analytic override | H4 |

Within hybrid rows the hand-back rule (overwrite versus flush) is counter-balanced. Rows A3 and S2 additionally sweep the entry threshold over 150, 180 and 220 mm; the primary comparison uses the shipped value of 180 mm.

## Budget

- Minimum 30 in-distribution trials per row, 10 per key position.
- Sequential cap of 60 in-distribution trials per row. Stopping follows the sequential test named in the analysis plan.
- Out-of-distribution conditions are a secondary endpoint at 10 trials per condition per surviving row.

Thirty trials gives about 85% power for a 40-point gap and about 30% for a 20-point gap (two-sided Fisher test, α = 0.05). Effects under 20 points are exploratory unless the sequential test terminates.

## Procedure controls

- **Order.** Rows are interleaved within a session (A1, A3, S1, S2, and so on), not blocked, so lighting, battery and operator drift are not confounded with policy.
- **Reset.** The arm returns to the demonstration start pose before every trial. The scene is reset to taped marks and the initial frame is saved with the trial log.
- **Blinding.** The scorer is not told which row produced a trial. Where the motion itself reveals the row, the trial is marked unblinded in the log and reported separately.
- **Tuning leakage.** Controller parameters are fixed on a separate set of ten unscored rollouts before the first scored trial. The primary comparison does not use a threshold chosen on scored data.
- **Checkpoints.** Each learned policy is the `best_val` checkpoint from one training run with the seed recorded in `split.json`. No checkpoint is swapped after scoring starts.

## What is logged per trial

Binary success, primary failure class (grasp instability, repetition loop, state mismatch, precision misalignment, false trigger, stale chunk), recovery opportunities and successes, time to first closure, steps, and for hybrid rows the depth error at closure. Counts are stored alongside rates.

## Amendments

### 2026-11 — Bag-clip protocol and platform instrumentation (additive)

- **Added** evaluation protocol `config/eval/bag_clip_in_bowl.yaml` and demonstration dataset `so101_bag_clip_bowl` (~48 teleoperated RGB-D episodes) as an active real-robot task alongside keys-in-cup. Default training/eval stack: ACT, ResNet-18 encoder, depth tokens, chunk length 100, checkpoint selection `best_val`.
- **Added** logging path for scored runs: `outputs/eval/<policy>/<timestamp>/` via `python -m so101 eval --protocol config/eval/bag_clip_in_bowl.yaml` (and dashboard eval launch). Successful trials may append trajectories to `so101_bag_clip_bowl_eval` when marked successful.
- **Added** interactive eval control recipes without changing success criteria: `config/recipes/act25.yaml`, `act200.yaml`, `reactive_act1.yaml`.
- **Noted** platform features now shipped: ZMQ startup failover (Ethernet before Wi-Fi in `ZMQ_HOSTS`), HG-DAgger scripts `scripts/18_dagger.sh` and `scripts/19_train_dagger.sh`.
- **Unchanged:** minimum 30 ID trials per matrix row, sequential cap 60, blinding and stopping rules for H1–H4 still refer to the keys protocol and arms A1–S3 until a future amendment designates bag-clip as the primary powered endpoint with the same budget.
- **Transparency:** incomplete exploratory bag-clip ACT protocol scores (~15% success on runs logged so far) are **not** powered results and must not be used in place of the pre-registered keys comparison.
