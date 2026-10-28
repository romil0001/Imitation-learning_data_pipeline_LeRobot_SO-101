# Experiment pre-registration

Frozen before any scored trial. Changes after the first scored trial are amendments and are logged at the bottom with a date, not edited in place.

Companion documents: [`research_proposal.md`](research_proposal.md), [`analysis_plan.md`](analysis_plan.md), [`paper/main.pdf`](paper/main.pdf) section 5.

## Task and success criterion

Task string: *Place keys into metal cup*. A trial succeeds only if the keys are inside the metal cup, the gripper is clear of the keys, and nothing was knocked over. The criterion is the one committed in `config/eval/pen_to_keys.yaml` and is not reinterpreted per trial.

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

None yet.
