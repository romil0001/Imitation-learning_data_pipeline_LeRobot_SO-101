# Analysis plan

Specifies every comparison before the data exist. Estimators, tests and decision rules below are the ones the paper will report. Anything not listed here is exploratory and is labelled as such.

## Estimands

For each row and each split (in distribution, out of distribution):

- Success rate p, reported as successes over trials, never as a percentage alone.
- Per-condition success rate for each condition in the protocol.
- Failure-class distribution over failed trials.
- Recovery rate: successful recoveries over recovery opportunities.
- For hybrid rows, the depth error at closure, e = d − d*, in millimetres.

The primary estimand for H1, H2 and H4 is the in-distribution success-rate difference between a row and its baseline (A3−A1, A2−A1, A4−A3, S2−S1, S3−S2). The primary estimand for H3 is A5−A1 in distribution and, secondarily, how much of the A3−A1 gain A5 retains. The primary estimand for the out-of-distribution question is the interaction: the out-of-distribution gain minus the in-distribution gain, per row.

## Estimates and intervals

- Point estimate: successes divided by trials.
- Interval: 95% Wilson score interval. Intervals are the default display; a bar without an interval is not a result.
- Posterior: Beta(k+1, n−k+1) from a uniform prior, reported as P(p_row > p_baseline | data).

## Tests

- Pairwise success comparison: two-sided Fisher exact test on the pooled trials of that split.
- Sequential stopping: the sequential policy-comparison test of Snyder et al. (RSS 2025), capped at 60 in-distribution trials. The test, not a peek at the point estimate, decides whether to stop.
- Failure-class and recovery comparisons are descriptive. They are not a second primary endpoint and are not thresholded.

## Decision rules

| Hypothesis | Accepted as supported when | Otherwise |
|---|---|---|
| H1 | Hybrid minus base is at least 20 points in distribution, the sequential test has terminated in favour of the hybrid, and the gain is concentrated in grasp-instability trials. | Reported as not supported, with the interval. |
| H2 | RGB-D ACT is not better than RGB ACT in distribution: the interval on the difference covers zero or favours RGB. | If RGB-D wins in distribution, H2 is rejected and that result is the finding. |
| H3 | The distilled RGB policy recovers at least half of the hybrid's in-distribution gain, and a smaller share of its out-of-distribution gain. | Reported as partial or not supported. |
| H4 | The learned gate has no more false-trigger or stale-chunk events than the threshold gate, success is not lower, and it degrades less across the 150/180/220 mm sweep. | Reported per error type; no composite score is invented to rescue it. |

A result that misses its rule is reported in full. Rules are not relaxed after seeing the data.

## Reporting

Each comparison is a row in one table: successes/trials for both arms, Wilson intervals, the posterior probability, whether the sequential test terminated, and the failure-class breakdown. Fixture data currently in `assets/eval/` is illustrative and is never mixed with scored trials. Figures are regenerated from the trial logs with `research/scripts/make_figures.py --eval-json`.

## What this plan does not do

- No test on training loss. Training loss is a diagnostic and is not a manipulation result.
- No pooling of in-distribution and out-of-distribution trials into one success rate.
- No claim about arms, tasks or sensors that were not in the pre-registration.
