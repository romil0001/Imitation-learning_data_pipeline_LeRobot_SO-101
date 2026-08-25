# Research outcomes and evaluation graphs

This document maps **measurable outcomes** produced by the platform and how they
flow from data collection to reported metrics. See [`methods.md`](methods.md) for
methodological detail and [`reproduction.md`](reproduction.md) for hardware setup.

## Primary research outcomes

| Outcome | Definition | Where stored |
|---------|------------|--------------|
| **Demonstration corpus** | Teleoperated episodes with synchronized proprioception and video (+ optional depth) | `data/datasets/<name>/` |
| **Imitation policy** | ACT or fine-tuned SmolVLA checkpoint | `outputs/train/` |
| **Autonomous rollout** | Closed-loop follower execution under a loaded policy | `outputs/eval/<checkpoint>/<run_id>/trials/` |
| **Success rate** | Fraction of trials scored successful under a fixed protocol | `summary.json` → `pc_success` |
| **Per-condition breakdown** | Success rate stratified by manipulation condition (e.g. object layout) | `summary.json` → `by_condition` |

Train loss (`metrics.jsonl`) is a **training diagnostic**, not a reported outcome.
Reported manipulation performance uses **protocol eval** only.

Outcome graphs (mermaid): below. **Committed result plots:** [`assets/graphs/`](../assets/graphs/).

![Overall success rate](../assets/graphs/success_overall.png)

![Success by condition](../assets/graphs/success_by_condition.png)

## End-to-end outcome flow

```mermaid
flowchart LR
  subgraph acquisition["I. Data acquisition"]
    T[Leader teleoperation]
    F[Follower execution]
    C[Wrist RGB + depth<br/>Overhead RGB]
    T --> F
    C --> D[(Episode store)]
    F --> D
  end

  subgraph learning["II. Policy learning"]
    D --> ACT[ACT training]
    D --> CONV[LeRobot convert]
    CONV --> SV[SmolVLA fine-tune]
    ACT --> CKPT[(Checkpoints)]
    SV --> CKPT
  end

  subgraph deployment["III. Closed-loop eval"]
    CKPT --> POL{Policy}
    POL --> ACTe[ACT rollout]
    POL --> SVe[SmolVLA rollout]
    POL --> HY[Hybrid + depth grasp]
    ACTe --> TRIAL[Trial log]
    SVe --> TRIAL
    HY --> TRIAL
  end

  subgraph metrics["IV. Reported outcomes"]
    TRIAL --> SCORE[Human success score y/n]
    SCORE --> PC["pc_success"]
    SCORE --> COND["by_condition"]
    SCORE --> ART[(summary.json)]
  end
```

## Policy comparison graph

Three policy families share the same eval harness and protocol; they differ in
observation modality and contact-phase control.

```mermaid
flowchart TB
  OBS["Observations at eval<br/>wrist RGB · overhead RGB · wrist depth Z16 · qpos"]

  OBS --> ACTp["ACT<br/>RGB (+ depth in model)<br/>action chunks · temporal ensemble"]
  OBS --> SVp["SmolVLA<br/>RGB + language task<br/>no depth in backbone"]
  OBS --> HYp["Hybrid<br/>SmolVLA or ACT base<br/>+ DepthGraspController near contact"]

  ACTp --> OUT["Same eval protocol<br/>pen_to_keys.yaml"]
  SVp --> OUT
  HYp --> OUT

  OUT --> M1["Trial success bool"]
  OUT --> M2["Steps / early stop"]
  OUT --> M3["pc_success aggregate"]
```

**Hypothesis the hybrid stack tests:** language-driven visuomotor policies handle
approach and transport; metric wrist depth improves contact-phase height and
grasp reliability without feeding depth into the VLA backbone.

## Evaluation protocol graph

Default task: *Place keys into metal cup*. Protocol:
`config/eval/pen_to_keys.yaml`.

```mermaid
flowchart TD
  P[Load protocol YAML] --> GRID["Condition grid<br/>keys=left · center · right"]
  GRID --> N["n_trials_per_condition = 5"]
  N --> LOOP{For each trial}
  LOOP --> HOME[Interpolate to home qpos]
  HOME --> RESET[Human scene reset]
  RESET --> ROLL[Policy rollout ≤ timeout_s]
  ROLL --> SCORE["Operator scores y/n"]
  SCORE --> LOG["trials/NNNN/<br/>success · steps · condition_id"]
  LOG --> LOOP
  LOOP --> SUM["summary.json<br/>pc_success · by_condition"]
```

### Success criterion (fixed at eval time)

Keys inside the metal cup, gripper clear of the keys, nothing knocked over.
Each trial is scored independently after a fresh reset, so success rates reflect
real task completion and can be compared fairly across policies and conditions.

### Example outcome structure

After `python -m so101 eval --protocol config/eval/pen_to_keys.yaml`:

```text
outputs/eval/<checkpoint>/<run_id>/
  protocol.json      # frozen protocol + checkpoint metadata
  summary.json       # pc_success, by_condition, trial counts
  trials/0001/
    trial.json       # success, steps, condition_id, early_stop
  trials/0002/
    ...
```

`summary.json` fields used in papers/reports:

- `pc_success` — overall success rate in [0, 1]
- `by_condition` — map from condition id → `{successes, trials, rate}`
- `n_trials` — total scored trials

## Depth-assisted grasp outcome

Hybrid eval adds a **contact-phase sub-outcome**: whether depth-based vertical
correction reduced grasp height error before lift.

```mermaid
stateDiagram-v2
  [*] --> Approach: base policy
  Approach --> GraspAssist: depth < enter_mm
  GraspAssist --> GraspAssist: adjust shoulder_lift / elbow_flex
  GraspAssist --> Transport: depth > exit_mm OR gripper closed
  Transport --> Place: base policy
  Place --> [*]
```

Tune thresholds in `config/hybrid_grasp.yaml`. Logged trial metrics still use
the same binary task success; depth grasp is an **intervention**, not a separate
score, unless you add ablation conditions to the protocol YAML.

## Suggested reporting table (thesis / paper)

| Method | Observations | Contact phase | Metric |
|--------|--------------|---------------|--------|
| ACT | RGB (+ depth optional) | learned end-to-end | `pc_success`, `by_condition` |
| SmolVLA | RGB + language | learned end-to-end | same |
| Hybrid (SmolVLA + depth) | RGB + language; depth at grasp only | analytic depth controller | same |
| Hybrid (ACT + depth) | RGB + depth; depth at grasp only | analytic depth controller | same |

Hold camera config, task string, and protocol YAML constant across rows.
