# Research documents

This folder holds the machine-learning research programme built on the SO-101
platform. It is written as a PhD-style research project: questions, hypotheses,
formal problem statement, pre-registered experiments and analysis plan. The
engineering pipeline itself is documented in [`../docs/`](../docs/).

| Document | Purpose |
|---|---|
| [`paper/main.tex`](paper/main.tex) → [`paper/main.pdf`](paper/main.pdf) | Full research proposal: formulation, method (four uses of one depth stream), pre-registered design, pilot analysis, research plan. |
| [`research_proposal.md`](research_proposal.md) | Short-form proposal: thesis statement, research questions, hypotheses, contributions, scope. |
| [`experiment_preregistration.md`](experiment_preregistration.md) | The experiment matrix, conditions, trial budgets, blinding and stopping rules, frozen before scored trials begin. |
| [`analysis_plan.md`](analysis_plan.md) | Metrics, estimators, tests, decision rules and reporting format for every pre-registered comparison. |
| [`../docs/literature_review.md`](../docs/literature_review.md) | Literature review and gap analysis the programme is built on. |
| [`scripts/make_figures.py`](scripts/make_figures.py) | Regenerates every quantitative figure in `paper/figures/` (drives the real `DepthGraspController`). |

## Research question in one sentence

On a force-blind, position-controlled low-cost arm, does metric wrist depth
improve grasp-phase reliability of RGB imitation policies more effectively as an
analytic contact-phase override, as a model input, or as a training-time
teacher, and does the benefit grow out of distribution?

## Platform snapshot (November 2026)

- **Tasks:** keys-in-cup (`config/eval/pen_to_keys.yaml`, pre-registered powered matrix) and **bag-clip-in-bowl** (`config/eval/bag_clip_in_bowl.yaml`, active real-robot development).
- **Demonstrations:** `so101_bag_clip_bowl` (~48 episodes, RGB-D); default CLI/train/eval dataset name in the repo.
- **Baseline policy:** ACT + ResNet-18 + depth tokens, chunk 100, `best_val` under `outputs/train/`.
- **Eval:** protocol eval → `outputs/eval/<run>/`; dashboard launch; recipes `config/recipes/act25.yaml`, `act200.yaml`, `reactive_act1.yaml`.
- **Infrastructure:** Omen ZMQ publisher with Ethernet-then-Wi-Fi failover (`ZMQ_HOSTS`); successful eval trajectories can append to `so101_bag_clip_bowl_eval`; HG-DAgger via `scripts/18_dagger.sh` / `19_train_dagger.sh`.
- **Honest status:** exploratory bag-clip ACT protocol runs are logged but **incomplete** (~15% success on runs to date). No pre-registered powered comparison has been completed; paper fixtures remain illustrative.

Code: [github.com/romil0001/SO101_Visuomotor_Manipulation](https://github.com/romil0001/SO101_Visuomotor_Manipulation)

## Building the paper

```bash
pip install matplotlib numpy opencv-python-headless pyyaml   # figures
python research/scripts/make_figures.py                       # writes paper/figures/
cd research/paper && make                                     # needs TeX Live incl. texlive-science
```

After a real protocol run, rebuild the result figures from the logs instead of
the illustrative fixtures:

```bash
python research/scripts/make_figures.py \
  --eval-json outputs/eval/<act_ckpt>/<run>/summary.json \
  --eval-json outputs/eval/<hybrid_ckpt>/<run>/summary.json
```

## Figures

| File | Source |
|---|---|
| `overhead_real.jpg`, `wrist_real.jpg` | Real frames from the cell (`assets/frames/`). |
| `roi_overlay.pdf` | Repository's placeholder depth rendering with the controller ROI drawn on it. Replace with a real D405 frame once exported. |
| `ci_width.pdf`, `power.pdf` | Wilson interval width and Fisher-test power versus trial count. |
| `controller_response.pdf` | Closed-loop response of the shipped `DepthGraspController` against a 1-D plant. |
| `chunk_override.pdf` | Schematic of the override / action-chunk interaction. |
| `success_with_ci.pdf`, `posteriors.pdf` | Reporting pipeline applied to `assets/eval/` fixtures. Labelled illustrative; not measurements. |
