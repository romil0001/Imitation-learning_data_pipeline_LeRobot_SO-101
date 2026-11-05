# Eval and session logs

Short pointer for operators — details in [`reproduction.md`](reproduction.md) and
[`methods.md`](methods.md).

## Protocol eval output

Completed scored runs write under:

```text
outputs/eval/<checkpoint_slug>/<run_id>/
  protocol.json
  summary.json       # pc_success, per_condition, n_trials — only when the grid finishes
  trials/0001/trial.json
  ...
```

Launch interactively: `bash scripts/11_eval_act.sh` (or `15_eval_hybrid.sh`).
Non-interactive: `INTERACTIVE=0 POLICY=act POLICY_PATH=... CONTROL_MODE=act25 bash scripts/11_eval_act.sh`.

Partial sessions (quit early, timed abort) may leave `trials/` without
`summary.json`. Compare two finished runs:

```bash
python -m so101 compare-eval --baseline outputs/eval/.../summary.json --candidate outputs/eval/.../summary.json
```

After DAgger fine-tune, `bash scripts/20_rescore_compare.sh` prints the same paths.

## Session log

Numbered scripts append stdout/stderr via `run_logged`:

```text
outputs/session.log
```

Includes train, record, eval, and DAgger wrappers — useful when a terminal scrollback
is lost.

## Success trajectories

When protocol eval saves successes (default), episodes land in:

```text
data/datasets/<dataset>_eval/
```

Tagged `eval_success` in episode metadata.
