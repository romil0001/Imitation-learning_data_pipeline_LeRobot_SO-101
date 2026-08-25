# Public visualization assets

Committed figures for the GitHub README and documentation.

## Camera frames (`frames/`)

**Real-robot photos (committed):**

- `overhead_real.jpg` — scene view: gripper, metal cup, keys
- `wrist_real.jpg` — ego / wrist view looking toward the workspace

Illustrative synthetic layouts (optional condition sketches) may also appear from
`scripts/generate_public_assets.py`. Prefer the real photos for the README.

Replace or add more from recordings:

```bash
python scripts/generate_public_assets.py --dataset-dir data/datasets/so101_d405_demo
git add assets/frames/
```

## Eval rate graphs (`graphs/`)

Bar charts of **pc_success** overall and per condition (keys left / center / right).
Built from fixture summaries in `eval/` until you have real eval logs.

Regenerate from a real eval run:

```bash
python scripts/generate_public_assets.py \
  --eval-dir outputs/eval/act_so101_d405_demo/<run_id> \
  --eval-dir outputs/eval/hybrid_<run_id>
```

Or refresh illustrative fixtures + graphs:

```bash
python scripts/generate_public_assets.py
```

Requires: `pip install matplotlib opencv-python-headless`
