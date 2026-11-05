# Public visualization assets

Committed figures for the GitHub README and documentation.

## Camera frames (`frames/`)

**Real-robot photos (committed):**

- `overhead_real.jpg` — scene view: gripper, metal cup, keys (legacy framing)
- `wrist_real.jpg` — ego / wrist view looking toward the workspace

Primary task today is **bag clip in bowl** (`so101_bag_clip_bowl`). Prefer fresh
frames from that dataset for README updates.

Replace or add from recordings:

```bash
python scripts/generate_public_assets.py --dataset-dir data/datasets/so101_bag_clip_bowl
git add assets/frames/
```

Legacy keys demos: `data/datasets/so101_d405_demo`.

## Eval rate graphs (`graphs/`)

Bar charts of **pc_success** overall and per condition. The committed PNGs were
built from **pen-to-keys fixture** summaries in `eval/` — illustrative only.

Regenerate from a real bag-clip (or other) protocol run:

```bash
python scripts/generate_public_assets.py \
  --eval-dir outputs/eval/act_so101_bag_clip_bowl/<run_id>
```

Or refresh illustrative fixtures + graphs:

```bash
python scripts/generate_public_assets.py
```

Requires: `pip install matplotlib opencv-python-headless`
