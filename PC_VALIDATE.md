# Validation runbook — run the Veilaudit test suite on your own PC

Your 5800X3D / 32GB is the right machine for this. The VM here choked (2 CPUs,
7GB RAM); the full loop fits comfortably on your box. Total: ~10 minutes of
your time, then it runs itself.

## What's being validated

Two 45s 1080p test clips (`testdata/videos/club_a.mp4`, `club_b.mp4`), composited
from real FairFace photos (mostly Black men and white women) on nightclub-style
backgrounds. 8 programmed censorship violations: faces that unblur mid-video,
pixelation cases, and weak-blur edge cases. Per-frame ground truth lives in
`testdata/ground_truth/*.json`. The loop: baseline → score → auto-label → learn
→ re-run → score again, until event recall is complete and false positives are zero.

## 1. Copy the project to your PC

Skip `testdata/fairface/` (thousands of source photos — only needed to re-render,
not to validate) and `__pycache__`:

```bash
rsync -avz --exclude 'testdata/fairface' --exclude '__pycache__' --exclude '.venv' \
  <source>:~/workspace/Veilaudit/ ~/Veilaudit/
```

## 2. Set up

```bash
cd ~/Veilaudit
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

First run downloads the InsightFace model pack (~300MB → `~/.insightface`).
One-time, needs network briefly. Fully offline after that.

## 3. One config edit — force CPU

`onnxruntime` pip wheels have no ROCm build, so your RX 6800 can't run the
detector. CPU is genuinely fine here (detection runs on 640px frames every 3rd
frame). In `tests/test_config.yaml` set:

```yaml
detection:
  use_gpu: false
```

## 4. Baseline run

```bash
.venv/bin/python main.py -c tests/test_config.yaml --workers 4
```

Expected: a few minutes for both clips. Output lands in `testdata/audit_out/`
(flagged frames, `report.csv`/`report.json`, per-video `_tracks.json`).

## 5. Score the baseline

```bash
.venv/bin/python tests/evaluate.py \
  --gt testdata/ground_truth --out testdata/audit_out
```

Read `testdata/audit_out/eval_report.json`: event recall (want: all 8),
false positives (want: 0), track consistency (person #2 stays person #2).

## 6. Auto-label the review queue, then learn

```bash
.venv/bin/python tests/auto_label.py \
  --gt testdata/ground_truth --review-db testdata/audit_out/review.sqlite
.venv/bin/python learn.py tune
```

`learn.py tune` fits the face-confidence mapping and grid-searches the blur
thresholds against ground truth, writing `learned_params.json` — picked up
automatically on the next run.

## 7. Re-run and re-score

```bash
.venv/bin/python main.py -c tests/test_config.yaml --workers 4 --reset
.venv/bin/python tests/evaluate.py \
  --gt testdata/ground_truth --out testdata/audit_out
```

Iterate steps 6–7 until `eval_report.json` shows full event recall with zero
false positives. If a metric is stuck near a threshold, look at the flagged
crops in `testdata/audit_out/review/` — that's what `calibrate.py` is for.

## 8. Optional: hand-label the review UI instead of auto-labeling

For a feel of the human loop:

```bash
.venv/bin/python review_server.py   # opens http://127.0.0.1:8471
```

Answer "face or not" (F/N) and "blurred or sharp" (B/H) on the uncertain items,
then `learn.py tune` again. Real labels beat synthetic ground truth.

## Definition of done

- 8/8 violation events caught (event recall = 1.0)
- 0 false-positive frames
- Track IDs stable per person across each clip (check `_tracks.json`)

When the scorecard is clean, the pipeline is validated — then calibrate
thresholds on a sample of your *real* footage before the big batch, because
nightclub compression shifts the blur metrics.
