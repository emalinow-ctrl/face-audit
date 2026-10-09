# Veilaudit

Offline batch pipeline that audits folders of video for **improperly censored (unblurred) faces**.
Everything runs locally — InsightFace/ONNX face detection, classic CV blur metrics,
OpenCV video I/O. No cloud APIs, no content filters, no data leaves the machine.

## How it works

1. **Decode** each video with OpenCV/FFmpeg, sampling every Nth frame (configurable).
2. **Detect** faces with InsightFace's SCRFD detector (`buffalo_l` bundle, ONNX Runtime).
   Detection runs on a downscaled frame; blur is always measured on the **full-resolution** crop.
3. **Judge** each face: Laplacian variance + FFT high-frequency ratio + edge density,
   plus a solid-mask check (black bars). A face failing the blur check is "unblurred".
4. **Per-person tracking**: each distinct face gets a stable track ID (IoU tracker,
   `tracker.py`) with its own sharp/blurred vote history. A person is flagged only
   after looking sharp in `min_fail_frames` of their recent sampled frames — so a
   blurred face never masks (or inherits the verdict of) an unblurred face next to
   it, and motion blur can't condemn someone on a single frame. Faces that look
   sharp but aren't confirmed yet are drawn orange ("REVIEW #id").
5. **Export** only flagged frames (red boxes on unblurred faces, labeled with track
   IDs) and write `report.csv` / `report.json`: video, frame, timestamp, track IDs,
   boxes, metric values — plus `<video>_tracks.json` with a per-person summary
   (first/last seen, sharp vs censored counts, confirmed verdict).
6. **Checkpoint** every video's frame position to SQLite — Ctrl-C or crash resumes cleanly.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# NVIDIA GPU? swap the CPU runtime for the GPU one (never install both):
#   pip uninstall onnxruntime && pip install onnxruntime-gpu
```

First run downloads the InsightFace model pack once (~300 MB → `~/.insightface`);
after that the machine can be fully air-gapped.

## Calibrate first (important)

Blur thresholds depend on your source quality. Don't trust the defaults blindly:

```bash
python calibrate.py --video /data/videos/sample.mp4 --out calib_out --max-faces 400
```

This dumps face crops named like `f000123_i0_LAP0042_HF0.031_ED0.012.jpg` plus a
`metrics.csv`. Eyeball which crops you consider blurred vs sharp, find the
Laplacian / HF-ratio values that separate them, and set
`blur_check.laplacian_threshold` / `blur_check.hfratio_threshold` in `config.yaml`.

## Run

```bash
python main.py --input-dir /data/videos --output-dir /data/audit_out --workers 4
# resume is automatic; --reset clears checkpoints and reprocesses everything
```

## Human review queue & learning

The pipeline doesn't guess when it's unsure — it asks you:

- **Face confidence.** Every detection gets a `face_confidence` score (detector
  score × 5-point landmark geometry plausibility). Below 0.80
  (`review_queue.face_confidence_threshold`), the crop goes to the review queue
  instead of being auto-judged.
- **Borderline blur.** If a blur metric lands within `blur_margin` (25%) of its
  threshold, the pipeline's blurred/sharp guess goes to the queue for confirmation.

Review in the browser (stdlib only, no dependencies):

```bash
python review_server.py              # -> http://127.0.0.1:8471
```

Keyboard-driven: `F` = contains a face, `N` = not a face, then `B`/`H` for
blurred vs. sharp. Each item stores the crop, a frame thumbnail, the pipeline's
guess, and the features behind it.

Then make the pipeline learn from your corrections:

```bash
python learn.py report               # how well current thresholds match your labels
python learn.py tune                  # fit face-confidence mapping + blur thresholds
                                      # -> learned_params.json
python main.py --reset ...            # re-run so learned params apply everywhere
```

`tune` fits a tiny logistic regression on your face/not-face labels (replacing
the heuristic confidence once you have 50+ labels) and grid-searches the blur
thresholds for best F1 on your blurred/sharp labels (30+ needed). `main.py`
picks up `learned_params.json` automatically and logs what it applied.

To feed corrections into the v2 classifier instead:

```bash
python learn.py export-dataset --out data/review_labels
# -> face_sharp/ face_blurred/ not_face/ — merge into train_classifier.py training
```

Outputs under the output dir:

- `<video_stem>/frame_00000123_t12.40s.jpg` — flagged frames, red = unblurred face
- `report.csv` — one row per flagged frame: video, frame, timestamp, boxes, metrics
- `report.json` — same, structured
- `stats.json` — per-video counts (frames, faces seen/failed/skipped)
- `checkpoints.sqlite` — resume state

## Config reference (`config.yaml`)

| Section | Key knobs |
|---|---|
| `video` | `sample_every_n_frames` (2–5), `checkpoint_every_n_frames` |
| `detection` | `det_size`, `det_thresh`, `use_gpu`, `min_face_px` (skip tiny faces) |
| `blur_check` | `laplacian_threshold`, `hfratio_threshold`, per-track `min_fail_frames`/`window_frames` |
| `tracking` | `enabled`, `iou_thresh`, `max_missed_frames` |
| `export` | box drawing, JPEG quality |
| `run` | `workers` (one process per video) |

## Performance

- SCRFD at 640 px does ~15–30 fps on a modern CPU; sampling every 3rd frame
  processes a 1-hour video in roughly 10–20 min on CPU, near real-time on GPU.
- Multiprocessing is per-video; one detector instance per worker.

## Upgrading to a learned classifier

If hand-tuned metrics are too noisy, `train_classifier.py` trains a MobileNetV3-Small
binary classifier on public face crops with *synthesized* censorship
(gaussian blur, mosaic pixelation, black bars) — no real footage needed for training —
and exports ONNX. The module docstring shows the ~10 lines needed to plug it into
`blurcheck.py` as a tiebreaker.

## Limitations (honest)

- Faces smaller than `min_face_px` are skipped and counted, not judged.
- Heavy motion blur / compression artifacts can mimic censorship blur; temporal
  smoothing mitigates but doesn't eliminate this.
- MP4 resume seeks are keyframe-approximate — fine for sparse sampling, not frame-exact.
- Partial occlusions (hand over face) are not censorship; only blur/pixelation/masking
  of detected faces is judged.

## Privacy

All processing is local. Flagged-frame exports contain faces by design — treat the
output directory with the same care as the source footage.
