#!/usr/bin/env python3
"""Veilaudit: offline batch audit of videos for unblurred faces.

    python main.py --input-dir /data/videos --output-dir /data/audit_out

Everything runs locally (InsightFace/ONNX + OpenCV). Progress is checkpointed
to SQLite, so Ctrl-C / crashes resume where they left off.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# insightface internals raise a deprecation FutureWarning on import; not ours to fix.
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import cv2
import typer
import yaml
from tqdm import tqdm

from blurcheck import assess_face
from db import CheckpointDB
from detector import FaceDetector
from exporter import Exporter
from learn import apply_learned_params
from review_store import ReviewDB
from tracker import IoUTracker
from utils import expand_bbox

app = typer.Typer(add_completion=False)
log = logging.getLogger("Veilaudit")

_DETECTOR: FaceDetector | None = None


def _get_detector(det_cfg: dict, learned_params: str | None = None) -> FaceDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FaceDetector(
            model_pack=det_cfg["model_pack"],
            det_size=tuple(det_cfg["det_size"]),
            det_thresh=det_cfg["det_thresh"],
            use_gpu=det_cfg["use_gpu"],
            learned_params=learned_params,
        )
        log.info("detector ready (providers=%s)", _DETECTOR.active_providers)
    return _DETECTOR


def _worker_init(det_cfg: dict, learned_params: str | None = None) -> None:
    _get_detector(det_cfg, learned_params)  # warm one detector per worker process


def _is_borderline(metrics: dict, bc_cfg: dict, margin: float) -> bool:
    """True when a blur metric sits within `margin` of its threshold (uncertain call)."""
    et = bc_cfg["edge_threshold"]
    if abs(metrics["edge_den"] - et) / max(et, 1e-9) < margin:
        return True
    pft, plt = bc_cfg["pixel_flat_threshold"], bc_cfg["pixel_lap_threshold"]
    if (abs(metrics["flat_frac"] - pft) / max(pft, 1e-9) < margin
            and metrics["lap_var"] > plt * 0.5):
        return True
    return False


def _queue_review(review_db: ReviewDB, crops_dir: Path, thumb_cache: dict,
                  video_stem: str, video_name: str, frame, idx: int, fps: float,
                  w: int, h: int, bbox: tuple, reason: str,
                  pipeline_blur: str | None, metrics: dict | None,
                  det_score: float, fc: float, feats: dict | None,
                  track_id: int | None, n_queued: list) -> None:
    """Persist a crop + frame thumbnail and enqueue a human review item."""
    x1, y1, x2, y2 = bbox
    n_queued[0] += 1
    crop_path = crops_dir / "crops" / f"{video_stem}_f{idx:08d}_q{n_queued[0]:03d}.jpg"
    cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
    if idx not in thumb_cache:
        s = min(1.0, 960 / max(1, w))
        small = cv2.resize(frame, (int(w * s), int(h * s)))
        fp = crops_dir / "frames" / f"{video_stem}_f{idx:08d}.jpg"
        cv2.imwrite(str(fp), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumb_cache[idx] = fp
    rel_area = (x2 - x1) * (y2 - y1) / max(1, w * h)
    review_db.add_item(
        video=video_name, frame=idx, timestamp_s=idx / fps,
        crop_path=str(crop_path), frame_path=str(thumb_cache[idx]),
        bbox=bbox, track_id=track_id,
        det_score=det_score, face_confidence=fc,
        kps_feats=feats, rel_area=rel_area,
        reason=reason, pipeline_blur=pipeline_blur, metrics=metrics)


def iter_videos(input_dir: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in exts)


def _write_status(output_dir: str, video_stem: str, payload: dict) -> None:
    """Publish a tiny live-status JSON for the dashboard.

    Called once per checkpoint (~every 90 frames) plus once at completion.
    Best-effort: status is advisory and must never break the pipeline.
    """
    try:
        sdir = Path(output_dir) / "status"
        sdir.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        (sdir / f"{video_stem}.json").write_text(json.dumps(payload))
    except OSError:
        pass


def _process_video(video: str, cfg: dict, db_path: str, output_dir: str) -> dict:
    """Process one video end-to-end. Must be top-level for pickling."""
    db = CheckpointDB(db_path)
    if db.is_done(video):
        return {"video": video, "skipped": True, "rows": [], "stats": {}}

    det_cfg, vid_cfg, bc_cfg = cfg["detection"], cfg["video"], cfg["blur_check"]
    exporter = Exporter(output_dir, cfg)
    detector = _get_detector(det_cfg, cfg["paths"].get("learned_params"))

    rq_cfg = cfg.get("review_queue", {})
    review_enabled = bool(rq_cfg.get("enabled", True))
    review_thresh = float(rq_cfg.get("face_confidence_threshold", 0.80))
    blur_margin = float(rq_cfg.get("blur_margin", 0.25))
    review_db = ReviewDB(rq_cfg.get("db", "review.sqlite")) if review_enabled else None
    crops_dir = Path(rq_cfg.get("crops_dir", "review"))
    (crops_dir / "crops").mkdir(parents=True, exist_ok=True)
    (crops_dir / "frames").mkdir(parents=True, exist_ok=True)
    thumb_cache: dict[int, Path] = {}
    n_queued = [0]

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {"video": video, "error": "cannot open", "rows": [], "stats": {}}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start = db.last_frame(video)
    if start > 0:
        # MP4 seeks are keyframe-approximate; acceptable since we sample sparsely.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    sample_n = vid_cfg["sample_every_n_frames"]
    ckpt_n = vid_cfg["checkpoint_every_n_frames"]
    need = bc_cfg["min_fail_frames"]
    trk_cfg = cfg.get("tracking", {})
    use_tracking = bool(trk_cfg.get("enabled", True))
    tracker = IoUTracker(iou_thresh=trk_cfg.get("iou_thresh", 0.3),
                         max_missed=trk_cfg.get("max_missed_frames", 5),
                         window=bc_cfg["window_frames"])
    recent: deque[bool] = deque(maxlen=bc_cfg["window_frames"])  # fallback if tracking off

    rows: list[dict] = []
    stats = {"frames_read": 0, "frames_sampled": 0, "faces_seen": 0,
             "faces_failed": 0, "faces_skipped_small": 0, "tracks_total": 0,
             "review_queued": 0}
    recent_conf: deque[float] = deque(maxlen=12)  # rolling face-confidence window
    video_stem = Path(video).stem
    idx = start
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_n == 0:
                stats["frames_sampled"] += 1
                assessed = []  # [(entry, sharp, borderline, det, metrics)]
                for f in detector.detect(frame):
                    fc = f["face_confidence"]
                    recent_conf.append(round(fc, 3))
                    x1, y1, x2, y2 = expand_bbox(f["bbox"], h, w)
                    if min(x2 - x1, y2 - y1) < det_cfg["min_face_px"]:
                        stats["faces_skipped_small"] += 1
                        continue
                    # Low face confidence -> human review queue, never auto-judged.
                    if review_enabled and fc < review_thresh:
                        _queue_review(review_db, crops_dir, thumb_cache, video_stem,
                                      Path(video).name, frame, idx, fps, w, h,
                                      (x1, y1, x2, y2), "low_face_conf", None, None,
                                      f["score"], fc, f["kps_feats"], None, n_queued)
                        stats["review_queued"] += 1
                        continue
                    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    res = assess_face(roi, bc_cfg)
                    entry = {"bbox": (x1, y1, x2, y2), "score": round(f["score"], 3),
                             "metrics": res["metrics"], "track_id": None}
                    stats["faces_seen"] += 1
                    sharp = not res["censored"]
                    borderline = review_enabled and _is_borderline(
                        res["metrics"], bc_cfg, blur_margin)
                    assessed.append((entry, sharp, borderline, f, res["metrics"]))

                failures, passes, pending = [], [], []
                if use_tracking:
                    # Per-person verdicts: each distinct face gets a track ID and
                    # its own sharp/censored vote history. A track is flagged only
                    # after looking sharp in `need` recent sampled frames — so a
                    # blurred person never masks (or inherits the verdict of) an
                    # unblurred person next to them, and motion blur can't condemn
                    # a track on a single frame.
                    dets = [{"bbox": e["bbox"], "sharp": s, "borderline": b,
                             "entry": e, "det": d, "metrics": m}
                            for e, s, b, d, m in assessed]
                    for track, det in tracker.update(dets, idx):
                        entry = det["entry"]
                        entry["track_id"] = track.id
                        if det["borderline"]:
                            # Uncertain blur call -> human confirms the guess.
                            # Still counts as a vote so the track stays stable.
                            _queue_review(
                                review_db, crops_dir, thumb_cache, video_stem,
                                Path(video).name, frame, idx, fps, w, h,
                                entry["bbox"], "borderline_blur",
                                "sharp" if det["sharp"] else "blurred",
                                det["metrics"], det["det"]["score"],
                                det["det"]["face_confidence"],
                                det["det"]["kps_feats"], track.id, n_queued)
                            stats["review_queued"] += 1
                            pending.append(entry)
                        elif not det["sharp"]:
                            passes.append(entry)
                        elif track.is_confirmed(need):
                            failures.append(entry)
                        else:
                            pending.append(entry)  # sharp but not yet confirmed
                    stats["tracks_total"] = tracker.count
                else:
                    for entry, sharp, borderline, d, m in assessed:
                        if borderline:
                            _queue_review(review_db, crops_dir, thumb_cache, video_stem,
                                          Path(video).name, frame, idx, fps, w, h,
                                          entry["bbox"], "borderline_blur",
                                          "sharp" if sharp else "blurred",
                                          m, d["score"], d["face_confidence"],
                                          d["kps_feats"], None, n_queued)
                            stats["review_queued"] += 1
                            pending.append(entry)
                        elif sharp:
                            failures.append(entry)
                        else:
                            passes.append(entry)
                    recent.append(len(failures) > 0)
                    if not (failures and sum(recent) >= need):
                        failures = []  # hold export until persistence confirmed
                stats["faces_failed"] += len(failures)
                if failures:
                    ts = idx / fps
                    row = exporter.save_frame(frame, failures, passes, pending,
                                              video_stem, idx, ts)
                    row["video"] = Path(video).name
                    rows.append(row)
            stats["frames_read"] += 1
            if idx % ckpt_n == 0:
                db.checkpoint(video, idx)
                _write_status(output_dir, video_stem, {
                    "video": Path(video).name, "frame": idx,
                    "total_frames": total, "fps": round(fps, 2),
                    "status": "processing", "stats": dict(stats),
                    "recent_confidence": list(recent_conf),
                })
            idx += 1
    finally:
        cap.release()
    db.mark_done(video, idx)
    _write_status(output_dir, video_stem, {
        "video": Path(video).name, "frame": idx,
        "total_frames": total, "fps": round(fps, 2),
        "status": "done", "stats": dict(stats),
        "recent_confidence": list(recent_conf),
    })
    if use_tracking:
        stats["tracks_total"] = tracker.count
        tracks_path = Path(output_dir) / f"{video_stem}_tracks.json"
        tracks_path.write_text(json.dumps(tracker.summary(need), indent=1))
    log.info("%s: %d frames, %d faces, %d failed -> %d exported",
             Path(video).name, stats["frames_read"], stats["faces_seen"],
             stats["faces_failed"], len(rows))
    return {"video": video, "rows": rows, "stats": stats,
            "total_frames": total, "fps": fps}


def _write_csv(rows: list[dict], out_path: Path) -> None:
    fields = ["video", "frame", "timestamp_s", "timestamp_hms", "image",
              "faces_total", "faces_failed", "tracks_failed", "tracks_pending",
              "fail_boxes", "fail_metrics"]
    with open(out_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            rr = dict(r)
            for k in ("tracks_failed", "tracks_pending", "fail_boxes", "fail_metrics"):
                rr[k] = json.dumps(rr.get(k, []))
            wr.writerow(rr)


@app.command()
def run(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c",
                                help="Path to config.yaml"),
    input_dir: Path | None = typer.Option(None, "--input-dir", "-i"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    workers: int | None = typer.Option(None, "--workers", "-w",
                                       help="Worker processes (videos are the unit of work)"),
    reset: bool = typer.Option(False, "--reset",
                               help="Clear checkpoints and reprocess everything"),
):
    cfg = yaml.safe_load(config.read_text())
    if input_dir:
        cfg["paths"]["input_dir"] = str(input_dir)
    if output_dir:
        cfg["paths"]["output_dir"] = str(output_dir)
    if workers:
        cfg["run"]["workers"] = workers

    logging.basicConfig(level=cfg["run"].get("log_level", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")

    for note in apply_learned_params(cfg, cfg["paths"].get("learned_params",
                                                            "learned_params.json")):
        log.info("learned params: %s", note)

    in_dir = Path(cfg["paths"]["input_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    db = CheckpointDB(cfg["paths"]["checkpoint_db"])
    if reset:
        db.reset()
        log.info("checkpoints cleared")

    videos = iter_videos(in_dir, cfg["video"]["extensions"])
    if not videos:
        log.error("no videos found under %s", in_dir)
        raise typer.Exit(1)
    log.info("found %d videos", len(videos))

    n_workers = max(1, int(cfg["run"]["workers"]))
    all_rows: list[dict] = []
    per_video_stats: dict[str, dict] = {}

    if n_workers == 1:
        for v in tqdm(videos, desc="videos"):
            res = _process_video(str(v), cfg, str(db.path), str(out_dir))
            all_rows.extend(res.get("rows", []))
            per_video_stats[v.name] = res.get("stats", {})
    else:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_worker_init,
                                 initargs=(cfg["detection"],
                                           cfg["paths"].get("learned_params"))) as ex:
            futs = {ex.submit(_process_video, str(v), cfg, str(db.path), str(out_dir)): v
                    for v in videos}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="videos"):
                v = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001 — keep the batch alive
                    log.exception("worker failed on %s: %s", v.name, e)
                    continue
                all_rows.extend(res.get("rows", []))
                per_video_stats[v.name] = res.get("stats", {})

    all_rows.sort(key=lambda r: (r["video"], r["frame"]))
    _write_csv(all_rows, out_dir / "report.csv")
    (out_dir / "report.json").write_text(json.dumps(all_rows, indent=1))
    (out_dir / "stats.json").write_text(json.dumps(per_video_stats, indent=1))
    log.info("done: %d flagged frames across %d videos -> %s",
             len(all_rows), len(videos), out_dir)


if __name__ == "__main__":
    sys.exit(app())
