#!/usr/bin/env python3
"""Learn from human review labels.

    python learn.py tune --config config.yaml
        Fits a logistic-regression face-confidence mapping from face/not-face
        labels, and grid-searches blur thresholds from blurred/sharp labels.
        Writes learned_params.json (merged with any existing file).

    python learn.py export-dataset --config config.yaml --out data/review_labels
        Copies labeled crops into face_sharp/ face_blurred/ not_face/ for
        retraining the v2 classifier (see train_classifier.py).

    python learn.py report --config config.yaml
        Prints what the labels say about current thresholds.

Workflow: run pipeline -> review in the browser -> tune -> re-run pipeline
with --reset so the learned parameters apply to everything.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import typer
import yaml

from detector import FACE_FEATURES
from review_store import ReviewDB

app = typer.Typer(add_completion=False)
log = logging.getLogger("learn")

MIN_FACE_SAMPLES = 50
MIN_BLUR_SAMPLES = 30


# ------------------------------------------------------------- logistic regression
def _fit_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.5,
                iters: int = 3000, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    """Plain-numpy binary logistic regression; returns (weights_with_bias, train_acc)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xs = (X - mu) / sd
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * grad
    acc = float(((1.0 / (1.0 + np.exp(-Xb @ w)) >= 0.5) == y).mean())
    return w, acc, mu, sd


def _face_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in rows:
        if r["label_face"] is None:
            continue
        feats = [r["det_score"], r["eye_ratio"], r["kps_inside"], r["order_ok"]]
        if any(v is None for v in feats):
            continue
        X.append(feats)
        y.append(r["label_face"])
    return np.array(X, dtype=float), np.array(y, dtype=float)


# ------------------------------------------------------------- blur threshold search
def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / max(1e-9, prec + rec)


def _predict_blurred(laps: np.ndarray, flats: np.ndarray, eds: np.ndarray,
                     et: float, plt: float, pft: float) -> np.ndarray:
    """Mirror of blurcheck.assess_face decision: pixelated OR blurred => 1."""
    pixelated = (flats > pft) & (laps > plt)
    blurred = eds < et
    return (pixelated | blurred).astype(int)


def _tune_blur(rows: list[dict]) -> dict | None:
    """Grid-search edge/pixel thresholds maximizing F1 on human blur labels."""
    laps, flats, eds, ys = [], [], [], []
    for r in rows:
        if r["label_face"] != 1 or r["label_blur"] not in ("blurred", "sharp"):
            continue
        if r["lap_var"] is None or r["flat_frac"] is None or r["edge_den"] is None:
            continue
        laps.append(r["lap_var"])
        flats.append(r["flat_frac"])
        eds.append(r["edge_den"])
        ys.append(1 if r["label_blur"] == "blurred" else 0)
    laps, flats, eds, ys = map(np.array, (laps, flats, eds, ys))
    if len(ys) < MIN_BLUR_SAMPLES or ys.min() == ys.max():
        return None
    best = {"f1": -1.0}
    q = np.linspace(0.05, 0.95, 13)
    for et in np.quantile(eds, q):
        for plt in np.quantile(laps, q[::2]):
            for pft in np.quantile(flats, q[::2]):
                f1 = _f1(ys, _predict_blurred(laps, flats, eds, et, plt, pft))
                if f1 > best["f1"]:
                    best = {"f1": f1, "edge_threshold": float(et),
                            "pixel_lap_threshold": float(plt),
                            "pixel_flat_threshold": float(pft)}
    best["n"] = int(len(ys))
    return best


# ------------------------------------------------------------- learned params file
def load_learned(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save_learned(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))


def apply_learned_params(cfg: dict, path: str | Path) -> list[str]:
    """Override config thresholds with learned ones. Returns human-readable notes."""
    notes = []
    data = load_learned(path)
    blur = data.get("blur") or {}
    if blur.get("n", 0) >= MIN_BLUR_SAMPLES:
        bc = cfg["blur_check"]
        bc["edge_threshold"] = blur["edge_threshold"]
        bc["pixel_lap_threshold"] = blur["pixel_lap_threshold"]
        bc["pixel_flat_threshold"] = blur["pixel_flat_threshold"]
        notes.append(
            f"blur thresholds <- learned (edge<{blur['edge_threshold']:.4f}, "
            f"pixel: flat>{blur['pixel_flat_threshold']:.3f} & "
            f"lap>{blur['pixel_lap_threshold']:.1f}) from {blur['n']} labels")
    fc = data.get("face_conf") or {}
    if fc.get("n", 0) >= MIN_FACE_SAMPLES:
        notes.append(f"face-confidence mapping <- learned from {fc['n']} labels "
                     f"(train acc {fc.get('train_acc', 0):.3f})")
    return notes


# ------------------------------------------------------------- commands
def _load_cfg(config: Path) -> tuple[dict, ReviewDB]:
    cfg = yaml.safe_load(config.read_text())
    rq = cfg.get("review_queue", {})
    return cfg, ReviewDB(rq.get("db", "review.sqlite"))


@app.command()
def tune(config: Path = typer.Option(Path("config.yaml"), "--config", "-c")):
    """Fit face-confidence mapping + blur thresholds from review labels."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    cfg, db = _load_cfg(config)
    rows = db.labeled()
    log.info("%d labeled items", len(rows))
    if not rows:
        log.warning("no labels yet — review some items first (review_server.py)")
        raise typer.Exit(1)

    params = load_learned(cfg["paths"].get("learned_params", "learned_params.json"))

    X, y = _face_features(rows)
    if len(y) >= MIN_FACE_SAMPLES and y.min() != y.max():
        w, acc, mu, sd = _fit_logreg(X, y)
        params["face_conf"] = {
            "weights": [float(v) for v in w],
            "mean": [float(v) for v in mu],
            "std": [float(v) for v in sd],
            "features": FACE_FEATURES,
            "n": int(len(y)),
            "train_acc": acc,
        }
        log.info("face-confidence mapping fitted on %d labels (train acc %.3f)",
                 len(y), acc)
    else:
        log.info("face labels: %d (need %d with both classes) — keeping heuristic",
                 len(y), MIN_FACE_SAMPLES)

    blur = _tune_blur(rows)
    if blur:
        params["blur"] = blur
        log.info("blur thresholds tuned: edge<%.4f pixel(flat>%.3f & lap>%.1f) "
                 "(F1 %.3f on %d labels)",
                 blur["edge_threshold"], blur["pixel_flat_threshold"],
                 blur["pixel_lap_threshold"], blur["f1"], blur["n"])
    else:
        log.info("blur labels insufficient (need %d with both classes)", MIN_BLUR_SAMPLES)

    from datetime import datetime, timezone
    params["updated_at"] = datetime.now(timezone.utc).isoformat()
    out = cfg["paths"].get("learned_params", "learned_params.json")
    save_learned(out, params)
    log.info("wrote %s — re-run main.py --reset to apply", out)


@app.command()
def report(config: Path = typer.Option(Path("config.yaml"), "--config", "-c")):
    """Show what current labels say about the active thresholds (no changes)."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    cfg, db = _load_cfg(config)
    rows = db.labeled()
    log.info("labeled items: %d", len(rows))

    X, y = _face_features(rows)
    if len(y):
        pred = (X[:, 0] >= 0.8).astype(int)  # det_score vs review policy
        log.info("face labels=%d: det_score>=0.8 agreement=%.3f", len(y), float((pred == y).mean()))

    blur_rows = [r for r in rows
                 if r["label_face"] == 1 and r["label_blur"] in ("blurred", "sharp")
                 and r["lap_var"] is not None and r["edge_den"] is not None]
    if blur_rows:
        bc = cfg["blur_check"]
        ys = np.array([1 if r["label_blur"] == "blurred" else 0 for r in blur_rows])
        pred = _predict_blurred(
            np.array([r["lap_var"] for r in blur_rows]),
            np.array([r.get("flat_frac") or 0.0 for r in blur_rows]),
            np.array([r["edge_den"] for r in blur_rows]),
            bc["edge_threshold"], bc["pixel_lap_threshold"],
            bc["pixel_flat_threshold"])
        log.info("blur labels=%d: current thresholds F1=%.3f", len(ys), _f1(ys, pred))


@app.command(name="export-dataset")
def export_dataset(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    out: Path = typer.Option(Path("data/review_labels"), "--out", "-o"),
):
    """Copy labeled crops into face_sharp/ face_blurred/ not_face/ for retraining."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    _, db = _load_cfg(config)
    counts = {"face_sharp": 0, "face_blurred": 0, "not_face": 0}
    for r in db.labeled():
        src = Path(r["crop_path"])
        if not src.exists():
            continue
        if r["label_face"] == 0:
            sub = "not_face"
        elif r["label_blur"] == "sharp":
            sub = "face_sharp"
        elif r["label_blur"] == "blurred":
            sub = "face_blurred"
        else:
            continue
        dst = out / sub / f"{r['id']:06d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        counts[sub] += 1
    log.info("exported %s", counts)
    log.info("retrain the v2 classifier with e.g.: "
             "python train_classifier.py --faces data/review_labels/face_sharp "
             "--out models/blur_classifier.onnx  (merge with FFHQ crops for volume)")


if __name__ == "__main__":
    sys.exit(app())
