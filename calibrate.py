#!/usr/bin/env python3
"""Threshold calibration for the blur check.

Detects faces in a sample clip and dumps every face crop with its metrics
baked into the filename, plus a metrics.csv for spreadsheet inspection:

    f000123_i0_LAP0042_HF0.031_ED0.012.jpg

Eyeball the crops: find the Laplacian / HF-ratio values that separate the
faces you consider blurred from the ones you consider sharp, then set
`laplacian_threshold` / `hfratio_threshold` in config.yaml.

    python calibrate.py --video /data/videos/sample.mp4 --out calib_out --max-faces 400
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import cv2
import typer
import yaml
from tqdm import tqdm

from blurcheck import edge_density, flat_frac, highfreq_ratio, laplacian_var
from detector import FaceDetector
from utils import expand_bbox

app = typer.Typer(add_completion=False)
log = logging.getLogger("calibrate")


@app.command()
def calibrate(
    video: Path = typer.Option(..., "--video", help="Representative sample clip"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    out: Path = typer.Option(Path("calib_out"), "--out", "-o"),
    sample_every: int = typer.Option(5, "--sample-every"),
    max_faces: int = typer.Option(400, "--max-faces"),
):
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(config.read_text())["detection"]
    out.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(model_pack=cfg["model_pack"],
                            det_size=tuple(cfg["det_size"]),
                            det_thresh=cfg["det_thresh"],
                            use_gpu=cfg["use_gpu"])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log.error("cannot open %s", video)
        raise typer.Exit(1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    rows, saved, idx = [], 0, 0
    pbar = tqdm(desc="scanning", unit="frames")
    while saved < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        if idx % sample_every != 0:
            idx += 1
            continue
        for i, f in enumerate(detector.detect(frame)):
            x1, y1, x2, y2 = expand_bbox(f["bbox"], h, w)
            if min(x2 - x1, y2 - y1) < cfg["min_face_px"]:
                continue
            roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            lap, hf, ed = laplacian_var(roi), highfreq_ratio(roi), edge_density(roi)
            ff = flat_frac(roi)
            name = f"f{idx:06d}_i{i}_LAP{lap:04.0f}_HF{hf:.3f}_ED{ed:.3f}.jpg"
            cv2.imwrite(str(out / name), roi)
            rows.append({"file": name, "frame": idx, "lap_var": round(lap, 2),
                         "hf_ratio": round(hf, 4), "edge_den": round(ed, 4),
                         "flat_frac": round(ff, 4)})
            saved += 1
            if saved >= max_faces:
                break
        idx += 1
    pbar.close()
    cap.release()

    with open(out / "metrics.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "frame", "lap_var", "hf_ratio", "edge_den", "flat_frac"])
        wr.writeheader()
        wr.writerows(rows)
    log.info("saved %d face crops to %s — inspect filenames, then set thresholds in config.yaml",
             saved, out)


if __name__ == "__main__":
    sys.exit(app())
