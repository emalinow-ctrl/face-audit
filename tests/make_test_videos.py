#!/usr/bin/env python3
"""Generate synthetic 1080p nightclub-style test footage with KNOWN blur schedules.

Real face photos (FairFace val) are composited onto dark nightclub backgrounds.
Each actor follows a blur schedule: blurred -> sharp (censorship violation) ->
blurred, etc. Ground truth (per-frame boxes + blur state) is written as JSON
for scoring the pipeline.

Usage:
    python tests/make_test_videos.py --fairface testdata/fairface \\
        --out testdata/videos --gt testdata/ground_truth --seed 7
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1920, 1080, 24
DUR = 45  # seconds per video
NFRAMES = DUR * FPS

ADULT_AGES = {"20-29", "30-39", "40-49", "50-59", "60-69"}

# (t0, t1, state) per actor. "sharp" = censorship violation (must be flagged).
# "weak" = inadequate blur (must be flagged OR sent to review, never silently passed).
SCHEDULES = {
    "club_a.mp4": [
        [(0, 12, "blur"), (12, 25, "sharp"), (25, 45, "blur")],
        [(0, 8, "sharp"), (8, 45, "blur")],
        [(0, 45, "blur")],
        [(0, 30, "pixel"), (30, 45, "sharp")],
        [(0, 20, "blur"), (20, 30, "weak"), (30, 45, "blur")],
        [(0, 45, "sharp")],
    ],
    "club_b.mp4": [
        [(0, 15, "blur"), (15, 28, "sharp"), (28, 45, "blur")],
        [(0, 45, "blur")],
        [(0, 10, "sharp"), (10, 45, "pixel")],
        [(0, 35, "blur"), (35, 45, "sharp")],
        [(0, 45, "weak")],
        [(0, 22, "blur"), (22, 32, "sharp"), (32, 45, "blur")],
    ],
}

# (race, gender, count) — mostly Black men and white women, per request.
CAST_SPECS = [("Black", "Male", 5), ("White", "Female", 5),
              ("Black", "Female", 1), ("White", "Male", 1)]

# Loose grid so faces don't overlap.
GRID = [(0.16, 0.32), (0.36, 0.58), (0.56, 0.32), (0.76, 0.58),
        (0.27, 0.74), (0.68, 0.24)]


def find_image_dir(fairface: Path) -> Path:
    for cand in (fairface / "face_images" / "val", fairface / "faces" / "val",
                 fairface / "val"):
        if cand.is_dir() and len(list(cand.glob("*.jpg"))) > 1000:
            return cand
    # fallback: any dir with many jpgs whose names look like val/
    for p in fairface.rglob("val"):
        if p.is_dir() and len(list(p.glob("*.jpg"))) > 1000:
            return p
    raise SystemExit(f"could not find FairFace val images under {fairface}")


def select_cast(fairface: Path, seed: int) -> list[dict]:
    rng = random.Random(seed)
    imgdir = find_image_dir(fairface)
    with open(fairface / "fairface_label_val.csv") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["age"] in ADULT_AGES and (imgdir / Path(r["file"]).name).exists()]
    cast = []
    for race, gender, count in CAST_SPECS:
        pool = [r for r in rows if r["race"] == race and r["gender"] == gender]
        pool = sorted(pool, key=lambda r: r["file"])
        if len(pool) < count:
            raise SystemExit(f"only {len(pool)} {race} {gender} faces available")
        step = len(pool) // count
        for i in range(count):
            r = pool[(i * step + rng.randrange(step)) % len(pool)]
            cast.append({"file": (imgdir / Path(r["file"]).name).as_posix(),
                         "race": race, "gender": gender})
    rng.shuffle(cast)
    return cast


def make_background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, H)[:, None]
    bg = (yy * np.array([10, 10, 30]) + (1 - yy) * np.array([4, 4, 14]))
    bg = np.repeat(bg[:, :, None] if False else bg.reshape(H, 1, 3), W, axis=1)
    bg = bg.astype(np.float32)
    palette = [(170, 40, 40), (200, 120, 30), (50, 60, 180), (120, 40, 140),
               (30, 120, 150)]
    Y, X = np.ogrid[:H, :W]
    for _ in range(16):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(120, 340)
        col = np.array(palette[rng.integers(len(palette))], np.float32)
        d2 = ((X - cx) ** 2 + (Y - cy) ** 2) / r ** 2
        glow = np.exp(-d2 * 3)[..., None]
        bg += glow * col * rng.uniform(0.25, 0.6)
    # vignette
    d2 = (((X - W / 2) / (W / 2)) ** 2 + ((Y - H / 2) / (H / 2)) ** 2) / 2
    bg *= np.clip(1 - d2 * 0.55, 0.35, 1)[..., None]
    return np.clip(bg, 0, 255).astype(np.uint8)


def soft_mask(size: int, feather: int = 14) -> np.ndarray:
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, size // 2), (size // 2 - 2, size // 2 - 2), 0, 0, 360, 1.0, -1)
    k = feather | 1
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def variants(img: np.ndarray, size: int) -> dict[str, np.ndarray]:
    base = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    out = {"sharp": base}
    out["blur"] = cv2.GaussianBlur(base, (61, 61), 0)
    small = cv2.resize(base, (10, 10), interpolation=cv2.INTER_LINEAR)
    out["pixel"] = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
    out["weak"] = cv2.GaussianBlur(base, (15, 15), 0)
    return out


def state_at(schedule, t: float) -> str:
    for t0, t1, s in schedule:
        if t0 <= t < t1:
            return s
    return schedule[-1][2]


def render_video(out_path: Path, bg: np.ndarray, actors: list[dict],
                 schedule_list: list, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt_frames = []
    n = len(actors)
    for fi in range(NFRAMES):
        t = fi / FPS
        frame = bg.astype(np.int16) + rng.integers(-5, 6, (H, W, 3)).astype(np.int16)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        f_actors = []
        for ai, a in enumerate(actors):
            cx = (a["x0"] + a["ax"] * math.sin(2 * math.pi * t / a["Tx"] + a["phx"])) * W
            cy = (a["y0"] + a["ay"] * math.sin(2 * math.pi * t / a["Ty"] + a["phy"])) * H
            state = state_at(schedule_list[ai], t)
            img = a["variants"][state]
            s = a["size"]
            x1, y1 = int(cx - s / 2), int(cy - s / 2)
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + s), min(H, y1 + s)
            if x2c > x1c and y2c > y1c:
                m = a["mask"][y1c - y1:y2c - y1, x1c - x1:x2c - x1][..., None]
                roi = frame[y1c:y2c, x1c:x2c].astype(np.float32)
                pasted = img[y1c - y1:y2c - y1, x1c - x1:x2c - x1].astype(np.float32)
                frame[y1c:y2c, x1c:x2c] = (roi * (1 - m) + pasted * m).astype(np.uint8)
            f_actors.append({"actor": ai, "bbox": [x1c, y1c, x2c, y2c], "state": state})
        gt_frames.append({"t": round(t, 3), "actors": f_actors})
        vw.write(frame)
        if fi % 240 == 0:
            print(f"  frame {fi}/{NFRAMES}", flush=True)
    vw.release()
    return {"video": out_path.name, "fps": FPS, "frames": NFRAMES,
            "actors": [{"actor": ai, **{k: a[k] for k in ("race", "gender", "file")},
                        "schedule": schedule_list[ai]} for ai, a in enumerate(actors)],
            "frame_data": gt_frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fairface", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.gt.mkdir(parents=True, exist_ok=True)

    cast = select_cast(args.fairface, args.seed)
    print(f"cast: {len(cast)} people")
    for c in cast:
        print(f"  {Path(c['file']).name} {c['race']} {c['gender']}")

    rng = random.Random(args.seed)
    for vi, (vname, schedules) in enumerate(SCHEDULES.items()):
        print(f"rendering {vname} ...")
        bg = make_background(args.seed * 100 + vi)
        people = cast[vi * 6:(vi + 1) * 6]
        actors = []
        for ai, p in enumerate(people):
            img = cv2.imread(p["file"])
            size = rng.randint(200, 260)
            gx, gy = GRID[ai]
            actors.append({
                "race": p["race"], "gender": p["gender"], "file": p["file"],
                "variants": variants(img, size), "mask": soft_mask(size),
                "size": size,
                "x0": gx + rng.uniform(-0.03, 0.03), "y0": gy + rng.uniform(-0.03, 0.03),
                "ax": rng.uniform(0.04, 0.07), "ay": rng.uniform(0.03, 0.05),
                "Tx": rng.uniform(9, 20), "Ty": rng.uniform(11, 22),
                "phx": rng.uniform(0, 6.28), "phy": rng.uniform(0, 6.28),
            })
        gt = render_video(args.out / vname, bg, actors, schedules, args.seed * 1000 + vi)
        with open(args.gt / f"{Path(vname).stem}.json", "w") as fh:
            json.dump(gt, fh)
        print(f"wrote {vname} + ground truth")


if __name__ == "__main__":
    main()
