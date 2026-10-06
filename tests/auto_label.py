#!/usr/bin/env python3
"""Simulate the human reviewer using synthetic ground truth.

For each pending review item, finds the actor whose ground-truth box best
overlaps the queued crop box at that frame:
  - IoU > 0.3  -> label_face=1; label_blur from the actor's scheduled state
                  (blurred/pixel -> 'blurred'; sharp/weak -> 'sharp',
                  weak counts as inadequate censorship)
  - IoU <= 0.3 -> label_face=0 (genuine false detection)

This lets us exercise learn.py's tuning loop without manual clicking.
A real deployment would use review_server.py instead.

Usage:
    python tests/auto_label.py --gt testdata/ground_truth \\
        --review-db /data/audit_out/review.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--review-db", required=True, type=Path)
    args = ap.parse_args()

    gts = {}
    for p in args.gt.glob("*.json"):
        g = json.loads(p.read_text())
        # frame index -> actors
        frames = {i: fr["actors"] for i, fr in enumerate(g["frame_data"])}
        gts[g["video"]] = frames

    con = sqlite3.connect(str(args.review_db))
    con.row_factory = sqlite3.Row
    items = [dict(r) for r in con.execute(
        "SELECT * FROM review_items WHERE status='pending'").fetchall()]

    n_face = n_notface = 0
    for it in items:
        frames = gts.get(it["video"], {})
        actors = frames.get(it["frame"], [])
        bbox = json.loads(it["bbox"])
        best, best_state = 0.0, None
        for a in actors:
            v = iou(bbox, a["bbox"])
            if v > best:
                best, best_state = v, a["state"]
        if best > 0.3:
            label_face = 1
            label_blur = "blurred" if best_state in ("blur", "pixel") else "sharp"
            n_face += 1
        else:
            label_face, label_blur = 0, None
            n_notface += 1
        con.execute(
            "UPDATE review_items SET status='reviewed', label_face=?, label_blur=?,"
            " reviewed_at=datetime('now') WHERE id=?",
            (label_face, label_blur, it["id"]),
        )
    con.commit()
    con.close()
    print(f"labeled {len(items)} items: {n_face} faces, {n_notface} not-faces")


if __name__ == "__main__":
    main()
