#!/usr/bin/env python3
"""Score Veilaudit outputs against synthetic ground truth.

Checks, per video:
  - EVENT RECALL: every scheduled "sharp" (violation) segment got >=1 exported
    frame within [t0+1s, t1] (1s grace for temporal confirmation).
  - WEAK COVERAGE: every "weak" blur segment was exported OR sent to review
    (never silently passed).
  - FALSE POSITIVES: exported frames at times when no actor was sharp/weak
    (1.5s grace after a segment ends for confirmation lag).
  - TRACK CONSISTENCY: distinct track IDs flagged per violation segment (want 1).

Usage:
    python tests/evaluate.py --gt testdata/ground_truth --out /data/audit_out \\
        --review-db /data/audit_out/review.sqlite
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


def load_report(out: Path) -> list[dict]:
    rows = []
    p = out / "report.csv"
    if not p.exists():
        return rows
    with open(p) as fh:
        for r in csv.DictReader(fh):
            r["frame"] = int(r["frame"])
            r["timestamp_s"] = float(r["timestamp_s"])
            r["tracks_failed"] = json.loads(r["tracks_failed"] or "[]")
            rows.append(r)
    return rows


def review_items(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM review_items").fetchall()]
    con.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--review-db", type=Path, default=None)
    args = ap.parse_args()

    rows = load_report(args.out)
    reviews = review_items(args.review_db) if args.review_db else []

    total_events = total_recalled = total_fp = 0
    weak_total = weak_covered = 0
    report = {"videos": {}}

    for gt_path in sorted(args.gt.glob("*.json")):
        gt = json.loads(gt_path.read_text())
        vname = gt["video"]
        vrows = [r for r in rows if r["video"] == vname]
        vreviews = [r for r in reviews if r["video"] == vname]
        vrep = {"events": [], "false_positives": 0}

        # collect sharp / weak segments across actors
        sharp_segs, weak_segs = [], []
        for a in gt["actors"]:
            for t0, t1, s in a["schedule"]:
                if s == "sharp":
                    sharp_segs.append((a["actor"], t0, t1))
                elif s == "weak":
                    weak_segs.append((a["actor"], t0, t1))

        def in_any(ts, segs, post_grace=0.0):
            return any(t0 <= ts <= t1 + post_grace for _, t0, t1 in segs)

        for actor, t0, t1 in sharp_segs:
            total_events += 1
            hits = [r for r in vrows if t0 + 1.0 <= r["timestamp_s"] <= t1]
            recalled = len(hits) > 0
            total_recalled += recalled
            tracks = sorted({t for r in hits for t in r["tracks_failed"]})
            vrep["events"].append({
                "actor": actor, "t0": t0, "t1": t1, "type": "sharp",
                "recalled": recalled, "n_exports": len(hits),
                "tracks_flagged": tracks, "track_consistent": len(tracks) <= 2,
            })

        for actor, t0, t1 in weak_segs:
            weak_total += 1
            exported = any(t0 + 1.0 <= r["timestamp_s"] <= t1 for r in vrows)
            queued = any(t0 <= r["timestamp_s"] <= t1 and r["reason"] == "borderline_blur"
                         for r in vreviews)
            covered = exported or queued
            weak_covered += covered
            vrep["events"].append({
                "actor": actor, "t0": t0, "t1": t1, "type": "weak",
                "covered": covered, "exported": exported, "queued_for_review": queued,
            })

        for r in vrows:
            if not in_any(r["timestamp_s"], sharp_segs + weak_segs, post_grace=1.5):
                total_fp += 1
                vrep["false_positives"] += 1

        report["videos"][vname] = vrep

    report["summary"] = {
        "event_recall": f"{total_recalled}/{total_events}",
        "event_recall_rate": round(total_recalled / max(1, total_events), 3),
        "weak_coverage": f"{weak_covered}/{weak_total}",
        "weak_coverage_rate": round(weak_covered / max(1, weak_total), 3),
        "false_positive_frames": total_fp,
        "review_items": len(reviews),
    }
    print(json.dumps(report, indent=1))
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
