"""Selective frame export + report row assembly."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils import fmt_timestamp


class Exporter:
    def __init__(self, output_dir: str | Path, cfg: dict):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg["export"]

    @staticmethod
    def _tag(track_id) -> str:
        return f" #{track_id}" if track_id is not None else ""

    def save_frame(
        self,
        frame_bgr: np.ndarray,
        failures: list[dict],
        passes: list[dict],
        pending: list[dict],
        video_stem: str,
        frame_idx: int,
        timestamp_s: float,
    ) -> dict:
        """Write the annotated frame; return the report row for it."""
        out = frame_bgr.copy()
        if self.cfg["draw_boxes"]:
            for f in failures:  # red = unblurred (audit failure)
                x1, y1, x2, y2 = f["bbox"]
                cv2.rectangle(out, (x1, y1), (x2, y2), tuple(self.cfg["box_color_fail"]), 2)
                cv2.putText(out, f"UNBLURRED{self._tag(f['track_id'])} {f['metrics']['lap_var']:.0f}",
                            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, tuple(self.cfg["box_color_fail"]), 2)
            for f in pending:  # orange = sharp but not yet confirmed for its track
                x1, y1, x2, y2 = f["bbox"]
                cv2.rectangle(out, (x1, y1), (x2, y2), tuple(self.cfg["box_color_pending"]), 2)
                cv2.putText(out, f"REVIEW{self._tag(f['track_id'])}",
                            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, tuple(self.cfg["box_color_pending"]), 2)
            if self.cfg["draw_pass_boxes"]:
                for f in passes:  # green = blurred (debug only)
                    x1, y1, x2, y2 = f["bbox"]
                    cv2.rectangle(out, (x1, y1), (x2, y2), tuple(self.cfg["box_color_pass"]), 1)

        vdir = self.root / video_stem
        vdir.mkdir(parents=True, exist_ok=True)
        fname = f"frame_{frame_idx:08d}_t{timestamp_s:.2f}s.jpg"
        path = vdir / fname
        cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["jpeg_quality"]])

        return {
            "image": str(path.relative_to(self.root)),
            "frame": frame_idx,
            "timestamp_s": round(timestamp_s, 2),
            "timestamp_hms": fmt_timestamp(timestamp_s),
            "faces_total": len(failures) + len(passes) + len(pending),
            "faces_failed": len(failures),
            "tracks_failed": sorted({f["track_id"] for f in failures
                                     if f["track_id"] is not None}),
            "tracks_pending": sorted({f["track_id"] for f in pending
                                      if f["track_id"] is not None}),
            "fail_boxes": [f["bbox"] for f in failures],
            "fail_metrics": [
                {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in f["metrics"].items()}
                for f in failures
            ],
        }
