"""Lightweight IoU-based face tracker: persistent per-person IDs across frames.

Why: a frame may contain several faces — some blurred, some not. Per-frame
detection alone can't tell "the unblurred person in frame 100" from "the
unblurred person in frame 400", nor keep one blurred individual's verdict
separate from an unblurred person standing next to them. This tracker assigns
each distinct face a stable track ID and accumulates that *person's* blur
verdicts over time, so the report attributes failures to individuals.

Faces move little between sampled frames, so greedy IoU matching is reliable.
No dependencies beyond the standard library. Fully offline.
"""
from __future__ import annotations

from collections import deque


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


class FaceTrack:
    """One distinct person, observed across sampled frames."""

    def __init__(self, track_id: int, bbox: tuple, frame_idx: int, window: int):
        self.id = track_id
        self.bbox = bbox
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.missed = 0                      # sampled frames since last seen
        self.recent: deque[bool] = deque(maxlen=window)  # True = sharp (fail)
        self.n_sharp = 0
        self.n_censored = 0

    def update(self, bbox: tuple, frame_idx: int, sharp: bool) -> None:
        self.bbox = bbox
        self.last_frame = frame_idx
        self.missed = 0
        self.recent.append(bool(sharp))
        if sharp:
            self.n_sharp += 1
        else:
            self.n_censored += 1

    def is_confirmed(self, min_fail: int) -> bool:
        """Sharp in >= min_fail of recent sampled frames -> confirmed unblurred.

        Per-person temporal smoothing: a single motion-blurred frame can't
        condemn a track, but a persistently sharp face gets flagged.
        """
        return sum(self.recent) >= min_fail


class IoUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 5, window: int = 3):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed      # sampled frames a person may vanish
        self.window = window
        self.tracks: list[FaceTrack] = []
        self._next_id = 1

    @property
    def count(self) -> int:
        return self._next_id - 1

    def update(self, detections: list[dict], frame_idx: int) -> list[tuple[FaceTrack, dict]]:
        """Match detections to tracks.

        Each detection dict needs 'bbox', 'sharp' (bool), 'entry' (report dict).
        Returns [(track, detection), ...] with track IDs assigned.
        """
        for t in self.tracks:
            t.missed += 1

        # Greedy best-IoU-first matching above threshold.
        candidates = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                iou = _iou(t.bbox, d["bbox"])
                if iou >= self.iou_thresh:
                    candidates.append((iou, ti, di))
        candidates.sort(key=lambda c: c[0], reverse=True)

        used_t, used_d = set(), set()
        pairs: list[tuple[FaceTrack, dict]] = []
        for _, ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            track = self.tracks[ti]
            det = detections[di]
            track.update(det["bbox"], frame_idx, det["sharp"])
            pairs.append((track, det))

        # Unmatched detections start new tracks (new people entering frame).
        for di, det in enumerate(detections):
            if di not in used_d:
                track = FaceTrack(self._next_id, det["bbox"], frame_idx, self.window)
                self._next_id += 1
                track.update(det["bbox"], frame_idx, det["sharp"])
                self.tracks.append(track)
                pairs.append((track, det))

        # Retire people gone longer than max_missed sampled frames.
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return pairs

    def summary(self, min_fail: int) -> dict[int, dict]:
        """Per-person audit summary for the report."""
        out = {}
        for t in self.tracks:
            out[t.id] = {
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
                "frames_seen": t.n_sharp + t.n_censored,
                "n_sharp": t.n_sharp,
                "n_censored": t.n_censored,
                "confirmed_unblurred": t.n_sharp >= min_fail,
            }
        return out
