"""Small geometry helpers shared by the pipeline."""
from __future__ import annotations


def clip_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))


def expand_bbox(
    bbox: tuple[int, int, int, int], h: int, w: int, pad: float = 0.15
) -> tuple[int, int, int, int]:
    """Grow a bbox by `pad` fraction on each side (helps blur metrics see edges)."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(x1 - bw * pad)
    y1 = int(y1 - bh * pad)
    x2 = int(x2 + bw * pad)
    y2 = int(y2 + bh * pad)
    return clip_bbox((x1, y1, x2, y2), h, w)


def fmt_timestamp(seconds: float) -> str:
    s = max(0.0, seconds)
    hh, rem = divmod(int(s), 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{int((s % 1) * 100):02d}"
