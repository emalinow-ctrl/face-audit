"""SQLite-backed manual review queue.

Items land here when the pipeline isn't sure:
  - low_face_conf   : face_confidence < threshold — human decides face vs. not-a-face
  - borderline_blur : blur metrics within `blur_margin` of a threshold —
                      human confirms the pipeline's blurred/sharp guess

Every correction is stored with the features the pipeline used, so `learn.py`
can retune the face-confidence mapping and blur thresholds from real labels.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

REASONS = ("low_face_conf", "borderline_blur")

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video TEXT NOT NULL,
  frame INTEGER NOT NULL,
  timestamp_s REAL NOT NULL,
  crop_path TEXT NOT NULL,
  frame_path TEXT NOT NULL,
  bbox TEXT NOT NULL,
  track_id INTEGER,
  det_score REAL NOT NULL,
  face_confidence REAL NOT NULL,
  eye_ratio REAL,
  kps_inside REAL,
  order_ok REAL,
  rel_area REAL,
  reason TEXT NOT NULL,
  pipeline_face INTEGER NOT NULL,   -- 1: pipeline suspected a face
  pipeline_blur TEXT,               -- blurred | sharp | NULL
  lap_var REAL,
  hf_ratio REAL,
  edge_den REAL,
  status TEXT NOT NULL DEFAULT 'pending',
  label_face INTEGER,               -- 1 / 0 / NULL
  label_blur TEXT,                  -- blurred | sharp / NULL
  created_at TEXT NOT NULL,
  reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(status);
"""


class ReviewDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(SCHEMA)
            con.execute("PRAGMA journal_mode=WAL")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_item(self, *,
                 video: str, frame: int, timestamp_s: float,
                 crop_path: str, frame_path: str,
                 bbox: tuple, track_id: int | None,
                 det_score: float, face_confidence: float,
                 kps_feats: dict | None, rel_area: float,
                 reason: str, pipeline_blur: str | None,
                 metrics: dict | None) -> int:
        assert reason in REASONS, reason
        feats = kps_feats or {}
        with self._lock, self._connect() as con:
            cur = con.execute(
                """INSERT INTO review_items(
                     video, frame, timestamp_s, crop_path, frame_path, bbox, track_id,
                     det_score, face_confidence, eye_ratio, kps_inside, order_ok, rel_area,
                     reason, pipeline_face, pipeline_blur, lap_var, hf_ratio, edge_den,
                     created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (video, frame, timestamp_s, crop_path, frame_path,
                 json.dumps(list(bbox)), track_id,
                 det_score, face_confidence,
                 feats.get("eye_ratio"), feats.get("kps_inside"), feats.get("order_ok"),
                 rel_area, reason, 1, pipeline_blur,
                 (metrics or {}).get("lap_var"), (metrics or {}).get("hf_ratio"),
                 (metrics or {}).get("edge_den"), self._now()),
            )
            return int(cur.lastrowid)

    def pending(self, limit: int | None = None) -> list[dict]:
        q = "SELECT * FROM review_items WHERE status='pending' ORDER BY id"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._connect() as con:
            return [dict(r) for r in con.execute(q).fetchall()]

    def get(self, item_id: int) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM review_items WHERE id=?", (item_id,)).fetchone()
        return dict(row) if row else None

    def set_label(self, item_id: int, label_face: int | None, label_blur: str | None) -> None:
        assert label_face in (0, 1, None)
        assert label_blur in ("blurred", "sharp", None)
        with self._lock, self._connect() as con:
            con.execute(
                """UPDATE review_items
                   SET status='reviewed', label_face=?, label_blur=?, reviewed_at=?
                   WHERE id=?""",
                (label_face, label_blur, self._now(), item_id),
            )

    def reopen(self, item_id: int) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE review_items SET status='pending', label_face=NULL, label_blur=NULL WHERE id=?",
                (item_id,),
            )

    def stats(self) -> dict:
        with self._connect() as con:
            n_pending = con.execute("SELECT COUNT(*) FROM review_items WHERE status='pending'").fetchone()[0]
            n_done = con.execute("SELECT COUNT(*) FROM review_items WHERE status='reviewed'").fetchone()[0]
            n_face = con.execute("SELECT COUNT(*) FROM review_items WHERE label_face=1").fetchone()[0]
            n_notface = con.execute("SELECT COUNT(*) FROM review_items WHERE label_face=0").fetchone()[0]
        return {"pending": n_pending, "reviewed": n_done,
                "labeled_face": n_face, "labeled_not_face": n_notface}

    def labeled(self) -> list[dict]:
        """All human-labeled items, for learn.py."""
        with self._connect() as con:
            return [dict(r) for r in con.execute(
                "SELECT * FROM review_items WHERE status='reviewed' AND label_face IS NOT NULL"
            ).fetchall()]
