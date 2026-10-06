"""SQLite checkpoint store: resume interrupted runs without reprocessing.

Schema: videos(path TEXT PRIMARY KEY, last_frame INT, status TEXT, updated_at TEXT)
status is 'in_progress' or 'done'.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class CheckpointDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS videos(
                       path TEXT PRIMARY KEY,
                       last_frame INTEGER NOT NULL DEFAULT 0,
                       status TEXT NOT NULL DEFAULT 'in_progress',
                       updated_at TEXT NOT NULL)"""
            )
            con.execute("PRAGMA journal_mode=WAL")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def last_frame(self, video: str) -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT last_frame FROM videos WHERE path=?", (video,)
            ).fetchone()
        return int(row[0]) if row else 0

    def is_done(self, video: str) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT status FROM videos WHERE path=?", (video,)
            ).fetchone()
        return bool(row and row[0] == "done")

    def checkpoint(self, video: str, frame_idx: int) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """INSERT INTO videos(path, last_frame, status, updated_at)
                   VALUES(?, ?, 'in_progress', ?)
                   ON CONFLICT(path) DO UPDATE SET
                     last_frame=excluded.last_frame,
                     status='in_progress',
                     updated_at=excluded.updated_at""",
                (video, frame_idx, self._now()),
            )

    def mark_done(self, video: str, frame_idx: int = -1) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """INSERT INTO videos(path, last_frame, status, updated_at)
                   VALUES(?, ?, 'done', ?)
                   ON CONFLICT(path) DO UPDATE SET
                     last_frame=excluded.last_frame,
                     status='done',
                     updated_at=excluded.updated_at""",
                (video, frame_idx, self._now()),
            )

    def reset(self, video: str | None = None) -> None:
        """Clear checkpoints (all, or one video) to force reprocessing."""
        with self._lock, self._connect() as con:
            if video:
                con.execute("DELETE FROM videos WHERE path=?", (video,))
            else:
                con.execute("DELETE FROM videos")
