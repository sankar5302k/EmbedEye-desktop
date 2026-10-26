"""
Staging store for EmbedEye using SQLite (./data/staging.db).
Holds OCR text and Florence-2 captions before committing to Qdrant vector database.
"""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.security import is_subpath

DEFAULT_STAGING_DB = Path(__file__).resolve().parent.parent / "data" / "staging.db"


class StagingStore:
    """
    Thread-safe SQLite store for staged OCR and captions.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_STAGING_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS staged_entries (
                    path TEXT PRIMARY KEY,
                    file_mtime REAL NOT NULL,
                    file_size INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    ocr_text TEXT,
                    caption TEXT,
                    staged_at TEXT NOT NULL,
                    extra_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_staged_mode ON staged_entries (mode)"
            )
            conn.commit()

    def upsert(
        self,
        path: str,
        mtime: float,
        size: int,
        mode: str,
        ocr_text: Optional[str] = None,
        caption: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Insert or replace a staged entry."""
        norm_path = os.path.normpath(path)
        now_iso = datetime.now(timezone.utc).isoformat()
        extra_str = json.dumps(extra) if extra else None

        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO staged_entries (path, file_mtime, file_size, mode, ocr_text, caption, staged_at, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_mtime = excluded.file_mtime,
                    file_size = excluded.file_size,
                    mode = excluded.mode,
                    ocr_text = excluded.ocr_text,
                    caption = excluded.caption,
                    staged_at = excluded.staged_at,
                    extra_json = excluded.extra_json
                """,
                (norm_path, mtime, size, mode, ocr_text, caption, now_iso, extra_str),
            )
            conn.commit()

    def is_staged_and_unchanged(self, path: str, current_mtime: float) -> bool:
        """Check if file was already staged with the same mtime."""
        norm_path = os.path.normpath(path)
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "SELECT file_mtime FROM staged_entries WHERE path = ?", (norm_path,)
            )
            row = cursor.fetchone()
            if row and abs(row["file_mtime"] - current_mtime) < 1e-4:
                return True
        return False

    def get_entries_in_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Retrieve all staged rows whose path is inside folder_path (recursive)."""
        with self._lock, self._connection() as conn:
            cursor = conn.execute("SELECT * FROM staged_entries")
            rows = cursor.fetchall()
            matching = []
            for row in rows:
                if is_subpath(row["path"], folder_path):
                    matching.append(dict(row))
            return matching

    def count_in_folder(self, folder_path: str) -> Dict[str, int]:
        """
        Return breakdown counts of staged items in folder:
        total, ocr_only, ocr_caption
        """
        entries = self.get_entries_in_folder(folder_path)
        ocr_count = sum(1 for e in entries if e["mode"] == "ocr")
        caption_count = sum(1 for e in entries if e["mode"] == "ocr_caption")
        return {
            "total": len(entries),
            "ocr": ocr_count,
            "ocr_caption": caption_count,
        }

    def delete_in_folder(self, folder_path: str) -> int:
        """Delete all staged rows inside folder_path (recursive)."""
        entries = self.get_entries_in_folder(folder_path)
        if not entries:
            return 0

        paths_to_delete = [e["path"] for e in entries]
        with self._lock, self._connection() as conn:
            conn.executemany(
                "DELETE FROM staged_entries WHERE path = ?",
                [(p,) for p in paths_to_delete],
            )
            conn.commit()
        return len(paths_to_delete)


# Global singleton instance
staging_store = StagingStore()
