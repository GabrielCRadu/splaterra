from __future__ import annotations

import sqlite3
from pathlib import Path

from app.paths import data_root

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    raw_video_path TEXT,
    frames_dir TEXT,
    frame_count INTEGER,
    recommendation_json TEXT,
    colmap_sparse_dir TEXT,
    undistorted_images_dir TEXT,
    splat_path TEXT
);

CREATE TABLE IF NOT EXISTS job_stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    log TEXT,
    error TEXT,
    progress_percent REAL,
    UNIQUE(job_id, stage)
);
"""


def db_path() -> Path:
    return data_root() / "splatprep.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI dispatches sync dependencies (get_job_store) to a worker
    # threadpool, and does not guarantee the same thread handles both the
    # generator's setup and its teardown - so a connection created for one
    # request can be closed from a different thread. Each call to connect()
    # still returns a connection used by exactly one request, never shared
    # across concurrent requests, so disabling sqlite3's same-thread check
    # here is safe and does not introduce real concurrent-access risk.
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn
