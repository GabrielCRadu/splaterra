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
    splat_path TEXT,
    export_path TEXT,
    export_format TEXT,
    device_type TEXT NOT NULL DEFAULT 'standard'
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
    settings_json TEXT,
    eval_metrics_json TEXT,
    UNIQUE(job_id, stage)
);
"""


def db_path() -> Path:
    return data_root() / "splaterra.db"


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
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    # CREATE TABLE IF NOT EXISTS never adds columns to an already-existing
    # table. Earlier schema changes could just delete the dev DB since it had
    # no real data yet - data/splaterra.db now holds real trained jobs, so
    # new nullable columns get an idempotent ALTER TABLE here instead.
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    for column in ("export_path", "export_format"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
    # device_type is NOT NULL with a real default ("standard"), unlike the
    # nullable columns above where NULL is the legitimate "not done yet"
    # value - a bare `ADD COLUMN device_type TEXT` would leave every
    # pre-existing job's device_type as None, forcing None-coalescing
    # everywhere device_type is compared against "standard".
    if "device_type" not in existing_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN device_type TEXT NOT NULL DEFAULT 'standard'")

    existing_stage_columns = {row["name"] for row in conn.execute("PRAGMA table_info(job_stages)")}
    if "settings_json" not in existing_stage_columns:
        conn.execute("ALTER TABLE job_stages ADD COLUMN settings_json TEXT")
    if "eval_metrics_json" not in existing_stage_columns:
        conn.execute("ALTER TABLE job_stages ADD COLUMN eval_metrics_json TEXT")
    conn.commit()
