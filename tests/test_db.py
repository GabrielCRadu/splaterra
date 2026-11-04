import sqlite3

from app import db


def test_connect_creates_schema(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"jobs", "job_stages"} <= tables
    conn.close()


def test_connect_migrates_old_schema_without_losing_data(tmp_path):
    # Simulates data/splaterra.db as it existed before export_path/export_format
    # were added - connect() must add the missing columns in place rather than
    # requiring the file to be deleted, since it can hold real trained jobs.
    path = tmp_path / "old.db"
    old_conn = sqlite3.connect(path)
    old_conn.execute(
        """
        CREATE TABLE jobs (
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
        )
        """
    )
    old_conn.execute(
        "INSERT INTO jobs (id, name, created_at, splat_path) VALUES (?, ?, ?, ?)",
        ("job-1", "MasinaPoarta", "2026-07-26T10:10:33+00:00", "training/splat_30000.ply"),
    )
    old_conn.commit()
    old_conn.close()

    conn = db.connect(path)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert "export_path" in columns
    assert "export_format" in columns
    assert "device_type" in columns

    row = conn.execute("SELECT * FROM jobs WHERE id = ?", ("job-1",)).fetchone()
    assert row["name"] == "MasinaPoarta"
    assert row["splat_path"] == "training/splat_30000.ply"
    assert row["export_path"] is None
    # device_type is NOT NULL with a real default, unlike export_path/
    # export_format above where NULL is the legitimate "not done yet" value -
    # a pre-existing job must backfill to "standard", not None.
    assert row["device_type"] == "standard"
    conn.close()


def test_connect_migrates_job_stages_without_settings_json(tmp_path):
    path = tmp_path / "old.db"
    old_conn = sqlite3.connect(path)
    old_conn.execute(
        """
        CREATE TABLE job_stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            log TEXT,
            error TEXT,
            progress_percent REAL,
            UNIQUE(job_id, stage)
        )
        """
    )
    old_conn.execute(
        "INSERT INTO job_stages (job_id, stage, status) VALUES (?, ?, ?)",
        ("job-1", "training", "done"),
    )
    old_conn.commit()
    old_conn.close()

    conn = db.connect(path)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(job_stages)")}
    assert "settings_json" in columns

    row = conn.execute("SELECT * FROM job_stages WHERE job_id = ?", ("job-1",)).fetchone()
    assert row["status"] == "done"
    assert row["settings_json"] is None
    conn.close()


def test_connect_migrates_job_stages_without_eval_metrics_json(tmp_path):
    path = tmp_path / "old.db"
    old_conn = sqlite3.connect(path)
    old_conn.execute(
        """
        CREATE TABLE job_stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            log TEXT,
            error TEXT,
            progress_percent REAL,
            settings_json TEXT,
            UNIQUE(job_id, stage)
        )
        """
    )
    old_conn.execute(
        "INSERT INTO job_stages (job_id, stage, status) VALUES (?, ?, ?)",
        ("job-1", "training", "done"),
    )
    old_conn.commit()
    old_conn.close()

    conn = db.connect(path)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(job_stages)")}
    assert "eval_metrics_json" in columns

    row = conn.execute("SELECT * FROM job_stages WHERE job_id = ?", ("job-1",)).fetchone()
    assert row["status"] == "done"
    assert row["eval_metrics_json"] is None
    conn.close()
