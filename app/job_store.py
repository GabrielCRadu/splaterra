from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from app.models import Job, JobStage


class JobStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create_job(self, name: str) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._conn.execute(
            "INSERT INTO jobs (id, name, created_at) VALUES (?, ?, ?)",
            (job.id, job.name, job.created_at),
        )
        self._conn.commit()
        return job

    def get_job(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self) -> list[Job]:
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def update_job(self, job_id: str, **fields: object) -> None:
        if not fields:
            return
        columns = ", ".join(f"{key} = ?" for key in fields)
        values = [*fields.values(), job_id]
        self._conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", values)
        self._conn.commit()

    def upsert_stage(
        self,
        job_id: str,
        stage: str,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        log: str | None = None,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO job_stages (job_id, stage, status, started_at, finished_at, log, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, stage) DO UPDATE SET
                status = excluded.status,
                started_at = COALESCE(excluded.started_at, job_stages.started_at),
                finished_at = excluded.finished_at,
                log = excluded.log,
                error = excluded.error
            """,
            (job_id, stage, status, started_at, finished_at, log, error),
        )
        self._conn.commit()

    def get_stage(self, job_id: str, stage: str) -> JobStage | None:
        row = self._conn.execute(
            "SELECT * FROM job_stages WHERE job_id = ? AND stage = ?",
            (job_id, stage),
        ).fetchone()
        return self._row_to_stage(row) if row else None

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            raw_video_path=row["raw_video_path"],
            frames_dir=row["frames_dir"],
            frame_count=row["frame_count"],
        )

    @staticmethod
    def _row_to_stage(row: sqlite3.Row) -> JobStage:
        return JobStage(
            job_id=row["job_id"],
            stage=row["stage"],
            status=row["status"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            log=row["log"],
            error=row["error"],
        )
