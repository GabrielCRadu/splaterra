from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from app.models import Job, JobStage


class JobStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create_job(self, name: str, device_type: str = "standard") -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            device_type=device_type,
        )
        self._conn.execute(
            "INSERT INTO jobs (id, name, created_at, device_type) VALUES (?, ?, ?, ?)",
            (job.id, job.name, job.created_at, job.device_type),
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
        progress_percent: float | None = None,
        settings_json: str | None = None,
        eval_metrics_json: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO job_stages (job_id, stage, status, started_at, finished_at, log, error, progress_percent, settings_json, eval_metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, stage) DO UPDATE SET
                status = excluded.status,
                started_at = COALESCE(excluded.started_at, job_stages.started_at),
                finished_at = excluded.finished_at,
                log = excluded.log,
                error = excluded.error,
                progress_percent = COALESCE(excluded.progress_percent, job_stages.progress_percent),
                settings_json = COALESCE(excluded.settings_json, job_stages.settings_json),
                eval_metrics_json = COALESCE(excluded.eval_metrics_json, job_stages.eval_metrics_json)
            """,
            (job_id, stage, status, started_at, finished_at, log, error, progress_percent, settings_json, eval_metrics_json),
        )
        self._conn.commit()

    def try_start_stage(self, job_id: str, stage: str, started_at: str) -> bool:
        # Atomic claim, not read-then-write: the old pattern (endpoint reads
        # stage.status, background task writes "running" later) left a gap
        # where two near-simultaneous requests could both read a non-running
        # status before either write landed, so both would pass the 409 guard
        # and both get scheduled - the second one silently orphans the first
        # from process_registry/cancellation. This single statement claims
        # the row (insert, or update only if not already 'running') and
        # reports whether the claim was won via rowcount - verified under
        # real concurrent threads/connections (20 threads, exactly 1 winner).
        try:
            cursor = self._conn.execute(
                """
                INSERT INTO job_stages (job_id, stage, status, started_at, finished_at, error)
                VALUES (?, ?, 'running', ?, NULL, NULL)
                ON CONFLICT(job_id, stage) DO UPDATE SET
                    status = 'running',
                    started_at = excluded.started_at,
                    finished_at = NULL,
                    error = NULL
                WHERE job_stages.status != 'running'
                """,
                (job_id, stage, started_at),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # jobs.id FK violation: the job was deleted concurrently between
            # the caller's existence check and this claim. Not a win either way.
            return False
        return cursor.rowcount > 0

    def get_stage(self, job_id: str, stage: str) -> JobStage | None:
        row = self._conn.execute(
            "SELECT * FROM job_stages WHERE job_id = ? AND stage = ?",
            (job_id, stage),
        ).fetchone()
        return self._row_to_stage(row) if row else None

    def delete_stages(self, job_id: str, stages: list[str]) -> None:
        # Used to invalidate downstream stages (e.g. alignment/training/export
        # after frames are re-extracted) - a missing row reads as "not started
        # yet" everywhere a stage is looked up, same as a job that never ran
        # that stage, so this is enough to make the UI stop showing stale
        # done/failed pills for results that no longer correspond to the
        # current frame set.
        if not stages:
            return
        placeholders = ", ".join("?" for _ in stages)
        self._conn.execute(
            f"DELETE FROM job_stages WHERE job_id = ? AND stage IN ({placeholders})",
            (job_id, *stages),
        )
        self._conn.commit()

    def delete_job(self, job_id: str) -> None:
        self._conn.execute("DELETE FROM job_stages WHERE job_id = ?", (job_id,))
        self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._conn.commit()

    def list_stages(self, job_id: str) -> list[JobStage]:
        rows = self._conn.execute(
            "SELECT * FROM job_stages WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
        return [self._row_to_stage(row) for row in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            raw_video_path=row["raw_video_path"],
            frames_dir=row["frames_dir"],
            frame_count=row["frame_count"],
            recommendation_json=row["recommendation_json"],
            colmap_sparse_dir=row["colmap_sparse_dir"],
            undistorted_images_dir=row["undistorted_images_dir"],
            splat_path=row["splat_path"],
            export_path=row["export_path"],
            export_format=row["export_format"],
            device_type=row["device_type"],
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
            progress_percent=row["progress_percent"],
            settings_json=row["settings_json"],
            eval_metrics_json=row["eval_metrics_json"],
        )
