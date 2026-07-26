from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.lichtfeld import convert_splat
from app.paths import training_dir


class ExportError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_export(store: JobStore, job_id: str, *, format: str) -> Path:
    job = store.get_job(job_id)
    if job is None:
        # No stage write: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row
        # to attach a stage to for a job_id that doesn't exist.
        raise ExportError(f"Unknown job: {job_id}")
    if job.splat_path is None:
        message = "Job has no trained splat yet"
        now = _now()
        store.upsert_stage(job_id, "export", "failed", started_at=now, finished_at=now, error=message)
        raise ExportError(message)

    output_path = training_dir(job_id) / f"export.{format}"
    store.upsert_stage(job_id, "export", "running", started_at=_now())

    try:
        result_path = convert_splat(Path(job.splat_path), output_path, format=format)
    except Exception as exc:
        store.upsert_stage(job_id, "export", "failed", finished_at=_now(), error=str(exc))
        raise ExportError(str(exc)) from exc

    store.update_job(job_id, export_path=str(result_path), export_format=format)
    store.upsert_stage(job_id, "export", "done", finished_at=_now())
    return result_path
