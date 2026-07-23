from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import alignment_dir
from app.realityscan import AlignmentResult, run_alignment


class AlignmentError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_alignment(store: JobStore, job_id: str) -> AlignmentResult:
    job = store.get_job(job_id)
    if job is None:
        # No stage write: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row
        # to attach a stage to for a job_id that doesn't exist.
        raise AlignmentError(f"Unknown job: {job_id}")
    if job.frames_dir is None:
        message = "Job has no extracted frames yet"
        now = _now()
        store.upsert_stage(job_id, "alignment", "failed", started_at=now, finished_at=now, error=message)
        raise AlignmentError(message)

    output_dir = alignment_dir(job_id)
    progress_path = output_dir / "progress.txt"
    store.upsert_stage(job_id, "alignment", "running", started_at=_now())

    def report_progress(percent: float) -> None:
        store.upsert_stage(job_id, "alignment", "running", progress_percent=percent)

    try:
        result = run_alignment(
            Path(job.frames_dir), output_dir, progress_path=progress_path, on_progress=report_progress
        )
    except Exception as exc:
        store.upsert_stage(job_id, "alignment", "failed", finished_at=_now(), error=str(exc))
        raise AlignmentError(str(exc)) from exc

    store.update_job(
        job_id,
        colmap_sparse_dir=str(result.sparse_dir),
        undistorted_images_dir=str(result.images_dir),
    )
    store.upsert_stage(job_id, "alignment", "done", finished_at=_now(), log=result.log)
    return result
