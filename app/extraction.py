from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import frames_dir
from app.sharp_frames import ExtractionResult, SharpFramesError, run_extraction


class ExtractionError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_frames(
    store: JobStore,
    job_id: str,
    *,
    selection_method: str = "best-n",
    num_frames: int = 300,
    fps: int = 10,
) -> int:
    job = store.get_job(job_id)
    if job is None:
        # No stage write here: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row to
        # attach a stage to for a job_id that doesn't exist.
        raise ExtractionError(f"Unknown job: {job_id}")
    if job.raw_video_path is None:
        now = _now()
        store.upsert_stage(
            job_id, "frame_extraction", "failed", started_at=now, finished_at=now,
            error="Job has no ingested video yet",
        )
        raise ExtractionError("Job has no ingested video yet")

    output_dir = frames_dir(job_id)
    store.upsert_stage(job_id, "frame_extraction", "running", started_at=_now())

    def report_progress(percent: float) -> None:
        store.upsert_stage(job_id, "frame_extraction", "running", progress_percent=percent)

    try:
        result: ExtractionResult = run_extraction(
            Path(job.raw_video_path),
            output_dir,
            selection_method=selection_method,
            num_frames=num_frames,
            fps=fps,
            on_progress=report_progress,
        )
    except SharpFramesError as exc:
        store.upsert_stage(job_id, "frame_extraction", "failed", finished_at=_now(), error=str(exc))
        raise ExtractionError(str(exc)) from exc

    store.update_job(job_id, frames_dir=str(output_dir), frame_count=len(result.frames))
    store.upsert_stage(job_id, "frame_extraction", "done", finished_at=_now(), log=result.log)
    return len(result.frames)
