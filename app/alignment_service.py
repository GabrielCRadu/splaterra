from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app import process_registry
from app.job_store import JobStore
from app.paths import alignment_dir
from app.realityscan import AlignmentResult, run_alignment


class AlignmentError(Exception):
    pass


# app/realityscan.py's run_alignment() defaults to a flat 600s timeout, which
# doesn't scale with how many frames RealityScan has to align - fine for a
# single clip's worth of frames, but a multi-clip job can have hundreds of
# frames and legitimately take much longer. The only real timing data point
# so far (MasinaPoarta: 78 frames, single clip) took ~65s, so 5s/frame leaves
# generous headroom; recalibrate once more multi-clip runs are logged.
ALIGNMENT_BASE_TIMEOUT_SECONDS = 600
ALIGNMENT_SECONDS_PER_FRAME = 5


def _alignment_timeout(frame_count: int | None) -> int:
    if not frame_count:
        return ALIGNMENT_BASE_TIMEOUT_SECONDS
    return max(ALIGNMENT_BASE_TIMEOUT_SECONDS, frame_count * ALIGNMENT_SECONDS_PER_FRAME)


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

    def report_log(text: str) -> None:
        store.upsert_stage(job_id, "alignment", "running", log=text)

    try:
        result = run_alignment(
            Path(job.frames_dir),
            output_dir,
            progress_path=progress_path,
            on_progress=report_progress,
            timeout_seconds=_alignment_timeout(job.frame_count),
            on_process_start=lambda p: process_registry.register(job_id, "alignment", p),
            on_log=report_log,
        )
    except Exception as exc:
        message = (
            "Alignment anulat de utilizator"
            if process_registry.was_cancelled(job_id, "alignment")
            else str(exc)
        )
        store.upsert_stage(job_id, "alignment", "failed", finished_at=_now(), error=message)
        raise AlignmentError(message) from exc
    finally:
        process_registry.unregister(job_id, "alignment")

    store.update_job(
        job_id,
        colmap_sparse_dir=str(result.sparse_dir),
        undistorted_images_dir=str(result.images_dir),
    )
    store.upsert_stage(job_id, "alignment", "done", finished_at=_now(), log=result.log)
    return result
