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


# RealityScan can exit 0 and write valid (if nearly empty) COLMAP files even
# when it only registered a tiny fraction of the input frames - e.g. a real
# job with 1666 extracted frames came back with just 2 registered images
# (insufficient overlap/features in the footage). alignment_service's
# existing "do the expected files exist" check passed fine, so the job sailed
# through to training, which crashed 5 seconds in on the degenerate 2-camera/
# 42-point reconstruction with a cryptic CUDA error. Catch it here instead,
# with a message that actually points at the real problem.
MIN_REGISTERED_IMAGES = 10
MIN_REGISTRATION_RATIO = 0.2


def _registration_coverage_error(images_dir: Path, frame_count: int | None) -> str | None:
    registered = sum(1 for _ in images_dir.iterdir())
    too_few_absolute = registered < MIN_REGISTERED_IMAGES
    too_few_relative = bool(frame_count) and registered / frame_count < MIN_REGISTRATION_RATIO
    if not (too_few_absolute or too_few_relative):
        return None
    if frame_count:
        percent = round(registered / frame_count * 100)
        return (
            f"Alignment a inregistrat doar {registered} din {frame_count} cadre ({percent}%) - "
            "prea putine pentru o reconstructie utila. Verifica daca footage-ul are suficient "
            "overlap intre cadre."
        )
    return (
        f"Alignment a inregistrat doar {registered} cadre - prea putine pentru o "
        "reconstructie utila. Verifica daca footage-ul are suficient overlap intre cadre."
    )


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

        # Kept inside the same try/except as run_alignment (not after it):
        # images_dir.iterdir() below is a filesystem read that - even though
        # run_alignment already confirmed the directory exists right before
        # returning - could still race against an external deletion in this
        # narrow window. Left unguarded, that would raise past this function
        # entirely (only caught by the background task's outer catch-all in
        # app/deps.py), leaving the stage stuck on "running" forever with no
        # visible failure instead of the "failed" status every other error
        # path here produces.
        coverage_error = _registration_coverage_error(result.images_dir, job.frame_count)
        if coverage_error is not None:
            store.upsert_stage(job_id, "alignment", "failed", finished_at=_now(), error=coverage_error)
            raise AlignmentError(coverage_error)

        store.update_job(
            job_id,
            colmap_sparse_dir=str(result.sparse_dir),
            undistorted_images_dir=str(result.images_dir),
        )
        store.upsert_stage(job_id, "alignment", "done", finished_at=_now(), log=result.log)
    except AlignmentError:
        raise  # already recorded as "failed" above - avoid a redundant duplicate write
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

    return result
