from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from app import process_registry
from app.ingestion import list_raw_videos
from app.job_store import JobStore
from app.paths import frames_dir, raw_dir
from app.sharp_frames import run_extraction
from app.three_sixty_extractor import run_360_extraction

# Matches app/api/frames.py's _IMAGE_SUFFIXES - kept as a separate constant since
# extraction.py is a service module and shouldn't import from the API layer.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
    interval_seconds: float = 1.0,
    ai_mask: bool = False,
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

    def report_log(text: str) -> None:
        store.upsert_stage(job_id, "frame_extraction", "running", log=text)

    def register_process(process) -> None:
        process_registry.register(job_id, "frame_extraction", process)

    # 2+ real video files on disk means a multi-clip job: sharp-frames auto-detects
    # a directory input and processes every video, combining frame candidates from
    # all of them (see app/sharp_frames.py). 0 or 1 real file (including tests that
    # set raw_video_path without creating a real file) keeps the original behavior.
    # Multi-clip isn't supported for 360 jobs yet (run_360_extraction takes a single
    # video) - device_type != "standard" always uses job.raw_video_path directly.
    videos = list_raw_videos(job_id)
    is_360 = job.device_type != "standard"
    input_path = (
        raw_dir(job_id) if len(videos) > 1 and not is_360 else Path(job.raw_video_path)
    )

    try:
        if is_360:
            result = run_360_extraction(
                input_path,
                output_dir,
                interval_seconds=interval_seconds,
                ai_mask=ai_mask,
                on_progress=report_progress,
                on_log=report_log,
                on_process_start=register_process,
            )
        else:
            result = run_extraction(
                input_path,
                output_dir,
                selection_method=selection_method,
                num_frames=num_frames,
                fps=fps,
                on_progress=report_progress,
                on_log=report_log,
                on_process_start=register_process,
            )
    except Exception as exc:
        message = (
            "Extragere anulata de utilizator"
            if process_registry.was_cancelled(job_id, "frame_extraction")
            else str(exc)
        )
        store.upsert_stage(job_id, "frame_extraction", "failed", finished_at=_now(), error=message)
        raise ExtractionError(message) from exc
    finally:
        process_registry.unregister(job_id, "frame_extraction")

    store.update_job(job_id, frames_dir=str(output_dir), frame_count=len(result.frames))
    store.upsert_stage(job_id, "frame_extraction", "done", finished_at=_now(), log=result.log)
    return len(result.frames)


def import_local_frames(store: JobStore, job_id: str, source_dir: Path) -> int:
    # Alternative to extract_frames(): skips sharp-frames entirely for callers
    # who already have a folder of hand-picked/pre-extracted sharp frames.
    # Deliberately independent of raw_video_path/ingestion - a frames-only
    # workflow with no ingested video is a valid use case.
    job = store.get_job(job_id)
    if job is None:
        # No stage write here: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row to
        # attach a stage to for a job_id that doesn't exist.
        raise ExtractionError(f"Unknown job: {job_id}")
    if not source_dir.is_dir():
        message = f"Source directory not found: {source_dir}"
        now = _now()
        store.upsert_stage(job_id, "frame_extraction", "failed", started_at=now, finished_at=now, error=message)
        raise ExtractionError(message)

    images = sorted(
        p for p in source_dir.glob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not images:
        message = f"No image files found in {source_dir}"
        now = _now()
        store.upsert_stage(job_id, "frame_extraction", "failed", started_at=now, finished_at=now, error=message)
        raise ExtractionError(message)

    output_dir = frames_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    store.upsert_stage(job_id, "frame_extraction", "running", started_at=_now())

    try:
        for image in images:
            shutil.copy2(image, output_dir / image.name)
    except OSError as exc:
        store.upsert_stage(job_id, "frame_extraction", "failed", finished_at=_now(), error=str(exc))
        raise ExtractionError(f"Failed to copy frames: {exc}") from exc

    store.update_job(job_id, frames_dir=str(output_dir), frame_count=len(images))
    store.upsert_stage(
        job_id, "frame_extraction", "done", finished_at=_now(),
        log=f"Imported {len(images)} pre-extracted frames from {source_dir}",
    )
    return len(images)
