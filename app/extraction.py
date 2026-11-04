from __future__ import annotations

import json
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


def _invalidate_downstream_results(store: JobStore, job_id: str, job) -> None:
    # If alignment/training/export already ran against the OLD frame set,
    # leaving their job_stages rows and job.colmap_sparse_dir/splat_path/
    # export_path in place after a re-extraction would silently lie to the
    # user: the UI would keep showing "done" for stages whose output no
    # longer corresponds to the frames now in frames_dir, and the download
    # endpoints would keep serving a splat/export built from frames that no
    # longer exist. Only touches anything when there's actually something
    # downstream to invalidate, so the normal first-time extraction path
    # doesn't pay for an extra write.
    if job.colmap_sparse_dir is None:
        return
    store.update_job(
        job_id,
        colmap_sparse_dir=None,
        undistorted_images_dir=None,
        splat_path=None,
        export_path=None,
        export_format=None,
    )
    store.delete_stages(job_id, ["alignment", "training", "export"])


def extract_frames(
    store: JobStore,
    job_id: str,
    *,
    selection_method: str = "best-n",
    num_frames: int = 300,
    fps: int = 10,
    interval_seconds: float = 1.0,
    ai_mask: bool = False,
    skip_up_frames: bool = False,
    layout: str = "cube",
    camera_count: int = 6,
    resolution: int | None = None,
    nadir_mask: bool = False,
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
    # Clear stale output before each run: sharp-frames handles this itself via
    # --force-overwrite, but run_360_extraction has no such flag - a retry with
    # different settings (e.g. a new interval_seconds) would otherwise leave
    # old frames from the previous attempt mixed in with the new ones.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2+ real video files on disk means a multi-clip job: sharp-frames auto-detects
    # a directory input and processes every video, combining frame candidates from
    # all of them (see app/sharp_frames.py). 0 or 1 real file (including tests that
    # set raw_video_path without creating a real file) keeps the original behavior.
    # 360Extractor's own CLI does the same for a directory input (os.walk +
    # per-file <stem>_processed/ output subfolder under the shared --output,
    # confirmed against the real installed tool's src/main.py/core/processor.py) -
    # so device_type != "standard" can use the exact same raw_dir(job_id) path.
    videos = list_raw_videos(job_id)
    is_360 = job.device_type != "standard"
    input_path = raw_dir(job_id) if len(videos) > 1 else Path(job.raw_video_path)

    settings_json = json.dumps(
        {
            "interval_seconds": interval_seconds,
            "ai_mask": ai_mask,
            "skip_up_frames": skip_up_frames,
            "layout": layout,
            "camera_count": camera_count,
            "resolution": resolution,
            "nadir_mask": nadir_mask,
        }
        if is_360
        else {"selection_method": selection_method, "num_frames": num_frames, "fps": fps}
    )
    store.upsert_stage(
        job_id, "frame_extraction", "running", started_at=_now(), settings_json=settings_json
    )

    def report_progress(percent: float) -> None:
        store.upsert_stage(job_id, "frame_extraction", "running", progress_percent=percent)

    def report_log(text: str) -> None:
        store.upsert_stage(job_id, "frame_extraction", "running", log=text)

    def register_process(process) -> None:
        process_registry.register(job_id, "frame_extraction", process)

    try:
        if is_360:
            result = run_360_extraction(
                input_path,
                output_dir,
                interval_seconds=interval_seconds,
                ai_mask=ai_mask,
                skip_up_frames=skip_up_frames,
                layout=layout,
                camera_count=camera_count,
                resolution=resolution,
                nadir_mask=nadir_mask,
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

    _invalidate_downstream_results(store, job_id, job)
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

    _invalidate_downstream_results(store, job_id, job)
    store.update_job(job_id, frames_dir=str(output_dir), frame_count=len(images))
    store.upsert_stage(
        job_id, "frame_extraction", "done", finished_at=_now(),
        log=f"Imported {len(images)} pre-extracted frames from {source_dir}",
    )
    return len(images)
