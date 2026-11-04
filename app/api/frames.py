from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.deps import cancel_or_force_fail, claim_stage_or_409, get_job_store, run_with_new_store
from app.extraction import extract_frames, import_local_frames
from app.job_store import JobStore
from app.paths import frames_dir

router = APIRouter(prefix="/jobs", tags=["frames"])

# sharp-frames writes a selected_metadata.json alongside the extracted images
# in the same output directory - only list actual image files, matching the
# extensions app/sharp_frames.py's run_extraction() already globs for its result.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class ExtractFramesRequest(BaseModel):
    selection_method: str = "best-n"
    num_frames: int = Field(default=300, gt=0)
    fps: int = Field(default=10, gt=0)
    # Used only for 360 jobs (job.device_type != "standard") - ignored by the
    # sharp-frames path, same as selection_method/num_frames/fps are ignored
    # by the 360 path. See app/extraction.py's device_type branch.
    interval_seconds: float = Field(default=1.0, gt=0)
    ai_mask: bool = False
    skip_up_frames: bool = False
    layout: Literal["cube", "ring", "fibonacci"] = "cube"
    # 2-36 per the real tool's documented range (docs/SETTINGS.md); ignored
    # by the tool itself for layout="cube", which always forces 6.
    camera_count: int = Field(default=6, ge=2, le=36)
    resolution: int | None = Field(default=None, gt=0)
    nadir_mask: bool = False


class ImportFramesRequest(BaseModel):
    source_path: str


@router.post("/{job_id}/extract-frames", status_code=202)
def extract_frames_endpoint(
    job_id: str,
    body: ExtractFramesRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    claim_stage_or_409(
        store, job_id, "frame_extraction", "Frame extraction is already running for this job"
    )
    background_tasks.add_task(
        run_with_new_store,
        extract_frames,
        job_id,
        selection_method=body.selection_method,
        num_frames=body.num_frames,
        fps=body.fps,
        interval_seconds=body.interval_seconds,
        ai_mask=body.ai_mask,
        skip_up_frames=body.skip_up_frames,
        layout=body.layout,
        camera_count=body.camera_count,
        resolution=body.resolution,
        nadir_mask=body.nadir_mask,
    )
    return {"status": "started"}


@router.post("/{job_id}/extract-frames/cancel")
def cancel_extract_frames_endpoint(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> dict[str, bool]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"cancelled": cancel_or_force_fail(store, job_id, "frame_extraction")}


@router.post("/{job_id}/import-frames", status_code=202)
def import_frames_endpoint(
    job_id: str,
    body: ImportFramesRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    claim_stage_or_409(
        store, job_id, "frame_extraction", "Frame extraction is already running for this job"
    )
    background_tasks.add_task(run_with_new_store, import_local_frames, job_id, Path(body.source_path))
    return {"status": "started"}


@router.get("/{job_id}/frames")
def list_frames(job_id: str, store: JobStore = Depends(get_job_store)) -> dict[str, list[str]]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.frames_dir is None:
        return {"frames": []}
    directory = frames_dir(job_id)
    names = sorted(
        p.name for p in directory.glob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    return {"frames": names}


@router.delete("/{job_id}/frames/{filename}")
def delete_frame(
    job_id: str, filename: str, store: JobStore = Depends(get_job_store)
) -> dict[str, bool]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", "..") or safe_name != filename:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename!r}")
    directory = frames_dir(job_id)
    frame_path = directory / safe_name
    if not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found")
    frame_path.unlink()

    remaining = sum(
        1 for p in directory.glob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    store.update_job(job_id, frame_count=remaining)
    return {"deleted": True}


@router.get("/{job_id}/frames/{filename}")
def get_frame(
    job_id: str, filename: str, store: JobStore = Depends(get_job_store)
) -> FileResponse:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", "..") or safe_name != filename:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename!r}")
    frame_path = frames_dir(job_id) / safe_name
    if not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path)
