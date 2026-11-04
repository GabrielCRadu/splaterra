from __future__ import annotations

import logging
import shutil
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app.deps import get_job_store
from app.job_store import JobStore
from app.paths import job_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


class CreateJobRequest(BaseModel):
    name: str
    device_type: Literal["standard", "dji_osmo_360", "dji_avata_360"] = "standard"


class JobResponse(BaseModel):
    id: str
    name: str
    created_at: str
    raw_video_path: str | None
    frames_dir: str | None
    frame_count: int | None
    recommendation_json: str | None
    colmap_sparse_dir: str | None
    undistorted_images_dir: str | None
    splat_path: str | None
    export_path: str | None
    export_format: str | None
    device_type: str


class JobStageResponse(BaseModel):
    stage: str
    status: str
    started_at: str | None
    finished_at: str | None
    log: str | None
    error: str | None
    progress_percent: float | None
    settings_json: str | None
    eval_metrics_json: str | None


@router.post("", response_model=JobResponse)
def create_job(
    body: CreateJobRequest, store: JobStore = Depends(get_job_store)
) -> JobResponse:
    job = store.create_job(body.name, device_type=body.device_type)
    return JobResponse(**job.__dict__)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> JobResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**job.__dict__)


@router.get("", response_model=list[JobResponse])
def list_jobs(store: JobStore = Depends(get_job_store)) -> list[JobResponse]:
    return [JobResponse(**job.__dict__) for job in store.list_jobs()]


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str, store: JobStore = Depends(get_job_store)) -> Response:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if any(stage.status == "running" for stage in store.list_stages(job_id)):
        raise HTTPException(status_code=409, detail="Cannot delete a job with a stage still running")

    store.delete_job(job_id)
    directory = job_dir(job_id)
    if directory.is_dir():
        try:
            shutil.rmtree(directory)
        except OSError:
            # The DB record (what the job list actually reflects) is already
            # gone - a locked file (open download, antivirus scan, etc.) on
            # Windows shouldn't turn "delete succeeded" into an unhandled 500
            # for something the user can't do anything about anyway. Leaves
            # an orphaned directory on disk rather than a stuck/half-deleted
            # job the user can't retry.
            logger.exception("Failed to remove job directory %s after delete (non-fatal)", directory)
    return Response(status_code=204)


@router.get("/{job_id}/stages", response_model=list[JobStageResponse])
def get_job_stages(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> list[JobStageResponse]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return [JobStageResponse(**stage.__dict__) for stage in store.list_stages(job_id)]
