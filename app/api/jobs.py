from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app.deps import get_job_store
from app.job_store import JobStore
from app.paths import job_dir

router = APIRouter(prefix="/jobs", tags=["jobs"])


class CreateJobRequest(BaseModel):
    name: str


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


class JobStageResponse(BaseModel):
    stage: str
    status: str
    started_at: str | None
    finished_at: str | None
    log: str | None
    error: str | None
    progress_percent: float | None


@router.post("", response_model=JobResponse)
def create_job(
    body: CreateJobRequest, store: JobStore = Depends(get_job_store)
) -> JobResponse:
    job = store.create_job(body.name)
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
        shutil.rmtree(directory)
    return Response(status_code=204)


@router.get("/{job_id}/stages", response_model=list[JobStageResponse])
def get_job_stages(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> list[JobStageResponse]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return [JobStageResponse(**stage.__dict__) for stage in store.list_stages(job_id)]
