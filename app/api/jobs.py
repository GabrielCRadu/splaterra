from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store
from app.job_store import JobStore

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
