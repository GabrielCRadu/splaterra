from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import process_registry
from app.deps import get_job_store, run_with_new_store
from app.job_store import JobStore
from app.training_service import compute_training

router = APIRouter(prefix="/jobs", tags=["training"])


class TrainRequest(BaseModel):
    variable_exposure: bool = False


@router.post("/{job_id}/train", status_code=202)
def train_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks,
    body: TrainRequest = TrainRequest(),
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    background_tasks.add_task(
        run_with_new_store, compute_training, job_id, variable_exposure=body.variable_exposure
    )
    return {"status": "started"}


@router.post("/{job_id}/train/cancel")
def cancel_train_endpoint(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> dict[str, bool]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"cancelled": process_registry.cancel(job_id, "training")}


@router.get("/{job_id}/training")
def get_training(job_id: str, store: JobStore = Depends(get_job_store)) -> dict[str, str | None]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.splat_path is None:
        raise HTTPException(status_code=404, detail="Training not computed yet")
    return {"splat_path": job.splat_path}


@router.get("/{job_id}/training/download")
def download_training_result(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> FileResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.splat_path is None:
        raise HTTPException(status_code=404, detail="Training not computed yet")
    splat_path = Path(job.splat_path)
    if not splat_path.is_file():
        raise HTTPException(status_code=404, detail="Splat file not found on disk")
    return FileResponse(splat_path, filename=splat_path.name)
