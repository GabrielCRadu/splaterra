from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.deps import cancel_or_force_fail, claim_stage_or_409, get_job_store, run_with_new_store
from app.job_store import JobStore
from app.training_service import compute_training

router = APIRouter(prefix="/jobs", tags=["training"])


class TrainRequest(BaseModel):
    variable_exposure: bool = False
    max_gaussians: int | None = Field(default=None, gt=0)
    iterations: int | None = Field(default=None, gt=0)
    resize_factor: Literal[1, 2, 4, 8] | None = None
    eval: bool = False
    test_every: int | None = Field(default=None, gt=0)
    # UNVERIFIED against a real RunPod/vast.ai account - see docs/TODO.md.
    # image_name/gpu_type are required when use_cloud is set (checked in
    # app/training_service.py, not here, since the requirement is
    # conditional on use_cloud rather than a fixed field constraint).
    use_cloud: bool = False
    cloud_provider: Literal["runpod", "vastai"] = "runpod"
    cloud_image: str | None = None
    cloud_gpu_type: str | None = None


@router.post("/{job_id}/train", status_code=202)
def train_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks,
    body: TrainRequest = TrainRequest(),
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    claim_stage_or_409(store, job_id, "training", "Training is already running for this job")
    background_tasks.add_task(
        run_with_new_store,
        compute_training,
        job_id,
        variable_exposure=body.variable_exposure,
        max_gaussians=body.max_gaussians,
        iterations=body.iterations,
        resize_factor=body.resize_factor,
        eval=body.eval,
        test_every=body.test_every,
        use_cloud=body.use_cloud,
        cloud_provider=body.cloud_provider,
        cloud_image=body.cloud_image,
        cloud_gpu_type=body.cloud_gpu_type,
    )
    return {"status": "started"}


@router.post("/{job_id}/train/cancel")
def cancel_train_endpoint(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> dict[str, bool]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"cancelled": cancel_or_force_fail(store, job_id, "training")}


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
