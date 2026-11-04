from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.alignment_service import compute_alignment
from app.deps import cancel_or_force_fail, claim_stage_or_409, get_job_store, run_with_new_store
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["alignment"])


@router.post("/{job_id}/align", status_code=202)
def align_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    claim_stage_or_409(store, job_id, "alignment", "Alignment is already running for this job")
    background_tasks.add_task(run_with_new_store, compute_alignment, job_id)
    return {"status": "started"}


@router.post("/{job_id}/align/cancel")
def cancel_align_endpoint(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> dict[str, bool]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"cancelled": cancel_or_force_fail(store, job_id, "alignment")}


@router.get("/{job_id}/alignment")
def get_alignment(job_id: str, store: JobStore = Depends(get_job_store)) -> dict[str, str | None]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.colmap_sparse_dir is None:
        raise HTTPException(status_code=404, detail="Alignment not computed yet")
    return {
        "colmap_sparse_dir": job.colmap_sparse_dir,
        "undistorted_images_dir": job.undistorted_images_dir,
    }
