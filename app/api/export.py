from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.deps import cancel_or_force_fail, claim_stage_or_409, get_job_store, run_with_new_store
from app.export_service import compute_export
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["export"])


class ExportRequest(BaseModel):
    format: Literal["ply", "sog", "spz", "html"]


@router.post("/{job_id}/export", status_code=202)
def export_endpoint(
    job_id: str,
    body: ExportRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    claim_stage_or_409(store, job_id, "export", "Export is already running for this job")
    background_tasks.add_task(run_with_new_store, compute_export, job_id, format=body.format)
    return {"status": "started"}


@router.post("/{job_id}/export/cancel")
def cancel_export_endpoint(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> dict[str, bool]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # convert_splat() is a blocking subprocess.run() with its own timeout,
    # never registered in process_registry - see cancel_or_force_fail, this
    # can only ever clear a stuck "running" row, not interrupt a live convert.
    return {"cancelled": cancel_or_force_fail(store, job_id, "export")}


@router.get("/{job_id}/export")
def get_export(job_id: str, store: JobStore = Depends(get_job_store)) -> dict[str, str | None]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.export_path is None:
        raise HTTPException(status_code=404, detail="Export not computed yet")
    return {"export_path": job.export_path, "export_format": job.export_format}


@router.get("/{job_id}/export/download")
def download_export(job_id: str, store: JobStore = Depends(get_job_store)) -> FileResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.export_path is None:
        raise HTTPException(status_code=404, detail="Export not computed yet")
    export_path = Path(job.export_path)
    if not export_path.is_file():
        raise HTTPException(status_code=404, detail="Export file not found on disk")
    return FileResponse(export_path, filename=export_path.name)
