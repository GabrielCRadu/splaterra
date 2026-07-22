from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store
from app.ingestion import IngestionError, import_local_file
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["ingestion"])


class LocalImportRequest(BaseModel):
    source_path: str


@router.post("/{job_id}/ingest/local")
def ingest_local(
    job_id: str,
    body: LocalImportRequest,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        destination = import_local_file(store, job_id, Path(body.source_path))
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"raw_video_path": str(destination)}
