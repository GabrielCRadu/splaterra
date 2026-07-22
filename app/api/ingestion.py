from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.deps import get_job_store
from app.ingestion import IngestionError, import_local_file
from app.job_store import JobStore
from app.uploads import UploadError, complete_upload, received_chunks, save_chunk, start_upload

router = APIRouter(prefix="/jobs", tags=["ingestion"])


class LocalImportRequest(BaseModel):
    source_path: str


class StartUploadRequest(BaseModel):
    filename: str
    total_chunks: int


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


@router.post("/{job_id}/uploads")
def create_upload(
    job_id: str, body: StartUploadRequest, store: JobStore = Depends(get_job_store)
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    upload_id = start_upload(job_id, body.filename, body.total_chunks)
    return {"upload_id": upload_id}


@router.put("/{job_id}/uploads/{upload_id}/chunks/{chunk_index}")
async def upload_chunk(job_id: str, upload_id: str, chunk_index: int, request: Request) -> dict[str, int]:
    data = await request.body()
    try:
        save_chunk(job_id, upload_id, chunk_index, data)
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"received": chunk_index}


@router.get("/{job_id}/uploads/{upload_id}/chunks")
def list_received_chunks(job_id: str, upload_id: str) -> dict[str, list[int]]:
    try:
        return {"received_chunks": received_chunks(job_id, upload_id)}
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{job_id}/uploads/{upload_id}/complete")
def finish_upload(
    job_id: str, upload_id: str, store: JobStore = Depends(get_job_store)
) -> dict[str, str]:
    try:
        destination = complete_upload(store, job_id, upload_id)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"raw_video_path": str(destination)}
