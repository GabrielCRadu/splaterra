from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from app.deps import get_job_store, run_with_new_store
from app.ingestion import import_local_file, list_raw_videos, probe_ingested_video
from app.job_store import JobStore
from app.uploads import UploadError, complete_upload, received_chunks, save_chunk, start_upload
from app.video_metadata import VideoMetadataError

router = APIRouter(prefix="/jobs", tags=["ingestion"])


class LocalImportRequest(BaseModel):
    source_path: str


class StartUploadRequest(BaseModel):
    filename: str
    total_chunks: int


@router.post("/{job_id}/ingest/local", status_code=202)
def ingest_local(
    job_id: str,
    body: LocalImportRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    background_tasks.add_task(run_with_new_store, import_local_file, job_id, Path(body.source_path))
    return {"status": "started"}


@router.get("/{job_id}/raw-videos")
def get_raw_videos(job_id: str, store: JobStore = Depends(get_job_store)) -> dict[str, list[str]]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"videos": [p.name for p in list_raw_videos(job_id)]}


@router.get("/{job_id}/video-info")
def get_video_info(job_id: str, store: JobStore = Depends(get_job_store)) -> dict[str, float | int]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.raw_video_path is None:
        raise HTTPException(status_code=404, detail="Job has no ingested video yet")
    try:
        meta = probe_ingested_video(job_id, job.raw_video_path)
    except VideoMetadataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "fps": meta.fps,
        "duration_seconds": meta.duration_seconds,
        "total_frames": round(meta.duration_seconds * meta.fps),
    }


@router.post("/{job_id}/uploads")
def create_upload(
    job_id: str, body: StartUploadRequest, store: JobStore = Depends(get_job_store)
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        upload_id = start_upload(job_id, body.filename, body.total_chunks)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


@router.post("/{job_id}/uploads/{upload_id}/complete", status_code=202)
def finish_upload(
    job_id: str,
    upload_id: str,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    background_tasks.add_task(run_with_new_store, complete_upload, job_id, upload_id)
    return {"status": "started"}
