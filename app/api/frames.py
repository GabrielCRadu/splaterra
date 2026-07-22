from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store, run_with_new_store
from app.extraction import extract_frames
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["frames"])


class ExtractFramesRequest(BaseModel):
    selection_method: str = "best-n"
    num_frames: int = 300
    fps: int = 10


@router.post("/{job_id}/extract-frames", status_code=202)
def extract_frames_endpoint(
    job_id: str,
    body: ExtractFramesRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    background_tasks.add_task(
        run_with_new_store,
        extract_frames,
        job_id,
        selection_method=body.selection_method,
        num_frames=body.num_frames,
        fps=body.fps,
    )
    return {"status": "started"}
