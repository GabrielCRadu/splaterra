from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store
from app.extraction import ExtractionError, extract_frames
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["frames"])


class ExtractFramesRequest(BaseModel):
    selection_method: str = "best-n"
    num_frames: int = 300
    fps: int = 10


@router.post("/{job_id}/extract-frames")
def extract_frames_endpoint(
    job_id: str,
    body: ExtractFramesRequest,
    store: JobStore = Depends(get_job_store),
) -> dict[str, int]:
    try:
        count = extract_frames(
            store,
            job_id,
            selection_method=body.selection_method,
            num_frames=body.num_frames,
            fps=body.fps,
        )
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"frame_count": count}
