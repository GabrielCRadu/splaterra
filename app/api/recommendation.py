from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store
from app.job_store import JobStore
from app.recommendation_service import RecommendationError, compute_recommendation

router = APIRouter(prefix="/jobs", tags=["recommendation"])


class RecommendRequest(BaseModel):
    srt_path: str | None = None


@router.post("/{job_id}/recommend")
def recommend_endpoint(
    job_id: str, body: RecommendRequest, store: JobStore = Depends(get_job_store)
) -> dict:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    srt_path = Path(body.srt_path) if body.srt_path else None
    try:
        recommendation = compute_recommendation(store, job_id, srt_path)
    except RecommendationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dataclasses.asdict(recommendation)


@router.get("/{job_id}/recommendation")
def get_recommendation(job_id: str, store: JobStore = Depends(get_job_store)) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.recommendation_json is None:
        raise HTTPException(status_code=404, detail="Recommendation not computed yet")
    return json.loads(job.recommendation_json)
