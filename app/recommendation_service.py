from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from app.drone_telemetry import SrtParseError, parse_srt
from app.hardware import detect_hardware
from app.job_store import JobStore
from app.recommendation import Recommendation, recommend
from app.video_metadata import VideoMetadataError, probe_video


class RecommendationError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_recommendation(
    store: JobStore, job_id: str, srt_path: Path | None = None
) -> Recommendation:
    job = store.get_job(job_id)
    if job is None:
        # No stage write: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row
        # to attach a stage to for a job_id that doesn't exist.
        raise RecommendationError(f"Unknown job: {job_id}")
    if job.raw_video_path is None:
        message = "Job has no ingested video yet"
        now = _now()
        store.upsert_stage(
            job_id, "recommendation", "failed", started_at=now, finished_at=now, error=message
        )
        raise RecommendationError(message)

    store.upsert_stage(job_id, "recommendation", "running", started_at=_now())
    try:
        video_meta = probe_video(Path(job.raw_video_path))
        drone = parse_srt(srt_path) if srt_path is not None else None
        hardware = detect_hardware()
        recommendation = recommend(video_meta, drone, hardware)
    except (VideoMetadataError, SrtParseError) as exc:
        store.upsert_stage(job_id, "recommendation", "failed", finished_at=_now(), error=str(exc))
        raise RecommendationError(str(exc)) from exc

    payload = json.dumps(dataclasses.asdict(recommendation))
    store.update_job(job_id, recommendation_json=payload)
    store.upsert_stage(job_id, "recommendation", "done", finished_at=_now())
    return recommendation
