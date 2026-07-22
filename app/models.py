from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Job:
    id: str
    name: str
    created_at: str
    raw_video_path: str | None = None
    frames_dir: str | None = None
    frame_count: int | None = None
    recommendation_json: str | None = None


@dataclass
class JobStage:
    job_id: str
    stage: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    log: str | None = None
    error: str | None = None
