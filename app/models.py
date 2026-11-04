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
    colmap_sparse_dir: str | None = None
    undistorted_images_dir: str | None = None
    splat_path: str | None = None
    export_path: str | None = None
    export_format: str | None = None
    device_type: str = "standard"


@dataclass
class JobStage:
    job_id: str
    stage: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    log: str | None = None
    error: str | None = None
    progress_percent: float | None = None
    settings_json: str | None = None
    eval_metrics_json: str | None = None
