from __future__ import annotations

import json
from datetime import datetime, timezone

from app.job_store import JobStore
from app.lichtfeld import TrainingResult, run_training
from app.paths import alignment_dir, training_dir


class TrainingError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_gaussians_from_recommendation(recommendation_json: str | None) -> int | None:
    if recommendation_json is None:
        return None
    try:
        return json.loads(recommendation_json)["lichtfeld"]["max_gaussians"]
    except (KeyError, TypeError, ValueError):
        return None


def compute_training(store: JobStore, job_id: str) -> TrainingResult:
    job = store.get_job(job_id)
    if job is None:
        # No stage write: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row
        # to attach a stage to for a job_id that doesn't exist.
        raise TrainingError(f"Unknown job: {job_id}")
    if job.colmap_sparse_dir is None:
        message = "Job has no alignment computed yet"
        now = _now()
        store.upsert_stage(job_id, "training", "failed", started_at=now, finished_at=now, error=message)
        raise TrainingError(message)

    # alignment_dir(job_id) is the COLMAP dataset root LichtFeld's --data-path
    # expects (parent of both sparse/0/ and images/) - it's the same output_dir
    # app/alignment_service.py already passed to run_alignment, recomputed here
    # directly via the same path helper rather than parsed out of
    # colmap_sparse_dir.
    data_path = alignment_dir(job_id)
    output_dir = training_dir(job_id)
    max_gaussians = _max_gaussians_from_recommendation(job.recommendation_json)

    store.upsert_stage(job_id, "training", "running", started_at=_now())

    def report_progress(percent: float) -> None:
        store.upsert_stage(job_id, "training", "running", progress_percent=percent)

    try:
        result = run_training(
            data_path, output_dir, max_gaussians=max_gaussians, on_progress=report_progress
        )
    except Exception as exc:
        store.upsert_stage(job_id, "training", "failed", finished_at=_now(), error=str(exc))
        raise TrainingError(str(exc)) from exc

    store.update_job(job_id, splat_path=str(result.splat_path))
    store.upsert_stage(job_id, "training", "done", finished_at=_now(), log=result.log)
    return result
