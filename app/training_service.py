from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Literal

from app import process_registry
from app.cloud_training import run_cloud_training
from app.hardware import UsageSnapshot, detect_hardware, sample_usage
from app.job_store import JobStore
from app.lichtfeld import TrainingResult, run_training
from app.paths import alignment_dir, training_dir
from app.recommendation import training_resize_factor
from app.training_metrics import TrainingMetrics, append_training_metrics, parse_training_log

logger = logging.getLogger(__name__)


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


def _log_training_metrics(
    job_id: str,
    *,
    status: str,
    error: str | None,
    log: str,
    variable_exposure: bool,
    max_gaussians: int | None,
    usage_samples: list[UsageSnapshot],
) -> None:
    # Diagnostic data for calibrating app/recommendation.py's constants (see
    # docs/ISSUES.md) - must never mask the real training outcome, so any
    # failure here is logged and swallowed rather than raised.
    try:
        parsed = parse_training_log(log) if log else dict.fromkeys(
            (
                "num_images",
                "num_initial_points",
                "sh_degree",
                "iterations_configured",
                "final_gaussians",
                "duration_seconds",
                "avg_iter_per_sec",
            )
        )
        hardware = detect_hardware()
        vram_values = [s.vram_used_mb for s in usage_samples if s.vram_used_mb is not None]
        cpu_values = [s.cpu_percent for s in usage_samples]
        ram_values = [s.ram_used_gb for s in usage_samples]

        metrics = TrainingMetrics(
            job_id=job_id,
            timestamp=_now(),
            status=status,
            error=error,
            variable_exposure=variable_exposure,
            requested_max_gaussians=max_gaussians,
            gpu_name=hardware.gpu.name if hardware.gpu else None,
            vram_total_mb=hardware.gpu.vram_mb if hardware.gpu else None,
            vram_peak_used_mb=max(vram_values) if vram_values else None,
            vram_avg_used_mb=int(sum(vram_values) / len(vram_values)) if vram_values else None,
            cpu_peak_percent=max(cpu_values) if cpu_values else None,
            ram_peak_used_gb=max(ram_values) if ram_values else None,
            **parsed,
        )
        append_training_metrics(metrics)
    except Exception:
        logger.exception("Failed to log training metrics for job %s (non-fatal)", job_id)


def compute_training(
    store: JobStore,
    job_id: str,
    *,
    variable_exposure: bool = False,
    max_gaussians: int | None = None,
    iterations: int | None = None,
    resize_factor: int | None = None,
    eval: bool = False,
    test_every: int | None = None,
    use_cloud: bool = False,
    cloud_provider: Literal["runpod", "vastai"] = "runpod",
    cloud_image: str | None = None,
    cloud_gpu_type: str | None = None,
) -> TrainingResult:
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
    if use_cloud and (not cloud_image or not cloud_gpu_type):
        message = "Cloud training requires both an image name and a GPU type"
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
    # An explicit slider value from the UI always wins; falling back to the
    # recommendation keeps the old behavior for any caller that doesn't pass one.
    if max_gaussians is None:
        max_gaussians = _max_gaussians_from_recommendation(job.recommendation_json)
    if resize_factor is None:
        resize_factor = training_resize_factor(job.frame_count)

    settings_json = json.dumps(
        {
            "variable_exposure": variable_exposure,
            "max_gaussians": max_gaussians,
            "iterations": iterations,
            "resize_factor": resize_factor,
            "eval": eval,
            "test_every": test_every,
            "use_cloud": use_cloud,
            "cloud_provider": cloud_provider if use_cloud else None,
            "cloud_image": cloud_image if use_cloud else None,
            "cloud_gpu_type": cloud_gpu_type if use_cloud else None,
        }
    )
    store.upsert_stage(job_id, "training", "running", started_at=_now(), settings_json=settings_json)

    usage_samples: list[UsageSnapshot] = []

    def report_progress(percent: float) -> None:
        store.upsert_stage(job_id, "training", "running", progress_percent=percent)
        try:
            usage_samples.append(sample_usage())
        except Exception:
            # Best-effort diagnostic sampling must never break a real training run.
            logger.exception("Failed to sample hardware usage during training (non-fatal)")

    def report_log(text: str) -> None:
        store.upsert_stage(job_id, "training", "running", log=text)

    try:
        if use_cloud:
            # No live progress/log streaming yet (cloud_job.run_cloud_job runs
            # the remote command as a single blocking SSH call, not a
            # Popen+reader-thread like the local path) - report_progress/
            # report_log are simply never called, same graceful-degradation
            # the frontend already handles when LichtFeld's own log format
            # doesn't match (indeterminate progress bar). Cancel from the UI
            # also can't interrupt an in-flight cloud run yet (see
            # cancel_or_force_fail in app/deps.py): it'll mark the stage
            # failed locally, but the remote pod keeps running/billing until
            # the command finishes or times out - terminate it manually from
            # the provider's dashboard if that happens. UNVERIFIED against a
            # real account (see docs/TODO.md).
            result = run_cloud_training(
                data_path,
                output_dir,
                provider=cloud_provider,
                image_name=cloud_image,
                gpu_type_id=cloud_gpu_type,
                max_gaussians=max_gaussians,
                iterations=iterations,
                bilateral_grid=variable_exposure,
                resize_factor=resize_factor,
                eval=eval,
                test_every=test_every,
            )
        else:
            result = run_training(
                data_path,
                output_dir,
                max_gaussians=max_gaussians,
                iterations=iterations,
                bilateral_grid=variable_exposure,
                resize_factor=resize_factor,
                eval=eval,
                test_every=test_every,
                on_progress=report_progress,
                on_log=report_log,
                on_process_start=lambda p: process_registry.register(job_id, "training", p),
            )
    except Exception as exc:
        message = (
            "Antrenare anulata de utilizator"
            if process_registry.was_cancelled(job_id, "training")
            else str(exc)
        )
        store.upsert_stage(job_id, "training", "failed", finished_at=_now(), error=message)
        _log_training_metrics(
            job_id,
            status="failed",
            error=message,
            # LichtFeldError's message already embeds the raw CLI log (see
            # app/lichtfeld.py's run_training) - parsing it even on failure
            # captures num_images/num_initial_points for exactly the OOM
            # crashes app/recommendation.py's constants need calibrating
            # against, instead of losing that data on every failed run.
            log=message,
            variable_exposure=variable_exposure,
            max_gaussians=max_gaussians,
            usage_samples=usage_samples,
        )
        raise TrainingError(message) from exc
    finally:
        process_registry.unregister(job_id, "training")

    store.update_job(job_id, splat_path=str(result.splat_path))
    eval_metrics_json = json.dumps(result.eval_metrics) if result.eval_metrics is not None else None
    store.upsert_stage(
        job_id, "training", "done", finished_at=_now(), log=result.log,
        eval_metrics_json=eval_metrics_json,
    )
    _log_training_metrics(
        job_id,
        status="done",
        error=None,
        log=result.log,
        variable_exposure=variable_exposure,
        max_gaussians=max_gaussians,
        usage_samples=usage_samples,
    )
    return result
