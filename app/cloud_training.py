from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from app.cloud_job import CloudJobError, run_cloud_job
from app.lichtfeld import LichtFeldError, TrainingResult, build_training_command, find_result_splat, parse_eval_metrics

# app/paths.py's alignment_dir(job_id)/training_dir(job_id) are only ever
# passed as local_input_dir/output_dir - the remote paths just need to be
# fixed strings inside whatever container the pod's image provides, matching
# app/cloud_job.py's own remote_workspace default.
_REMOTE_DATA_DIR = "/workspace/input"
_REMOTE_OUTPUT_DIR = "/workspace/output"


def run_cloud_training(
    data_path: Path,
    output_dir: Path,
    *,
    provider: Literal["runpod", "vastai"],
    image_name: str,
    gpu_type_id: str,
    max_gaussians: int | None = None,
    iterations: int | None = None,
    bilateral_grid: bool = False,
    resize_factor: int | None = None,
    eval: bool = False,
    test_every: int | None = None,
    ready_timeout_seconds: int = 300,
    command_timeout_seconds: int = 7200,
) -> TrainingResult:
    # UNVERIFIED against a real RunPod/vast.ai account (see docs/TODO.md) -
    # built from each provider's public docs/SDK, exercised only against
    # mocks in tests/test_cloud_training.py. image_name must point at a
    # Docker image with LichtFeld-Studio already on PATH (this app doesn't
    # build or publish one) - there's no local-install-dir style discovery
    # possible on a remote container the way _lichtfeld_binary() does locally.
    command_parts = build_training_command(
        "LichtFeld-Studio", _REMOTE_DATA_DIR, _REMOTE_OUTPUT_DIR,
        max_gaussians=max_gaussians, iterations=iterations, bilateral_grid=bilateral_grid,
        resize_factor=resize_factor, eval=eval, test_every=test_every,
    )
    command = " ".join(command_parts)

    # Same staleness guard as run_training()'s local path: output_dir is
    # app/paths.py's training_dir(job_id), which could already hold a stale
    # .ply from an earlier LOCAL run of the same job if the owner switches a
    # job from local to cloud training - download_directory() doesn't clear
    # the destination first, it just scp's alongside whatever's already there.
    run_started_at = time.time()
    try:
        result = run_cloud_job(
            data_path,
            output_dir,
            command,
            image_name=image_name,
            gpu_type_id=gpu_type_id,
            provider=provider,
            remote_workspace="/workspace",
            ready_timeout_seconds=ready_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
        )
    except CloudJobError as exc:
        # Reusing LichtFeldError (rather than a parallel cloud-specific error
        # type) keeps app/training_service.py's exception handling identical
        # regardless of whether a run happened locally or in the cloud.
        raise LichtFeldError(str(exc)) from exc

    log = result.remote_command_output
    splat_path = find_result_splat(output_dir, run_started_at, log)
    return TrainingResult(splat_path=splat_path, log=log, eval_metrics=parse_eval_metrics(output_dir))
