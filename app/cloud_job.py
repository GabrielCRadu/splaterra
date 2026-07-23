from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.remote_exec import RemoteExecError, download_directory, run_remote_command, upload_directory
from app.runpod_client import RunPodError, create_gpu_pod, terminate_pod, wait_for_pod_ready

logger = logging.getLogger(__name__)


class CloudJobError(Exception):
    pass


@dataclass
class CloudJobResult:
    pod_id: str
    remote_command_output: str
    output_dir: Path


def run_cloud_job(
    local_input_dir: Path,
    output_dir: Path,
    command: str,
    *,
    image_name: str,
    gpu_type_id: str,
    remote_workspace: str = "/workspace",
    ready_timeout_seconds: int = 300,
    command_timeout_seconds: int = 3600,
) -> CloudJobResult:
    job_name = f"splatprep-{uuid.uuid4().hex[:8]}"
    try:
        pod_id = create_gpu_pod(job_name, image_name=image_name, gpu_type_id=gpu_type_id)
    except RunPodError as exc:
        raise CloudJobError(str(exc)) from exc

    try:
        connection = wait_for_pod_ready(pod_id, timeout_seconds=ready_timeout_seconds)
        upload_directory(connection, local_input_dir, remote_workspace)
        output = run_remote_command(connection, command, timeout_seconds=command_timeout_seconds)
        download_directory(connection, f"{remote_workspace}/output", output_dir)
        return CloudJobResult(pod_id=pod_id, remote_command_output=output, output_dir=output_dir)
    except (RunPodError, RemoteExecError) as exc:
        raise CloudJobError(str(exc)) from exc
    finally:
        try:
            terminate_pod(pod_id)
        except Exception:
            logger.exception(
                "Failed to terminate RunPod pod %s - it may still be running and billing. "
                "Terminate it manually from the RunPod dashboard.",
                pod_id,
            )
