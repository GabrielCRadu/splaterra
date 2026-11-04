from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from app import runpod_client, vast_client
from app.remote_exec import RemoteExecError, download_directory, run_remote_command, upload_directory
from app.runpod_client import RunPodError
from app.vast_client import VastAIError

logger = logging.getLogger(__name__)


class CloudJobError(Exception):
    pass


class _Provider(NamedTuple):
    module: object
    error: type[Exception]
    label: str


# Both clients expose the same create_gpu_pod/wait_for_pod_ready/terminate_pod
# shape (see app/vast_client.py's create_gpu_pod docstring for how vast.ai's
# very different rental model - renting a specific already-listed offer,
# not a GPU type directly like RunPod - is squeezed into it), so dispatch is
# just picking which module's functions to call.
_PROVIDERS: dict[str, _Provider] = {
    "runpod": _Provider(runpod_client, RunPodError, "RunPod"),
    "vastai": _Provider(vast_client, VastAIError, "vast.ai"),
}


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
    provider: Literal["runpod", "vastai"] = "runpod",
    remote_workspace: str = "/workspace",
    ready_timeout_seconds: int = 300,
    command_timeout_seconds: int = 3600,
) -> CloudJobResult:
    if provider not in _PROVIDERS:
        raise CloudJobError(f"Unknown cloud provider: {provider!r}. Expected one of {sorted(_PROVIDERS)}")
    client, provider_error, label = _PROVIDERS[provider]

    job_name = f"splaterra-{uuid.uuid4().hex[:8]}"
    try:
        pod_id = client.create_gpu_pod(job_name, image_name=image_name, gpu_type_id=gpu_type_id)
    except provider_error as exc:
        raise CloudJobError(str(exc)) from exc

    try:
        connection = client.wait_for_pod_ready(pod_id, timeout_seconds=ready_timeout_seconds)
        upload_directory(connection, local_input_dir, remote_workspace)
        output = run_remote_command(connection, command, timeout_seconds=command_timeout_seconds)
        download_directory(connection, f"{remote_workspace}/output", output_dir)
        return CloudJobResult(pod_id=pod_id, remote_command_output=output, output_dir=output_dir)
    except (provider_error, RemoteExecError) as exc:
        raise CloudJobError(str(exc)) from exc
    finally:
        try:
            client.terminate_pod(pod_id)
        except Exception:
            logger.exception(
                "Failed to terminate %s pod %s - it may still be running and billing. "
                "Terminate it manually from the %s dashboard.",
                label, pod_id, label,
            )
