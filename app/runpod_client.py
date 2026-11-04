from __future__ import annotations

import os
import time

import runpod

from app.remote_exec import PodConnection


class RunPodError(Exception):
    pass


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunPodError(
            "RUNPOD_API_KEY not set. Get an API key from your RunPod account settings "
            "and set it as an environment variable."
        )
    return key


def create_gpu_pod(
    name: str,
    *,
    image_name: str,
    gpu_type_id: str,
    container_disk_in_gb: int = 20,
    volume_in_gb: int = 0,
) -> str:
    runpod.api_key = _api_key()
    try:
        pod = runpod.create_pod(
            name,
            image_name,
            gpu_type_id,
            cloud_type="ALL",
            support_public_ip=True,
            start_ssh=True,
            container_disk_in_gb=container_disk_in_gb,
            volume_in_gb=volume_in_gb,
            ports="22/tcp",
        )
    except Exception as exc:
        raise RunPodError(f"Failed to create RunPod pod: {exc}") from exc
    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError(f"RunPod did not return a pod id: {pod}")
    return pod_id


def wait_for_pod_ready(
    pod_id: str, *, timeout_seconds: int = 300, poll_interval_seconds: int = 5
) -> PodConnection:
    runpod.api_key = _api_key()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            pod = runpod.get_pod(pod_id)
        except Exception as exc:
            raise RunPodError(f"Failed to query RunPod pod {pod_id}: {exc}") from exc
        connection = _extract_ssh_connection(pod_id, pod)
        if connection is not None:
            return connection
        time.sleep(poll_interval_seconds)
    raise RunPodError(f"Pod {pod_id} did not become SSH-ready within {timeout_seconds}s")


def _extract_ssh_connection(pod_id: str, pod: dict) -> PodConnection | None:
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return PodConnection(pod_id=pod_id, ssh_host=port["ip"], ssh_port=int(port["publicPort"]))
    return None


def terminate_pod(pod_id: str) -> None:
    runpod.api_key = _api_key()
    try:
        runpod.terminate_pod(pod_id)
    except Exception as exc:
        raise RunPodError(f"Failed to terminate RunPod pod {pod_id}: {exc}") from exc
