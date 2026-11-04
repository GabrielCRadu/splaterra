from __future__ import annotations

import os
import time

import requests

from app.remote_exec import PodConnection

# Verified against docs.vast.ai's REST API reference (api-reference/search/
# search-offers, api-reference/instances/{create,show,destroy}-instance) -
# never exercised against a real vast.ai account (no API key available in
# this session, see docs/TODO.md). Treat as unverified until a real rent/
# destroy round-trip confirms it.
_BASE_URL = "https://console.vast.ai"


class VastAIError(Exception):
    pass


def _api_key() -> str:
    key = os.environ.get("VAST_API_KEY")
    if not key:
        raise VastAIError(
            "VAST_API_KEY not set. Get an API key from your vast.ai account settings "
            "and set it as an environment variable. Also register an SSH public key "
            "with your vast.ai account before creating an instance - unlike RunPod, "
            "vast.ai does not accept one at instance-creation time."
        )
    return key


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}


def search_offers(gpu_name: str, *, limit: int = 20) -> list[dict]:
    try:
        response = requests.post(
            f"{_BASE_URL}/api/v0/bundles",
            headers=_headers(),
            json={
                "limit": limit,
                "type": "ondemand",
                "gpu_name": {"eq": gpu_name},
                "order": [["dph_total", "asc"]],
                "rentable": {"eq": True},
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise VastAIError(f"Failed to search vast.ai offers: {exc}") from exc
    return response.json().get("offers", [])


def create_gpu_pod(
    name: str,
    *,
    image_name: str,
    gpu_type_id: str,
    disk_gb: int = 20,
) -> str:
    # gpu_type_id doubles as either a GPU name filter (e.g. "RTX_4090" - the
    # cheapest rentable matching offer is auto-selected via search_offers) or
    # a specific numeric offer id (skips the search, rents that exact
    # machine) - matches app/runpod_client.py's create_gpu_pod(gpu_type_id=...)
    # shape so app/cloud_job.py can dispatch to either provider identically,
    # even though the two platforms' underlying rental models differ (RunPod
    # rents a GPU type directly; vast.ai rents a specific already-listed
    # machine/offer).
    if gpu_type_id.isdigit():
        offer_id = gpu_type_id
    else:
        offers = search_offers(gpu_type_id, limit=1)
        if not offers:
            raise VastAIError(f"No rentable vast.ai offers found for GPU {gpu_type_id!r}")
        offer_id = str(offers[0]["id"])

    try:
        response = requests.put(
            f"{_BASE_URL}/api/v0/asks/{offer_id}",
            headers=_headers(),
            json={"image": image_name, "disk": disk_gb, "runtype": "ssh"},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        raise VastAIError(f"Failed to create vast.ai instance from offer {offer_id}: {exc}") from exc
    if not body.get("success"):
        raise VastAIError(f"vast.ai rejected instance creation: {body}")
    instance_id = body.get("new_contract")
    if not instance_id:
        raise VastAIError(f"vast.ai did not return an instance id: {body}")
    # name (this app's own job-scoped label, e.g. "splaterra-abc123") isn't
    # sent anywhere - the create-instance endpoint has no free-text label
    # field documented beyond "label", left unset since nothing in this app
    # reads it back.
    return str(instance_id)


def wait_for_pod_ready(
    pod_id: str, *, timeout_seconds: int = 300, poll_interval_seconds: int = 5
) -> PodConnection:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"{_BASE_URL}/api/v0/instances/{pod_id}", headers=_headers(), timeout=30
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise VastAIError(f"Failed to query vast.ai instance {pod_id}: {exc}") from exc
        instance = response.json().get("instances") or {}
        status = instance.get("actual_status")
        # These states never transition to "running" on their own (confirmed
        # in docs.vast.ai's show-instance reference) - waiting out the full
        # timeout on one would just waste it.
        if status in ("exited", "unknown", "offline"):
            raise VastAIError(f"vast.ai instance {pod_id} entered status {status!r} and will not start")
        if status == "running" and instance.get("ssh_host") and instance.get("ssh_port"):
            return PodConnection(
                pod_id=pod_id,
                ssh_host=instance["ssh_host"],
                ssh_port=int(instance["ssh_port"]),
            )
        time.sleep(poll_interval_seconds)
    raise VastAIError(f"Instance {pod_id} did not become SSH-ready within {timeout_seconds}s")


def terminate_pod(pod_id: str) -> None:
    try:
        response = requests.delete(
            f"{_BASE_URL}/api/v0/instances/{pod_id}", headers=_headers(), timeout=30
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise VastAIError(f"Failed to terminate vast.ai instance {pod_id}: {exc}") from exc
