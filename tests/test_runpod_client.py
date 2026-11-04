from unittest.mock import MagicMock

import pytest

from app import runpod_client
from app.remote_exec import PodConnection
from app.runpod_client import RunPodError


def test_create_gpu_pod_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RunPodError, match="RUNPOD_API_KEY"):
        runpod_client.create_gpu_pod("test", image_name="img", gpu_type_id="gpu")


def test_create_gpu_pod_returns_pod_id(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(runpod_client.runpod, "create_pod", MagicMock(return_value={"id": "pod-123"}))

    pod_id = runpod_client.create_gpu_pod("test", image_name="img", gpu_type_id="gpu")

    assert pod_id == "pod-123"


def test_create_gpu_pod_raises_when_no_id_returned(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(runpod_client.runpod, "create_pod", MagicMock(return_value={}))

    with pytest.raises(RunPodError, match="did not return a pod id"):
        runpod_client.create_gpu_pod("test", image_name="img", gpu_type_id="gpu")


def test_create_gpu_pod_wraps_sdk_errors_as_runpod_error(monkeypatch):
    # Real-world case observed live: RunPod's SDK raises its own
    # runpod.error.QueryError (e.g. "account balance is too low"), which is
    # not a RunPodError - callers up the stack (cloud_job.run_cloud_job) only
    # catch RunPodError, so an unwrapped SDK exception would escape uncaught.
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(
        runpod_client.runpod,
        "create_pod",
        MagicMock(side_effect=RuntimeError("account balance is too low")),
    )

    with pytest.raises(RunPodError, match="account balance is too low"):
        runpod_client.create_gpu_pod("test", image_name="img", gpu_type_id="gpu")


def test_wait_for_pod_ready_returns_connection_once_ssh_port_mapped(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    ready_pod = {
        "runtime": {
            "ports": [
                {"privatePort": 22, "publicPort": 40001, "ip": "1.2.3.4", "isIpPublic": True, "type": "tcp"}
            ]
        }
    }
    monkeypatch.setattr(runpod_client.runpod, "get_pod", MagicMock(return_value=ready_pod))
    monkeypatch.setattr(runpod_client.time, "sleep", lambda _: None)

    connection = runpod_client.wait_for_pod_ready("pod-123", timeout_seconds=30)

    assert connection == PodConnection(pod_id="pod-123", ssh_host="1.2.3.4", ssh_port=40001)


def test_wait_for_pod_ready_polls_until_ssh_appears(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    not_ready = {"runtime": None}
    ready = {
        "runtime": {
            "ports": [
                {"privatePort": 22, "publicPort": 40001, "ip": "1.2.3.4", "isIpPublic": True, "type": "tcp"}
            ]
        }
    }
    monkeypatch.setattr(
        runpod_client.runpod, "get_pod", MagicMock(side_effect=[not_ready, not_ready, ready])
    )
    monkeypatch.setattr(runpod_client.time, "sleep", lambda _: None)

    connection = runpod_client.wait_for_pod_ready("pod-123", timeout_seconds=30)

    assert connection.ssh_port == 40001


def test_wait_for_pod_ready_raises_on_timeout(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(runpod_client.runpod, "get_pod", MagicMock(return_value={"runtime": None}))
    monkeypatch.setattr(runpod_client.time, "sleep", lambda _: None)

    fake_clock = [0.0]

    def fake_monotonic():
        fake_clock[0] += 100.0
        return fake_clock[0]

    monkeypatch.setattr(runpod_client.time, "monotonic", fake_monotonic)

    with pytest.raises(RunPodError, match="did not become SSH-ready"):
        runpod_client.wait_for_pod_ready("pod-123", timeout_seconds=30, poll_interval_seconds=0)


def test_wait_for_pod_ready_wraps_sdk_errors_as_runpod_error(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(
        runpod_client.runpod, "get_pod", MagicMock(side_effect=RuntimeError("network unreachable"))
    )

    with pytest.raises(RunPodError, match="network unreachable"):
        runpod_client.wait_for_pod_ready("pod-123", timeout_seconds=30)


def test_terminate_pod_calls_sdk(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    mock_terminate = MagicMock()
    monkeypatch.setattr(runpod_client.runpod, "terminate_pod", mock_terminate)

    runpod_client.terminate_pod("pod-123")

    mock_terminate.assert_called_once_with("pod-123")


def test_terminate_pod_wraps_sdk_errors_as_runpod_error(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(
        runpod_client.runpod, "terminate_pod", MagicMock(side_effect=RuntimeError("gateway timeout"))
    )

    with pytest.raises(RunPodError, match="gateway timeout"):
        runpod_client.terminate_pod("pod-123")
