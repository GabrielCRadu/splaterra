from unittest.mock import MagicMock

import pytest

from app import cloud_job
from app.remote_exec import PodConnection, RemoteExecError
from app.runpod_client import RunPodError
from app.vast_client import VastAIError

CONNECTION = PodConnection(pod_id="pod-123", ssh_host="1.2.3.4", ssh_port=40001)


def _patch_happy_path(monkeypatch):
    monkeypatch.setattr(cloud_job.runpod_client, "create_gpu_pod", MagicMock(return_value="pod-123"))
    monkeypatch.setattr(cloud_job.runpod_client, "wait_for_pod_ready", MagicMock(return_value=CONNECTION))
    monkeypatch.setattr(cloud_job, "upload_directory", MagicMock())
    monkeypatch.setattr(cloud_job, "run_remote_command", MagicMock(return_value="training done"))
    monkeypatch.setattr(cloud_job, "download_directory", MagicMock())


def test_run_cloud_job_terminates_pod_on_success(tmp_path, monkeypatch):
    _patch_happy_path(monkeypatch)
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job.runpod_client, "terminate_pod", terminate_mock)

    result = cloud_job.run_cloud_job(
        tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
    )

    assert result.pod_id == "pod-123"
    assert result.remote_command_output == "training done"
    terminate_mock.assert_called_once_with("pod-123")


def test_run_cloud_job_terminates_pod_even_when_command_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_job.runpod_client, "create_gpu_pod", MagicMock(return_value="pod-123"))
    monkeypatch.setattr(cloud_job.runpod_client, "wait_for_pod_ready", MagicMock(return_value=CONNECTION))
    monkeypatch.setattr(cloud_job, "upload_directory", MagicMock())
    monkeypatch.setattr(
        cloud_job, "run_remote_command", MagicMock(side_effect=RemoteExecError("training crashed"))
    )
    monkeypatch.setattr(cloud_job, "download_directory", MagicMock())
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job.runpod_client, "terminate_pod", terminate_mock)

    with pytest.raises(cloud_job.CloudJobError, match="training crashed"):
        cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    terminate_mock.assert_called_once_with("pod-123")


def test_run_cloud_job_terminates_pod_even_when_upload_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_job.runpod_client, "create_gpu_pod", MagicMock(return_value="pod-123"))
    monkeypatch.setattr(cloud_job.runpod_client, "wait_for_pod_ready", MagicMock(return_value=CONNECTION))
    monkeypatch.setattr(
        cloud_job, "upload_directory", MagicMock(side_effect=RemoteExecError("upload failed"))
    )
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job.runpod_client, "terminate_pod", terminate_mock)

    with pytest.raises(cloud_job.CloudJobError, match="upload failed"):
        cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    terminate_mock.assert_called_once_with("pod-123")


def test_run_cloud_job_does_not_terminate_when_pod_creation_itself_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_job.runpod_client, "create_gpu_pod", MagicMock(side_effect=RunPodError("no capacity")))
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job.runpod_client, "terminate_pod", terminate_mock)

    with pytest.raises(cloud_job.CloudJobError, match="no capacity"):
        cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    terminate_mock.assert_not_called()


def test_run_cloud_job_logs_but_does_not_raise_when_termination_itself_fails(tmp_path, monkeypatch, caplog):
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        cloud_job.runpod_client, "terminate_pod", MagicMock(side_effect=RunPodError("dashboard unreachable"))
    )

    with caplog.at_level("ERROR"):
        result = cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    assert result.pod_id == "pod-123"
    assert "may still be running and billing" in caplog.text


def test_run_cloud_job_logs_but_does_not_raise_when_termination_raises_unexpected_error(tmp_path, monkeypatch, caplog):
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        cloud_job.runpod_client, "terminate_pod", MagicMock(side_effect=ConnectionError("network unreachable"))
    )

    with caplog.at_level("ERROR"):
        result = cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    assert result.pod_id == "pod-123"
    assert "may still be running and billing" in caplog.text


def test_run_cloud_job_dispatches_to_vastai_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_job.vast_client, "create_gpu_pod", MagicMock(return_value="vast-456"))
    monkeypatch.setattr(cloud_job.vast_client, "wait_for_pod_ready", MagicMock(return_value=CONNECTION))
    monkeypatch.setattr(cloud_job, "upload_directory", MagicMock())
    monkeypatch.setattr(cloud_job, "run_remote_command", MagicMock(return_value="training done"))
    monkeypatch.setattr(cloud_job, "download_directory", MagicMock())
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job.vast_client, "terminate_pod", terminate_mock)
    # The RunPod client must never be touched when provider="vastai" is requested.
    runpod_create_mock = MagicMock()
    monkeypatch.setattr(cloud_job.runpod_client, "create_gpu_pod", runpod_create_mock)

    result = cloud_job.run_cloud_job(
        tmp_path / "input", tmp_path / "output", "train.sh",
        image_name="img", gpu_type_id="RTX_4090", provider="vastai",
    )

    assert result.pod_id == "vast-456"
    terminate_mock.assert_called_once_with("vast-456")
    runpod_create_mock.assert_not_called()


def test_run_cloud_job_wraps_vastai_errors_as_cloud_job_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cloud_job.vast_client, "create_gpu_pod", MagicMock(side_effect=VastAIError("no offers"))
    )
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job.vast_client, "terminate_pod", terminate_mock)

    with pytest.raises(cloud_job.CloudJobError, match="no offers"):
        cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh",
            image_name="img", gpu_type_id="RTX_4090", provider="vastai",
        )

    terminate_mock.assert_not_called()


def test_run_cloud_job_rejects_unknown_provider(tmp_path):
    with pytest.raises(cloud_job.CloudJobError, match="Unknown cloud provider"):
        cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh",
            image_name="img", gpu_type_id="gpu", provider="some-other-cloud",
        )
