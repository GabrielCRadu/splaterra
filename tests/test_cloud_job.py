from unittest.mock import MagicMock

import pytest

from app import cloud_job
from app.remote_exec import RemoteExecError
from app.runpod_client import PodConnection, RunPodError

CONNECTION = PodConnection(pod_id="pod-123", ssh_host="1.2.3.4", ssh_port=40001)


def _patch_happy_path(monkeypatch):
    monkeypatch.setattr(cloud_job, "create_gpu_pod", MagicMock(return_value="pod-123"))
    monkeypatch.setattr(cloud_job, "wait_for_pod_ready", MagicMock(return_value=CONNECTION))
    monkeypatch.setattr(cloud_job, "upload_directory", MagicMock())
    monkeypatch.setattr(cloud_job, "run_remote_command", MagicMock(return_value="training done"))
    monkeypatch.setattr(cloud_job, "download_directory", MagicMock())


def test_run_cloud_job_terminates_pod_on_success(tmp_path, monkeypatch):
    _patch_happy_path(monkeypatch)
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job, "terminate_pod", terminate_mock)

    result = cloud_job.run_cloud_job(
        tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
    )

    assert result.pod_id == "pod-123"
    assert result.remote_command_output == "training done"
    terminate_mock.assert_called_once_with("pod-123")


def test_run_cloud_job_terminates_pod_even_when_command_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_job, "create_gpu_pod", MagicMock(return_value="pod-123"))
    monkeypatch.setattr(cloud_job, "wait_for_pod_ready", MagicMock(return_value=CONNECTION))
    monkeypatch.setattr(cloud_job, "upload_directory", MagicMock())
    monkeypatch.setattr(
        cloud_job, "run_remote_command", MagicMock(side_effect=RemoteExecError("training crashed"))
    )
    monkeypatch.setattr(cloud_job, "download_directory", MagicMock())
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job, "terminate_pod", terminate_mock)

    with pytest.raises(cloud_job.CloudJobError, match="training crashed"):
        cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    terminate_mock.assert_called_once_with("pod-123")


def test_run_cloud_job_terminates_pod_even_when_upload_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_job, "create_gpu_pod", MagicMock(return_value="pod-123"))
    monkeypatch.setattr(cloud_job, "wait_for_pod_ready", MagicMock(return_value=CONNECTION))
    monkeypatch.setattr(
        cloud_job, "upload_directory", MagicMock(side_effect=RemoteExecError("upload failed"))
    )
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job, "terminate_pod", terminate_mock)

    with pytest.raises(cloud_job.CloudJobError, match="upload failed"):
        cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    terminate_mock.assert_called_once_with("pod-123")


def test_run_cloud_job_does_not_terminate_when_pod_creation_itself_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_job, "create_gpu_pod", MagicMock(side_effect=RunPodError("no capacity")))
    terminate_mock = MagicMock()
    monkeypatch.setattr(cloud_job, "terminate_pod", terminate_mock)

    with pytest.raises(cloud_job.CloudJobError, match="no capacity"):
        cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    terminate_mock.assert_not_called()


def test_run_cloud_job_logs_but_does_not_raise_when_termination_itself_fails(tmp_path, monkeypatch, caplog):
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        cloud_job, "terminate_pod", MagicMock(side_effect=RunPodError("dashboard unreachable"))
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
        cloud_job, "terminate_pod", MagicMock(side_effect=ConnectionError("network unreachable"))
    )

    with caplog.at_level("ERROR"):
        result = cloud_job.run_cloud_job(
            tmp_path / "input", tmp_path / "output", "train.sh", image_name="img", gpu_type_id="gpu"
        )

    assert result.pod_id == "pod-123"
    assert "may still be running and billing" in caplog.text
