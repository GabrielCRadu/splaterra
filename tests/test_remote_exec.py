from unittest.mock import MagicMock

import pytest

from app import remote_exec
from app.remote_exec import PodConnection

CONNECTION = PodConnection(pod_id="pod-123", ssh_host="1.2.3.4", ssh_port=40001)


def test_run_remote_command_raises_when_ssh_missing(monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: None)
    with pytest.raises(remote_exec.RemoteExecError, match="ssh not found"):
        remote_exec.run_remote_command(CONNECTION, "echo hi")


def test_run_remote_command_returns_output_on_success(monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: "/usr/bin/ssh")

    def fake_run(command, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "hi\n"
        result.stderr = ""
        return result

    monkeypatch.setattr(remote_exec.subprocess, "run", fake_run)

    output = remote_exec.run_remote_command(CONNECTION, "echo hi")

    assert "hi" in output


def test_run_remote_command_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: "/usr/bin/ssh")

    def fake_run(command, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "command not found"
        return result

    monkeypatch.setattr(remote_exec.subprocess, "run", fake_run)

    with pytest.raises(remote_exec.RemoteExecError, match="command not found"):
        remote_exec.run_remote_command(CONNECTION, "bad-command")


def test_run_remote_command_raises_on_timeout(monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: "/usr/bin/ssh")

    def fake_run(command, **kwargs):
        raise remote_exec.subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

    monkeypatch.setattr(remote_exec.subprocess, "run", fake_run)

    with pytest.raises(remote_exec.RemoteExecError, match="did not finish within"):
        remote_exec.run_remote_command(CONNECTION, "sleep 999", timeout_seconds=5)


def test_upload_directory_raises_when_scp_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: None)
    with pytest.raises(remote_exec.RemoteExecError, match="scp not found"):
        remote_exec.upload_directory(CONNECTION, tmp_path, "/workspace")


def test_upload_directory_builds_correct_scp_command(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: "/usr/bin/scp")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(remote_exec.subprocess, "run", fake_run)

    remote_exec.upload_directory(CONNECTION, tmp_path, "/workspace/data")

    command = captured["command"]
    assert "-P" in command
    assert str(CONNECTION.ssh_port) in command
    assert f"root@{CONNECTION.ssh_host}:/workspace/data" in command
    assert str(tmp_path) in command


def test_download_directory_creates_local_dir_and_builds_command(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: "/usr/bin/scp")
    local_dir = tmp_path / "results"

    def fake_run(command, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(remote_exec.subprocess, "run", fake_run)

    remote_exec.download_directory(CONNECTION, "/workspace/output", local_dir)

    assert local_dir.is_dir()
