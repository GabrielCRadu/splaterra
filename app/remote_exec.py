from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from app.runpod_client import PodConnection


class RemoteExecError(Exception):
    pass


def _ssh_key_path() -> Path:
    return Path(os.environ.get("SPLATPREP_SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa")))


def _ssh_binary() -> str:
    binary = shutil.which("ssh")
    if binary is None:
        raise RemoteExecError("ssh not found on PATH. Install the OpenSSH client.")
    return binary


def _scp_binary() -> str:
    binary = shutil.which("scp")
    if binary is None:
        raise RemoteExecError("scp not found on PATH. Install the OpenSSH client.")
    return binary


def _ssh_common_args() -> list[str]:
    # Every RunPod pod is a fresh ephemeral instance with a new host key and a
    # new IP:port each time, so strict host-key checking would prompt
    # interactively and hang forever in this project's headless/remote-access
    # context. This is standard practice for automating ephemeral cloud
    # instances, not a carelessly-skipped security check.
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", f"UserKnownHostsFile={os.devnull}",
        "-o", "LogLevel=ERROR",
    ]


def upload_directory(
    connection: PodConnection, local_dir: Path, remote_dir: str, *, timeout_seconds: int = 1800
) -> None:
    command = [
        _scp_binary(),
        *_ssh_common_args(),
        "-P", str(connection.ssh_port),
        "-i", str(_ssh_key_path()),
        "-r", str(local_dir),
        f"root@{connection.ssh_host}:{remote_dir}",
    ]
    _run(command, timeout_seconds)


def download_directory(
    connection: PodConnection, remote_dir: str, local_dir: Path, *, timeout_seconds: int = 1800
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        _scp_binary(),
        *_ssh_common_args(),
        "-P", str(connection.ssh_port),
        "-i", str(_ssh_key_path()),
        "-r", f"root@{connection.ssh_host}:{remote_dir}",
        str(local_dir),
    ]
    _run(command, timeout_seconds)


def run_remote_command(
    connection: PodConnection, command: str, *, timeout_seconds: int = 3600
) -> str:
    ssh_command = [
        _ssh_binary(),
        *_ssh_common_args(),
        "-p", str(connection.ssh_port),
        "-i", str(_ssh_key_path()),
        f"root@{connection.ssh_host}",
        command,
    ]
    return _run(ssh_command, timeout_seconds)


def _run(command: list[str], timeout_seconds: int) -> str:
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        raise RemoteExecError(
            f"{command[0]} did not finish within {timeout_seconds}s and was terminated"
        ) from exc

    log = process.stdout + process.stderr
    if process.returncode != 0:
        raise RemoteExecError(f"{command[0]} exited with {process.returncode}:\n{log}")
    return log
