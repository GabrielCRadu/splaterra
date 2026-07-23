# RunPod Cloud GPU Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build generic, reusable infrastructure to run an arbitrary command against an arbitrary local input folder on a RunPod GPU pod and get the result back — with the pod **always** torn down afterward, success or failure. This plan is deliberately **not** wired to LichtFeld, any job/stage, any API endpoint, or the UI — it produces a standalone `run_cloud_job(...)` function that a future plan will call once LichtFeld's own CLI/dataset-format details are worked out.

**Architecture:** Three layers, each independently testable: `app/runpod_client.py` wraps the official `runpod` Python SDK for pod lifecycle (create, wait-until-SSH-ready, terminate). `app/remote_exec.py` wraps `ssh`/`scp` via subprocess (same pattern as `app/realityscan.py`) for file transfer and remote command execution once a pod is reachable. `app/cloud_job.py` composes both into one entry point, `run_cloud_job(...)`, with pod teardown in a `finally` block so no pod is ever left running (and billing) after a failure.

**Tech Stack:** Python 3.11+ (this project's floor; the `runpod` PyPI package itself requires 3.10+, no conflict), the `runpod` package (official SDK, added this plan), the system's `ssh`/`scp` binaries (OpenSSH client, subprocess only — never a Python SSH library, to stay consistent with this project's "wrap real CLI tools via subprocess" convention).

## Global Constraints

- **Real pod creation costs real money.** Only Task 4's test creates a real pod, and it is **double-gated**: skipped unless both `RUNPOD_API_KEY` is set AND `SPLATPREP_RUN_REAL_RUNPOD_TEST=1` is explicitly set. This is deliberate: the project's `.env` pattern means a user's real API key will often be present in the environment for normal app use, and a test suite that spins up billable GPU pods just because a key happens to be set would be a serious, easy-to-trigger-by-accident cost bug. Tasks 1-3's tests are 100% mocked, no exceptions.
- **A pod, once created, is always torn down.** `run_cloud_job` calls `terminate_pod` in a `finally` block wrapping every step after pod creation (wait-ready, upload, remote command, download). If termination itself fails, that failure is logged loudly (`logger.exception`, mentioning the pod may still be billing) rather than silently swallowed or allowed to mask the original error — but it never raises over an already-in-flight exception from the `try` block.
- **`RUNPOD_API_KEY` is read from the environment only** (`os.environ.get("RUNPOD_API_KEY")`) — never hardcoded, never logged, never included in an error message verbatim (error messages reference the env var *name*, not its value).
- **SSH access to a RunPod pod requires the user's SSH public key to already be registered in their RunPod account settings**, and a local private key available at `SPLATPREP_SSH_KEY_PATH` (env var, default `~/.ssh/id_rsa`) — this is a one-time, account-level manual step this codebase cannot automate. Document it in Task 4.
- **`ssh`/`scp` subprocess calls are time-bounded**, same reasoning and same pattern as `app/realityscan.py`'s `run_alignment`: the project owner works over remote access and cannot manually intervene if a remote command hangs. `subprocess.TimeoutExpired` is always caught and converted to `RemoteExecError`.
- **`StrictHostKeyChecking=no` and a null `UserKnownHostsFile` are deliberate, not a security shortcut taken carelessly.** Every RunPod pod is a fresh ephemeral instance with a new host key and a new IP:port each time, so strict host-key checking would either prompt interactively (hanging forever in a headless/remote-access context — the exact failure mode this project avoids everywhere else) or require pre-seeding known_hosts with a key that doesn't exist yet. This is standard practice for automating ephemeral cloud instances (the same tradeoff CI systems and infra-as-code tools make).
- This plan adds **no new API endpoints, no UI, no job/stage integration**. `run_cloud_job` is called directly in its own tests only.

---

## Task 1: RunPod client — pod lifecycle (create, wait-ready, terminate)

**Files:**
- Modify: `requirements.txt`
- Create: `app/runpod_client.py`
- Test: `tests/test_runpod_client.py`

**Interfaces:**
- Consumes: nothing (wraps the third-party `runpod` package directly).
- Produces: `PodConnection` (dataclass: `pod_id: str, ssh_host: str, ssh_port: int`); `create_gpu_pod(name: str, *, image_name: str, gpu_type_id: str, container_disk_in_gb: int = 20, volume_in_gb: int = 0) -> str` (returns pod id); `wait_for_pod_ready(pod_id: str, *, timeout_seconds: int = 300, poll_interval_seconds: int = 5) -> PodConnection`; `terminate_pod(pod_id: str) -> None`; `RunPodError`. Task 3 consumes all of these directly by name (imported into its module namespace, not called as `runpod_client.xxx`, so its tests can monkeypatch them).

- [ ] **Step 1: Add `runpod` to `requirements.txt`**

Append to `requirements.txt`:
```
runpod>=1.7
```

- [ ] **Step 2: Install the new dependency**

Run: `python -m pip install -r requirements.txt`
Expected: `runpod` installs without errors.

- [ ] **Step 3: Write the failing tests**

`tests/test_runpod_client.py`:
```python
from unittest.mock import MagicMock

import pytest

from app import runpod_client
from app.runpod_client import PodConnection, RunPodError


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


def test_terminate_pod_calls_sdk(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    mock_terminate = MagicMock()
    monkeypatch.setattr(runpod_client.runpod, "terminate_pod", mock_terminate)

    runpod_client.terminate_pod("pod-123")

    mock_terminate.assert_called_once_with("pod-123")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_runpod_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.runpod_client'`

- [ ] **Step 5: Write `app/runpod_client.py`**

```python
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import runpod


class RunPodError(Exception):
    pass


@dataclass
class PodConnection:
    pod_id: str
    ssh_host: str
    ssh_port: int


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
        pod = runpod.get_pod(pod_id)
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
    runpod.terminate_pod(pod_id)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_runpod_client.py -v`
Expected: PASS (7 passed)

- [ ] **Step 7: Commit**

```bash
git add requirements.txt app/runpod_client.py tests/test_runpod_client.py
git commit -m "feat: add RunPod client for GPU pod lifecycle (create, wait-ready, terminate)"
```

---

## Task 2: SSH-based remote execution and file transfer

**Files:**
- Create: `app/remote_exec.py`
- Test: `tests/test_remote_exec.py`

**Interfaces:**
- Consumes: `PodConnection` (Task 1).
- Produces: `RemoteExecError`; `upload_directory(connection: PodConnection, local_dir: Path, remote_dir: str, *, timeout_seconds: int = 1800) -> None`; `download_directory(connection: PodConnection, remote_dir: str, local_dir: Path, *, timeout_seconds: int = 1800) -> None`; `run_remote_command(connection: PodConnection, command: str, *, timeout_seconds: int = 3600) -> str` (returns combined stdout+stderr). Task 3 consumes all three functions and `RemoteExecError` directly by name.

- [ ] **Step 1: Write the failing tests**

`tests/test_remote_exec.py`:
```python
from unittest.mock import MagicMock

import pytest

from app import remote_exec
from app.runpod_client import PodConnection

CONNECTION = PodConnection(pod_id="pod-123", ssh_host="1.2.3.4", ssh_port=40001)


def test_run_remote_command_raises_when_ssh_missing(monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: None)
    with pytest.raises(remote_exec.RemoteExecError, match="ssh not found"):
        remote_exec.run_remote_command(CONNECTION, "echo hi")


def test_run_remote_command_returns_output_on_success(monkeypatch):
    monkeypatch.setattr(remote_exec.shutil, "which", lambda name: "/usr/bin/ssh")

    def fake_run(command, capture_output, text, check, timeout):
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

    def fake_run(command, capture_output, text, check, timeout):
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

    def fake_run(command, capture_output, text, check, timeout):
        raise remote_exec.subprocess.TimeoutExpired(cmd=command, timeout=timeout)

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

    def fake_run(command, capture_output, text, check, timeout):
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

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(remote_exec.subprocess, "run", fake_run)

    remote_exec.download_directory(CONNECTION, "/workspace/output", local_dir)

    assert local_dir.is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_remote_exec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.remote_exec'`

- [ ] **Step 3: Write `app/remote_exec.py`**

```python
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
```

Note `scp` uses `-P` (capital) for the port flag while `ssh` uses `-p` (lowercase) — this is standard OpenSSH tooling, not a typo; both are used correctly above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_remote_exec.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add app/remote_exec.py tests/test_remote_exec.py
git commit -m "feat: add SSH-based remote command execution and file transfer"
```

---

## Task 3: Cloud job orchestration with guaranteed teardown

**Files:**
- Create: `app/cloud_job.py`
- Test: `tests/test_cloud_job.py`

**Interfaces:**
- Consumes: `create_gpu_pod`, `wait_for_pod_ready`, `terminate_pod`, `RunPodError` (Task 1); `upload_directory`, `run_remote_command`, `download_directory`, `RemoteExecError` (Task 2).
- Produces: `CloudJobResult` (dataclass: `pod_id: str, remote_command_output: str, output_dir: Path`); `run_cloud_job(local_input_dir: Path, output_dir: Path, command: str, *, image_name: str, gpu_type_id: str, remote_workspace: str = "/workspace", ready_timeout_seconds: int = 300, command_timeout_seconds: int = 3600) -> CloudJobResult`; `CloudJobError`. This is the single entry point a future plan will call to wire in LichtFeld.

- [ ] **Step 1: Write the failing tests**

`tests/test_cloud_job.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cloud_job.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.cloud_job'`

- [ ] **Step 3: Write `app/cloud_job.py`**

```python
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
        except RunPodError:
            logger.exception(
                "Failed to terminate RunPod pod %s - it may still be running and billing. "
                "Terminate it manually from the RunPod dashboard.",
                pod_id,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cloud_job.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all pass, 139 passed (120 pre-existing + 7 from Task 1 + 7 from Task 2 + 5 from Task 3 — exact count isn't load-bearing, zero failures is)

- [ ] **Step 6: Commit**

```bash
git add app/cloud_job.py tests/test_cloud_job.py
git commit -m "feat: add cloud job orchestration with guaranteed pod teardown"
```

---

## Task 4: Real RunPod smoke test (opt-in only) + setup documentation

**Files:**
- Test: `tests/test_cloud_job_real_runpod.py` (new)
- Modify: `README.md`

**Interfaces:**
- Consumes: `run_cloud_job` (Task 3). No new production interfaces — this task only proves the real, unmocked plumbing works when a user deliberately opts in, and documents the prerequisites.

- [ ] **Step 1: Write the opt-in-gated real test**

`tests/test_cloud_job_real_runpod.py`:
```python
import os
import shutil

import pytest

from app.cloud_job import run_cloud_job

_RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
_OPTED_IN = os.environ.get("SPLATPREP_RUN_REAL_RUNPOD_TEST") == "1"
_SSH = shutil.which("ssh")
_SCP = shutil.which("scp")


@pytest.mark.skipif(
    not (_RUNPOD_API_KEY and _OPTED_IN and _SSH and _SCP),
    reason=(
        "Real RunPod smoke test skipped by default - it creates and briefly bills a real "
        "GPU pod. To run it deliberately: set RUNPOD_API_KEY, add your SSH public key to "
        "your RunPod account (Settings -> SSH Public Keys), and set "
        "SPLATPREP_RUN_REAL_RUNPOD_TEST=1. Verify the gpu_type_id below is still a valid, "
        "orderable GPU type on your RunPod account before running - GPU catalogs change."
    ),
)
def test_real_cloud_job_round_trip(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "hello.txt").write_text("hello from splat prep")

    result = run_cloud_job(
        input_dir,
        tmp_path / "output",
        "mkdir -p /workspace/output && cp /workspace/input/hello.txt /workspace/output/",
        image_name="runpod/base:0.6.2-cuda12.4.1",
        gpu_type_id="NVIDIA GeForce RTX 3070",
    )

    assert result.pod_id
    assert (tmp_path / "output" / "hello.txt").read_text() == "hello from splat prep"
```

- [ ] **Step 2: Run it (it will skip by default)**

Run: `python -m pytest tests/test_cloud_job_real_runpod.py -v`
Expected: SKIPPED (1 skipped), with the reason message printed — this confirms the gate itself works correctly without spending any money.

- [ ] **Step 3: Add a RunPod setup section to `README.md`**

Add this section to `README.md` (after the existing "Acces remote (Tailscale)" section if present, otherwise at the end):

```markdown
## Cloud GPU (RunPod)

Training pe LichtFeld cere un GPU NVIDIA — daca PC-ul tau nu are unul (sau nu e suficient de
puternic), aplicatia poate rula training-ul pe un pod GPU temporar in cloud, prin RunPod.

Pasi de configurare (o singura data):

1. Creeaza cont pe [runpod.io](https://www.runpod.io) si genereaza un API key din
   Settings -> API Keys.
2. Seteaza-l ca variabila de mediu: `RUNPOD_API_KEY=<cheia-ta>` (nu il pune niciodata in cod
   sau in git).
3. Adauga cheia ta publica SSH (`~/.ssh/id_rsa.pub` sau echivalent) in contul RunPod, la
   Settings -> SSH Public Keys — fara asta, aplicatia nu se poate conecta la pod-urile create.
4. Daca folosesti alta cale pentru cheia privata SSH decat cea default (`~/.ssh/id_rsa`),
   seteaza `SPLATPREP_SSH_KEY_PATH` catre calea corecta.

Costul e per-ora, doar cat timp ruleaza pod-ul — aplicatia opreste automat pod-ul la finalul
fiecarui job (sau la orice eroare), ca sa nu platesti timp mort.
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_cloud_job_real_runpod.py README.md
git commit -m "test: add opt-in real RunPod smoke test and setup docs"
```
