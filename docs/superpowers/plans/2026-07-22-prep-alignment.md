# Prep Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Orchestrate RealityScan's headless CLI to turn a job's extracted frames into aligned cameras + a sparse point cloud, wired into the same job/stage/background-task/polling infrastructure already built (Plans 1, 2, 5), plus a UI screen to trigger and view it.

**Architecture:** `app/realityscan.py` is a subprocess wrapper around RealityScan's real, verified CLI command sequence (see Global Constraints — this was tested against the actual installed binary on this machine, not written from documentation alone). `app/alignment_service.py` orchestrates it against a job using the same `JobStore`/stage pattern as every other stage. `app/api/alignment.py` exposes it as a backgrounded `202` endpoint plus a results `GET`, following Plan 5's established pattern exactly. `static/job.html` gets a new button/link pair, and a new `static/alignment.html` shows the result.

**Tech Stack:** Same as the rest of the project (Python 3.11+, FastAPI, SQLite, pytest) plus RealityScan (`C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe` on this machine — Windows-only, proprietary, invoked via subprocess only).

## Global Constraints

- **The RealityScan command sequence below is empirically verified, not documentation-derived.** It was run against the real installed binary on this development machine (`RealityScan.exe -headless -silent <crash_dir> -set "appQuitOnError=true" -newScene -writeProgress <path> -addFolder <frames_dir> -align -exportRegistration <reg_path> -exportSparsePointCloud <ply_path> -save <project_path> -quit`) with a small set of synthetic (non-photogrammetry-realistic) test frames. Confirmed behavior: process starts and exits cleanly within seconds (no hang, no interactive popup, no crash dialog); a failed reconstruction (as expected from synthetic frames with no real parallax) produces a **non-zero exit code** (observed: `5`) rather than always exiting `0`; `-writeProgress` produces a real, readable file with lines of the form `<task_id> <fraction 0-1> <col3> <col4> #started|#progress|#completed`. Treat non-zero exit as a genuine failure signal — it is not a documentation gap or a flaky test artifact.
- **RealityScan is not on `PATH` on this machine but is installed** at `C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe`. The binary lookup must check `PATH` first (via `shutil.which`, respecting explicit user setup) and fall back to globbing `C:\Program Files\Epic Games\RealityScan_*\RealityScan.exe` (newest version first) — this fallback is what makes the binary discoverable without any manual setup, on this machine and for any implementer/test subagent working in this same environment.
- **The subprocess call is time-bounded (`timeout_seconds`, default 600).** The user runs this project over remote access and cannot manually dismiss a hung GUI process — an unbounded `subprocess.run` call would risk blocking a background task (and thus this project's whole background-task thread pool) forever. `subprocess.TimeoutExpired` must be caught and turned into a clear `RealityScanError`, not left to propagate or hang.
- Alignment operates on **`job.frames_dir`** (the extracted-frames folder from Plan 1), not `job.raw_video_path` — matches the design spec's pipeline ordering (alignment follows frame extraction; it does not depend on the recommendation stage, which only needs the raw video).
- Same binding pre-flight/failed-stage rule established in Plans 1, 2, and 5: any stage failure — including a pre-flight validation failure before the stage's "running" upsert — is recorded as a `failed` row in `job_stages`, **except** when `job_id` itself doesn't exist (no row to attach to, since `job_stages.job_id` has a foreign-key reference to `jobs.id` under `PRAGMA foreign_keys = ON`).
- `POST /jobs/{id}/align` follows the exact background-task pattern established in Plan 5: `status_code=202`, synchronous 404 pre-check, `background_tasks.add_task(run_with_new_store, compute_alignment, job_id)`, returns `{"status": "started"}` immediately. `GET /jobs/{id}/alignment` is a plain synchronous read, like `GET /jobs/{id}/recommendation`.
- The `-writeProgress` file's raw content is appended to the stage's `log` field as-is (matching the existing `sharp_frames.py`/`extraction.py` log pattern) — it is **not** parsed into a structured percentage. The column semantics beyond the fraction value (2nd column) aren't documented anywhere and weren't reverse-engineered beyond what the one real test run showed; building a parser on top of that would be guessing, not verified behavior.
- Every external-tool subprocess wrapper is tested via mocks in its unit tests (spec §9 discipline, same as `sharp_frames.py`). Task 6 additionally includes a **real** end-to-end test against the actual installed RealityScan binary — unlike Plans 3/4 as originally scoped, RealityScan does not need to be treated as unverifiable on this machine.

---

## Task 1: RealityScan CLI wrapper + `alignment_dir` path helper

**Files:**
- Modify: `app/paths.py`
- Create: `app/realityscan.py`
- Test: `tests/test_paths.py` (add test)
- Test: `tests/test_realityscan.py` (new)

**Interfaces:**
- Consumes: nothing (pure subprocess wrapper, same shape as `app/sharp_frames.py`).
- Produces: `paths.alignment_dir(job_id: str) -> Path`; `run_alignment(frames_dir: Path, output_dir: Path, *, progress_path: Path | None = None, timeout_seconds: int = 600) -> AlignmentResult` (dataclass: `registration_path: Path, point_cloud_path: Path, log: str`) and `RealityScanError`. Task 3 (alignment service) consumes all of these.

- [ ] **Step 1: Write the failing test for the path helper**

Add to `tests/test_paths.py`:
```python
def test_alignment_dir_uses_data_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATPREP_DATA_DIR", str(tmp_path))
    assert paths.alignment_dir("abc123") == tmp_path.resolve() / "jobs" / "abc123" / "alignment"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL with `AttributeError: module 'app.paths' has no attribute 'alignment_dir'`

- [ ] **Step 3: Replace the full contents of `app/paths.py`**

```python
from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("SPLATPREP_DATA_DIR", "./data")).resolve()


def job_dir(job_id: str) -> Path:
    return data_root() / "jobs" / job_id


def raw_dir(job_id: str) -> Path:
    return job_dir(job_id) / "raw"


def frames_dir(job_id: str) -> Path:
    return job_dir(job_id) / "frames"


def alignment_dir(job_id: str) -> Path:
    return job_dir(job_id) / "alignment"


def upload_chunks_dir(job_id: str, upload_id: str) -> Path:
    return job_dir(job_id) / "uploads" / upload_id / "chunks"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paths.py -v`
Expected: PASS (3 passed — the 2 pre-existing plus the new one)

- [ ] **Step 5: Write the failing tests for the RealityScan wrapper**

`tests/test_realityscan.py`:
```python
import subprocess
from unittest.mock import MagicMock

import pytest

from app import realityscan


def test_run_alignment_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: None)
    monkeypatch.setattr(realityscan, "_EPIC_GAMES_DIR", tmp_path)  # empty dir, nothing to glob
    with pytest.raises(realityscan.RealityScanError, match="not found"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_realityscan_binary_falls_back_to_epic_games_install(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: None)
    fake_exe = tmp_path / "RealityScan_2.2" / "RealityScan.exe"
    fake_exe.parent.mkdir(parents=True)
    fake_exe.write_text("")
    monkeypatch.setattr(realityscan, "_EPIC_GAMES_DIR", tmp_path)

    binary = realityscan._realityscan_binary()

    assert binary == str(fake_exe)


def test_run_alignment_returns_result_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    output_dir = tmp_path / "out"

    def fake_run(command, capture_output, text, check, timeout):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "registration.csv").write_text("frame_001.jpg\n")
        (output_dir / "points.ply").write_bytes(b"fake ply")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "aligned ok"
        result.stderr = ""
        return result

    monkeypatch.setattr(realityscan.subprocess, "run", fake_run)

    result = realityscan.run_alignment(tmp_path / "frames", output_dir)

    assert result.registration_path == output_dir / "registration.csv"
    assert result.point_cloud_path == output_dir / "points.ply"
    assert "aligned ok" in result.log


def test_run_alignment_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 5
        result.stdout = ""
        result.stderr = "export failed: no valid reconstruction"
        return result

    monkeypatch.setattr(realityscan.subprocess, "run", fake_run)

    with pytest.raises(realityscan.RealityScanError, match="export failed"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_when_output_files_missing_despite_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(realityscan.subprocess, "run", fake_run)

    with pytest.raises(realityscan.RealityScanError, match="did not produce"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    def fake_run(command, capture_output, text, check, timeout):
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr(realityscan.subprocess, "run", fake_run)

    with pytest.raises(realityscan.RealityScanError, match="did not finish within"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out", timeout_seconds=5)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/test_realityscan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.realityscan'`

- [ ] **Step 7: Write `app/realityscan.py`**

```python
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_EPIC_GAMES_DIR = Path(r"C:\Program Files\Epic Games")


class RealityScanError(Exception):
    pass


@dataclass
class AlignmentResult:
    registration_path: Path
    point_cloud_path: Path
    log: str


def _realityscan_binary() -> str:
    binary = shutil.which("RealityScan")
    if binary is not None:
        return binary
    candidates = sorted(_EPIC_GAMES_DIR.glob("RealityScan_*/RealityScan.exe"), reverse=True)
    if candidates:
        return str(candidates[0])
    raise RealityScanError(
        "RealityScan not found on PATH or in the default Epic Games install location. "
        "Install RealityScan, or add its install folder to PATH."
    )


def run_alignment(
    frames_dir: Path,
    output_dir: Path,
    *,
    progress_path: Path | None = None,
    timeout_seconds: int = 600,
) -> AlignmentResult:
    binary = _realityscan_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    crash_dir = output_dir / "crash_reports"
    crash_dir.mkdir(parents=True, exist_ok=True)

    registration_path = output_dir / "registration.csv"
    point_cloud_path = output_dir / "points.ply"
    project_path = output_dir / "project.rsproj"

    command = [
        binary,
        "-headless",
        "-silent",
        str(crash_dir),
        "-set",
        "appQuitOnError=true",
        "-newScene",
    ]
    if progress_path is not None:
        command += ["-writeProgress", str(progress_path)]
    command += [
        "-addFolder",
        str(frames_dir),
        "-align",
        "-exportRegistration",
        str(registration_path),
        "-exportSparsePointCloud",
        str(point_cloud_path),
        "-save",
        str(project_path),
        "-quit",
    ]

    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        raise RealityScanError(
            f"RealityScan did not finish within {timeout_seconds}s and was terminated"
        ) from exc

    log = process.stdout + process.stderr
    if progress_path is not None and progress_path.is_file():
        log += "\n--- progress ---\n" + progress_path.read_text(errors="ignore")

    if process.returncode != 0:
        raise RealityScanError(f"RealityScan exited with {process.returncode}:\n{log}")

    if not registration_path.is_file() or not point_cloud_path.is_file():
        raise RealityScanError(f"RealityScan did not produce expected output files:\n{log}")

    return AlignmentResult(
        registration_path=registration_path,
        point_cloud_path=point_cloud_path,
        log=log,
    )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_realityscan.py -v`
Expected: PASS (6 passed)

- [ ] **Step 9: Commit**

```bash
git add app/paths.py app/realityscan.py tests/test_paths.py tests/test_realityscan.py
git commit -m "feat: add RealityScan CLI wrapper (verified against real binary)"
```

---

## Task 2: Extend job schema with `registration_path` / `point_cloud_path`

**Files:**
- Modify: `app/db.py`
- Modify: `app/models.py`
- Modify: `app/job_store.py`
- Modify: `app/api/jobs.py`
- Test: `tests/test_job_store.py` (add test)
- Test: `tests/test_api_jobs.py` (add test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Job.registration_path: str | None` and `Job.point_cloud_path: str | None` now exist and round-trip through `JobStore` exactly like every other nullable `Job` field, and are now included in the `GET /jobs`/`GET /jobs/{id}` `JobResponse` (closing the same kind of gap Plan 5 Task 1 closed for `recommendation_json`). Task 3 sets these via `store.update_job(job_id, registration_path=..., point_cloud_path=...)`.

Note: if a local `data/splatprep.db` file exists on disk from manual testing, delete it after this task — `CREATE TABLE IF NOT EXISTS` does not alter an existing table's columns, and this project has no production data (the `data/` directory is gitignored).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_job_store.py`:
```python
def test_update_job_sets_registration_and_point_cloud_paths(job_store):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        registration_path="/data/alignment/registration.csv",
        point_cloud_path="/data/alignment/points.ply",
    )
    fetched = job_store.get_job(job.id)
    assert fetched.registration_path == "/data/alignment/registration.csv"
    assert fetched.point_cloud_path == "/data/alignment/points.ply"
```

Add to `tests/test_api_jobs.py`:
```python
def test_get_job_includes_alignment_fields(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}")
    body = response.json()
    assert "registration_path" in body
    assert "point_cloud_path" in body
    assert body["registration_path"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_job_store.py tests/test_api_jobs.py -v`
Expected: FAIL — `test_update_job_sets_registration_and_point_cloud_paths` with a `TypeError` (unexpected keyword args aren't rejected by `update_job`, but the columns don't exist yet, so the `UPDATE` will raise `sqlite3.OperationalError: no such column`); `test_get_job_includes_alignment_fields` with `KeyError`/`AssertionError` (field missing from response).

- [ ] **Step 3: Replace the full contents of `app/db.py`**

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.paths import data_root

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    raw_video_path TEXT,
    frames_dir TEXT,
    frame_count INTEGER,
    recommendation_json TEXT,
    registration_path TEXT,
    point_cloud_path TEXT
);

CREATE TABLE IF NOT EXISTS job_stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    log TEXT,
    error TEXT,
    UNIQUE(job_id, stage)
);
"""


def db_path() -> Path:
    return data_root() / "splatprep.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn
```

- [ ] **Step 4: Replace the full contents of `app/models.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Job:
    id: str
    name: str
    created_at: str
    raw_video_path: str | None = None
    frames_dir: str | None = None
    frame_count: int | None = None
    recommendation_json: str | None = None
    registration_path: str | None = None
    point_cloud_path: str | None = None


@dataclass
class JobStage:
    job_id: str
    stage: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    log: str | None = None
    error: str | None = None
```

- [ ] **Step 5: Update `_row_to_job` in `app/job_store.py`**

Find the `_row_to_job` static method and add the two new fields:

```python
    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            raw_video_path=row["raw_video_path"],
            frames_dir=row["frames_dir"],
            frame_count=row["frame_count"],
            recommendation_json=row["recommendation_json"],
            registration_path=row["registration_path"],
            point_cloud_path=row["point_cloud_path"],
        )
```

Every other method in `app/job_store.py` stays exactly as it is — `update_job(**fields)` already handles arbitrary columns generically and needs no change.

- [ ] **Step 6: Add the two fields to `JobResponse` in `app/api/jobs.py`**

Find the `JobResponse` class and add the two fields (keep every other field and every route in the file exactly as-is):

```python
class JobResponse(BaseModel):
    id: str
    name: str
    created_at: str
    raw_video_path: str | None
    frames_dir: str | None
    frame_count: int | None
    recommendation_json: str | None
    registration_path: str | None
    point_cloud_path: str | None
```

- [ ] **Step 7: Delete any stale local database file**

Run: `rm -f data/splatprep.db` (harmless if it doesn't exist).

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_db.py tests/test_job_store.py tests/test_api_jobs.py -v`
Expected: PASS (all tests in these three files)

- [ ] **Step 9: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all pass, 107 passed (98 pre-existing + 6 from Task 1 + 3 new here — exact count isn't load-bearing, zero failures is)

- [ ] **Step 10: Commit**

```bash
git add app/db.py app/models.py app/job_store.py app/api/jobs.py tests/test_job_store.py tests/test_api_jobs.py
git commit -m "feat: add registration_path/point_cloud_path columns to jobs table"
```

---

## Task 3: Alignment service

**Files:**
- Create: `app/alignment_service.py`
- Test: `tests/test_alignment_service.py`

**Interfaces:**
- Consumes: `JobStore` (Plan 1), `paths.alignment_dir` (Task 1), `run_alignment`/`AlignmentResult`/`RealityScanError` (Task 1).
- Produces: `compute_alignment(store: JobStore, job_id: str) -> AlignmentResult` and `AlignmentError`. Sets `job.registration_path`/`job.point_cloud_path` and the `alignment` stage to `done` (or `failed`, per the binding pre-flight rule — except when `job_id` doesn't exist at all). Task 4 (API) calls this directly.

- [ ] **Step 1: Write the failing tests**

`tests/test_alignment_service.py`:
```python
from unittest.mock import MagicMock

import pytest

from app import alignment_service
from app.realityscan import AlignmentResult, RealityScanError


def test_compute_alignment_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    fake_result = AlignmentResult(
        registration_path=tmp_path / "registration.csv",
        point_cloud_path=tmp_path / "points.ply",
        log="aligned ok",
    )
    monkeypatch.setattr(alignment_service, "run_alignment", MagicMock(return_value=fake_result))

    result = alignment_service.compute_alignment(job_store, job.id)

    assert result is fake_result
    fetched = job_store.get_job(job.id)
    assert fetched.registration_path == str(tmp_path / "registration.csv")
    assert fetched.point_cloud_path == str(tmp_path / "points.ply")

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "done"
    assert stage.log == "aligned ok"


def test_compute_alignment_without_extracted_frames_raises(job_store):
    job = job_store.create_job("Test")

    with pytest.raises(alignment_service.AlignmentError, match="no extracted frames"):
        alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "failed"


def test_compute_alignment_unknown_job_raises_without_stage_write(job_store):
    with pytest.raises(alignment_service.AlignmentError, match="Unknown job"):
        alignment_service.compute_alignment(job_store, "does-not-exist")

    assert job_store.get_stage("does-not-exist", "alignment") is None


def test_compute_alignment_marks_stage_failed_on_realityscan_error(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    monkeypatch.setattr(
        alignment_service, "run_alignment", MagicMock(side_effect=RealityScanError("export failed"))
    )

    with pytest.raises(alignment_service.AlignmentError, match="export failed"):
        alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "failed"
    assert stage.error == "export failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_alignment_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.alignment_service'`

- [ ] **Step 3: Write `app/alignment_service.py`**

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import alignment_dir
from app.realityscan import AlignmentResult, RealityScanError, run_alignment


class AlignmentError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_alignment(store: JobStore, job_id: str) -> AlignmentResult:
    job = store.get_job(job_id)
    if job is None:
        # No stage write: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row
        # to attach a stage to for a job_id that doesn't exist.
        raise AlignmentError(f"Unknown job: {job_id}")
    if job.frames_dir is None:
        message = "Job has no extracted frames yet"
        now = _now()
        store.upsert_stage(job_id, "alignment", "failed", started_at=now, finished_at=now, error=message)
        raise AlignmentError(message)

    output_dir = alignment_dir(job_id)
    progress_path = output_dir / "progress.txt"
    store.upsert_stage(job_id, "alignment", "running", started_at=_now())
    try:
        result = run_alignment(Path(job.frames_dir), output_dir, progress_path=progress_path)
    except RealityScanError as exc:
        store.upsert_stage(job_id, "alignment", "failed", finished_at=_now(), error=str(exc))
        raise AlignmentError(str(exc)) from exc

    store.update_job(
        job_id,
        registration_path=str(result.registration_path),
        point_cloud_path=str(result.point_cloud_path),
    )
    store.upsert_stage(job_id, "alignment", "done", finished_at=_now(), log=result.log)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_alignment_service.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/alignment_service.py tests/test_alignment_service.py
git commit -m "feat: add alignment service orchestrating RealityScan against a job"
```

---

## Task 4: Alignment API endpoints

**Files:**
- Create: `app/api/alignment.py`
- Modify: `app/main.py`
- Test: `tests/test_api_alignment.py`

**Interfaces:**
- Consumes: `compute_alignment`/`AlignmentError` (Task 3), `get_job_store`/`run_with_new_store` (Plan 1/Plan 5).
- Produces: `POST /jobs/{job_id}/align` (202, `{"status": "started"}`, backgrounded); `GET /jobs/{job_id}/alignment` (returns `{"registration_path": str, "point_cloud_path": str}`, 404 if job doesn't exist or alignment not computed yet). Task 5 (frontend) calls both.

- [ ] **Step 1: Write the failing tests**

`tests/test_api_alignment.py`:
```python
def test_align_endpoint_returns_202_and_invokes_service(client, monkeypatch):
    from app.api import alignment as alignment_api

    calls = []

    def fake_compute_alignment(store, job_id):
        calls.append(job_id)

    monkeypatch.setattr(alignment_api, "compute_alignment", fake_compute_alignment)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/align")

    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [job_id]


def test_align_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/align")
    assert response.status_code == 404


def test_get_alignment_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/alignment")
    assert response.status_code == 404


def test_get_alignment_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/alignment")
    assert response.status_code == 404


def test_get_alignment_returns_paths_once_computed(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).update_job(
        job_id, registration_path="/data/reg.csv", point_cloud_path="/data/points.ply"
    )
    conn.close()

    response = client.get(f"/jobs/{job_id}/alignment")
    assert response.status_code == 200
    assert response.json() == {
        "registration_path": "/data/reg.csv",
        "point_cloud_path": "/data/points.ply",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_alignment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.alignment'`

- [ ] **Step 3: Write `app/api/alignment.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.alignment_service import compute_alignment
from app.deps import get_job_store, run_with_new_store
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["alignment"])


@router.post("/{job_id}/align", status_code=202)
def align_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    background_tasks.add_task(run_with_new_store, compute_alignment, job_id)
    return {"status": "started"}


@router.get("/{job_id}/alignment")
def get_alignment(job_id: str, store: JobStore = Depends(get_job_store)) -> dict[str, str | None]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.registration_path is None:
        raise HTTPException(status_code=404, detail="Alignment not computed yet")
    return {
        "registration_path": job.registration_path,
        "point_cloud_path": job.point_cloud_path,
    }
```

- [ ] **Step 4: Wire the router into the app**

Replace the full contents of `app/main.py`:

```python
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.alignment import router as alignment_router
from app.api.frames import router as frames_router
from app.api.hardware import router as hardware_router
from app.api.ingestion import router as ingestion_router
from app.api.jobs import router as jobs_router
from app.api.recommendation import router as recommendation_router

app = FastAPI(title="Splat Prep")
app.include_router(jobs_router)
app.include_router(ingestion_router)
app.include_router(frames_router)
app.include_router(hardware_router)
app.include_router(recommendation_router)
app.include_router(alignment_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
```

The static mount stays registered last, after every `include_router` call — do not change that ordering.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_alignment.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all pass, 116 passed (107 from Tasks 1-2 + 4 from Task 3 + 5 new here — exact count isn't load-bearing, zero failures is)

- [ ] **Step 7: Commit**

```bash
git add app/api/alignment.py app/main.py tests/test_api_alignment.py
git commit -m "feat: add alignment API endpoints"
```

---

## Task 5: Frontend — alignment controls on the job page + results page

**Files:**
- Modify: `static/job.html`
- Modify: `static/index.html`
- Create: `static/alignment.html`
- Test: `tests/test_static_pages.py` (add tests)

**Interfaces:**
- Consumes: `POST /jobs/{id}/align`, `GET /jobs/{id}/alignment` (Task 4), the already-existing `STAGE_LABELS`/`pollStages`/`apiGet`/`apiPost`/`escapeHtml` helpers (Plan 5, including its post-review fix round that added `escapeHtml` to `static/js/app.js` and stopped `job.html`'s poll loop once no stage is running — **do not remove or regress either of those**).

- [ ] **Step 1: Read the current files first**

Read `static/job.html` and `static/index.html` in full before making any change — both were modified after their original Plan 5 tasks (a fix round added `escapeHtml(...)` calls and a poll-stop condition to `job.html`, and `escapeHtml(...)` to `index.html`). Your edits must preserve all of that; you are adding to these files, not replacing them wholesale.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_static_pages.py`:
```python
def test_job_page_contains_align_button(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="btn-align"' in response.text


def test_alignment_page_served(client):
    response = client.get("/alignment.html")
    assert response.status_code == 200
    assert 'id="content"' in response.text
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_static_pages.py -v`
Expected: FAIL — `test_job_page_contains_align_button` because `id="btn-align"` doesn't exist yet; `test_alignment_page_served` because `/alignment.html` doesn't exist yet (404).

- [ ] **Step 4: Add the `STAGE_LABELS` entry in both `static/index.html` and `static/job.html`**

In both files, find the `STAGE_LABELS` object (currently `{ ingestion: "Ingestie", frame_extraction: "Extragere cadre", recommendation: "Recomandare" }`) and add an `alignment` entry so it reads:

```javascript
  const STAGE_LABELS = {
    ingestion: "Ingestie",
    frame_extraction: "Extragere cadre",
    alignment: "Alignment",
    recommendation: "Recomandare",
  };
```

- [ ] **Step 5: Add the align button/link markup to `static/job.html`**

Find the existing action elements block (it currently has, in this order: `btn-extract`, `link-frames`, `btn-recommend`, `link-recommendation`). Insert a new button and link **between `link-frames` and `btn-recommend`**, so the block reads, in order: `btn-extract`, `link-frames`, `btn-align`, `link-alignment`, `btn-recommend`, `link-recommendation`:

```html
<button class="btn" id="btn-align" style="display:none;">Porneste alignment</button>
<a class="btn-ghost" id="link-alignment" style="display:none;">Vezi alignment</a>
```

- [ ] **Step 6: Add alignment gating logic to `updateActions` in `static/job.html`**

Inside the `updateActions(stages)` function, alongside the existing `ingestionDone`/`extractionDone`/`recommendationDone` constants, add:

```javascript
    const alignmentDone = stageStatus(stages, "alignment") === "done";
```

Change the `btn-extract` visibility condition's neighbor block to also show/hide the new elements — add these two lines in the same style as the existing `link-frames`/`recommendationLink` blocks:

```javascript
    document.getElementById("btn-align").style.display =
      extractionDone && !alignmentDone ? "block" : "none";

    const alignmentLink = document.getElementById("link-alignment");
    alignmentLink.style.display = alignmentDone ? "block" : "none";
    alignmentLink.href = `/alignment.html?id=${jobId}`;
```

- [ ] **Step 7: Add the click handler for `btn-align` in `static/job.html`**

Alongside the existing `btn-extract`/`btn-recommend` click handlers, add:

```javascript
  document.getElementById("btn-align").addEventListener("click", async () => {
    await apiPost(`/jobs/${jobId}/align`, {});
    refresh();
  });
```

- [ ] **Step 8: Create `static/alignment.html`**

```html
<!doctype html>
<html lang="ro">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Alignment — Splat Prep</title>
<link rel="stylesheet" href="/css/app.css" />
</head>
<body>
<h1>Alignment</h1>
<p class="subtitle" id="status">Se incarca...</p>

<div id="content"></div>

<script src="/js/app.js"></script>
<script>
  const jobId = jobIdFromQuery();

  async function load() {
    try {
      const alignment = await apiGet(`/jobs/${jobId}/alignment`);
      document.getElementById("status").textContent = "";
      document.getElementById("content").innerHTML = `
        <div class="card">
          <div class="job-name" style="margin-bottom:8px;">Rezultat</div>
          <div class="job-meta">Registration: ${escapeHtml(alignment.registration_path)}</div>
          <div class="job-meta">Nor de puncte: ${escapeHtml(alignment.point_cloud_path)}</div>
        </div>
      `;
    } catch (err) {
      document.getElementById("status").textContent = "Alignment-ul nu a fost calculat inca.";
    }
  }

  load();
</script>
</body>
</html>
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_static_pages.py -v`
Expected: PASS (all tests in this file)

- [ ] **Step 10: Visual verification**

Start the dev server, open `/job.html?id=<a job that has completed frame extraction>` in a browser (or via `chrome-devtools` tooling if available), confirm "Porneste alignment" appears once extraction is done, click it, and confirm the stage list updates and eventually shows a link to `/alignment.html`. Note in your report whether you were able to do this.

- [ ] **Step 11: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all pass

- [ ] **Step 12: Commit**

```bash
git add static/job.html static/index.html static/alignment.html tests/test_static_pages.py
git commit -m "feat: add alignment controls to job page and a results page"
```

---

## Task 6: End-to-end test against the real RealityScan binary

**Files:**
- Test: `tests/test_alignment_end_to_end.py` (new)

**Interfaces:**
- Consumes: the full app (`app.main.app`) and every module built in Tasks 1-5, plus the real `ffmpeg`, `sharp-frames`, and RealityScan binaries already confirmed present on this machine. No new production interfaces.

- [ ] **Step 1: Write the end-to-end test**

`tests/test_alignment_end_to_end.py`:
```python
import shutil
import subprocess

import pytest

from app.realityscan import RealityScanError, _realityscan_binary

FFMPEG = shutil.which("ffmpeg")
SHARP_FRAMES = shutil.which("sharp-frames")

try:
    _realityscan_binary()
    _REALITYSCAN_AVAILABLE = True
except RealityScanError:
    _REALITYSCAN_AVAILABLE = False


@pytest.mark.skipif(
    FFMPEG is None or SHARP_FRAMES is None or not _REALITYSCAN_AVAILABLE,
    reason="requires ffmpeg, sharp-frames, and RealityScan installed",
)
def test_full_ingest_extract_align_flow(client, tmp_path):
    video_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=10",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    job_id = client.post("/jobs", json={"name": "E2E alignment"}).json()["id"]

    ingest_response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(video_path)}
    )
    assert ingest_response.status_code == 202

    extract_response = client.post(
        f"/jobs/{job_id}/extract-frames",
        json={"selection_method": "batched", "fps": 5},
    )
    assert extract_response.status_code == 202
    job_after_extract = client.get(f"/jobs/{job_id}").json()
    assert job_after_extract["frame_count"] > 0

    align_response = client.post(f"/jobs/{job_id}/align")
    assert align_response.status_code == 202

    stages = client.get(f"/jobs/{job_id}/stages").json()
    alignment_stage = next(s for s in stages if s["stage"] == "alignment")
    # Real photogrammetry alignment needs overlapping real-world footage;
    # the synthetic testsrc pattern used here has no true parallax, so
    # RealityScan may legitimately fail to reconstruct anything (this is
    # exactly what was observed during manual verification while writing
    # this plan). This test verifies the orchestration reaches a terminal
    # state without hanging or crashing — not that alignment quality is
    # good, which depends on the real footage the user provides.
    assert alignment_stage["status"] in ("done", "failed")
    if alignment_stage["status"] == "failed":
        assert alignment_stage["error"]
    else:
        job = client.get(f"/jobs/{job_id}").json()
        assert job["registration_path"] is not None
        assert job["point_cloud_path"] is not None
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_alignment_end_to_end.py -v`
Expected: PASS — either branch (`done` or `failed`) is an acceptable outcome per the test's own assertions; SKIPPED only if `ffmpeg`, `sharp-frames`, or RealityScan aren't found in the environment this runs in. Never a bare failure.

- [ ] **Step 3: Run the full suite one more time**

Run: `python -m pytest -v`
Expected: all tests pass (or pass+skip cleanly), zero failures — 120 passed on this machine (116 from Tasks 1-5 + 1 new here + the 3 new tests from Task 5 already counted in Task 4's post-suite total... exact count isn't load-bearing, what matters is zero failures)

- [ ] **Step 4: Commit**

```bash
git add tests/test_alignment_end_to_end.py
git commit -m "test: add end-to-end ingest-extract-align integration test against real RealityScan"
```
