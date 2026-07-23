# Progress Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a live percentage + progress bar for frame extraction and RealityScan alignment (the two stages with a real, parseable progress signal), and an indeterminate animated bar for every other running stage, on the job detail page.

**Architecture:** Add a nullable `progress_percent` column to `job_stages`. `app/sharp_frames.py` switches from a single blocking `subprocess.run` call to `subprocess.Popen` plus a background reader thread that live-parses `tqdm`-style `NN%|` lines from stdout (Windows can't reliably poll a pipe without blocking, so a reader thread feeding a `queue.Queue` is the standard safe pattern — not a workaround, the correct tool here) and enforces a manual timeout. `app/realityscan.py` switches to `subprocess.Popen` with a simple polling loop that periodically re-reads the `-writeProgress` file it already writes — no reader thread needed there, since polling a file is not the same problem as reading a live pipe. Both wrappers gain an `on_progress: Callable[[float], None] | None` parameter; the two service layers use it to push live updates into `job_stages`. The frontend renders a determinate bar when `progress_percent` is present, an indeterminate animated one otherwise.

**Tech Stack:** Same as the rest of the project (Python 3.11+, stdlib `subprocess`/`threading`/`queue`, FastAPI, SQLite, pytest) — no new dependencies.

## Global Constraints

- **Every subprocess call in this project must stay time-bounded.** The project owner works over remote access and cannot manually kill a hung process. `run_extraction` currently has **no timeout at all** — this plan closes that pre-existing gap (`timeout_seconds: int = 1800` default) as part of the Popen rewrite, not as a separate task, since the rewrite touches the exact same code path. `run_alignment`'s existing `timeout_seconds=600` default and its "kill on timeout" guarantee must survive the Popen rewrite unchanged.
- **Live progress is a live signal, not a promise of continuous polling.** The frontend already polls `GET /jobs/{id}/stages` every 2s (`pollStages` in `static/js/app.js`, unchanged by this plan) — that cadence is reused as-is; this plan does not add a faster poll loop.
- **Scope, per explicit user decision:** real numeric percent for **frame extraction** and **alignment** only. **Recommendation** and **ingestion** stages get the indeterminate bar (no backend change needed for them — they simply never set `progress_percent`, so it stays `NULL`, which the frontend already treats as "indeterminate").
- Same binding pre-flight/failed-stage rule established throughout this codebase: any stage failure is recorded as `failed` in `job_stages` before raising, except when `job_id` itself doesn't exist (FK constraint, no row to attach to).
- External-tool wrappers are tested via subprocess/Popen mocks in unit tests, never the real binary — except the dedicated end-to-end test in Task 6, which runs against the real installed tools on this machine and is allowed to skip if they're missing.
- `db.connect()` already sets `check_same_thread=False` (established fix) — the new reader thread in Task 2 must open **its own** connection if it ever needs the database (it doesn't: progress is reported via an in-memory callback, not a direct DB write from the thread — see Task 2's design note).

---

## Task 1: `progress_percent` column

**Files:**
- Modify: `app/db.py`
- Modify: `app/models.py`
- Modify: `app/job_store.py`
- Modify: `app/api/jobs.py`
- Test: `tests/test_job_store.py` (add test)
- Test: `tests/test_api_jobs.py` (add test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `JobStage.progress_percent: float | None`; `JobStore.upsert_stage(..., progress_percent: float | None = None)` now accepts and persists it, preserving the last known value across calls that don't pass one (via `COALESCE`, the same pattern already used for `started_at`); `JobStageResponse` (API model) includes `progress_percent`. Tasks 4 and 5 rely on this.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_job_store.py`:
```python
def test_upsert_stage_sets_and_preserves_progress_percent(job_store):
    job = job_store.create_job("Test")

    job_store.upsert_stage(job.id, "frame_extraction", "running", started_at="t0", progress_percent=12.5)
    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.progress_percent == 12.5

    # A later call that doesn't pass progress_percent must not reset it to None
    job_store.upsert_stage(job.id, "frame_extraction", "running")
    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.progress_percent == 12.5

    # An explicit new value does update it
    job_store.upsert_stage(job.id, "frame_extraction", "running", progress_percent=47.0)
    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.progress_percent == 47.0
```

Add to `tests/test_api_jobs.py`:
```python
def test_get_job_stages_includes_progress_percent(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "frame_extraction", "running", started_at="t0", progress_percent=33.0)
    conn.close()

    response = client.get(f"/jobs/{job_id}/stages")
    assert response.status_code == 200
    stage = response.json()[0]
    assert stage["progress_percent"] == 33.0


def test_get_job_stages_progress_percent_defaults_to_none(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    from app import db
    from app.job_store import JobStore

    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "ingestion", "done")
    conn.close()

    response = client.get(f"/jobs/{job_id}/stages")
    assert response.json()[0]["progress_percent"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_job_store.py tests/test_api_jobs.py -v`
Expected: FAIL — `TypeError: upsert_stage() got an unexpected keyword argument 'progress_percent'`

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
    progress_percent REAL,
    UNIQUE(job_id, stage)
);
"""


def db_path() -> Path:
    return data_root() / "splatprep.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI dispatches sync dependencies (get_job_store) to a worker
    # threadpool, and does not guarantee the same thread handles both the
    # generator's setup and its teardown - so a connection created for one
    # request can be closed from a different thread. Each call to connect()
    # still returns a connection used by exactly one request, never shared
    # across concurrent requests, so disabling sqlite3's same-thread check
    # here is safe and does not introduce real concurrent-access risk.
    conn = sqlite3.connect(target, check_same_thread=False)
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
    progress_percent: float | None = None
```

- [ ] **Step 5: Update `upsert_stage` and `_row_to_stage` in `app/job_store.py`**

Find `upsert_stage` and replace it with:
```python
    def upsert_stage(
        self,
        job_id: str,
        stage: str,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        log: str | None = None,
        error: str | None = None,
        progress_percent: float | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO job_stages (job_id, stage, status, started_at, finished_at, log, error, progress_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, stage) DO UPDATE SET
                status = excluded.status,
                started_at = COALESCE(excluded.started_at, job_stages.started_at),
                finished_at = excluded.finished_at,
                log = excluded.log,
                error = excluded.error,
                progress_percent = COALESCE(excluded.progress_percent, job_stages.progress_percent)
            """,
            (job_id, stage, status, started_at, finished_at, log, error, progress_percent),
        )
        self._conn.commit()
```

Find `_row_to_stage` and replace it with:
```python
    @staticmethod
    def _row_to_stage(row: sqlite3.Row) -> JobStage:
        return JobStage(
            job_id=row["job_id"],
            stage=row["stage"],
            status=row["status"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            log=row["log"],
            error=row["error"],
            progress_percent=row["progress_percent"],
        )
```

Every other method in `app/job_store.py` stays exactly as it is.

- [ ] **Step 6: Add `progress_percent` to `JobStageResponse` in `app/api/jobs.py`**

Find `JobStageResponse` and replace it with:
```python
class JobStageResponse(BaseModel):
    stage: str
    status: str
    started_at: str | None
    finished_at: str | None
    log: str | None
    error: str | None
    progress_percent: float | None
```

Every route and every other class in the file stays exactly as it is.

- [ ] **Step 7: Delete any stale local database file**

Run: `rm -f data/splatprep.db` (harmless if it doesn't exist — `CREATE TABLE IF NOT EXISTS` doesn't add columns to an existing table, and this project has no production data).

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_job_store.py tests/test_api_jobs.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 9: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all pass — this run also covers the real opt-in RunPod test being correctly skipped/passed depending on this machine's env vars; zero failures either way is what matters, not the exact total

- [ ] **Step 10: Commit**

```bash
git add app/db.py app/models.py app/job_store.py app/api/jobs.py tests/test_job_store.py tests/test_api_jobs.py
git commit -m "feat: add progress_percent column to job_stages"
```

---

## Task 2: Live progress for frame extraction (`app/sharp_frames.py`)

**Files:**
- Modify: `app/sharp_frames.py` (full replacement)
- Test: `tests/test_sharp_frames.py` (full replacement)

**Interfaces:**
- Consumes: nothing new.
- Produces: `run_extraction(video_path, output_dir, *, selection_method="best-n", num_frames=300, fps=10, on_progress: Callable[[float], None] | None = None, timeout_seconds: int = 1800) -> ExtractionResult`. `on_progress` is called with a `float` in `[0, 100]` each time a `tqdm`-style `NN%|` marker is parsed from the subprocess's combined stdout/stderr. `SharpFramesError` is now also raised on timeout (previously there was no timeout at all). Task 4 wires this into `app/extraction.py`.

**Design note — why a reader thread, not `select`:** Windows cannot reliably use `select()`/non-blocking reads on a subprocess pipe the way Unix can. The standard cross-platform-safe pattern for "read a live pipe AND enforce a timeout" is: a background thread does blocking reads and pushes complete lines onto a `queue.Queue`; the main thread polls that queue with a short timeout in a loop, checking the overall deadline between polls. `tqdm` updates its bar using `\r` (carriage return), not `\n`, so the reader thread splits on **both** `\r` and `\n` to see each intermediate percentage, not just the final one per line.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_sharp_frames.py`:
```python
import io
import subprocess
import time
from unittest.mock import MagicMock

import pytest

from app import sharp_frames


class FakeStdout(io.StringIO):
    """A StringIO that, once exhausted, keeps returning "" like a real closed pipe."""


def _fake_popen(stdout_text: str, returncode: int = 0) -> MagicMock:
    process = MagicMock()
    process.stdout = FakeStdout(stdout_text)
    process.wait = MagicMock(return_value=returncode)
    process.kill = MagicMock()
    return process


def test_run_extraction_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: None)
    with pytest.raises(sharp_frames.SharpFramesError, match="not found on PATH"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_extraction_returns_frames_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    output_dir = tmp_path / "out"

    def fake_popen(command, stdout, stderr, text, bufsize):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        (output_dir / "frame_0002.jpg").write_bytes(b"fake")
        return _fake_popen("extracted 2 frames\n", returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    result = sharp_frames.run_extraction(tmp_path / "video.mp4", output_dir)

    assert len(result.frames) == 2
    assert "extracted 2 frames" in result.log


def test_run_extraction_reports_progress_from_tqdm_style_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    output_dir = tmp_path / "out"
    # tqdm updates with \r between percentages, only a real \n at the end
    fake_output = (
        "Calculating sharpness for frames:  10%|1  | 1/10\r"
        "Calculating sharpness for frames:  50%|5  | 5/10\r"
        "Calculating sharpness for frames: 100%|##| 10/10\n"
    )

    def fake_popen(command, stdout, stderr, text, bufsize):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen(fake_output, returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    seen_percentages = []
    sharp_frames.run_extraction(
        tmp_path / "video.mp4", output_dir, on_progress=seen_percentages.append
    )

    assert seen_percentages == [10.0, 50.0, 100.0]


def test_run_extraction_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    def fake_popen(command, stdout, stderr, text, bufsize):
        return _fake_popen("ffmpeg not found\n", returncode=1)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    with pytest.raises(sharp_frames.SharpFramesError, match="ffmpeg not found"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_extraction_raises_when_no_frames_produced(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    def fake_popen(command, stdout, stderr, text, bufsize):
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    with pytest.raises(sharp_frames.SharpFramesError, match="produced no frames"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_extraction_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    # A stream that never reaches EOF, simulating a hung process. The reader
    # thread is daemon=True so it won't block the test process from exiting,
    # but without a small real sleep here it would busy-loop at 100% CPU on
    # one core for the rest of the test session (harmless but wasteful) -
    # this uses the real time.sleep (not app.sharp_frames.time, which the
    # test mocks separately for the main-thread deadline check below).
    class HangingStdout:
        def read(self, _n):
            import time as real_time

            real_time.sleep(0.01)
            return " "  # never "" (EOF), never \r or \n - buffers forever

    def fake_popen(command, stdout, stderr, text, bufsize):
        process = MagicMock()
        process.stdout = HangingStdout()
        process.kill = MagicMock()
        process.wait = MagicMock(return_value=1)
        return process

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    fake_clock = [0.0]

    def fake_monotonic():
        fake_clock[0] += 1000.0  # jump straight past any deadline
        return fake_clock[0]

    monkeypatch.setattr(sharp_frames.time, "monotonic", fake_monotonic)

    with pytest.raises(sharp_frames.SharpFramesError, match="did not finish within"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out", timeout_seconds=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sharp_frames.py -v`
Expected: FAIL — `AttributeError: module 'app.sharp_frames' has no attribute 'time'` (or similar, since the current module doesn't import `time`/use `Popen`/accept `on_progress`)

- [ ] **Step 3: Replace the full contents of `app/sharp_frames.py`**

```python
from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_PERCENT_RE = re.compile(r"(\d+)%\|")


class SharpFramesError(Exception):
    pass


@dataclass
class ExtractionResult:
    frames: list[Path]
    log: str


def _sharp_frames_binary() -> str:
    binary = shutil.which("sharp-frames")
    if binary is None:
        raise SharpFramesError(
            "sharp-frames CLI not found on PATH. Install it with 'pip install sharp-frames'."
        )
    return binary


def _reader_thread(stream, output_queue: queue.Queue) -> None:
    buffer = ""
    for char in iter(lambda: stream.read(1), ""):
        if char in ("\r", "\n"):
            if buffer:
                output_queue.put(buffer)
                buffer = ""
        else:
            buffer += char
    if buffer:
        output_queue.put(buffer)
    output_queue.put(None)  # sentinel: stream closed


def run_extraction(
    video_path: Path,
    output_dir: Path,
    *,
    selection_method: str = "best-n",
    num_frames: int = 300,
    fps: int = 10,
    on_progress: Callable[[float], None] | None = None,
    timeout_seconds: int = 1800,
) -> ExtractionResult:
    binary = _sharp_frames_binary()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        binary,
        str(video_path),
        str(output_dir),
        "--fps",
        str(fps),
        "--selection-method",
        selection_method,
        "--force-overwrite",
    ]
    if selection_method == "best-n":
        command += ["--num-frames", str(num_frames)]

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    output_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_reader_thread, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    log_lines: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            process.wait()
            raise SharpFramesError(
                f"sharp-frames did not finish within {timeout_seconds}s and was terminated"
            )
        try:
            item = output_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if item is None:
            break
        log_lines.append(item)
        match = _PERCENT_RE.search(item)
        if match and on_progress is not None:
            on_progress(float(match.group(1)))

    returncode = process.wait()
    log = "\n".join(log_lines)

    if returncode != 0:
        raise SharpFramesError(f"sharp-frames exited with {returncode}:\n{log}")

    frames = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))
    if not frames:
        raise SharpFramesError(f"sharp-frames produced no frames:\n{log}")

    return ExtractionResult(frames=frames, log=log)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sharp_frames.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add app/sharp_frames.py tests/test_sharp_frames.py
git commit -m "feat: live progress + timeout for sharp-frames extraction"
```

---

## Task 3: Live progress for alignment (`app/realityscan.py`)

**Files:**
- Modify: `app/realityscan.py` (full replacement)
- Test: `tests/test_realityscan.py` (full replacement)

**Interfaces:**
- Consumes: nothing new.
- Produces: `run_alignment(frames_dir, output_dir, *, progress_path=None, timeout_seconds=600, on_progress: Callable[[float], None] | None = None) -> AlignmentResult`. `on_progress` is called with a `float` in `[0, 100]` whenever the `-writeProgress` file has a newer fraction than last time it was polled (poll interval: 1s). The existing timeout-kills-the-process guarantee is preserved exactly. Task 4 wires this into `app/alignment_service.py`.

**Design note — no reader thread needed here:** unlike sharp-frames, RealityScan's progress comes from a **file** it writes on its own (`-writeProgress`), not from parsing live stdout. Polling a file periodically from the main thread while checking `process.poll()` is sufficient and simpler — no pipe-reading concurrency problem to solve.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_realityscan.py`:
```python
import subprocess
from unittest.mock import MagicMock

import pytest

from app import realityscan


def _fake_process(returncode: int, stdout_text: str = "") -> MagicMock:
    process = MagicMock()
    process.poll = MagicMock(side_effect=[None, None, returncode])
    process.returncode = returncode
    process.stdout = MagicMock()
    process.stdout.read = MagicMock(return_value=stdout_text)
    process.kill = MagicMock()
    process.wait = MagicMock(return_value=returncode)
    return process


def test_run_alignment_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: None)
    monkeypatch.setattr(realityscan, "_EPIC_GAMES_DIR", tmp_path)
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
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    output_dir = tmp_path / "out"

    def fake_popen(command, stdout, stderr, text):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "registration.csv").write_text("frame_001.jpg\n")
        (output_dir / "points.ply").write_bytes(b"fake ply")
        return _fake_process(0, "aligned ok")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    result = realityscan.run_alignment(tmp_path / "frames", output_dir)

    assert result.registration_path == output_dir / "registration.csv"
    assert result.point_cloud_path == output_dir / "points.ply"
    assert "aligned ok" in result.log


def test_run_alignment_reports_progress_from_progress_file(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    output_dir = tmp_path / "out"
    progress_path = tmp_path / "progress.txt"
    sleep_calls = {"count": 0}

    def fake_sleep(_seconds):
        # Simulate the progress file gaining a new line between polls. The
        # loop structure is: while poll() is None: [check file] -> sleep().
        # So the Nth sleep() call happens right after the Nth file check -
        # write here what the (N+1)th check should see.
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            progress_path.write_text("65536 0.30 0.02 0.05 #progress\n")
        elif sleep_calls["count"] == 2:
            progress_path.write_text(
                "65536 0.30 0.02 0.05 #progress\n65536 0.90 0.04 0.01 #progress\n"
            )

    monkeypatch.setattr(realityscan.time, "sleep", fake_sleep)

    # 4 poll() calls: the while-condition is checked before each of the 3
    # loop-body entries (None, None, None) and once more to exit (returncode).
    # Body entry 1: file doesn't exist yet -> no report -> sleep #1 writes "0.30".
    # Body entry 2: file has "0.30" -> reports 30.0 -> sleep #2 writes "0.30"+"0.90".
    # Body entry 3: file's latest line is "0.90" -> reports 90.0 -> sleep #3 (no-op write).
    # 4th poll() call returns the returncode, ending the loop.
    process = MagicMock()
    process.poll = MagicMock(side_effect=[None, None, None, 0])
    process.returncode = 0
    process.stdout = MagicMock()
    process.stdout.read = MagicMock(return_value="ok")
    process.kill = MagicMock()
    process.wait = MagicMock(return_value=0)

    def fake_popen(command, stdout, stderr, text):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "registration.csv").write_text("frame_001.jpg\n")
        (output_dir / "points.ply").write_bytes(b"fake ply")
        return process

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    seen_percentages = []
    realityscan.run_alignment(
        tmp_path / "frames", output_dir, progress_path=progress_path, on_progress=seen_percentages.append
    )

    assert seen_percentages == [30.0, 90.0]


def test_run_alignment_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    def fake_popen(command, stdout, stderr, text):
        return _fake_process(5, "export failed: no valid reconstruction")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    with pytest.raises(realityscan.RealityScanError, match="export failed"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_when_output_files_missing_despite_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    def fake_popen(command, stdout, stderr, text):
        return _fake_process(0, "")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    with pytest.raises(realityscan.RealityScanError, match="did not produce"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    def fake_popen(command, stdout, stderr, text):
        process = MagicMock()
        process.poll = MagicMock(return_value=None)  # never finishes
        process.kill = MagicMock()
        process.wait = MagicMock(return_value=1)
        return process

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    fake_clock = [0.0]

    def fake_monotonic():
        fake_clock[0] += 1000.0
        return fake_clock[0]

    monkeypatch.setattr(realityscan.time, "monotonic", fake_monotonic)

    with pytest.raises(realityscan.RealityScanError, match="did not finish within"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out", timeout_seconds=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_realityscan.py -v`
Expected: FAIL — the fixture `_fake_process` and tests reference `realityscan.subprocess.Popen`/`realityscan.time`, but the current module uses `subprocess.run` with a `timeout=` kwarg and doesn't import `time`, so `fake_popen` (registered as `Popen`) is never called, and `time.sleep`/`time.monotonic` patches have nothing to attach to in the way the tests expect — collection or assertion failures either way

- [ ] **Step 3: Replace the full contents of `app/realityscan.py`**

```python
from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_EPIC_GAMES_DIR = Path(r"C:\Program Files\Epic Games")
_PROGRESS_LINE_RE = re.compile(r"^\d+\s+([\d.]+)\s+")


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


def _latest_progress_percent(progress_path: Path) -> float | None:
    try:
        lines = progress_path.read_text(errors="ignore").strip().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        match = _PROGRESS_LINE_RE.match(line)
        if match:
            return float(match.group(1)) * 100
    return None


def run_alignment(
    frames_dir: Path,
    output_dir: Path,
    *,
    progress_path: Path | None = None,
    timeout_seconds: int = 600,
    on_progress: Callable[[float], None] | None = None,
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

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + timeout_seconds
    last_reported: float | None = None

    while process.poll() is None:
        if time.monotonic() > deadline:
            process.kill()
            process.wait()
            raise RealityScanError(
                f"RealityScan did not finish within {timeout_seconds}s and was terminated"
            )
        if progress_path is not None and on_progress is not None and progress_path.is_file():
            percent = _latest_progress_percent(progress_path)
            if percent is not None and percent != last_reported:
                on_progress(percent)
                last_reported = percent
        time.sleep(1.0)

    returncode = process.returncode
    stdout_text = process.stdout.read() if process.stdout is not None else ""
    log = stdout_text
    if progress_path is not None and progress_path.is_file():
        log += "\n--- progress ---\n" + progress_path.read_text(errors="ignore")

    if returncode != 0:
        raise RealityScanError(f"RealityScan exited with {returncode}:\n{log}")

    if not registration_path.is_file() or not point_cloud_path.is_file():
        raise RealityScanError(f"RealityScan did not produce expected output files:\n{log}")

    return AlignmentResult(
        registration_path=registration_path,
        point_cloud_path=point_cloud_path,
        log=log,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_realityscan.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add app/realityscan.py tests/test_realityscan.py
git commit -m "feat: live progress for RealityScan alignment via progress-file polling"
```

---

## Task 4: Wire progress into the two service layers

**Files:**
- Modify: `app/extraction.py` (full replacement)
- Modify: `app/alignment_service.py` (full replacement)
- Test: `tests/test_extraction.py` (add test)
- Test: `tests/test_alignment_service.py` (add test)

**Interfaces:**
- Consumes: `run_extraction(..., on_progress=...)` (Task 2), `run_alignment(..., on_progress=...)` (Task 3), `JobStore.upsert_stage(..., progress_percent=...)` (Task 1).
- Produces: while `extract_frames`/`compute_alignment` run, `job_stages.progress_percent` for their respective stage gets updated live as the underlying tool reports progress. No new public functions — same signatures as before.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_extraction.py`:
```python
def test_extract_frames_forwards_progress_to_job_store(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    def fake_run_extraction(video_path, output_dir, *, selection_method, num_frames, fps, on_progress):
        on_progress(25.0)
        on_progress(75.0)
        return extraction.ExtractionResult(frames=[Path("f1.jpg")], log="ok")

    monkeypatch.setattr(extraction, "run_extraction", fake_run_extraction)

    extraction.extract_frames(job_store, job.id)

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.progress_percent == 75.0
```

Make sure `tests/test_extraction.py` imports `Path` from `pathlib` and `app.sharp_frames.ExtractionResult` (or `extraction.ExtractionResult` if re-exported) — check the top of the existing file and reuse whatever import it already has for `ExtractionResult`; add `from pathlib import Path` if not already present.

Add to `tests/test_alignment_service.py`:
```python
def test_compute_alignment_forwards_progress_to_job_store(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    def fake_run_alignment(frames_dir, output_dir, *, progress_path, on_progress):
        on_progress(40.0)
        on_progress(95.0)
        return AlignmentResult(
            registration_path=tmp_path / "registration.csv",
            point_cloud_path=tmp_path / "points.ply",
            log="ok",
        )

    monkeypatch.setattr(alignment_service, "run_alignment", fake_run_alignment)

    alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.progress_percent == 95.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction.py tests/test_alignment_service.py -v`
Expected: FAIL — `stage.progress_percent` is `None` (nothing forwards progress yet), or a `TypeError` if the fake's keyword-only `on_progress` doesn't match the real call site

- [ ] **Step 3: Replace the full contents of `app/extraction.py`**

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import frames_dir
from app.sharp_frames import ExtractionResult, SharpFramesError, run_extraction


class ExtractionError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_frames(
    store: JobStore,
    job_id: str,
    *,
    selection_method: str = "best-n",
    num_frames: int = 300,
    fps: int = 10,
) -> int:
    job = store.get_job(job_id)
    if job is None:
        # No stage write here: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row to
        # attach a stage to for a job_id that doesn't exist.
        raise ExtractionError(f"Unknown job: {job_id}")
    if job.raw_video_path is None:
        now = _now()
        store.upsert_stage(
            job_id, "frame_extraction", "failed", started_at=now, finished_at=now,
            error="Job has no ingested video yet",
        )
        raise ExtractionError("Job has no ingested video yet")

    output_dir = frames_dir(job_id)
    store.upsert_stage(job_id, "frame_extraction", "running", started_at=_now())

    def report_progress(percent: float) -> None:
        store.upsert_stage(job_id, "frame_extraction", "running", progress_percent=percent)

    try:
        result: ExtractionResult = run_extraction(
            Path(job.raw_video_path),
            output_dir,
            selection_method=selection_method,
            num_frames=num_frames,
            fps=fps,
            on_progress=report_progress,
        )
    except SharpFramesError as exc:
        store.upsert_stage(job_id, "frame_extraction", "failed", finished_at=_now(), error=str(exc))
        raise ExtractionError(str(exc)) from exc

    store.update_job(job_id, frames_dir=str(output_dir), frame_count=len(result.frames))
    store.upsert_stage(job_id, "frame_extraction", "done", finished_at=_now(), log=result.log)
    return len(result.frames)
```

- [ ] **Step 4: Replace the full contents of `app/alignment_service.py`**

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import alignment_dir
from app.realityscan import AlignmentResult, run_alignment


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

    def report_progress(percent: float) -> None:
        store.upsert_stage(job_id, "alignment", "running", progress_percent=percent)

    try:
        result = run_alignment(
            Path(job.frames_dir), output_dir, progress_path=progress_path, on_progress=report_progress
        )
    except Exception as exc:
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

Note: `except Exception` here (unchanged from before this task) is deliberate and already correct — it was fixed in an earlier plan specifically so any error, not just `RealityScanError`, gets recorded as a `failed` stage instead of leaving it stuck on `running`. Do not narrow it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction.py tests/test_alignment_service.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 6: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all pass, zero failures

- [ ] **Step 7: Commit**

```bash
git add app/extraction.py app/alignment_service.py tests/test_extraction.py tests/test_alignment_service.py
git commit -m "feat: forward live extraction/alignment progress into job_stages"
```

---

## Task 5: Frontend — progress bars on the job page

**Files:**
- Modify: `static/css/app.css`
- Modify: `static/job.html`
- Test: `tests/test_static_pages.py` (add test)

**Interfaces:**
- Consumes: `stage.progress_percent` from `GET /jobs/{id}/stages` (Task 1), already polled every 2s by the existing `pollStages` helper in `static/js/app.js` (unchanged).
- Produces: a visible progress bar under any `running` stage row — determinate (filled to `progress_percent`%) when that value is present, indeterminate (sliding animation) otherwise. No new files.

- [ ] **Step 1: Read the current `static/job.html` in full before editing**

This file has been modified several times across earlier plans and fix rounds (polling restart logic, alignment controls, extraction settings panel). Read it fully first — the step below tells you exactly what to change inside `renderStages`, not to replace the whole file.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_static_pages.py`:
```python
def test_job_page_contains_progress_bar_styles_and_render_hook(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert "progress-track" in response.text
    assert "progress-fill" in response.text
    assert "progress-indeterminate" in response.text
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_static_pages.py -k progress_bar -v`
Expected: FAIL — none of those class names exist in `job.html` yet

- [ ] **Step 4: Add progress bar CSS to `static/css/app.css`**

Append to the end of the file:
```css
.progress-track {
  height: 4px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 100px;
  overflow: hidden;
  margin-top: 8px;
}
.progress-fill {
  height: 100%;
  background: var(--accent-green);
  border-radius: 100px;
  transition: width 0.3s ease;
}
.progress-indeterminate {
  height: 100%;
  width: 40%;
  background: var(--accent-green);
  border-radius: 100px;
  animation: progress-slide 1.2s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
  .progress-indeterminate { animation: none; width: 100%; opacity: 0.4; }
}
@keyframes progress-slide {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(250%); }
}
```

- [ ] **Step 5: Add a `progressBarHtml` helper and use it in `renderStages` in `static/job.html`**

Find the `renderStages` function (it currently builds `errorLine` and interpolates it into each stage row's `job-info` div). Add a new helper function right above `renderStages`:

```javascript
  function progressBarHtml(stage) {
    if (stage.status !== "running") return "";
    if (stage.progress_percent !== null && stage.progress_percent !== undefined) {
      const pct = Math.max(0, Math.min(100, stage.progress_percent));
      return `<div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>`;
    }
    return `<div class="progress-track"><div class="progress-indeterminate"></div></div>`;
  }
```

Then, inside `renderStages`, find the template literal that builds each stage row (it currently has `${errorLine}` inside the `job-info` div) and add `${progressBarHtml(s)}` right after `${errorLine}`, so the row reads:

```javascript
        return `
          <div class="job-row">
            <div class="job-info">
              <div class="job-name">${STAGE_LABELS[s.stage] || s.stage}</div>
              ${errorLine}
              ${progressBarHtml(s)}
            </div>
            <span class="pill ${pillClass}">${s.status}</span>
          </div>
        `;
```

Do not change anything else in `renderStages`, `updateActions`, the polling functions, or any button handler.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_static_pages.py -k progress_bar -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Visual verification**

Start the dev server, open `/job.html?id=<a job mid-extraction or mid-alignment>` in a browser (or via `chrome-devtools` tooling if available), and confirm a thin green progress bar appears under the running stage and fills up as extraction/alignment proceeds, or slides indeterminately for a running stage with no percent yet. Note in your report whether you were able to check this live.

- [ ] **Step 8: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add static/css/app.css static/job.html tests/test_static_pages.py
git commit -m "feat: render progress bars for running pipeline stages"
```

---

## Task 6: End-to-end verification against the real binaries

**Files:**
- Test: `tests/test_progress_end_to_end.py` (new)

**Interfaces:**
- Consumes: the full app and every module built in Tasks 1-5, plus the real `ffmpeg`, `sharp-frames`, and RealityScan binaries already confirmed present on this machine. No new production interfaces.

- [ ] **Step 1: Write the end-to-end test**

`tests/test_progress_end_to_end.py`:
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
def test_extraction_and_alignment_end_with_progress_recorded(client, tmp_path):
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

    job_id = client.post("/jobs", json={"name": "E2E progress"}).json()["id"]
    client.post(f"/jobs/{job_id}/ingest/local", json={"source_path": str(video_path)})
    client.post(f"/jobs/{job_id}/extract-frames", json={"selection_method": "batched", "fps": 5})

    stages = client.get(f"/jobs/{job_id}/stages").json()
    extraction_stage = next(s for s in stages if s["stage"] == "frame_extraction")
    assert extraction_stage["status"] == "done"
    # A real run must have reported at least one live percentage along the way.
    assert extraction_stage["progress_percent"] is not None

    client.post(f"/jobs/{job_id}/align")
    stages = client.get(f"/jobs/{job_id}/stages").json()
    alignment_stage = next(s for s in stages if s["stage"] == "alignment")
    # Same caveat as the alignment end-to-end test: synthetic footage has no
    # real parallax, so RealityScan may legitimately fail to reconstruct
    # anything - this test verifies progress reporting and a terminal state,
    # not reconstruction quality.
    assert alignment_stage["status"] in ("done", "failed")
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_progress_end_to_end.py -v`
Expected: PASS on this machine (all three tools are installed) — `extraction_stage["progress_percent"]` must be a real number, not `None`, proving the Popen-based reader thread actually parsed live `tqdm` output during a real run, not just in the mocked unit tests.

- [ ] **Step 3: Run the full suite one more time**

Run: `python -m pytest -v`
Expected: all tests pass (or pass+skip cleanly), zero failures

- [ ] **Step 4: Commit**

```bash
git add tests/test_progress_end_to_end.py
git commit -m "test: add end-to-end progress-reporting verification against real binaries"
```
