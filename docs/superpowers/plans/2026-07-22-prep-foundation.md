# Prep Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational service for the Gaussian Splatting prep app: a FastAPI backend with a SQLite-backed job store, video ingestion (local import + chunked/resumable upload), and sharp-frame extraction — producing a working, testable "upload a video, get sharp frames back" tool.

**Architecture:** A single Python package `app/` exposes a FastAPI application. Each concern (paths, DB, job store, ingestion, uploads, frame extraction) is its own module with one responsibility; API routers under `app/api/` are thin wrappers that call these modules. Job and per-stage state persist in SQLite so the service can restart without losing progress. External tools (`sharp-frames`) are invoked as subprocesses and tested via mocks, never actually run in unit tests.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, SQLite (stdlib `sqlite3`), pytest, httpx (via FastAPI `TestClient`), `sharp-frames` CLI (`pip install sharp-frames`, requires `ffmpeg`/`ffprobe` on PATH).

## Global Constraints

- Backend is Python + FastAPI; job/stage state persists in SQLite as a single file (spec §6).
- Video/frame/output files stay on local disk in per-job folders, never duplicated needlessly (spec §6).
- No artificial file-size limit on video ingestion — upload is chunked and resumable (spec §2, Etapa 0).
- Each pipeline stage is isolated: on failure the job stays in a `failed` state with the error visible, and only the failed stage is retried, not earlier completed stages (spec §4, Gestionare erori).
- External tools (`sharp-frames` here; RealityScan/LichtFeld in later plans) are always invoked as subprocesses and tested through subprocess mocks, since there is no official test CLI (spec §9).

---

## Task 1: Project scaffolding + health endpoint

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Produces: FastAPI app instance importable as `app.main.app`, with `GET /health` returning `{"status": "ok"}`. Later tasks call `app.include_router(...)` on this instance.

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.9
pytest>=8.3
httpx>=0.27
python-multipart>=0.0.12
sharp-frames>=1.0
```

- [ ] **Step 2: Install dependencies**

Run: `python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt`
Expected: packages install without errors (on Windows, activate via `.venv\Scripts\activate` for interactive shells; the direct `.venv/Scripts/python` path works from scripts/CI).

- [ ] **Step 3: Create empty package marker**

`app/__init__.py`:
```python
```

- [ ] **Step 4: Write the failing test**

`tests/test_main.py`:
```python
from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 5: Run test to verify it fails**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 6: Write minimal implementation**

`app/main.py`:
```python
from fastapi import FastAPI

app = FastAPI(title="Splat Prep")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS (1 passed)

- [ ] **Step 8: Commit**

```bash
git add requirements.txt app/__init__.py app/main.py tests/test_main.py
git commit -m "feat: scaffold FastAPI app with health endpoint"
```

---

## Task 2: Path helpers module

**Files:**
- Create: `app/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `data_root() -> Path`, `job_dir(job_id: str) -> Path`, `raw_dir(job_id: str) -> Path`, `frames_dir(job_id: str) -> Path`, `upload_chunks_dir(job_id: str, upload_id: str) -> Path`. Root directory controlled by env var `SPLATPREP_DATA_DIR` (default `./data`). Every later task that touches disk or the DB file uses these.

- [ ] **Step 1: Write the failing test**

`tests/test_paths.py`:
```python
from app import paths


def test_job_dir_uses_data_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATPREP_DATA_DIR", str(tmp_path))
    assert paths.job_dir("abc123") == tmp_path.resolve() / "jobs" / "abc123"
    assert paths.raw_dir("abc123") == tmp_path.resolve() / "jobs" / "abc123" / "raw"
    assert paths.frames_dir("abc123") == tmp_path.resolve() / "jobs" / "abc123" / "frames"
    assert paths.upload_chunks_dir("abc123", "up1") == (
        tmp_path.resolve() / "jobs" / "abc123" / "uploads" / "up1" / "chunks"
    )


def test_data_root_defaults_to_relative_data_dir(monkeypatch):
    monkeypatch.delenv("SPLATPREP_DATA_DIR", raising=False)
    from pathlib import Path

    assert paths.data_root() == Path("./data").resolve()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.paths'`

- [ ] **Step 3: Write minimal implementation**

`app/paths.py`:
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


def upload_chunks_dir(job_id: str, upload_id: str) -> Path:
    return job_dir(job_id) / "uploads" / upload_id / "chunks"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paths.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/paths.py tests/test_paths.py
git commit -m "feat: add job data-path helpers"
```

---

## Task 3: SQLite schema + JobStore + shared test fixtures

**Files:**
- Create: `app/db.py`
- Create: `app/models.py`
- Create: `app/job_store.py`
- Create: `tests/conftest.py`
- Test: `tests/test_db.py`
- Test: `tests/test_job_store.py`

**Interfaces:**
- Consumes: `app.paths.data_root` (Task 2).
- Produces: `db.connect(path: Path | None = None) -> sqlite3.Connection` (schema auto-created); `db.db_path() -> Path`. Dataclasses `Job(id, name, created_at, raw_video_path, frames_dir, frame_count)` and `JobStage(job_id, stage, status, started_at, finished_at, log, error)`. `JobStore` with methods `create_job(name) -> Job`, `get_job(job_id) -> Job | None`, `list_jobs() -> list[Job]`, `update_job(job_id, **fields) -> None`, `upsert_stage(job_id, stage, status, started_at=None, finished_at=None, log=None, error=None) -> None`, `get_stage(job_id, stage) -> JobStage | None`. Pytest fixtures `job_store` and `client` in `tests/conftest.py`, used by every later test file.

- [ ] **Step 1: Write the failing test for the schema**

`tests/test_db.py`:
```python
from app import db


def test_connect_creates_schema(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"jobs", "job_stages"} <= tables
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db'`

- [ ] **Step 3: Write minimal implementation**

`app/db.py`:
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
    frame_count INTEGER
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

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_db.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Add shared pytest fixtures**

`tests/conftest.py`:
```python
from __future__ import annotations

import pytest

from app import db
from app.job_store import JobStore


@pytest.fixture
def job_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATPREP_DATA_DIR", str(tmp_path))
    conn = db.connect(tmp_path / "test.db")
    return JobStore(conn)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATPREP_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
```

- [ ] **Step 6: Write the failing tests for JobStore**

`tests/test_job_store.py`:
```python
def test_create_and_get_job(job_store):
    job = job_store.create_job("Curte interioara")
    fetched = job_store.get_job(job.id)
    assert fetched is not None
    assert fetched.name == "Curte interioara"
    assert fetched.raw_video_path is None


def test_list_jobs_returns_newest_first(job_store):
    first = job_store.create_job("Primul")
    second = job_store.create_job("Al doilea")
    jobs = job_store.list_jobs()
    assert [job.id for job in jobs] == [second.id, first.id]


def test_update_job_sets_fields(job_store):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path="/data/raw/video.mp4", frame_count=42)
    fetched = job_store.get_job(job.id)
    assert fetched.raw_video_path == "/data/raw/video.mp4"
    assert fetched.frame_count == 42


def test_upsert_stage_creates_then_updates(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "ingestion", "running", started_at="t0")
    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "running"
    assert stage.started_at == "t0"

    job_store.upsert_stage(job.id, "ingestion", "done", finished_at="t1")
    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "done"
    assert stage.started_at == "t0"
    assert stage.finished_at == "t1"
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `python -m pytest tests/test_job_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.job_store'`

- [ ] **Step 8: Write `app/models.py`**

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

- [ ] **Step 9: Write `app/job_store.py`**

```python
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from app.models import Job, JobStage


class JobStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create_job(self, name: str) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._conn.execute(
            "INSERT INTO jobs (id, name, created_at) VALUES (?, ?, ?)",
            (job.id, job.name, job.created_at),
        )
        self._conn.commit()
        return job

    def get_job(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self) -> list[Job]:
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def update_job(self, job_id: str, **fields: object) -> None:
        if not fields:
            return
        columns = ", ".join(f"{key} = ?" for key in fields)
        values = [*fields.values(), job_id]
        self._conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", values)
        self._conn.commit()

    def upsert_stage(
        self,
        job_id: str,
        stage: str,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        log: str | None = None,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO job_stages (job_id, stage, status, started_at, finished_at, log, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, stage) DO UPDATE SET
                status = excluded.status,
                started_at = COALESCE(excluded.started_at, job_stages.started_at),
                finished_at = excluded.finished_at,
                log = excluded.log,
                error = excluded.error
            """,
            (job_id, stage, status, started_at, finished_at, log, error),
        )
        self._conn.commit()

    def get_stage(self, job_id: str, stage: str) -> JobStage | None:
        row = self._conn.execute(
            "SELECT * FROM job_stages WHERE job_id = ? AND stage = ?",
            (job_id, stage),
        ).fetchone()
        return self._row_to_stage(row) if row else None

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            raw_video_path=row["raw_video_path"],
            frames_dir=row["frames_dir"],
            frame_count=row["frame_count"],
        )

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
        )
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `python -m pytest tests/test_db.py tests/test_job_store.py -v`
Expected: PASS (5 passed)

- [ ] **Step 11: Commit**

```bash
git add app/db.py app/models.py app/job_store.py tests/conftest.py tests/test_db.py tests/test_job_store.py
git commit -m "feat: add SQLite schema and JobStore"
```

---

## Task 4: Jobs API (create / get / list)

**Files:**
- Create: `app/deps.py`
- Create: `app/api/__init__.py`
- Create: `app/api/jobs.py`
- Modify: `app/main.py`
- Test: `tests/test_api_jobs.py`

**Interfaces:**
- Consumes: `JobStore` (Task 3).
- Produces: FastAPI dependency `get_job_store()` (yields a `JobStore`, closes its connection after the request); router mounted at `/jobs` with `POST /jobs`, `GET /jobs/{job_id}`, `GET /jobs`, all returning `JobResponse` (`id, name, created_at, raw_video_path, frames_dir, frame_count`). Later tasks depend on `get_job_store` and reuse the `/jobs` prefix.

- [ ] **Step 1: Write the failing test**

`tests/test_api_jobs.py`:
```python
def test_create_and_get_job(client):
    create_response = client.post("/jobs", json={"name": "Curte interioara"})
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["name"] == "Curte interioara"
    assert created["frame_count"] is None

    get_response = client.get(f"/jobs/{created['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == created["id"]


def test_get_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_list_jobs_returns_created_jobs(client):
    client.post("/jobs", json={"name": "A"})
    client.post("/jobs", json={"name": "B"})

    response = client.get("/jobs")
    assert response.status_code == 200
    names = {job["name"] for job in response.json()}
    assert names == {"A", "B"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_jobs.py -v`
Expected: FAIL with `404 Not Found` on `POST /jobs` (route doesn't exist yet)

- [ ] **Step 3: Write `app/deps.py`**

```python
from __future__ import annotations

from collections.abc import Iterator

from app import db
from app.job_store import JobStore


def get_job_store() -> Iterator[JobStore]:
    conn = db.connect()
    try:
        yield JobStore(conn)
    finally:
        conn.close()
```

- [ ] **Step 4: Create `app/api/__init__.py`**

```python
```

- [ ] **Step 5: Write `app/api/jobs.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["jobs"])


class CreateJobRequest(BaseModel):
    name: str


class JobResponse(BaseModel):
    id: str
    name: str
    created_at: str
    raw_video_path: str | None
    frames_dir: str | None
    frame_count: int | None


@router.post("", response_model=JobResponse)
def create_job(
    body: CreateJobRequest, store: JobStore = Depends(get_job_store)
) -> JobResponse:
    job = store.create_job(body.name)
    return JobResponse(**job.__dict__)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> JobResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**job.__dict__)


@router.get("", response_model=list[JobResponse])
def list_jobs(store: JobStore = Depends(get_job_store)) -> list[JobResponse]:
    return [JobResponse(**job.__dict__) for job in store.list_jobs()]
```

- [ ] **Step 6: Wire the router into the app**

`app/main.py`:
```python
from fastapi import FastAPI

from app.api.jobs import router as jobs_router

app = FastAPI(title="Splat Prep")
app.include_router(jobs_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_jobs.py -v`
Expected: PASS (3 passed)

- [ ] **Step 8: Commit**

```bash
git add app/deps.py app/api/__init__.py app/api/jobs.py app/main.py tests/test_api_jobs.py
git commit -m "feat: add jobs API (create, get, list)"
```

---

## Task 5: Local video import

**Files:**
- Create: `app/ingestion.py`
- Create: `app/api/ingestion.py`
- Modify: `app/main.py`
- Test: `tests/test_ingestion.py`
- Test: `tests/test_api_ingestion_local.py`

**Interfaces:**
- Consumes: `JobStore` (Task 3), `paths.raw_dir` (Task 2), `get_job_store` (Task 4).
- Produces: `import_local_file(store: JobStore, job_id: str, source_path: Path) -> Path` and `IngestionError`. Sets `job.raw_video_path` and the `ingestion` stage to `done` (or `failed`). API: `POST /jobs/{job_id}/ingest/local` with body `{"source_path": str}`, returns `{"raw_video_path": str}`. Task 6 extends `app/api/ingestion.py` with upload routes; Task 8 relies on `job.raw_video_path` being set.

- [ ] **Step 1: Write the failing tests**

`tests/test_ingestion.py`:
```python
import pytest

from app.ingestion import IngestionError, import_local_file


def test_import_local_file_copies_and_records_job(job_store, tmp_path):
    job = job_store.create_job("Test")

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video bytes")

    destination = import_local_file(job_store, job.id, source)

    assert destination.exists()
    assert destination.read_bytes() == b"fake video bytes"

    fetched = job_store.get_job(job.id)
    assert fetched.raw_video_path == str(destination)

    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "done"


def test_import_local_file_missing_source_raises(job_store, tmp_path):
    job = job_store.create_job("Test")

    with pytest.raises(IngestionError):
        import_local_file(job_store, job.id, tmp_path / "missing.mp4")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ingestion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion'`

- [ ] **Step 3: Write `app/ingestion.py`**

```python
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import raw_dir


class IngestionError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_local_file(store: JobStore, job_id: str, source_path: Path) -> Path:
    if not source_path.is_file():
        raise IngestionError(f"Source file not found: {source_path}")

    destination_dir = raw_dir(job_id)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name

    store.upsert_stage(job_id, "ingestion", "running", started_at=_now())
    try:
        shutil.copy2(source_path, destination)
    except OSError as exc:
        store.upsert_stage(job_id, "ingestion", "failed", finished_at=_now(), error=str(exc))
        raise IngestionError(f"Failed to copy {source_path}: {exc}") from exc

    store.update_job(job_id, raw_video_path=str(destination))
    store.upsert_stage(job_id, "ingestion", "done", finished_at=_now())
    return destination
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ingestion.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the failing API test**

`tests/test_api_ingestion_local.py`:
```python
def test_ingest_local_sets_raw_video_path(client, tmp_path):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake video bytes")

    response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(source)}
    )
    assert response.status_code == 200
    assert response.json()["raw_video_path"].endswith("video.mp4")

    job = client.get(f"/jobs/{job_id}").json()
    assert job["raw_video_path"] is not None


def test_ingest_local_unknown_job_returns_404(client, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"data")
    response = client.post(
        "/jobs/does-not-exist/ingest/local", json={"source_path": str(source)}
    )
    assert response.status_code == 404


def test_ingest_local_missing_source_returns_400(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": "/no/such/file.mp4"}
    )
    assert response.status_code == 400
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_api_ingestion_local.py -v`
Expected: FAIL with `404 Not Found` on `POST /jobs/{job_id}/ingest/local` (route doesn't exist yet)

- [ ] **Step 7: Write `app/api/ingestion.py`**

```python
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store
from app.ingestion import IngestionError, import_local_file
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["ingestion"])


class LocalImportRequest(BaseModel):
    source_path: str


@router.post("/{job_id}/ingest/local")
def ingest_local(
    job_id: str,
    body: LocalImportRequest,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        destination = import_local_file(store, job_id, Path(body.source_path))
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"raw_video_path": str(destination)}
```

- [ ] **Step 8: Wire the router into the app**

`app/main.py`:
```python
from fastapi import FastAPI

from app.api.ingestion import router as ingestion_router
from app.api.jobs import router as jobs_router

app = FastAPI(title="Splat Prep")
app.include_router(jobs_router)
app.include_router(ingestion_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_ingestion_local.py -v`
Expected: PASS (3 passed)

- [ ] **Step 10: Commit**

```bash
git add app/ingestion.py app/api/ingestion.py app/main.py tests/test_ingestion.py tests/test_api_ingestion_local.py
git commit -m "feat: add local video import"
```

---

## Task 6: Chunked, resumable upload

**Files:**
- Create: `app/uploads.py`
- Modify: `app/api/ingestion.py`
- Test: `tests/test_uploads.py`
- Test: `tests/test_api_ingestion_upload.py`

**Interfaces:**
- Consumes: `JobStore` (Task 3), `paths.job_dir` / `paths.raw_dir` / `paths.upload_chunks_dir` (Task 2).
- Produces: `start_upload(job_id, filename, total_chunks) -> str` (returns `upload_id`), `save_chunk(job_id, upload_id, chunk_index, data: bytes) -> None`, `received_chunks(job_id, upload_id) -> list[int]`, `complete_upload(store, job_id, upload_id) -> Path`, `UploadError`. API: `POST /jobs/{job_id}/uploads`, `PUT /jobs/{job_id}/uploads/{upload_id}/chunks/{chunk_index}`, `GET /jobs/{job_id}/uploads/{upload_id}/chunks`, `POST /jobs/{job_id}/uploads/{upload_id}/complete`.

- [ ] **Step 1: Write the failing tests**

`tests/test_uploads.py`:
```python
import pytest

from app.uploads import (
    UploadError,
    complete_upload,
    received_chunks,
    save_chunk,
    start_upload,
)


def test_upload_round_trip(job_store):
    job = job_store.create_job("Test")

    upload_id = start_upload(job.id, "video.mp4", total_chunks=3)
    save_chunk(job.id, upload_id, 0, b"AAA")
    save_chunk(job.id, upload_id, 2, b"CCC")
    save_chunk(job.id, upload_id, 1, b"BBB")

    assert received_chunks(job.id, upload_id) == [0, 1, 2]

    destination = complete_upload(job_store, job.id, upload_id)
    assert destination.read_bytes() == b"AAABBBCCC"

    fetched = job_store.get_job(job.id)
    assert fetched.raw_video_path == str(destination)


def test_complete_upload_fails_when_chunks_missing(job_store):
    job = job_store.create_job("Test")

    upload_id = start_upload(job.id, "video.mp4", total_chunks=2)
    save_chunk(job.id, upload_id, 0, b"AAA")

    with pytest.raises(UploadError):
        complete_upload(job_store, job.id, upload_id)


def test_save_chunk_unknown_upload_raises(job_store):
    job = job_store.create_job("Test")
    with pytest.raises(UploadError):
        save_chunk(job.id, "no-such-upload", 0, b"AAA")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_uploads.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.uploads'`

- [ ] **Step 3: Write `app/uploads.py`**

```python
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import job_dir, raw_dir, upload_chunks_dir


class UploadError(Exception):
    pass


def _upload_dir(job_id: str, upload_id: str) -> Path:
    return job_dir(job_id) / "uploads" / upload_id


def _meta_path(job_id: str, upload_id: str) -> Path:
    return _upload_dir(job_id, upload_id) / "meta.json"


def start_upload(job_id: str, filename: str, total_chunks: int) -> str:
    upload_id = str(uuid.uuid4())
    chunks_dir = upload_chunks_dir(job_id, upload_id)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(job_id, upload_id).write_text(
        json.dumps({"filename": filename, "total_chunks": total_chunks})
    )
    return upload_id


def save_chunk(job_id: str, upload_id: str, chunk_index: int, data: bytes) -> None:
    chunks_dir = upload_chunks_dir(job_id, upload_id)
    if not chunks_dir.exists():
        raise UploadError(f"Unknown upload: {upload_id}")
    chunk_path = chunks_dir / f"{chunk_index:06d}.part"
    chunk_path.write_bytes(data)


def received_chunks(job_id: str, upload_id: str) -> list[int]:
    chunks_dir = upload_chunks_dir(job_id, upload_id)
    if not chunks_dir.exists():
        raise UploadError(f"Unknown upload: {upload_id}")
    return sorted(int(path.stem) for path in chunks_dir.glob("*.part"))


def complete_upload(store: JobStore, job_id: str, upload_id: str) -> Path:
    meta_path = _meta_path(job_id, upload_id)
    if not meta_path.exists():
        raise UploadError(f"Unknown upload: {upload_id}")
    meta = json.loads(meta_path.read_text())
    filename: str = meta["filename"]
    total_chunks: int = meta["total_chunks"]

    chunks_dir = upload_chunks_dir(job_id, upload_id)
    have = received_chunks(job_id, upload_id)
    expected = list(range(total_chunks))
    if have != expected:
        missing = sorted(set(expected) - set(have))
        raise UploadError(f"Missing chunks: {missing}")

    destination_dir = raw_dir(job_id)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename

    store.upsert_stage(
        job_id, "ingestion", "running", started_at=datetime.now(timezone.utc).isoformat()
    )
    with destination.open("wb") as output:
        for index in expected:
            chunk_path = chunks_dir / f"{index:06d}.part"
            output.write(chunk_path.read_bytes())

    store.update_job(job_id, raw_video_path=str(destination))
    store.upsert_stage(
        job_id, "ingestion", "done", finished_at=datetime.now(timezone.utc).isoformat()
    )

    for chunk_path in chunks_dir.glob("*.part"):
        chunk_path.unlink()
    meta_path.unlink()
    chunks_dir.rmdir()

    return destination
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_uploads.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the failing API tests**

`tests/test_api_ingestion_upload.py`:
```python
def test_upload_round_trip_via_api(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    upload_id = client.post(
        f"/jobs/{job_id}/uploads", json={"filename": "video.mp4", "total_chunks": 2}
    ).json()["upload_id"]

    put_response = client.put(
        f"/jobs/{job_id}/uploads/{upload_id}/chunks/0", content=b"AAA"
    )
    assert put_response.status_code == 200
    client.put(f"/jobs/{job_id}/uploads/{upload_id}/chunks/1", content=b"BBB")

    chunks_response = client.get(f"/jobs/{job_id}/uploads/{upload_id}/chunks")
    assert chunks_response.json() == {"received_chunks": [0, 1]}

    complete_response = client.post(f"/jobs/{job_id}/uploads/{upload_id}/complete")
    assert complete_response.status_code == 200

    job = client.get(f"/jobs/{job_id}").json()
    assert job["raw_video_path"] is not None


def test_complete_upload_missing_chunks_returns_400(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    upload_id = client.post(
        f"/jobs/{job_id}/uploads", json={"filename": "video.mp4", "total_chunks": 2}
    ).json()["upload_id"]
    client.put(f"/jobs/{job_id}/uploads/{upload_id}/chunks/0", content=b"AAA")

    response = client.post(f"/jobs/{job_id}/uploads/{upload_id}/complete")
    assert response.status_code == 400
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_api_ingestion_upload.py -v`
Expected: FAIL with `404 Not Found` on `POST /jobs/{job_id}/uploads` (route doesn't exist yet)

- [ ] **Step 7: Extend `app/api/ingestion.py` with upload routes**

Replace the full contents of `app/api/ingestion.py` (this supersedes the Task 5 version — it keeps the existing `ingest_local` route and adds the upload routes) with:

```python
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.deps import get_job_store
from app.ingestion import IngestionError, import_local_file
from app.job_store import JobStore
from app.uploads import UploadError, complete_upload, received_chunks, save_chunk, start_upload

router = APIRouter(prefix="/jobs", tags=["ingestion"])


class LocalImportRequest(BaseModel):
    source_path: str


class StartUploadRequest(BaseModel):
    filename: str
    total_chunks: int


@router.post("/{job_id}/ingest/local")
def ingest_local(
    job_id: str,
    body: LocalImportRequest,
    store: JobStore = Depends(get_job_store),
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        destination = import_local_file(store, job_id, Path(body.source_path))
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"raw_video_path": str(destination)}


@router.post("/{job_id}/uploads")
def create_upload(
    job_id: str, body: StartUploadRequest, store: JobStore = Depends(get_job_store)
) -> dict[str, str]:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    upload_id = start_upload(job_id, body.filename, body.total_chunks)
    return {"upload_id": upload_id}


@router.put("/{job_id}/uploads/{upload_id}/chunks/{chunk_index}")
async def upload_chunk(job_id: str, upload_id: str, chunk_index: int, request: Request) -> dict[str, int]:
    data = await request.body()
    try:
        save_chunk(job_id, upload_id, chunk_index, data)
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"received": chunk_index}


@router.get("/{job_id}/uploads/{upload_id}/chunks")
def list_received_chunks(job_id: str, upload_id: str) -> dict[str, list[int]]:
    try:
        return {"received_chunks": received_chunks(job_id, upload_id)}
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{job_id}/uploads/{upload_id}/complete")
def finish_upload(
    job_id: str, upload_id: str, store: JobStore = Depends(get_job_store)
) -> dict[str, str]:
    try:
        destination = complete_upload(store, job_id, upload_id)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"raw_video_path": str(destination)}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_ingestion_upload.py -v`
Expected: PASS (2 passed)

- [ ] **Step 9: Run the full test suite to check for regressions**

Run: `python -m pytest -v`
Expected: all tests pass (16 passed)

- [ ] **Step 10: Commit**

```bash
git add app/uploads.py app/api/ingestion.py tests/test_uploads.py tests/test_api_ingestion_upload.py
git commit -m "feat: add chunked resumable upload"
```

---

## Task 7: `sharp-frames` CLI wrapper

**Files:**
- Create: `app/sharp_frames.py`
- Test: `tests/test_sharp_frames.py`

**Interfaces:**
- Consumes: nothing (pure subprocess wrapper).
- Produces: `run_extraction(video_path: Path, output_dir: Path, *, selection_method: str = "best-n", num_frames: int = 300, fps: int = 10) -> ExtractionResult` (dataclass with `frames: list[Path]`, `log: str`), and `SharpFramesError`. Task 8 calls this directly.

- [ ] **Step 1: Write the failing tests**

`tests/test_sharp_frames.py`:
```python
from unittest.mock import MagicMock

import pytest

from app import sharp_frames


def test_run_extraction_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: None)
    with pytest.raises(sharp_frames.SharpFramesError, match="not found on PATH"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_extraction_returns_frames_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    output_dir = tmp_path / "out"

    def fake_run(command, capture_output, text, check):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        (output_dir / "frame_0002.jpg").write_bytes(b"fake")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "extracted 2 frames"
        result.stderr = ""
        return result

    monkeypatch.setattr(sharp_frames.subprocess, "run", fake_run)

    result = sharp_frames.run_extraction(tmp_path / "video.mp4", output_dir)

    assert len(result.frames) == 2
    assert "extracted 2 frames" in result.log


def test_run_extraction_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    def fake_run(command, capture_output, text, check):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "ffmpeg not found"
        return result

    monkeypatch.setattr(sharp_frames.subprocess, "run", fake_run)

    with pytest.raises(sharp_frames.SharpFramesError, match="ffmpeg not found"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sharp_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.sharp_frames'`

- [ ] **Step 3: Write `app/sharp_frames.py`**

```python
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def run_extraction(
    video_path: Path,
    output_dir: Path,
    *,
    selection_method: str = "best-n",
    num_frames: int = 300,
    fps: int = 10,
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

    process = subprocess.run(command, capture_output=True, text=True, check=False)
    log = process.stdout + process.stderr

    if process.returncode != 0:
        raise SharpFramesError(f"sharp-frames exited with {process.returncode}:\n{log}")

    frames = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))
    if not frames:
        raise SharpFramesError(f"sharp-frames produced no frames:\n{log}")

    return ExtractionResult(frames=frames, log=log)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sharp_frames.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/sharp_frames.py tests/test_sharp_frames.py
git commit -m "feat: add sharp-frames CLI wrapper"
```

---

## Task 8: Frame extraction service + API

**Files:**
- Create: `app/extraction.py`
- Create: `app/api/frames.py`
- Modify: `app/main.py`
- Test: `tests/test_extraction.py`
- Test: `tests/test_api_frames.py`

**Interfaces:**
- Consumes: `JobStore` (Task 3), `paths.frames_dir` (Task 2), `sharp_frames.run_extraction` / `ExtractionResult` / `SharpFramesError` (Task 7), `get_job_store` (Task 4).
- Produces: `extract_frames(store: JobStore, job_id: str, *, selection_method="best-n", num_frames=300, fps=10) -> int` (returns frame count, updates `job.frames_dir`/`job.frame_count`, sets `frame_extraction` stage) and `ExtractionError`. API: `POST /jobs/{job_id}/extract-frames` returns `{"frame_count": int}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_extraction.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app import extraction
from app.sharp_frames import ExtractionResult, SharpFramesError


def test_extract_frames_updates_job_and_stage(job_store, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path="/data/raw/video.mp4")

    fake_frames = [Path("f1.jpg"), Path("f2.jpg"), Path("f3.jpg")]
    monkeypatch.setattr(
        extraction,
        "run_extraction",
        MagicMock(return_value=ExtractionResult(frames=fake_frames, log="ok")),
    )

    count = extraction.extract_frames(job_store, job.id)

    assert count == 3
    fetched = job_store.get_job(job.id)
    assert fetched.frame_count == 3
    assert fetched.frames_dir is not None

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.status == "done"


def test_extract_frames_without_ingested_video_raises(job_store):
    job = job_store.create_job("Test")
    with pytest.raises(extraction.ExtractionError, match="no ingested video"):
        extraction.extract_frames(job_store, job.id)


def test_extract_frames_marks_stage_failed_on_error(job_store, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path="/data/raw/video.mp4")

    monkeypatch.setattr(
        extraction, "run_extraction", MagicMock(side_effect=SharpFramesError("boom"))
    )

    with pytest.raises(extraction.ExtractionError, match="boom"):
        extraction.extract_frames(job_store, job.id)

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.status == "failed"
    assert stage.error == "boom"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.extraction'`

- [ ] **Step 3: Write `app/extraction.py`**

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import frames_dir
from app.sharp_frames import SharpFramesError, run_extraction


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
        raise ExtractionError(f"Unknown job: {job_id}")
    if job.raw_video_path is None:
        raise ExtractionError("Job has no ingested video yet")

    output_dir = frames_dir(job_id)
    store.upsert_stage(job_id, "frame_extraction", "running", started_at=_now())
    try:
        result = run_extraction(
            Path(job.raw_video_path),
            output_dir,
            selection_method=selection_method,
            num_frames=num_frames,
            fps=fps,
        )
    except SharpFramesError as exc:
        store.upsert_stage(job_id, "frame_extraction", "failed", finished_at=_now(), error=str(exc))
        raise ExtractionError(str(exc)) from exc

    store.update_job(job_id, frames_dir=str(output_dir), frame_count=len(result.frames))
    store.upsert_stage(job_id, "frame_extraction", "done", finished_at=_now(), log=result.log)
    return len(result.frames)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the failing API tests**

`tests/test_api_frames.py`:
```python
from app.api import frames as frames_api


def test_extract_frames_endpoint_returns_count(client, monkeypatch):
    monkeypatch.setattr(frames_api, "extract_frames", lambda store, job_id, **kw: 5)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/extract-frames", json={})
    assert response.status_code == 200
    assert response.json() == {"frame_count": 5}


def test_extract_frames_endpoint_returns_400_on_error(client, monkeypatch):
    def raise_error(store, job_id, **kw):
        raise frames_api.ExtractionError("no ingested video")

    monkeypatch.setattr(frames_api, "extract_frames", raise_error)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/extract-frames", json={})
    assert response.status_code == 400
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_api_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.frames'`

- [ ] **Step 7: Write `app/api/frames.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store
from app.extraction import ExtractionError, extract_frames
from app.job_store import JobStore

router = APIRouter(prefix="/jobs", tags=["frames"])


class ExtractFramesRequest(BaseModel):
    selection_method: str = "best-n"
    num_frames: int = 300
    fps: int = 10


@router.post("/{job_id}/extract-frames")
def extract_frames_endpoint(
    job_id: str,
    body: ExtractFramesRequest,
    store: JobStore = Depends(get_job_store),
) -> dict[str, int]:
    try:
        count = extract_frames(
            store,
            job_id,
            selection_method=body.selection_method,
            num_frames=body.num_frames,
            fps=body.fps,
        )
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"frame_count": count}
```

- [ ] **Step 8: Wire the router into the app**

`app/main.py`:
```python
from fastapi import FastAPI

from app.api.frames import router as frames_router
from app.api.ingestion import router as ingestion_router
from app.api.jobs import router as jobs_router

app = FastAPI(title="Splat Prep")
app.include_router(jobs_router)
app.include_router(ingestion_router)
app.include_router(frames_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_frames.py -v`
Expected: PASS (2 passed)

- [ ] **Step 10: Commit**

```bash
git add app/extraction.py app/api/frames.py app/main.py tests/test_extraction.py tests/test_api_frames.py
git commit -m "feat: add frame extraction service and API"
```

---

## Task 9: End-to-end integration test

**Files:**
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: the full app (`app.main.app`) and every module built in Tasks 1-8. No new production interfaces are produced — this task only verifies the pieces work together.

- [ ] **Step 1: Write the end-to-end test**

`tests/test_end_to_end.py`:
```python
import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")
SHARP_FRAMES = shutil.which("sharp-frames")


@pytest.mark.skipif(
    FFMPEG is None or SHARP_FRAMES is None,
    reason="requires ffmpeg and sharp-frames installed on PATH",
)
def test_full_ingest_and_extract_flow(client, tmp_path):
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

    job_id = client.post("/jobs", json={"name": "E2E"}).json()["id"]

    ingest_response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(video_path)}
    )
    assert ingest_response.status_code == 200

    extract_response = client.post(
        f"/jobs/{job_id}/extract-frames",
        json={"selection_method": "batched", "fps": 5},
    )
    assert extract_response.status_code == 200
    assert extract_response.json()["frame_count"] > 0

    job = client.get(f"/jobs/{job_id}").json()
    assert job["frame_count"] > 0
    assert job["frames_dir"] is not None
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_end_to_end.py -v`
Expected: PASS if `ffmpeg` and `sharp-frames` are on PATH (1 passed); SKIPPED otherwise (1 skipped) — never a failure either way.

- [ ] **Step 3: Run the full suite one more time**

Run: `python -m pytest -v`
Expected: all tests pass or skip cleanly, no failures (24 passed, or 23 passed + 1 skipped if `sharp-frames`/`ffmpeg` are missing)

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: add end-to-end ingest-and-extract integration test"
```
