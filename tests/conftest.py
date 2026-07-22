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
