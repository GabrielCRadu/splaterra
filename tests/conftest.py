from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app import db
from app.job_store import JobStore


def pytest_configure(config):
    """Ensure Python Scripts directory is on PATH for console scripts like sharp-frames."""
    if sys.platform == "win32":
        import site

        # For user installations, scripts go to the user Scripts directory
        if site.ENABLE_USER_SITE:
            user_scripts = Path(site.USER_SITE).parent / "Scripts"
            if user_scripts.exists():
                if str(user_scripts) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{user_scripts}{os.pathsep}{os.environ.get('PATH', '')}"


@pytest.fixture
def job_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATERRA_DATA_DIR", str(tmp_path))
    conn = db.connect(tmp_path / "test.db")
    return JobStore(conn)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATERRA_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
