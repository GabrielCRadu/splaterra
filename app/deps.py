from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from app import db
from app.job_store import JobStore


def get_job_store() -> Iterator[JobStore]:
    conn = db.connect()
    try:
        yield JobStore(conn)
    finally:
        conn.close()


def run_with_new_store(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    conn = db.connect()
    store = JobStore(conn)
    try:
        func(store, *args, **kwargs)
    except Exception:
        # Service functions already record failures into job_stages before
        # raising; a background task has no caller to propagate the
        # exception to, so swallow it here — the job's stage status (polled
        # via GET /jobs/{id}/stages) is the UI's source of truth.
        logging.exception("Background task failed: %s", func.__name__)
    finally:
        conn.close()
