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
