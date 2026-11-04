from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app import db, process_registry
from app.job_store import JobStore


def get_job_store() -> Iterator[JobStore]:
    conn = db.connect()
    try:
        yield JobStore(conn)
    finally:
        conn.close()


def claim_stage_or_409(store: JobStore, job_id: str, stage: str, message: str) -> None:
    # Shared by every "start a stage" endpoint (extract/import-frames/align/
    # train/export): atomically claims (job_id, stage) as "running" via
    # JobStore.try_start_stage() instead of a separate read-then-schedule
    # check, which left a window where two near-simultaneous requests could
    # both read a non-running status before either background task wrote
    # "running" itself, letting both get scheduled and silently orphaning the
    # first from process_registry/cancellation.
    now = datetime.now(timezone.utc).isoformat()
    if store.try_start_stage(job_id, stage, now):
        return
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    raise HTTPException(status_code=409, detail=message)


def cancel_or_force_fail(store: JobStore, job_id: str, stage: str) -> bool:
    # Shared by every cancel endpoint. process_registry is in-memory and gets
    # wiped on every server restart - if the app (or the whole machine) was
    # restarted while a stage was "running", nothing ever transitions that
    # row away from "running" again, permanently stuck with no way to retry
    # or even delete the job (delete_job refuses while any stage shows
    # "running"). Real incident: an "ingestion" stage orphaned exactly like
    # this required a manual DB fix. process_registry.cancel() only handles
    # the case where a live process is actually registered; when it finds
    # nothing, treat a still-"running" stage as orphaned and force it to
    # "failed" instead of leaving it stuck. (ingestion/export never register
    # a process at all - copy2/chunk-reassembly and the blocking convert
    # subprocess.run aren't tracked - so for those two stages this is always
    # the only path that can ever clear a stuck "running" row.)
    if process_registry.cancel(job_id, stage):
        return True
    current = store.get_stage(job_id, stage)
    if current is not None and current.status == "running":
        store.upsert_stage(
            job_id,
            stage,
            "failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=(
                "Etapa a fost oprita manual (niciun proces activ gasit - probabil "
                "intrerupta de un restart de server)."
            ),
        )
        return True
    return False


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
