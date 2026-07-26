from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.job_store import JobStore
from app.paths import raw_dir

_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mts", ".m2ts"})


class IngestionError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_raw_videos(job_id: str) -> list[Path]:
    directory = raw_dir(job_id)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS)


def import_local_file(store: JobStore, job_id: str, source_path: Path) -> Path:
    if not source_path.is_file():
        message = f"Source file not found: {source_path}"
        store.upsert_stage(job_id, "ingestion", "failed", started_at=_now(), finished_at=_now(), error=message)
        raise IngestionError(message)

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
