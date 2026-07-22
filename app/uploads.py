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
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", "..") or safe_name != filename:
        raise UploadError(f"Invalid filename: {filename!r}")

    upload_id = str(uuid.uuid4())
    chunks_dir = upload_chunks_dir(job_id, upload_id)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(job_id, upload_id).write_text(
        json.dumps({"filename": safe_name, "total_chunks": total_chunks})
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
        error_msg = f"Unknown upload: {upload_id}"
        now = datetime.now(timezone.utc).isoformat()
        store.upsert_stage(
            job_id, "ingestion", "failed", started_at=now, finished_at=now, error=error_msg
        )
        raise UploadError(error_msg)
    meta = json.loads(meta_path.read_text())
    filename: str = meta["filename"]
    total_chunks: int = meta["total_chunks"]

    chunks_dir = upload_chunks_dir(job_id, upload_id)
    have = received_chunks(job_id, upload_id)
    expected = list(range(total_chunks))
    if have != expected:
        missing = sorted(set(expected) - set(have))
        now = datetime.now(timezone.utc).isoformat()
        error_msg = f"Missing chunks: {missing}"
        store.upsert_stage(
            job_id, "ingestion", "failed", started_at=now, finished_at=now, error=error_msg
        )
        raise UploadError(error_msg)

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
