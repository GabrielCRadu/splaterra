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
    if Path(safe_name).suffix.lower() == ".osv":
        # .OSV is the raw dual-fisheye format DJI Osmo 360/Avata 360 write -
        # rejected before a single chunk is uploaded (cheaper than failing
        # after gigabytes went over the wire) since this app only supports
        # already-stitched equirectangular footage in Phase 1 (see CLAUDE.md).
        raise UploadError(
            "Fisierele .OSV (dual-fisheye brut) nu sunt suportate. Stitch-uieste clipul in "
            "DJI Studio (export echirectangular) inainte sa il incarci aici."
        )

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
    # Real incident: reassembling a large chunked upload (8.92GB, 8K 360
    # footage) ran the destination drive out of space partway through -
    # output.write() raised OSError with nothing here to catch it, so the
    # exception propagated uncaught into run_with_new_store's broad
    # try/except, which only logs it server-side. The stage was left stuck
    # on "running" forever with zero indication to the user of what
    # happened. import_local_file() already guards its copy this way;
    # complete_upload()'s chunk-reassembly loop didn't.
    try:
        with destination.open("wb") as output:
            for index in expected:
                chunk_path = chunks_dir / f"{index:06d}.part"
                output.write(chunk_path.read_bytes())
    except OSError as exc:
        store.upsert_stage(
            job_id, "ingestion", "failed", finished_at=datetime.now(timezone.utc).isoformat(),
            error=f"Failed to reassemble upload: {exc}",
        )
        raise UploadError(f"Failed to reassemble upload: {exc}") from exc

    store.update_job(job_id, raw_video_path=str(destination))
    store.upsert_stage(
        job_id, "ingestion", "done", finished_at=datetime.now(timezone.utc).isoformat()
    )

    for chunk_path in chunks_dir.glob("*.part"):
        chunk_path.unlink()
    meta_path.unlink()
    chunks_dir.rmdir()

    return destination
