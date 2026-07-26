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


def alignment_dir(job_id: str) -> Path:
    return job_dir(job_id) / "alignment"


def training_dir(job_id: str) -> Path:
    return job_dir(job_id) / "training"


def upload_chunks_dir(job_id: str, upload_id: str) -> Path:
    return job_dir(job_id) / "uploads" / upload_id / "chunks"
