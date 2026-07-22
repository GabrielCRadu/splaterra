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


def test_extract_frames_without_ingested_video_records_failed_stage(job_store):
    job = job_store.create_job("Test")
    with pytest.raises(extraction.ExtractionError, match="no ingested video"):
        extraction.extract_frames(job_store, job.id)

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage is not None
    assert stage.status == "failed"
    assert stage.error is not None


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
