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


def test_extract_frames_marks_stage_failed_on_non_sharpframes_error(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    monkeypatch.setattr(
        extraction, "run_extraction", MagicMock(side_effect=OSError("disk full"))
    )

    with pytest.raises(extraction.ExtractionError, match="disk full"):
        extraction.extract_frames(job_store, job.id)

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.status == "failed"
    assert stage.error == "disk full"


def test_extract_frames_uses_raw_dir_when_multiple_clips_ingested(job_store, monkeypatch):
    from app.paths import raw_dir

    job = job_store.create_job("Test")
    directory = raw_dir(job.id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "a_clip.mp4").write_bytes(b"fake")
    (directory / "b_clip.mp4").write_bytes(b"fake")
    job_store.update_job(job.id, raw_video_path=str(directory / "a_clip.mp4"))

    captured = {}

    def fake_run_extraction(input_path, output_dir, *, selection_method, num_frames, fps, on_progress, on_log=None):
        captured["input_path"] = input_path
        return ExtractionResult(frames=[Path("f1.jpg")], log="ok")

    monkeypatch.setattr(extraction, "run_extraction", fake_run_extraction)

    extraction.extract_frames(job_store, job.id)

    assert captured["input_path"] == directory


def test_extract_frames_forwards_progress_to_job_store(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    def fake_run_extraction(video_path, output_dir, *, selection_method, num_frames, fps, on_progress, on_log=None):
        on_progress(25.0)
        on_progress(75.0)
        return ExtractionResult(frames=[Path("f1.jpg")], log="ok")

    monkeypatch.setattr(extraction, "run_extraction", fake_run_extraction)

    extraction.extract_frames(job_store, job.id)

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.progress_percent == 75.0


def test_extract_frames_forwards_log_to_job_store(job_store, tmp_path, monkeypatch):
    # Real-time visibility while the stage is still "running", not just the
    # final log written once it finishes.
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    seen_running_logs = []
    real_upsert_stage = job_store.upsert_stage

    def spy_upsert_stage(job_id, stage, status, **kwargs):
        if status == "running" and kwargs.get("log") is not None:
            seen_running_logs.append(kwargs["log"])
        real_upsert_stage(job_id, stage, status, **kwargs)

    monkeypatch.setattr(job_store, "upsert_stage", spy_upsert_stage)

    def fake_run_extraction(video_path, output_dir, *, selection_method, num_frames, fps, on_progress, on_log=None):
        on_log("partial output so far")
        return ExtractionResult(frames=[Path("f1.jpg")], log="final log")

    monkeypatch.setattr(extraction, "run_extraction", fake_run_extraction)

    extraction.extract_frames(job_store, job.id)

    assert seen_running_logs == ["partial output so far"]


def test_import_local_frames_copies_images_and_updates_job(job_store, tmp_path):
    job = job_store.create_job("Test")
    source = tmp_path / "existing_frames"
    source.mkdir()
    (source / "0001.jpg").write_bytes(b"fake")
    (source / "0002.png").write_bytes(b"fake")
    (source / "notes.txt").write_text("ignored")

    count = extraction.import_local_frames(job_store, job.id, source)

    assert count == 2
    fetched = job_store.get_job(job.id)
    assert fetched.frame_count == 2
    assert fetched.frames_dir is not None
    copied = sorted(p.name for p in Path(fetched.frames_dir).glob("*"))
    assert copied == ["0001.jpg", "0002.png"]

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.status == "done"


def test_import_local_frames_unknown_job_raises_without_stage_write(job_store, tmp_path):
    with pytest.raises(extraction.ExtractionError, match="Unknown job"):
        extraction.import_local_frames(job_store, "does-not-exist", tmp_path)

    assert job_store.get_stage("does-not-exist", "frame_extraction") is None


def test_import_local_frames_missing_directory_records_failed_stage(job_store, tmp_path):
    job = job_store.create_job("Test")

    with pytest.raises(extraction.ExtractionError, match="not found"):
        extraction.import_local_frames(job_store, job.id, tmp_path / "missing")

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.status == "failed"


def test_import_local_frames_empty_directory_records_failed_stage(job_store, tmp_path):
    job = job_store.create_job("Test")
    source = tmp_path / "empty"
    source.mkdir()

    with pytest.raises(extraction.ExtractionError, match="No image files found"):
        extraction.import_local_frames(job_store, job.id, source)

    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.status == "failed"
