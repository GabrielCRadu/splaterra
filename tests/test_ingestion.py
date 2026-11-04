from unittest.mock import MagicMock

import pytest

from app import ingestion
from app.ingestion import IngestionError, import_local_file, list_raw_videos, probe_ingested_video
from app.paths import raw_dir
from app.video_metadata import VideoMetadata


def test_import_local_file_copies_and_records_job(job_store, tmp_path):
    job = job_store.create_job("Test")

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video bytes")

    destination = import_local_file(job_store, job.id, source)

    assert destination.exists()
    assert destination.read_bytes() == b"fake video bytes"

    fetched = job_store.get_job(job.id)
    assert fetched.raw_video_path == str(destination)

    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "done"


def test_import_local_file_missing_source_raises(job_store, tmp_path):
    job = job_store.create_job("Test")

    with pytest.raises(IngestionError):
        import_local_file(job_store, job.id, tmp_path / "missing.mp4")


def test_import_local_file_missing_source_records_failed_stage(job_store, tmp_path):
    job = job_store.create_job("Test")

    with pytest.raises(IngestionError):
        import_local_file(job_store, job.id, tmp_path / "missing.mp4")

    stage = job_store.get_stage(job.id, "ingestion")
    assert stage is not None
    assert stage.status == "failed"
    assert stage.error is not None


def test_import_local_file_rejects_osv_extension(job_store, tmp_path):
    job = job_store.create_job("Test", device_type="dji_avata_360")
    source = tmp_path / "clip.OSV"
    source.write_bytes(b"raw dual-fisheye")

    with pytest.raises(IngestionError, match="OSV"):
        import_local_file(job_store, job.id, source)

    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "failed"
    assert "OSV" in stage.error


def test_list_raw_videos_returns_empty_when_no_raw_dir(job_store):
    job = job_store.create_job("Test")
    assert list_raw_videos(job.id) == []


def test_list_raw_videos_returns_single_video(job_store, tmp_path):
    job = job_store.create_job("Test")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake")
    import_local_file(job_store, job.id, source)

    videos = list_raw_videos(job.id)

    assert videos == [raw_dir(job.id) / "clip.mp4"]


def test_list_raw_videos_returns_multiple_videos_sorted(job_store, tmp_path):
    job = job_store.create_job("Test")
    directory = raw_dir(job.id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "b_clip.mp4").write_bytes(b"fake")
    (directory / "a_clip.mov").write_bytes(b"fake")

    videos = list_raw_videos(job.id)

    assert videos == [directory / "a_clip.mov", directory / "b_clip.mp4"]


def test_list_raw_videos_ignores_non_video_files(job_store):
    job = job_store.create_job("Test")
    directory = raw_dir(job.id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "clip.mp4").write_bytes(b"fake")
    (directory / "notes.txt").write_text("not a video")

    videos = list_raw_videos(job.id)

    assert videos == [directory / "clip.mp4"]


def test_probe_ingested_video_single_clip_probes_raw_video_path(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    video_path = tmp_path / "clip.mp4"

    fake_meta = VideoMetadata(width=1920, height=1080, fps=30.0, duration_seconds=10.0, codec="h264")
    fake_probe = MagicMock(return_value=fake_meta)
    monkeypatch.setattr(ingestion, "probe_video", fake_probe)

    result = probe_ingested_video(job.id, str(video_path))

    fake_probe.assert_called_once_with(video_path)
    assert result == fake_meta


def test_probe_ingested_video_aggregates_multiple_clips(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    directory = raw_dir(job.id)
    directory.mkdir(parents=True, exist_ok=True)
    clip_a = directory / "a_clip.mp4"
    clip_b = directory / "b_clip.mp4"
    clip_a.write_bytes(b"fake")
    clip_b.write_bytes(b"fake")

    probed = {
        clip_a: VideoMetadata(width=1920, height=1080, fps=30.0, duration_seconds=10.0, codec="h264"),
        clip_b: VideoMetadata(width=1280, height=720, fps=30.0, duration_seconds=20.0, codec="h264"),
    }
    monkeypatch.setattr(ingestion, "probe_video", lambda path: probed[path])

    result = probe_ingested_video(job.id, str(clip_a))

    assert result.duration_seconds == 30.0
    assert result.width == 1920
