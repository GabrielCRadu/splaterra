import json
from unittest.mock import MagicMock

import pytest

from app import recommendation_service
from app.drone_telemetry import SrtParseError
from app.hardware import HardwareInfo
from app.recommendation import LichtFeldSettings, Recommendation, RealityScanSettings, Verdict
from app.video_metadata import VideoMetadata, VideoMetadataError

FAKE_RECOMMENDATION = Recommendation(
    realityscan=RealityScanSettings(downscale=1, rolling_shutter_correction=False),
    lichtfeld=LichtFeldSettings(max_gaussians=1_000_000, bf16=True),
    verdict=Verdict(use_cloud=False, reason="Local GPU meets LichtFeld's minimum requirements"),
)


def test_compute_recommendation_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    monkeypatch.setattr(
        recommendation_service, "probe_video", MagicMock(return_value=VideoMetadata(
            width=1920, height=1080, fps=30.0, duration_seconds=10.0, codec="h264"
        ))
    )
    monkeypatch.setattr(recommendation_service, "detect_hardware", MagicMock(
        return_value=HardwareInfo(gpu=None, cpu_cores=8, ram_gb=16.0)
    ))
    monkeypatch.setattr(
        recommendation_service, "recommend", MagicMock(return_value=FAKE_RECOMMENDATION)
    )

    result = recommendation_service.compute_recommendation(job_store, job.id)

    assert result == FAKE_RECOMMENDATION

    fetched = job_store.get_job(job.id)
    assert fetched.recommendation_json is not None
    payload = json.loads(fetched.recommendation_json)
    assert payload["verdict"]["use_cloud"] is False
    assert payload["lichtfeld"]["max_gaussians"] == 1_000_000

    stage = job_store.get_stage(job.id, "recommendation")
    assert stage.status == "done"


def test_compute_recommendation_without_ingested_video_records_failed_stage(job_store):
    job = job_store.create_job("Test")

    with pytest.raises(recommendation_service.RecommendationError, match="no ingested video"):
        recommendation_service.compute_recommendation(job_store, job.id)

    stage = job_store.get_stage(job.id, "recommendation")
    assert stage is not None
    assert stage.status == "failed"
    assert stage.error is not None


def test_compute_recommendation_unknown_job_raises_without_stage_write(job_store):
    with pytest.raises(recommendation_service.RecommendationError, match="Unknown job"):
        recommendation_service.compute_recommendation(job_store, "does-not-exist")

    assert job_store.get_stage("does-not-exist", "recommendation") is None


def test_compute_recommendation_marks_stage_failed_on_video_metadata_error(
    job_store, tmp_path, monkeypatch
):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    monkeypatch.setattr(
        recommendation_service,
        "probe_video",
        MagicMock(side_effect=VideoMetadataError("ffprobe not found on PATH")),
    )

    with pytest.raises(recommendation_service.RecommendationError, match="ffprobe not found"):
        recommendation_service.compute_recommendation(job_store, job.id)

    stage = job_store.get_stage(job.id, "recommendation")
    assert stage.status == "failed"
    assert stage.error == "ffprobe not found on PATH"


def test_compute_recommendation_marks_stage_failed_on_srt_parse_error(
    job_store, tmp_path, monkeypatch
):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    monkeypatch.setattr(
        recommendation_service, "probe_video", MagicMock(return_value=VideoMetadata(
            width=1920, height=1080, fps=30.0, duration_seconds=10.0, codec="h264"
        ))
    )
    monkeypatch.setattr(
        recommendation_service,
        "parse_srt",
        MagicMock(side_effect=SrtParseError("SRT file not found: bad.srt")),
    )

    with pytest.raises(recommendation_service.RecommendationError, match="SRT file not found"):
        recommendation_service.compute_recommendation(job_store, job.id, srt_path=tmp_path / "bad.srt")

    stage = job_store.get_stage(job.id, "recommendation")
    assert stage.status == "failed"
