from unittest.mock import MagicMock

from app.video_metadata import VideoMetadata, VideoMetadataError


def test_video_info_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/video-info")
    assert response.status_code == 404


def test_video_info_no_ingested_video_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/video-info")
    assert response.status_code == 404


def test_video_info_returns_fps_duration_and_estimated_total_frames(client, monkeypatch, tmp_path):
    from app.api import ingestion as ingestion_api

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake")
    client.post(f"/jobs/{job_id}/ingest/local", json={"source_path": str(source)})

    monkeypatch.setattr(
        ingestion_api,
        "probe_ingested_video",
        MagicMock(
            return_value=VideoMetadata(
                width=1920, height=1080, fps=29.97, duration_seconds=10.0, codec="h264"
            )
        ),
    )

    response = client.get(f"/jobs/{job_id}/video-info")

    assert response.status_code == 200
    body = response.json()
    assert body["fps"] == 29.97
    assert body["duration_seconds"] == 10.0
    assert body["total_frames"] == 300


def test_video_info_returns_500_on_probe_error(client, monkeypatch, tmp_path):
    from app.api import ingestion as ingestion_api

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake")
    client.post(f"/jobs/{job_id}/ingest/local", json={"source_path": str(source)})

    monkeypatch.setattr(
        ingestion_api,
        "probe_ingested_video",
        MagicMock(side_effect=VideoMetadataError("ffprobe not found on PATH")),
    )

    response = client.get(f"/jobs/{job_id}/video-info")

    assert response.status_code == 500
