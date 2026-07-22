import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="requires ffmpeg/ffprobe installed on PATH"
)
def test_full_ingest_and_recommend_flow(client, tmp_path):
    video_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=10",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    job_id = client.post("/jobs", json={"name": "E2E recommendation"}).json()["id"]

    ingest_response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(video_path)}
    )
    assert ingest_response.status_code == 200

    recommend_response = client.post(f"/jobs/{job_id}/recommend", json={})
    assert recommend_response.status_code == 200

    body = recommend_response.json()
    assert body["realityscan"]["downscale"] in (1, 2)
    assert isinstance(body["verdict"]["use_cloud"], bool)
    assert isinstance(body["verdict"]["reason"], str) and body["verdict"]["reason"]

    stored = client.get(f"/jobs/{job_id}/recommendation").json()
    assert stored == body


def test_hardware_endpoint_returns_real_detection(client):
    response = client.get("/hardware")
    assert response.status_code == 200
    body = response.json()
    assert body["cpu_cores"] >= 1
    assert body["ram_gb"] > 0
    # gpu may legitimately be null on a machine with no NVIDIA GPU/driver —
    # both null and a populated dict are valid real-world results.
    assert body["gpu"] is None or "name" in body["gpu"]
