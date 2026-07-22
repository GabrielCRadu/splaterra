import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")
SHARP_FRAMES = shutil.which("sharp-frames")


@pytest.mark.skipif(
    FFMPEG is None or SHARP_FRAMES is None,
    reason="requires ffmpeg and sharp-frames installed on PATH",
)
def test_full_ingest_and_extract_flow(client, tmp_path):
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

    job_id = client.post("/jobs", json={"name": "E2E"}).json()["id"]

    ingest_response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(video_path)}
    )
    assert ingest_response.status_code == 200

    extract_response = client.post(
        f"/jobs/{job_id}/extract-frames",
        json={"selection_method": "batched", "fps": 5},
    )
    assert extract_response.status_code == 200
    assert extract_response.json()["frame_count"] > 0

    job = client.get(f"/jobs/{job_id}").json()
    assert job["frame_count"] > 0
    assert job["frames_dir"] is not None
