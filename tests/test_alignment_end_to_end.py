import shutil
import subprocess

import pytest

from app.realityscan import RealityScanError, _realityscan_binary

FFMPEG = shutil.which("ffmpeg")
SHARP_FRAMES = shutil.which("sharp-frames")

try:
    _realityscan_binary()
    _REALITYSCAN_AVAILABLE = True
except RealityScanError:
    _REALITYSCAN_AVAILABLE = False


@pytest.mark.skipif(
    FFMPEG is None or SHARP_FRAMES is None or not _REALITYSCAN_AVAILABLE,
    reason="requires ffmpeg, sharp-frames, and RealityScan installed",
)
def test_full_ingest_extract_align_flow(client, tmp_path):
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

    job_id = client.post("/jobs", json={"name": "E2E alignment"}).json()["id"]

    ingest_response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(video_path)}
    )
    assert ingest_response.status_code == 202

    extract_response = client.post(
        f"/jobs/{job_id}/extract-frames",
        json={"selection_method": "batched", "fps": 5},
    )
    assert extract_response.status_code == 202
    job_after_extract = client.get(f"/jobs/{job_id}").json()
    assert job_after_extract["frame_count"] > 0

    align_response = client.post(f"/jobs/{job_id}/align")
    assert align_response.status_code == 202

    stages = client.get(f"/jobs/{job_id}/stages").json()
    alignment_stage = next(s for s in stages if s["stage"] == "alignment")
    # Real photogrammetry alignment needs overlapping real-world footage;
    # the synthetic testsrc pattern used here has no true parallax, so
    # RealityScan may legitimately fail to reconstruct anything (this is
    # exactly what was observed during manual verification while writing
    # this plan). This test verifies the orchestration reaches a terminal
    # state without hanging or crashing — not that alignment quality is
    # good, which depends on the real footage the user provides.
    assert alignment_stage["status"] in ("done", "failed")
    if alignment_stage["status"] == "failed":
        assert alignment_stage["error"]
    else:
        job = client.get(f"/jobs/{job_id}").json()
        assert job["colmap_sparse_dir"] is not None
        assert job["undistorted_images_dir"] is not None
