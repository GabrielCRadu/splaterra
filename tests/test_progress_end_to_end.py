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
def test_extraction_and_alignment_end_with_progress_recorded(client, tmp_path):
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

    job_id = client.post("/jobs", json={"name": "E2E progress"}).json()["id"]
    client.post(f"/jobs/{job_id}/ingest/local", json={"source_path": str(video_path)})
    client.post(f"/jobs/{job_id}/extract-frames", json={"selection_method": "batched", "fps": 5})

    stages = client.get(f"/jobs/{job_id}/stages").json()
    extraction_stage = next(s for s in stages if s["stage"] == "frame_extraction")
    assert extraction_stage["status"] == "done"
    # A real run must have reported at least one live percentage along the way.
    assert extraction_stage["progress_percent"] is not None

    client.post(f"/jobs/{job_id}/align")
    stages = client.get(f"/jobs/{job_id}/stages").json()
    alignment_stage = next(s for s in stages if s["stage"] == "alignment")
    # Same caveat as the alignment end-to-end test: synthetic footage has no
    # real parallax, so RealityScan may legitimately fail to reconstruct
    # anything - this test verifies progress reporting and a terminal state,
    # not reconstruction quality.
    assert alignment_stage["status"] in ("done", "failed")
