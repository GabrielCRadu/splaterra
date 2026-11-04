import os
import shutil

import pytest

from app.cloud_job import run_cloud_job

_RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
_OPTED_IN = os.environ.get("SPLATERRA_RUN_REAL_RUNPOD_TEST") == "1"
_SSH = shutil.which("ssh")
_SCP = shutil.which("scp")


@pytest.mark.skipif(
    not (_RUNPOD_API_KEY and _OPTED_IN and _SSH and _SCP),
    reason=(
        "Real RunPod smoke test skipped by default - it creates and briefly bills a real "
        "GPU pod. To run it deliberately: set RUNPOD_API_KEY, add your SSH public key to "
        "your RunPod account (Settings -> SSH Public Keys), and set "
        "SPLATERRA_RUN_REAL_RUNPOD_TEST=1. Verify the gpu_type_id below is still a valid, "
        "orderable GPU type on your RunPod account before running - GPU catalogs change."
    ),
)
def test_real_cloud_job_round_trip(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "hello.txt").write_text("hello from splat prep")

    result = run_cloud_job(
        input_dir,
        tmp_path / "output",
        "mkdir -p /workspace/output && cp /workspace/input/hello.txt /workspace/output/",
        image_name="runpod/base:0.6.2-cuda12.4.1",
        gpu_type_id="NVIDIA GeForce RTX 3070",
    )

    assert result.pod_id
    assert (tmp_path / "output" / "hello.txt").read_text() == "hello from splat prep"
