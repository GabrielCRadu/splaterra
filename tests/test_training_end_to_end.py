import os
import shutil
import time
from pathlib import Path

import pytest

from app.lichtfeld import LichtFeldError, _lichtfeld_binary

try:
    _lichtfeld_binary()
    _LICHTFELD_AVAILABLE = True
except LichtFeldError:
    _LICHTFELD_AVAILABLE = False

_OPTED_IN = os.environ.get("SPLATERRA_RUN_REAL_LICHTFELD_TEST") == "1"
_ALIGNMENT_FIXTURE_DIR = os.environ.get("SPLATERRA_E2E_ALIGNMENT_DIR")


def _alignment_fixture_is_valid() -> bool:
    if not _ALIGNMENT_FIXTURE_DIR:
        return False
    root = Path(_ALIGNMENT_FIXTURE_DIR)
    sparse = root / "sparse" / "0"
    return (
        (sparse / "cameras.txt").is_file()
        and (sparse / "images.txt").is_file()
        and (sparse / "points3D.txt").is_file()
        and (root / "images").is_dir()
    )


@pytest.mark.skipif(
    not (_LICHTFELD_AVAILABLE and _OPTED_IN and _alignment_fixture_is_valid()),
    reason=(
        "Real LichtFeld training test skipped by default - a full headless training run can "
        "take a very long time (up to the 2-hour default timeout) and this codebase has never "
        "observed a real training run's stdout/output format. To run it deliberately: install "
        "LichtFeld Studio (or set SPLATERRA_LICHTFELD_DIR), set a real, previously-completed "
        "COLMAP alignment output's root directory (containing sparse/0/*.txt and images/) as "
        "SPLATERRA_E2E_ALIGNMENT_DIR - synthetic ffmpeg testsrc footage does not produce a "
        "valid reconstruction, so this cannot be auto-generated like the other e2e tests - and "
        "set SPLATERRA_RUN_REAL_LICHTFELD_TEST=1."
    ),
)
def test_real_training_run_against_prior_alignment_output(client, tmp_path):
    job_id = client.post("/jobs", json={"name": "E2E training"}).json()["id"]

    from app.paths import alignment_dir

    target = alignment_dir(job_id)
    shutil.copytree(Path(_ALIGNMENT_FIXTURE_DIR), target, dirs_exist_ok=True)

    from app import db
    from app.job_store import JobStore

    conn = db.connect()
    JobStore(conn).update_job(
        job_id,
        colmap_sparse_dir=str(target / "sparse" / "0"),
        undistorted_images_dir=str(target / "images"),
    )
    conn.close()

    train_response = client.post(f"/jobs/{job_id}/train")
    assert train_response.status_code == 202

    # This step deliberately blocks until the background task finishes - the
    # background task itself already enforces run_training's timeout_seconds,
    # so this will not hang forever even if it takes a very long time.
    training_stage = None
    for _ in range(7200):
        stages = client.get(f"/jobs/{job_id}/stages").json()
        training_stage = next((s for s in stages if s["stage"] == "training"), None)
        if training_stage is not None and training_stage["status"] in ("done", "failed"):
            break
        time.sleep(1)
    else:
        pytest.fail("training stage did not reach a terminal state within the polling window")

    # This test's primary goal is verifying the --headless --train invocation
    # shape and the output-file glob actually work against a real binary and
    # real data - not that training quality is good. A "failed" outcome here
    # (e.g. a flag this session couldn't verify turning out to be wrong, or a
    # genuinely unsupported dataset) is informative and should be inspected
    # via the stage's error/log, not silently treated as a passing skip.
    assert training_stage["status"] in ("done", "failed")
    if training_stage["status"] == "failed":
        assert training_stage["error"]
        pytest.fail(f"Real LichtFeld training run failed - inspect and fix: {training_stage['error']}")
    else:
        job = client.get(f"/jobs/{job_id}").json()
        assert job["splat_path"] is not None
        assert Path(job["splat_path"]).is_file()
