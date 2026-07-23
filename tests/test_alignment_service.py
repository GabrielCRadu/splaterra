from unittest.mock import MagicMock

import pytest

from app import alignment_service
from app.realityscan import AlignmentResult, RealityScanError


def test_compute_alignment_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    fake_result = AlignmentResult(
        sparse_dir=tmp_path / "sparse" / "0",
        images_dir=tmp_path / "images",
        log="aligned ok",
    )
    monkeypatch.setattr(alignment_service, "run_alignment", MagicMock(return_value=fake_result))

    result = alignment_service.compute_alignment(job_store, job.id)

    assert result is fake_result
    fetched = job_store.get_job(job.id)
    assert fetched.colmap_sparse_dir == str(tmp_path / "sparse" / "0")
    assert fetched.undistorted_images_dir == str(tmp_path / "images")

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "done"
    assert stage.log == "aligned ok"


def test_compute_alignment_without_extracted_frames_raises(job_store):
    job = job_store.create_job("Test")

    with pytest.raises(alignment_service.AlignmentError, match="no extracted frames"):
        alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "failed"


def test_compute_alignment_unknown_job_raises_without_stage_write(job_store):
    with pytest.raises(alignment_service.AlignmentError, match="Unknown job"):
        alignment_service.compute_alignment(job_store, "does-not-exist")

    assert job_store.get_stage("does-not-exist", "alignment") is None


def test_compute_alignment_marks_stage_failed_on_realityscan_error(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    monkeypatch.setattr(
        alignment_service, "run_alignment", MagicMock(side_effect=RealityScanError("export failed"))
    )

    with pytest.raises(alignment_service.AlignmentError, match="export failed"):
        alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "failed"
    assert stage.error == "export failed"


def test_compute_alignment_marks_stage_failed_on_non_realityscan_error(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    monkeypatch.setattr(
        alignment_service, "run_alignment", MagicMock(side_effect=OSError("disk full"))
    )

    with pytest.raises(alignment_service.AlignmentError, match="disk full"):
        alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "failed"
    assert stage.error == "disk full"


def test_compute_alignment_forwards_progress_to_job_store(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    def fake_run_alignment(frames_dir, output_dir, *, progress_path, on_progress):
        on_progress(40.0)
        on_progress(95.0)
        return AlignmentResult(
            sparse_dir=tmp_path / "sparse" / "0",
            images_dir=tmp_path / "images",
            log="ok",
        )

    monkeypatch.setattr(alignment_service, "run_alignment", fake_run_alignment)

    alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.progress_percent == 95.0
