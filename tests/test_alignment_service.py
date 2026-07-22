from unittest.mock import MagicMock

import pytest

from app import alignment_service
from app.realityscan import AlignmentResult, RealityScanError


def test_compute_alignment_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    fake_result = AlignmentResult(
        registration_path=tmp_path / "registration.csv",
        point_cloud_path=tmp_path / "points.ply",
        log="aligned ok",
    )
    monkeypatch.setattr(alignment_service, "run_alignment", MagicMock(return_value=fake_result))

    result = alignment_service.compute_alignment(job_store, job.id)

    assert result is fake_result
    fetched = job_store.get_job(job.id)
    assert fetched.registration_path == str(tmp_path / "registration.csv")
    assert fetched.point_cloud_path == str(tmp_path / "points.ply")

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
