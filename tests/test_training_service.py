import json
from unittest.mock import MagicMock

import pytest

from app import training_service
from app.lichtfeld import LichtFeldError, TrainingResult
from app.paths import alignment_dir


def test_compute_training_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    fake_result = TrainingResult(splat_path=tmp_path / "out" / "result.ply", log="training ok")
    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, on_progress):
        captured["data_path"] = data_path
        captured["output_dir"] = output_dir
        captured["max_gaussians"] = max_gaussians
        return fake_result

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    result = training_service.compute_training(job_store, job.id)

    assert result is fake_result
    assert captured["data_path"] == alignment_dir(job.id)
    fetched = job_store.get_job(job.id)
    assert fetched.splat_path == str(tmp_path / "out" / "result.ply")

    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "done"
    assert stage.log == "training ok"


def test_compute_training_without_alignment_raises(job_store):
    job = job_store.create_job("Test")

    with pytest.raises(training_service.TrainingError, match="no alignment"):
        training_service.compute_training(job_store, job.id)

    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "failed"


def test_compute_training_unknown_job_raises_without_stage_write(job_store):
    with pytest.raises(training_service.TrainingError, match="Unknown job"):
        training_service.compute_training(job_store, "does-not-exist")

    assert job_store.get_stage("does-not-exist", "training") is None


def test_compute_training_marks_stage_failed_on_lichtfeld_error(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    monkeypatch.setattr(
        training_service, "run_training", MagicMock(side_effect=LichtFeldError("CUDA out of memory"))
    )

    with pytest.raises(training_service.TrainingError, match="out of memory"):
        training_service.compute_training(job_store, job.id)

    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "failed"
    assert stage.error == "CUDA out of memory"


def test_compute_training_marks_stage_failed_on_non_lichtfeld_error(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    monkeypatch.setattr(training_service, "run_training", MagicMock(side_effect=OSError("disk full")))

    with pytest.raises(training_service.TrainingError, match="disk full"):
        training_service.compute_training(job_store, job.id)

    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "failed"


def test_compute_training_passes_max_gaussians_from_recommendation(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
        recommendation_json=json.dumps({"lichtfeld": {"max_gaussians": 3_000_000, "bf16": True}}),
    )

    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, on_progress):
        captured["max_gaussians"] = max_gaussians
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    assert captured["max_gaussians"] == 3_000_000


def test_compute_training_omits_max_gaussians_when_no_recommendation(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, on_progress):
        captured["max_gaussians"] = max_gaussians
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    assert captured["max_gaussians"] is None


def test_compute_training_forwards_progress_to_job_store(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    def fake_run_training(data_path, output_dir, *, max_gaussians, on_progress):
        on_progress(20.0)
        on_progress(80.0)
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    stage = job_store.get_stage(job.id, "training")
    assert stage.progress_percent == 80.0
