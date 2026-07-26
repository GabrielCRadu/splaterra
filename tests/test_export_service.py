from unittest.mock import MagicMock

import pytest

from app import export_service
from app.lichtfeld import LichtFeldError
from app.paths import training_dir


def test_compute_export_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, splat_path=str(tmp_path / "training" / "splat_30000.ply"))

    captured = {}

    def fake_convert_splat(input_path, output_path, *, format, sh_degree=-1):
        captured["input_path"] = input_path
        captured["output_path"] = output_path
        captured["format"] = format
        return output_path

    monkeypatch.setattr(export_service, "convert_splat", fake_convert_splat)

    result = export_service.compute_export(job_store, job.id, format="spz")

    assert captured["input_path"] == tmp_path / "training" / "splat_30000.ply"
    assert captured["format"] == "spz"
    assert result == training_dir(job.id) / "export.spz"

    fetched = job_store.get_job(job.id)
    assert fetched.export_path == str(training_dir(job.id) / "export.spz")
    assert fetched.export_format == "spz"

    stage = job_store.get_stage(job.id, "export")
    assert stage.status == "done"


def test_compute_export_without_training_raises(job_store):
    job = job_store.create_job("Test")

    with pytest.raises(export_service.ExportError, match="no trained splat"):
        export_service.compute_export(job_store, job.id, format="spz")

    stage = job_store.get_stage(job.id, "export")
    assert stage.status == "failed"


def test_compute_export_unknown_job_raises_without_stage_write(job_store):
    with pytest.raises(export_service.ExportError, match="Unknown job"):
        export_service.compute_export(job_store, "does-not-exist", format="spz")

    assert job_store.get_stage("does-not-exist", "export") is None


def test_compute_export_marks_stage_failed_on_lichtfeld_error(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, splat_path=str(tmp_path / "training" / "splat_30000.ply"))

    monkeypatch.setattr(
        export_service, "convert_splat", MagicMock(side_effect=LichtFeldError("conversion failed"))
    )

    with pytest.raises(export_service.ExportError, match="conversion failed"):
        export_service.compute_export(job_store, job.id, format="sog")

    stage = job_store.get_stage(job.id, "export")
    assert stage.status == "failed"
    assert stage.error == "conversion failed"


def test_compute_export_rejects_unsupported_format(job_store, tmp_path):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, splat_path=str(tmp_path / "training" / "splat_30000.ply"))

    with pytest.raises(export_service.ExportError, match="Unsupported"):
        export_service.compute_export(job_store, job.id, format="xyz")

    stage = job_store.get_stage(job.id, "export")
    assert stage.status == "failed"
