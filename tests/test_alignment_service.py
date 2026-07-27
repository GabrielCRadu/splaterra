from unittest.mock import MagicMock

import pytest

from app import alignment_service, process_registry
from app.realityscan import AlignmentResult, RealityScanError


def _write_fake_registered_images(images_dir, count=10):
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (images_dir / f"{i:05d}.png").write_bytes(b"fake")


def test_compute_alignment_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    _write_fake_registered_images(tmp_path / "images")
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


def test_compute_alignment_reports_friendly_message_when_cancelled(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    def fake_run_alignment(
        frames_dir, output_dir, *, progress_path, on_progress, timeout_seconds, on_process_start, on_log=None
    ):
        fake_process = MagicMock()
        fake_process.poll.return_value = None
        on_process_start(fake_process)
        process_registry.cancel(job.id, "alignment")
        raise RealityScanError("RealityScan exited with -9")

    monkeypatch.setattr(alignment_service, "run_alignment", fake_run_alignment)

    with pytest.raises(alignment_service.AlignmentError, match="anulat de utilizator"):
        alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "failed"
    assert stage.error == "Alignment anulat de utilizator"
    # registry entry must be cleaned up regardless of outcome
    assert process_registry.cancel(job.id, "alignment") is False


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

    def fake_run_alignment(
        frames_dir, output_dir, *, progress_path, on_progress, timeout_seconds, on_process_start=None, on_log=None
    ):
        on_progress(40.0)
        on_progress(95.0)
        _write_fake_registered_images(tmp_path / "images")
        return AlignmentResult(
            sparse_dir=tmp_path / "sparse" / "0",
            images_dir=tmp_path / "images",
            log="ok",
        )

    monkeypatch.setattr(alignment_service, "run_alignment", fake_run_alignment)

    alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.progress_percent == 95.0


def test_compute_alignment_forwards_log_to_job_store(job_store, tmp_path, monkeypatch):
    # Real-time visibility while the stage is still "running", not just the
    # final log written once it finishes.
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    seen_running_logs = []
    real_upsert_stage = job_store.upsert_stage

    def spy_upsert_stage(job_id, stage, status, **kwargs):
        if status == "running" and kwargs.get("log") is not None:
            seen_running_logs.append(kwargs["log"])
        real_upsert_stage(job_id, stage, status, **kwargs)

    monkeypatch.setattr(job_store, "upsert_stage", spy_upsert_stage)

    def fake_run_alignment(
        frames_dir, output_dir, *, progress_path, on_progress, timeout_seconds, on_process_start=None, on_log=None
    ):
        on_log("partial output so far")
        _write_fake_registered_images(tmp_path / "images")
        return AlignmentResult(
            sparse_dir=tmp_path / "sparse" / "0", images_dir=tmp_path / "images", log="final log"
        )

    monkeypatch.setattr(alignment_service, "run_alignment", fake_run_alignment)

    alignment_service.compute_alignment(job_store, job.id)

    assert seen_running_logs == ["partial output so far"]


def test_compute_alignment_scales_timeout_with_frame_count(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"), frame_count=1000)

    _write_fake_registered_images(tmp_path / "images", count=1000)
    fake_result = AlignmentResult(
        sparse_dir=tmp_path / "sparse" / "0", images_dir=tmp_path / "images", log="ok"
    )
    mock_run = MagicMock(return_value=fake_result)
    monkeypatch.setattr(alignment_service, "run_alignment", mock_run)

    alignment_service.compute_alignment(job_store, job.id)

    assert mock_run.call_args.kwargs["timeout_seconds"] == 5000


def test_compute_alignment_uses_base_timeout_when_frame_count_unknown(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    _write_fake_registered_images(tmp_path / "images")
    fake_result = AlignmentResult(
        sparse_dir=tmp_path / "sparse" / "0", images_dir=tmp_path / "images", log="ok"
    )
    mock_run = MagicMock(return_value=fake_result)
    monkeypatch.setattr(alignment_service, "run_alignment", mock_run)

    alignment_service.compute_alignment(job_store, job.id)

    assert mock_run.call_args.kwargs["timeout_seconds"] == alignment_service.ALIGNMENT_BASE_TIMEOUT_SECONDS


def test_compute_alignment_fails_on_near_empty_registration(job_store, tmp_path, monkeypatch):
    # Real crash this guards against: RealityScan exited 0 and wrote valid
    # COLMAP files while registering only 2 of 1666 extracted frames
    # (insufficient overlap in the footage) - the job sailed through to
    # training, which then crashed on the degenerate 2-camera reconstruction
    # with a cryptic CUDA error instead of a message pointing at the real
    # problem.
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"), frame_count=1666)

    _write_fake_registered_images(tmp_path / "images", count=2)
    fake_result = AlignmentResult(
        sparse_dir=tmp_path / "sparse" / "0", images_dir=tmp_path / "images", log="ok"
    )
    monkeypatch.setattr(alignment_service, "run_alignment", MagicMock(return_value=fake_result))

    with pytest.raises(alignment_service.AlignmentError, match="2 din 1666"):
        alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "failed"
    assert "2 din 1666" in stage.error
    # A failed coverage check must not leave stale colmap paths on the job.
    fetched = job_store.get_job(job.id)
    assert fetched.colmap_sparse_dir is None


def test_compute_alignment_fails_below_absolute_minimum_even_without_frame_count(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"))

    _write_fake_registered_images(tmp_path / "images", count=3)
    fake_result = AlignmentResult(
        sparse_dir=tmp_path / "sparse" / "0", images_dir=tmp_path / "images", log="ok"
    )
    monkeypatch.setattr(alignment_service, "run_alignment", MagicMock(return_value=fake_result))

    with pytest.raises(alignment_service.AlignmentError, match="doar 3 cadre"):
        alignment_service.compute_alignment(job_store, job.id)

    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "failed"


def test_compute_alignment_allows_partial_but_reasonable_registration(job_store, tmp_path, monkeypatch):
    # Some drop-off between extracted and registered frames is normal
    # (blurry/duplicate/no-overlap frames legitimately fail to register even
    # in a healthy reconstruction) - only catastrophic coverage should fail.
    job = job_store.create_job("Test")
    job_store.update_job(job.id, frames_dir=str(tmp_path / "frames"), frame_count=100)

    _write_fake_registered_images(tmp_path / "images", count=60)
    fake_result = AlignmentResult(
        sparse_dir=tmp_path / "sparse" / "0", images_dir=tmp_path / "images", log="ok"
    )
    monkeypatch.setattr(alignment_service, "run_alignment", MagicMock(return_value=fake_result))

    result = alignment_service.compute_alignment(job_store, job.id)

    assert result is fake_result
    stage = job_store.get_stage(job.id, "alignment")
    assert stage.status == "done"
