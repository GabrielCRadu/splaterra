import json
from unittest.mock import MagicMock

import pytest

from app import process_registry, training_service
from app.hardware import GpuInfo, HardwareInfo, UsageSnapshot
from app.lichtfeld import LichtFeldError, TrainingResult
from app.paths import alignment_dir

FAKE_HARDWARE = HardwareInfo(
    gpu=GpuInfo(name="NVIDIA GeForce RTX 3060", vram_mb=12288, compute_capability=8.6),
    cpu_cores=16,
    ram_gb=15.9,
)


def _patch_diagnostics(monkeypatch, usage_sequence=None):
    """Mocks the hardware/metrics side-channel so tests never touch a real
    binary (nvidia-smi) or write outside test isolation, matching this
    project's convention of never invoking real external tools in unit tests.
    """
    monkeypatch.setattr(training_service, "detect_hardware", lambda: FAKE_HARDWARE)
    if usage_sequence is not None:
        monkeypatch.setattr(training_service, "sample_usage", MagicMock(side_effect=usage_sequence))
    else:
        monkeypatch.setattr(
            training_service,
            "sample_usage",
            lambda: UsageSnapshot(
                cpu_percent=10.0, ram_used_gb=4.0, ram_total_gb=16.0,
                gpu_percent=50.0, vram_used_mb=5000, vram_total_mb=12288,
            ),
        )
    recorded_metrics = []
    monkeypatch.setattr(training_service, "append_training_metrics", recorded_metrics.append)
    return recorded_metrics


def test_compute_training_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    fake_result = TrainingResult(splat_path=tmp_path / "out" / "result.ply", log="training ok")
    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
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
    _patch_diagnostics(monkeypatch)
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


def test_compute_training_reports_friendly_message_when_cancelled(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start, on_log=None):
        fake_process = MagicMock()
        fake_process.poll.return_value = None
        on_process_start(fake_process)
        process_registry.cancel(job.id, "training")
        raise LichtFeldError("LichtFeld Studio exited with -9")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    with pytest.raises(training_service.TrainingError, match="anulata de utilizator"):
        training_service.compute_training(job_store, job.id)

    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "failed"
    assert stage.error == "Antrenare anulata de utilizator"
    # registry entry must be cleaned up regardless of outcome
    assert process_registry.cancel(job.id, "training") is False


def test_compute_training_marks_stage_failed_on_non_lichtfeld_error(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
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
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
        recommendation_json=json.dumps({"lichtfeld": {"max_gaussians": 3_000_000, "bf16": True}}),
    )

    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        captured["max_gaussians"] = max_gaussians
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    assert captured["max_gaussians"] == 3_000_000


def test_compute_training_max_gaussians_override_wins_over_recommendation(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
        recommendation_json=json.dumps({"lichtfeld": {"max_gaussians": 3_000_000, "bf16": True}}),
    )

    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        captured["max_gaussians"] = max_gaussians
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id, max_gaussians=1_500_000)

    assert captured["max_gaussians"] == 1_500_000


def test_compute_training_passes_iterations_override(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        captured["iterations"] = iterations
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id, iterations=15000)

    assert captured["iterations"] == 15000


def test_compute_training_downscales_images_instead_of_gaussians_for_large_jobs(
    job_store, tmp_path, monkeypatch
):
    # Real crash: a 12GB RTX 3060 hit CUDA OOM during image loading (not a
    # gaussian-tensor allocation) at max_gaussians=4_800_000 on an 830-image
    # job. The fix downscales training images (LichtFeld's --resize_factor)
    # rather than shrinking the gaussian budget, which would cost detail.
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
        frame_count=830,
        recommendation_json=json.dumps({"lichtfeld": {"max_gaussians": 4_800_000, "bf16": True}}),
    )

    captured = {}

    def fake_run_training(
        data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None
    ):
        captured["max_gaussians"] = max_gaussians
        captured["resize_factor"] = resize_factor
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    assert captured["max_gaussians"] == 4_800_000
    assert captured["resize_factor"] > 1


def test_compute_training_uses_full_resolution_for_small_jobs(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
        frame_count=78,
        recommendation_json=json.dumps({"lichtfeld": {"max_gaussians": 4_800_000, "bf16": True}}),
    )

    captured = {}

    def fake_run_training(
        data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None
    ):
        captured["resize_factor"] = resize_factor
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    assert captured["resize_factor"] == 1


def test_compute_training_resize_factor_override_wins_over_recommendation(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
        frame_count=830,
        recommendation_json=json.dumps({"lichtfeld": {"max_gaussians": 4_800_000, "bf16": True}}),
    )

    captured = {}

    def fake_run_training(
        data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None
    ):
        captured["resize_factor"] = resize_factor
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id, resize_factor=1)

    # frame_count=830 would normally push resize_factor above 1 (see the
    # downscale test above) - an explicit slider value must still win.
    assert captured["resize_factor"] == 1


def test_compute_training_writes_settings_json_to_running_stage(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(
        job_store, job.id, variable_exposure=True, max_gaussians=1_500_000, iterations=15000,
        resize_factor=2, eval=True, test_every=4,
    )

    stage = job_store.get_stage(job.id, "training")
    settings = json.loads(stage.settings_json)
    assert settings == {
        "variable_exposure": True,
        "max_gaussians": 1_500_000,
        "iterations": 15000,
        "resize_factor": 2,
        "eval": True,
        "test_every": 4,
        "use_cloud": False,
        "cloud_provider": None,
        "cloud_image": None,
        "cloud_gpu_type": None,
    }


def test_compute_training_rejects_cloud_without_image_or_gpu_type(job_store, tmp_path):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    with pytest.raises(training_service.TrainingError, match="image name and a GPU type"):
        training_service.compute_training(job_store, job.id, use_cloud=True)

    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "failed"


def test_compute_training_dispatches_to_cloud_when_use_cloud(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    captured = {}
    local_run_training = MagicMock()

    def fake_run_cloud_training(data_path, output_dir, *, provider, image_name, gpu_type_id, **kwargs):
        captured["provider"] = provider
        captured["image_name"] = image_name
        captured["gpu_type_id"] = gpu_type_id
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_cloud_training", fake_run_cloud_training)
    monkeypatch.setattr(training_service, "run_training", local_run_training)

    training_service.compute_training(
        job_store, job.id, use_cloud=True, cloud_provider="vastai",
        cloud_image="ghcr.io/example/lichtfeld:cuda", cloud_gpu_type="RTX_4090",
    )

    local_run_training.assert_not_called()
    assert captured["provider"] == "vastai"
    assert captured["image_name"] == "ghcr.io/example/lichtfeld:cuda"
    assert captured["gpu_type_id"] == "RTX_4090"

    stage = job_store.get_stage(job.id, "training")
    settings = json.loads(stage.settings_json)
    assert settings["use_cloud"] is True
    assert settings["cloud_provider"] == "vastai"
    assert settings["cloud_image"] == "ghcr.io/example/lichtfeld:cuda"
    assert settings["cloud_gpu_type"] == "RTX_4090"


def test_compute_training_cloud_error_marks_stage_failed(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    monkeypatch.setattr(
        training_service, "run_cloud_training", MagicMock(side_effect=LichtFeldError("no capacity"))
    )

    with pytest.raises(training_service.TrainingError, match="no capacity"):
        training_service.compute_training(
            job_store, job.id, use_cloud=True, cloud_provider="runpod",
            cloud_image="img", cloud_gpu_type="gpu",
        )

    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "failed"
    assert stage.error == "no capacity"


def test_compute_training_forwards_eval_and_test_every(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        captured["eval"] = eval
        captured["test_every"] = test_every
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id, eval=True, test_every=4)

    assert captured["eval"] is True
    assert captured["test_every"] == 4


def test_compute_training_writes_eval_metrics_json_when_present(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    eval_metrics = [{"iteration": 7000, "psnr": 24.1, "ssim": 0.81, "time_per_image": 0.05, "num_gaussians": 812345}]

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok", eval_metrics=eval_metrics)

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id, eval=True)

    stage = job_store.get_stage(job.id, "training")
    assert json.loads(stage.eval_metrics_json) == eval_metrics


def test_compute_training_eval_metrics_json_is_none_without_eval(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok", eval_metrics=None)

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    stage = job_store.get_stage(job.id, "training")
    assert stage.eval_metrics_json is None


def test_compute_training_omits_max_gaussians_when_no_recommendation(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        captured["max_gaussians"] = max_gaussians
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    assert captured["max_gaussians"] is None


def test_compute_training_forwards_progress_to_job_store(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        on_progress(20.0)
        on_progress(80.0)
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    stage = job_store.get_stage(job.id, "training")
    assert stage.progress_percent == 80.0


def test_compute_training_forwards_log_to_job_store(job_store, tmp_path, monkeypatch):
    # Real-time visibility while the stage is still "running", not just the
    # final log written once it finishes.
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    seen_running_logs = []
    real_upsert_stage = job_store.upsert_stage

    def spy_upsert_stage(job_id, stage, status, **kwargs):
        if status == "running" and kwargs.get("log") is not None:
            seen_running_logs.append(kwargs["log"])
        real_upsert_stage(job_id, stage, status, **kwargs)

    monkeypatch.setattr(job_store, "upsert_stage", spy_upsert_stage)

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        on_log("partial output so far")
        return TrainingResult(splat_path=tmp_path / "result.ply", log="final log")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id)

    assert seen_running_logs == ["partial output so far"]


def test_compute_training_passes_variable_exposure_as_bilateral_grid(job_store, tmp_path, monkeypatch):
    _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    captured = {}

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        captured["bilateral_grid"] = bilateral_grid
        return TrainingResult(splat_path=tmp_path / "result.ply", log="ok")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id, variable_exposure=True)

    assert captured["bilateral_grid"] is True


def test_compute_training_logs_metrics_on_success(job_store, tmp_path, monkeypatch):
    usage_sequence = [
        UsageSnapshot(cpu_percent=20.0, ram_used_gb=5.0, ram_total_gb=16.0, gpu_percent=90.0, vram_used_mb=6000, vram_total_mb=12288),
        UsageSnapshot(cpu_percent=40.0, ram_used_gb=6.0, ram_total_gb=16.0, gpu_percent=95.0, vram_used_mb=8000, vram_total_mb=12288),
    ]
    recorded = _patch_diagnostics(monkeypatch, usage_sequence=usage_sequence)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    def fake_run_training(data_path, output_dir, *, max_gaussians, iterations=None, bilateral_grid, resize_factor=1, eval=False, test_every=None, on_progress, on_process_start=None, on_log=None):
        on_progress(50.0)
        on_progress(100.0)
        return TrainingResult(splat_path=tmp_path / "result.ply", log="Final splats: 4800000")

    monkeypatch.setattr(training_service, "run_training", fake_run_training)

    training_service.compute_training(job_store, job.id, variable_exposure=True)

    assert len(recorded) == 1
    metrics = recorded[0]
    assert metrics.job_id == job.id
    assert metrics.status == "done"
    assert metrics.variable_exposure is True
    assert metrics.final_gaussians == 4800000
    assert metrics.gpu_name == "NVIDIA GeForce RTX 3060"
    assert metrics.vram_total_mb == 12288
    assert metrics.vram_peak_used_mb == 8000
    assert metrics.vram_avg_used_mb == 7000
    assert metrics.cpu_peak_percent == 40.0
    assert metrics.ram_peak_used_gb == 6.0


def test_compute_training_logs_metrics_on_failure(job_store, tmp_path, monkeypatch):
    recorded = _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    monkeypatch.setattr(
        training_service, "run_training", MagicMock(side_effect=LichtFeldError("CUDA out of memory"))
    )

    with pytest.raises(training_service.TrainingError):
        training_service.compute_training(job_store, job.id)

    assert len(recorded) == 1
    metrics = recorded[0]
    assert metrics.status == "failed"
    assert metrics.error == "CUDA out of memory"


def test_compute_training_parses_num_images_from_failed_run_log(job_store, tmp_path, monkeypatch):
    # LichtFeldError's message embeds the raw CLI log (see app/lichtfeld.py) -
    # a failed/crashed run must still get parsed for calibration data instead
    # of throwing the log away, or exactly the OOM crashes that need
    # calibrating against never make it into data/training_runs.jsonl.
    recorded = _patch_diagnostics(monkeypatch)
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir=str(tmp_path / "sparse" / "0"),
        undistorted_images_dir=str(tmp_path / "images"),
    )

    monkeypatch.setattr(
        training_service,
        "run_training",
        MagicMock(
            side_effect=LichtFeldError(
                "LichtFeld Studio exited with 3221226505:\n"
                "colmap.cpp:927 Read 830 cameras and 830 images from COLMAP text files\n"
                "colmap_loader.cpp:227 Loaded 1064510 points from COLMAP text file\n"
                "CUDA out of memory: failed to allocate 79626240 bytes"
            )
        ),
    )

    with pytest.raises(training_service.TrainingError):
        training_service.compute_training(job_store, job.id)

    metrics = recorded[0]
    assert metrics.num_images == 830
    assert metrics.num_initial_points == 1064510
