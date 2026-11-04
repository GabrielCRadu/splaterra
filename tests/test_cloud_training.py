from unittest.mock import MagicMock

import pytest

from app import cloud_training
from app.cloud_job import CloudJobError, CloudJobResult
from app.lichtfeld import LichtFeldError


def test_run_cloud_training_builds_remote_command_and_returns_result(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "splat.ply").write_bytes(b"fake ply")

    captured = {}

    def fake_run_cloud_job(local_input_dir, out_dir, command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return CloudJobResult(pod_id="pod-1", remote_command_output="training done", output_dir=out_dir)

    monkeypatch.setattr(cloud_training, "run_cloud_job", fake_run_cloud_job)

    result = cloud_training.run_cloud_training(
        tmp_path / "data", output_dir,
        provider="runpod", image_name="img", gpu_type_id="gpu",
        max_gaussians=1_500_000, iterations=15000, eval=True, test_every=4,
    )

    assert result.splat_path == output_dir / "splat.ply"
    assert result.log == "training done"
    assert "LichtFeld-Studio" in captured["command"]
    assert "/workspace/input" in captured["command"]
    assert "/workspace/output" in captured["command"]
    assert "--max-cap 1500000" in captured["command"]
    assert "--iter 15000" in captured["command"]
    assert "--eval" in captured["command"]
    assert "--test-every 4" in captured["command"]
    assert captured["kwargs"]["provider"] == "runpod"
    assert captured["kwargs"]["image_name"] == "img"
    assert captured["kwargs"]["gpu_type_id"] == "gpu"


def test_run_cloud_training_wraps_cloud_job_error_as_lichtfeld_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cloud_training, "run_cloud_job", MagicMock(side_effect=CloudJobError("no capacity"))
    )

    with pytest.raises(LichtFeldError, match="no capacity"):
        cloud_training.run_cloud_training(
            tmp_path / "data", tmp_path / "out", provider="runpod", image_name="img", gpu_type_id="gpu"
        )


def test_run_cloud_training_raises_when_no_splat_produced(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run_cloud_job(local_input_dir, out_dir, command, **kwargs):
        return CloudJobResult(pod_id="pod-1", remote_command_output="no output files", output_dir=out_dir)

    monkeypatch.setattr(cloud_training, "run_cloud_job", fake_run_cloud_job)

    with pytest.raises(LichtFeldError, match="did not produce"):
        cloud_training.run_cloud_training(
            tmp_path / "data", output_dir, provider="runpod", image_name="img", gpu_type_id="gpu"
        )


def test_run_cloud_training_parses_eval_metrics_when_present(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "splat.ply").write_bytes(b"fake ply")
    (output_dir / "metrics.csv").write_text(
        "iteration,psnr,ssim,time_per_image,num_gaussians\n7000,24.5,0.81,0.04,800000\n"
    )

    def fake_run_cloud_job(local_input_dir, out_dir, command, **kwargs):
        return CloudJobResult(pod_id="pod-1", remote_command_output="ok", output_dir=out_dir)

    monkeypatch.setattr(cloud_training, "run_cloud_job", fake_run_cloud_job)

    result = cloud_training.run_cloud_training(
        tmp_path / "data", output_dir, provider="vastai", image_name="img", gpu_type_id="RTX_4090", eval=True
    )

    assert result.eval_metrics == [
        {"iteration": 7000, "psnr": 24.5, "ssim": 0.81, "time_per_image": 0.04, "num_gaussians": 800000}
    ]


def test_run_cloud_training_forwards_provider_to_run_cloud_job(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "splat.ply").write_bytes(b"fake ply")

    captured = {}

    def fake_run_cloud_job(local_input_dir, out_dir, command, **kwargs):
        captured["provider"] = kwargs["provider"]
        return CloudJobResult(pod_id="pod-1", remote_command_output="ok", output_dir=out_dir)

    monkeypatch.setattr(cloud_training, "run_cloud_job", fake_run_cloud_job)

    cloud_training.run_cloud_training(
        tmp_path / "data", output_dir, provider="vastai", image_name="img", gpu_type_id="RTX_4090"
    )

    assert captured["provider"] == "vastai"
