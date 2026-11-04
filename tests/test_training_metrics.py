import json

from app import training_metrics
from app.training_metrics import TrainingMetrics

# Trimmed excerpt from a real LichtFeld Studio training log (job MasinaPoarta,
# 2026-07-26) - regexes are validated against real output, not an idealized format.
REAL_LOG_EXCERPT = """
[13:13:09.207] [32m[info][0m colmap.cpp:927  Read 77 cameras and 77 images from COLMAP text files
[13:13:09.685] [32m[info][0m colmap_loader.cpp:227  Loaded 165917 points from COLMAP text file
[13:13:09.979] [32m[info][0m splat_data.cpp:871  Initialized SplatData: 165917 points, max SH degree: 3, SH coefficients: 16, sh0 shape: [165917, 1, 3], shN shape: [165917, 15, 3]
[13:13:09.982] [32m[info][0m trainer.cpp:345  Initializing trainer with 30000 iterations
* Training completed in 3596.100s (avg 8.3 iter/s)
* Final splats: 4800000
"""


def test_parse_training_log_extracts_known_fields_from_real_log():
    result = training_metrics.parse_training_log(REAL_LOG_EXCERPT)

    assert result["num_images"] == 77
    assert result["num_initial_points"] == 165917
    assert result["sh_degree"] == 3
    assert result["iterations_configured"] == 30000
    assert result["final_gaussians"] == 4800000
    assert result["duration_seconds"] == 3596.100
    assert result["avg_iter_per_sec"] == 8.3


def test_parse_training_log_returns_none_for_missing_fields():
    result = training_metrics.parse_training_log("nothing relevant here\n")

    assert result == {
        "num_images": None,
        "num_initial_points": None,
        "sh_degree": None,
        "iterations_configured": None,
        "final_gaussians": None,
        "duration_seconds": None,
        "avg_iter_per_sec": None,
    }


def test_append_training_metrics_writes_jsonl_line(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATERRA_DATA_DIR", str(tmp_path))

    metrics = TrainingMetrics(
        job_id="job-1",
        timestamp="2026-07-26T13:13:08+00:00",
        status="done",
        error=None,
        variable_exposure=False,
        requested_max_gaussians=4800000,
        num_images=77,
        num_initial_points=165917,
        sh_degree=3,
        iterations_configured=30000,
        final_gaussians=4800000,
        duration_seconds=3596.1,
        avg_iter_per_sec=8.3,
        gpu_name="NVIDIA GeForce RTX 3060",
        vram_total_mb=12288,
        vram_peak_used_mb=9637,
        vram_avg_used_mb=9200,
        cpu_peak_percent=55.0,
        ram_peak_used_gb=13.1,
    )

    training_metrics.append_training_metrics(metrics)

    path = training_metrics.training_metrics_path()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["job_id"] == "job-1"
    assert record["final_gaussians"] == 4800000


def test_append_training_metrics_appends_multiple_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATERRA_DATA_DIR", str(tmp_path))

    for job_id in ("job-1", "job-2"):
        training_metrics.append_training_metrics(
            TrainingMetrics(
                job_id=job_id,
                timestamp="2026-07-26T13:13:08+00:00",
                status="done",
                error=None,
                variable_exposure=False,
                requested_max_gaussians=None,
                num_images=None,
                num_initial_points=None,
                sh_degree=None,
                iterations_configured=None,
                final_gaussians=None,
                duration_seconds=None,
                avg_iter_per_sec=None,
                gpu_name=None,
                vram_total_mb=None,
                vram_peak_used_mb=None,
                vram_avg_used_mb=None,
                cpu_peak_percent=None,
                ram_peak_used_gb=None,
            )
        )

    path = training_metrics.training_metrics_path()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["job_id"] == "job-1"
    assert json.loads(lines[1])["job_id"] == "job-2"
