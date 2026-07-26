from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.paths import data_root

_NUM_IMAGES_RE = re.compile(r"Read \d+ cameras and (\d+) images")
_NUM_INITIAL_POINTS_RE = re.compile(r"Loaded (\d+) points from COLMAP")
_SH_DEGREE_RE = re.compile(r"max SH degree:\s*(\d+)")
_ITERATIONS_CONFIGURED_RE = re.compile(r"Initializing trainer with (\d+) iterations")
_FINAL_GAUSSIANS_RE = re.compile(r"Final splats:\s*(\d+)")
_DURATION_RE = re.compile(r"Training completed in ([\d.]+)s \(avg ([\d.]+) iter/s\)")


@dataclass
class TrainingMetrics:
    job_id: str
    timestamp: str
    status: str
    error: str | None
    variable_exposure: bool
    requested_max_gaussians: int | None
    num_images: int | None
    num_initial_points: int | None
    sh_degree: int | None
    iterations_configured: int | None
    final_gaussians: int | None
    duration_seconds: float | None
    avg_iter_per_sec: float | None
    gpu_name: str | None
    vram_total_mb: int | None
    vram_peak_used_mb: int | None
    vram_avg_used_mb: int | None
    cpu_peak_percent: float | None
    ram_peak_used_gb: float | None


def _search_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def parse_training_log(log: str) -> dict:
    """Best-effort extraction of calibration-relevant figures from LichtFeld's raw
    CLI log. Every field degrades to None on a missing/changed log line rather than
    raising - this is diagnostic data for tuning app/recommendation.py's constants,
    not a functional dependency of the training pipeline itself.
    """
    duration_match = _DURATION_RE.search(log)
    return {
        "num_images": _search_int(_NUM_IMAGES_RE, log),
        "num_initial_points": _search_int(_NUM_INITIAL_POINTS_RE, log),
        "sh_degree": _search_int(_SH_DEGREE_RE, log),
        "iterations_configured": _search_int(_ITERATIONS_CONFIGURED_RE, log),
        "final_gaussians": _search_int(_FINAL_GAUSSIANS_RE, log),
        "duration_seconds": float(duration_match.group(1)) if duration_match else None,
        "avg_iter_per_sec": float(duration_match.group(2)) if duration_match else None,
    }


def training_metrics_path() -> Path:
    return data_root() / "training_runs.jsonl"


def append_training_metrics(metrics: TrainingMetrics) -> None:
    path = training_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(metrics)) + "\n")
