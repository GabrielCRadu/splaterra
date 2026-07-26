from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class GpuInfo:
    name: str
    vram_mb: int
    compute_capability: float


@dataclass
class HardwareInfo:
    gpu: GpuInfo | None
    cpu_cores: int
    ram_gb: float


@dataclass
class UsageSnapshot:
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_percent: float | None
    vram_used_mb: int | None
    vram_total_mb: int | None


def _run_nvidia_smi(query: str, *, timeout: float = 5.0) -> str | None:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None

    command = [binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    if process.returncode != 0 or not process.stdout.strip():
        return None
    return process.stdout.strip().splitlines()[0]


def _detect_gpu() -> GpuInfo | None:
    first_line = _run_nvidia_smi("name,memory.total,compute_cap")
    if first_line is None:
        return None

    name, vram_mb, compute_cap = (part.strip() for part in first_line.split(","))
    return GpuInfo(name=name, vram_mb=int(float(vram_mb)), compute_capability=float(compute_cap))


def detect_hardware() -> HardwareInfo:
    return HardwareInfo(
        gpu=_detect_gpu(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=psutil.virtual_memory().total / (1024**3),
    )


def sample_usage() -> UsageSnapshot:
    mem = psutil.virtual_memory()
    gpu_percent: float | None = None
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None

    first_line = _run_nvidia_smi("utilization.gpu,memory.used,memory.total")
    if first_line is not None:
        util, used, total = (part.strip() for part in first_line.split(","))
        gpu_percent = float(util)
        vram_used_mb = int(float(used))
        vram_total_mb = int(float(total))

    return UsageSnapshot(
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_used_gb=(mem.total - mem.available) / (1024**3),
        ram_total_gb=mem.total / (1024**3),
        gpu_percent=gpu_percent,
        vram_used_mb=vram_used_mb,
        vram_total_mb=vram_total_mb,
    )
