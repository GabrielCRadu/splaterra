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


def _detect_gpu() -> GpuInfo | None:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None

    command = [
        binary,
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0 or not process.stdout.strip():
        return None

    first_line = process.stdout.strip().splitlines()[0]
    name, vram_mb, compute_cap = (part.strip() for part in first_line.split(","))
    return GpuInfo(name=name, vram_mb=int(float(vram_mb)), compute_capability=float(compute_cap))


def detect_hardware() -> HardwareInfo:
    return HardwareInfo(
        gpu=_detect_gpu(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=psutil.virtual_memory().total / (1024**3),
    )
