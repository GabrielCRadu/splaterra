from types import SimpleNamespace
from unittest.mock import MagicMock

from app import hardware


def test_detect_hardware_no_gpu_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        hardware.psutil, "virtual_memory", lambda: SimpleNamespace(total=17179869184)
    )

    info = hardware.detect_hardware()

    assert info.gpu is None
    assert info.cpu_cores == 8
    assert info.ram_gb == 16.0


def test_detect_hardware_parses_gpu_info(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        hardware.psutil, "virtual_memory", lambda: SimpleNamespace(total=34359738368)
    )

    def fake_run(command, capture_output, text, check):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "NVIDIA GeForce RTX 4080, 16376, 8.9\n"
        result.stderr = ""
        return result

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    info = hardware.detect_hardware()

    assert info.gpu is not None
    assert info.gpu.name == "NVIDIA GeForce RTX 4080"
    assert info.gpu.vram_mb == 16376
    assert info.gpu.compute_capability == 8.9
    assert info.cpu_cores == 16
    assert info.ram_gb == 32.0


def test_detect_hardware_no_gpu_when_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        hardware.psutil, "virtual_memory", lambda: SimpleNamespace(total=8589934592)
    )

    def fake_run(command, capture_output, text, check):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "NVIDIA-SMI has failed"
        return result

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    info = hardware.detect_hardware()

    assert info.gpu is None
    assert info.cpu_cores == 4
    assert info.ram_gb == 8.0
