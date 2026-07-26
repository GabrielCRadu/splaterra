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

    def fake_run(command, capture_output, text, check, timeout):
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

    def fake_run(command, capture_output, text, check, timeout):
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


def test_sample_usage_parses_cpu_ram_and_gpu(monkeypatch):
    monkeypatch.setattr(hardware.psutil, "cpu_percent", lambda interval=None: 42.5)
    monkeypatch.setattr(
        hardware.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=17179869184, available=8589934592),
    )
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "95, 9637, 12288\n"
        result.stderr = ""
        return result

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    usage = hardware.sample_usage()

    assert usage.cpu_percent == 42.5
    assert usage.ram_used_gb == 8.0
    assert usage.ram_total_gb == 16.0
    assert usage.gpu_percent == 95.0
    assert usage.vram_used_mb == 9637
    assert usage.vram_total_mb == 12288


def test_sample_usage_gpu_fields_none_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(hardware.psutil, "cpu_percent", lambda interval=None: 10.0)
    monkeypatch.setattr(
        hardware.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=8589934592, available=4294967296),
    )
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)

    usage = hardware.sample_usage()

    assert usage.gpu_percent is None
    assert usage.vram_used_mb is None
    assert usage.vram_total_mb is None


def test_sample_usage_gpu_fields_none_when_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(hardware.psutil, "cpu_percent", lambda interval=None: 10.0)
    monkeypatch.setattr(
        hardware.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=8589934592, available=4294967296),
    )
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "NVIDIA-SMI has failed"
        return result

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    usage = hardware.sample_usage()

    assert usage.gpu_percent is None
    assert usage.vram_used_mb is None
    assert usage.vram_total_mb is None
