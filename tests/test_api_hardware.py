from unittest.mock import MagicMock

from app.api import hardware as hardware_api
from app.hardware import GpuInfo, HardwareInfo


def test_get_hardware_returns_detected_info(client, monkeypatch):
    monkeypatch.setattr(
        hardware_api,
        "detect_hardware",
        MagicMock(
            return_value=HardwareInfo(
                gpu=GpuInfo(name="NVIDIA GeForce RTX 4080", vram_mb=16376, compute_capability=8.9),
                cpu_cores=16,
                ram_gb=32.0,
            )
        ),
    )

    response = client.get("/hardware")

    assert response.status_code == 200
    body = response.json()
    assert body["cpu_cores"] == 16
    assert body["ram_gb"] == 32.0
    assert body["gpu"]["name"] == "NVIDIA GeForce RTX 4080"


def test_get_hardware_reports_no_gpu(client, monkeypatch):
    monkeypatch.setattr(
        hardware_api,
        "detect_hardware",
        MagicMock(return_value=HardwareInfo(gpu=None, cpu_cores=8, ram_gb=16.0)),
    )

    response = client.get("/hardware")

    assert response.status_code == 200
    assert response.json()["gpu"] is None
