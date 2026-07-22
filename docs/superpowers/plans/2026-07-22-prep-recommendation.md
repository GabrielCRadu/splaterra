# Prep Recommendation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the recommendation engine for the Gaussian Splatting prep app: read technical video metadata and optional DJI drone telemetry, detect local hardware (GPU/CPU/RAM), and combine them into suggested RealityScan/LichtFeld settings plus a local-vs-cloud verdict — producing a working, testable "point me at a job's ingested video, get back settings + a cloud/local recommendation" feature, built on top of Plan 1's foundation (already merged to `master`).

**Architecture:** Three independent, side-effect-light modules feed a pure recommendation engine: `app/video_metadata.py` (ffprobe subprocess wrapper), `app/drone_telemetry.py` (pure DJI SRT text parser, no subprocess), `app/hardware.py` (nvidia-smi subprocess wrapper + psutil for CPU/RAM). `app/recommendation.py` is a pure function with zero I/O that combines their outputs into a `Recommendation`. `app/recommendation_service.py` orchestrates them against a job (same `JobStore`/stage pattern as Plan 1), and `app/api/hardware.py` / `app/api/recommendation.py` expose it over HTTP.

**Tech Stack:** Same as Plan 1 (Python 3.11+, FastAPI, SQLite, pytest), plus `psutil` for cross-platform RAM detection and the system's `ffprobe`/`nvidia-smi` binaries (subprocess, never a Python binding).

## Global Constraints

- Every external-tool wrapper (`ffprobe`, `nvidia-smi`) is invoked as a subprocess and tested via subprocess mocks, never actually run in unit tests (spec §9, same discipline as Plan 1's `sharp_frames.py`).
- `app/recommendation.py` (the engine itself) takes only in-memory dataclasses as input and does no I/O — fully unit-testable with no mocking.
- Any stage failure — including a pre-flight validation failure before the stage's "running" upsert — must be recorded as a `failed` row in `job_stages`, EXCEPT when `job_id` itself doesn't exist (no row to attach to, since `job_stages.job_id` has a foreign-key reference to `jobs.id` under `PRAGMA foreign_keys = ON`). This is the same binding rule established and applied consistently across Plan 1's ingestion, upload, and extraction code — apply it identically here.
- LichtFeld's documented minimum for local training — established via research during the design phase, used here as the verdict engine's threshold — is an NVIDIA GPU with compute capability ≥ 7.5 and ≥ 8 GB VRAM.
- Job/stage state persists in SQLite as a single file (spec §6); this plan extends the existing `jobs` table with one new nullable column (`recommendation_json`) rather than adding a new table.

---

## Task 1: Video metadata module (`ffprobe` wrapper)

**Files:**
- Create: `app/video_metadata.py`
- Test: `tests/test_video_metadata.py`

**Interfaces:**
- Consumes: nothing (pure subprocess wrapper, same shape as Plan 1's `app/sharp_frames.py`).
- Produces: `probe_video(video_path: Path) -> VideoMetadata` (dataclass: `width: int, height: int, fps: float, duration_seconds: float, codec: str`) and `VideoMetadataError`. Task 4 (recommendation engine) and Task 6 (recommendation service) consume `VideoMetadata` and `VideoMetadataError`.

- [ ] **Step 1: Write the failing tests**

`tests/test_video_metadata.py`:
```python
import json
from unittest.mock import MagicMock

import pytest

from app import video_metadata


FFPROBE_OUTPUT_1080P_30FPS = json.dumps(
    {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
            }
        ],
        "format": {"duration": "12.500000"},
    }
)


def test_probe_video_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(video_metadata.shutil, "which", lambda name: None)
    with pytest.raises(video_metadata.VideoMetadataError, match="not found on PATH"):
        video_metadata.probe_video(tmp_path / "video.mp4")


def test_probe_video_parses_resolution_fps_duration_codec(tmp_path, monkeypatch):
    monkeypatch.setattr(video_metadata.shutil, "which", lambda name: "/usr/bin/ffprobe")

    def fake_run(command, capture_output, text, check):
        result = MagicMock()
        result.returncode = 0
        result.stdout = FFPROBE_OUTPUT_1080P_30FPS
        result.stderr = ""
        return result

    monkeypatch.setattr(video_metadata.subprocess, "run", fake_run)

    meta = video_metadata.probe_video(tmp_path / "video.mp4")

    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.fps == 30.0
    assert meta.duration_seconds == 12.5
    assert meta.codec == "h264"


def test_probe_video_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(video_metadata.shutil, "which", lambda name: "/usr/bin/ffprobe")

    def fake_run(command, capture_output, text, check):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "Invalid data found when processing input"
        return result

    monkeypatch.setattr(video_metadata.subprocess, "run", fake_run)

    with pytest.raises(video_metadata.VideoMetadataError, match="Invalid data"):
        video_metadata.probe_video(tmp_path / "video.mp4")


def test_probe_video_raises_when_no_video_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(video_metadata.shutil, "which", lambda name: "/usr/bin/ffprobe")

    def fake_run(command, capture_output, text, check):
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"streams": [{"codec_type": "audio"}], "format": {}})
        result.stderr = ""
        return result

    monkeypatch.setattr(video_metadata.subprocess, "run", fake_run)

    with pytest.raises(video_metadata.VideoMetadataError, match="No video stream"):
        video_metadata.probe_video(tmp_path / "video.mp4")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_video_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.video_metadata'`

- [ ] **Step 3: Write `app/video_metadata.py`**

```python
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class VideoMetadataError(Exception):
    pass


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration_seconds: float
    codec: str


def _ffprobe_binary() -> str:
    binary = shutil.which("ffprobe")
    if binary is None:
        raise VideoMetadataError(
            "ffprobe not found on PATH. Install ffmpeg (which includes ffprobe)."
        )
    return binary


def probe_video(video_path: Path) -> VideoMetadata:
    binary = _ffprobe_binary()
    command = [
        binary,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise VideoMetadataError(f"ffprobe exited with {process.returncode}:\n{process.stderr}")

    try:
        data = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise VideoMetadataError(f"ffprobe produced invalid JSON: {exc}") from exc

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise VideoMetadataError(f"No video stream found in {video_path}")

    return VideoMetadata(
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=_parse_frame_rate(video_stream.get("r_frame_rate", "0/1")),
        duration_seconds=float(data.get("format", {}).get("duration", 0.0)),
        codec=str(video_stream.get("codec_name", "unknown")),
    )


def _parse_frame_rate(raw: str) -> float:
    numerator, _, denominator = raw.partition("/")
    denom = float(denominator) if denominator else 1.0
    if denom == 0:
        return 0.0
    return float(numerator) / denom
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_video_metadata.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/video_metadata.py tests/test_video_metadata.py
git commit -m "feat: add ffprobe video metadata wrapper"
```

---

## Task 2: DJI drone SRT telemetry parser

**Files:**
- Create: `app/drone_telemetry.py`
- Test: `tests/test_drone_telemetry.py`

**Interfaces:**
- Consumes: nothing (pure text parsing, no subprocess, no dependency on other Plan 2 modules).
- Produces: `parse_srt(srt_path: Path) -> DroneTelemetry` (dataclass: `sample_count: int, has_gps: bool, avg_altitude_m: float | None, min_altitude_m: float | None, max_altitude_m: float | None`) and `SrtParseError`. Task 4 and Task 6 consume `DroneTelemetry` and `SrtParseError`.

- [ ] **Step 1: Write the failing tests**

`tests/test_drone_telemetry.py`:
```python
import pytest

from app.drone_telemetry import SrtParseError, parse_srt

DJI_SRT_TWO_SAMPLES = """1
00:00:00,000 --> 00:00:00,033
<font size="36">SrtCnt : 1, DiffTime : 33ms
2024-01-01 12:00:00
[iso : 100] [shutter : 1/1000.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 24.00] [latitude: 22.543000] [longitude: 114.057000] [rel_alt: 50.100 abs_alt: 120.400] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="36">SrtCnt : 2, DiffTime : 33ms
2024-01-01 12:00:00
[iso : 100] [shutter : 1/1000.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 24.00] [latitude: 22.543010] [longitude: 114.057010] [rel_alt: 51.200 abs_alt: 121.500] </font>

"""

PLAIN_SUBTITLE_NO_TELEMETRY = """1
00:00:00,000 --> 00:00:02,000
Just a regular subtitle, no drone data here.

"""


def test_parse_srt_missing_file_raises(tmp_path):
    with pytest.raises(SrtParseError, match="not found"):
        parse_srt(tmp_path / "missing.srt")


def test_parse_srt_extracts_altitude_and_gps(tmp_path):
    srt_path = tmp_path / "flight.srt"
    srt_path.write_text(DJI_SRT_TWO_SAMPLES, encoding="utf-8")

    telemetry = parse_srt(srt_path)

    assert telemetry.sample_count == 2
    assert telemetry.has_gps is True
    assert telemetry.min_altitude_m == 50.1
    assert telemetry.max_altitude_m == 51.2
    assert telemetry.avg_altitude_m == pytest.approx(50.65)


def test_parse_srt_plain_subtitle_has_no_telemetry(tmp_path):
    srt_path = tmp_path / "captions.srt"
    srt_path.write_text(PLAIN_SUBTITLE_NO_TELEMETRY, encoding="utf-8")

    telemetry = parse_srt(srt_path)

    assert telemetry.sample_count == 0
    assert telemetry.has_gps is False
    assert telemetry.avg_altitude_m is None
    assert telemetry.min_altitude_m is None
    assert telemetry.max_altitude_m is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_drone_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.drone_telemetry'`

- [ ] **Step 3: Write `app/drone_telemetry.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class SrtParseError(Exception):
    pass


@dataclass
class DroneTelemetry:
    sample_count: int
    has_gps: bool
    avg_altitude_m: float | None
    min_altitude_m: float | None
    max_altitude_m: float | None


_ALTITUDE_RE = re.compile(r"rel_alt:\s*(-?\d+(?:\.\d+)?)")
_GPS_RE = re.compile(r"latitude:\s*(-?\d+(?:\.\d+)?)")


def parse_srt(srt_path: Path) -> DroneTelemetry:
    if not srt_path.is_file():
        raise SrtParseError(f"SRT file not found: {srt_path}")

    text = srt_path.read_text(encoding="utf-8", errors="ignore")

    altitudes = [float(match) for match in _ALTITUDE_RE.findall(text)]
    has_gps = _GPS_RE.search(text) is not None

    if not altitudes:
        return DroneTelemetry(
            sample_count=0,
            has_gps=has_gps,
            avg_altitude_m=None,
            min_altitude_m=None,
            max_altitude_m=None,
        )

    return DroneTelemetry(
        sample_count=len(altitudes),
        has_gps=has_gps,
        avg_altitude_m=sum(altitudes) / len(altitudes),
        min_altitude_m=min(altitudes),
        max_altitude_m=max(altitudes),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_drone_telemetry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/drone_telemetry.py tests/test_drone_telemetry.py
git commit -m "feat: add DJI drone SRT telemetry parser"
```

---

## Task 3: Hardware detection module

**Files:**
- Modify: `requirements.txt`
- Create: `app/hardware.py`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: nothing (subprocess wrapper for GPU + `psutil`/`os` for CPU/RAM).
- Produces: `detect_hardware() -> HardwareInfo` (dataclass: `gpu: GpuInfo | None, cpu_cores: int, ram_gb: float`); `GpuInfo` (dataclass: `name: str, vram_mb: int, compute_capability: float`). No exception type — GPU absence is a valid, non-error result (`gpu=None`). Task 4 and Task 6 consume `HardwareInfo` and `GpuInfo`.

- [ ] **Step 1: Add `psutil` to `requirements.txt`**

Append to `requirements.txt`:
```
psutil>=6.0
```

- [ ] **Step 2: Install the new dependency**

Run: `python -m pip install -r requirements.txt`
Expected: `psutil` installs without errors.

- [ ] **Step 3: Write the failing tests**

`tests/test_hardware.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_hardware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.hardware'`

- [ ] **Step 5: Write `app/hardware.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_hardware.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add requirements.txt app/hardware.py tests/test_hardware.py
git commit -m "feat: add GPU/CPU/RAM hardware detection"
```

---

## Task 4: Recommendation engine (pure function)

**Files:**
- Create: `app/recommendation.py`
- Test: `tests/test_recommendation.py`

**Interfaces:**
- Consumes: `VideoMetadata` (Task 1), `DroneTelemetry` (Task 2), `HardwareInfo`/`GpuInfo` (Task 3).
- Produces: `recommend(video, drone, hardware) -> Recommendation` (dataclass composing `RealityScanSettings(downscale: int, rolling_shutter_correction: bool)`, `LichtFeldSettings(max_gaussians: int, bf16: bool)`, `Verdict(use_cloud: bool, reason: str)`). Pure function, zero I/O. Task 6 (recommendation service) and Task 7 (API) consume `recommend` and all four dataclasses.

- [ ] **Step 1: Write the failing tests**

`tests/test_recommendation.py`:
```python
from app.drone_telemetry import DroneTelemetry
from app.hardware import GpuInfo, HardwareInfo
from app.recommendation import recommend
from app.video_metadata import VideoMetadata

VIDEO_1080P = VideoMetadata(width=1920, height=1080, fps=30.0, duration_seconds=10.0, codec="h264")
VIDEO_4K = VideoMetadata(width=3840, height=2160, fps=30.0, duration_seconds=10.0, codec="h264")
VIDEO_8K = VideoMetadata(width=7680, height=4320, fps=30.0, duration_seconds=10.0, codec="hevc")

NO_DRONE = None
DRONE_TELEMETRY = DroneTelemetry(
    sample_count=100, has_gps=True, avg_altitude_m=50.0, min_altitude_m=40.0, max_altitude_m=60.0
)

NO_GPU_HW = HardwareInfo(gpu=None, cpu_cores=8, ram_gb=16.0)
STRONG_GPU_HW = HardwareInfo(
    gpu=GpuInfo(name="NVIDIA GeForce RTX 4080", vram_mb=12288, compute_capability=8.9),
    cpu_cores=16,
    ram_gb=32.0,
)
WEAK_COMPUTE_GPU_HW = HardwareInfo(
    # compute_capability strictly below the 7.5 minimum, with plenty of VRAM
    # (12 GB) so the VRAM check would pass if reached — isolates the
    # compute-capability branch from the VRAM branch.
    gpu=GpuInfo(name="NVIDIA GeForce GTX 1660", vram_mb=12288, compute_capability=6.1),
    cpu_cores=8,
    ram_gb=16.0,
)
LOW_VRAM_GPU_HW = HardwareInfo(
    gpu=GpuInfo(name="NVIDIA GeForce RTX 3050", vram_mb=6144, compute_capability=8.6),
    cpu_cores=8,
    ram_gb=16.0,
)


def test_no_gpu_recommends_cloud():
    rec = recommend(VIDEO_1080P, NO_DRONE, NO_GPU_HW)
    assert rec.verdict.use_cloud is True
    assert "No NVIDIA GPU" in rec.verdict.reason
    assert rec.lichtfeld.max_gaussians == 0
    assert rec.lichtfeld.bf16 is False


def test_compute_capability_below_minimum_recommends_cloud():
    rec = recommend(VIDEO_1080P, NO_DRONE, WEAK_COMPUTE_GPU_HW)
    assert rec.verdict.use_cloud is True
    assert "compute capability" in rec.verdict.reason


def test_vram_below_minimum_recommends_cloud():
    rec = recommend(VIDEO_1080P, NO_DRONE, LOW_VRAM_GPU_HW)
    assert rec.verdict.use_cloud is True
    assert "VRAM" in rec.verdict.reason


def test_strong_gpu_recommends_local():
    rec = recommend(VIDEO_1080P, NO_DRONE, STRONG_GPU_HW)
    assert rec.verdict.use_cloud is False
    assert rec.lichtfeld.bf16 is True
    # 12288 MB = 12 GB * 400_000 gaussians/GB = 4_800_000, well under the 6M cap
    assert rec.lichtfeld.max_gaussians == 4_800_000


def test_high_res_video_downscales_realityscan():
    rec_1080p = recommend(VIDEO_1080P, NO_DRONE, STRONG_GPU_HW)
    rec_4k = recommend(VIDEO_4K, NO_DRONE, STRONG_GPU_HW)
    rec_8k = recommend(VIDEO_8K, NO_DRONE, STRONG_GPU_HW)

    assert rec_1080p.realityscan.downscale == 1
    assert rec_4k.realityscan.downscale == 1
    assert rec_8k.realityscan.downscale == 2


def test_drone_footage_enables_rolling_shutter_correction():
    rec_no_drone = recommend(VIDEO_1080P, NO_DRONE, STRONG_GPU_HW)
    rec_drone = recommend(VIDEO_1080P, DRONE_TELEMETRY, STRONG_GPU_HW)

    assert rec_no_drone.realityscan.rolling_shutter_correction is False
    assert rec_drone.realityscan.rolling_shutter_correction is True


def test_max_gaussians_capped_for_very_high_vram():
    huge_gpu_hw = HardwareInfo(
        gpu=GpuInfo(name="NVIDIA H100", vram_mb=81920, compute_capability=9.0),
        cpu_cores=32,
        ram_gb=128.0,
    )
    rec = recommend(VIDEO_1080P, NO_DRONE, huge_gpu_hw)
    assert rec.lichtfeld.max_gaussians == 6_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_recommendation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.recommendation'`

- [ ] **Step 3: Write `app/recommendation.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from app.drone_telemetry import DroneTelemetry
from app.hardware import GpuInfo, HardwareInfo
from app.video_metadata import VideoMetadata

LICHTFELD_MIN_COMPUTE_CAPABILITY = 7.5
LICHTFELD_MIN_VRAM_GB = 8.0
BF16_MIN_COMPUTE_CAPABILITY = 8.0
GAUSSIANS_PER_VRAM_GB = 400_000
MAX_GAUSSIANS_CAP = 6_000_000
HIGH_RES_PIXEL_THRESHOLD = 3840 * 2160


@dataclass
class RealityScanSettings:
    downscale: int
    rolling_shutter_correction: bool


@dataclass
class LichtFeldSettings:
    max_gaussians: int
    bf16: bool


@dataclass
class Verdict:
    use_cloud: bool
    reason: str


@dataclass
class Recommendation:
    realityscan: RealityScanSettings
    lichtfeld: LichtFeldSettings
    verdict: Verdict


def recommend(
    video: VideoMetadata,
    drone: DroneTelemetry | None,
    hardware: HardwareInfo,
) -> Recommendation:
    is_high_res = video.width * video.height > HIGH_RES_PIXEL_THRESHOLD
    realityscan = RealityScanSettings(
        downscale=2 if is_high_res else 1,
        rolling_shutter_correction=drone is not None,
    )

    gpu = hardware.gpu
    vram_gb = gpu.vram_mb / 1024 if gpu else 0.0
    max_gaussians = min(MAX_GAUSSIANS_CAP, int(vram_gb * GAUSSIANS_PER_VRAM_GB))
    bf16 = gpu is not None and gpu.compute_capability >= BF16_MIN_COMPUTE_CAPABILITY
    lichtfeld = LichtFeldSettings(max_gaussians=max_gaussians, bf16=bf16)

    verdict = _decide_verdict(gpu, vram_gb)

    return Recommendation(realityscan=realityscan, lichtfeld=lichtfeld, verdict=verdict)


def _decide_verdict(gpu: GpuInfo | None, vram_gb: float) -> Verdict:
    if gpu is None:
        return Verdict(use_cloud=True, reason="No NVIDIA GPU detected locally")
    if gpu.compute_capability < LICHTFELD_MIN_COMPUTE_CAPABILITY:
        return Verdict(
            use_cloud=True,
            reason=(
                f"GPU compute capability {gpu.compute_capability} is below "
                f"LichtFeld's minimum ({LICHTFELD_MIN_COMPUTE_CAPABILITY})"
            ),
        )
    if vram_gb < LICHTFELD_MIN_VRAM_GB:
        return Verdict(
            use_cloud=True,
            reason=(
                f"GPU VRAM {vram_gb:.1f} GB is below LichtFeld's minimum "
                f"({LICHTFELD_MIN_VRAM_GB} GB)"
            ),
        )
    return Verdict(use_cloud=False, reason="Local GPU meets LichtFeld's minimum requirements")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_recommendation.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add app/recommendation.py tests/test_recommendation.py
git commit -m "feat: add pure recommendation engine"
```

---

## Task 5: Extend job schema with `recommendation_json`

**Files:**
- Modify: `app/db.py`
- Modify: `app/models.py`
- Modify: `app/job_store.py`
- Test: `tests/test_job_store.py` (add one test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Job.recommendation_json: str | None` field now exists and round-trips through `JobStore.create_job`/`get_job`/`update_job` exactly like every other nullable `Job` field. Task 6 sets this field via `store.update_job(job_id, recommendation_json=...)`.

Note: if a local `data/splatprep.db` file already exists on disk from manual testing, delete it after this task — `CREATE TABLE IF NOT EXISTS` does not alter an existing table's columns, and this project has no production data yet, so recreating the file is the simplest path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_job_store.py` (alongside the existing tests in that file):
```python
def test_update_job_sets_recommendation_json(job_store):
    job = job_store.create_job("Test")
    assert job.recommendation_json is None

    job_store.update_job(job.id, recommendation_json='{"verdict": {"use_cloud": false}}')
    fetched = job_store.get_job(job.id)
    assert fetched.recommendation_json == '{"verdict": {"use_cloud": false}}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_job_store.py -v`
Expected: FAIL with `AttributeError: 'Job' object has no attribute 'recommendation_json'`

- [ ] **Step 3: Replace the full contents of `app/db.py`**

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.paths import data_root

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    raw_video_path TEXT,
    frames_dir TEXT,
    frame_count INTEGER,
    recommendation_json TEXT
);

CREATE TABLE IF NOT EXISTS job_stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    log TEXT,
    error TEXT,
    UNIQUE(job_id, stage)
);
"""


def db_path() -> Path:
    return data_root() / "splatprep.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn
```

- [ ] **Step 4: Replace the full contents of `app/models.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Job:
    id: str
    name: str
    created_at: str
    raw_video_path: str | None = None
    frames_dir: str | None = None
    frame_count: int | None = None
    recommendation_json: str | None = None


@dataclass
class JobStage:
    job_id: str
    stage: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    log: str | None = None
    error: str | None = None
```

- [ ] **Step 5: Update `_row_to_job` in `app/job_store.py`**

Find the `_row_to_job` static method (it currently constructs a `Job` from five columns) and add the sixth field:

```python
    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            raw_video_path=row["raw_video_path"],
            frames_dir=row["frames_dir"],
            frame_count=row["frame_count"],
            recommendation_json=row["recommendation_json"],
        )
```

Leave every other method in `app/job_store.py` (`create_job`, `get_job`, `list_jobs`, `update_job`, `upsert_stage`, `get_stage`, `_row_to_stage`) exactly as they are — `update_job(**fields)` already handles arbitrary columns generically and needs no change.

- [ ] **Step 6: Delete any stale local database file**

Run: `rm -f data/splatprep.db` (if the file exists; harmless if it doesn't — this repo's `data/` directory is gitignored and only ever holds local runtime state, never committed).

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_db.py tests/test_job_store.py tests/test_api_jobs.py -v`
Expected: PASS (9 passed — the 8 pre-existing plus the new `recommendation_json` test)

- [ ] **Step 8: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all pass, 57 passed (39 from Plan 1 + 17 from Tasks 1-4 + 1 new in this task)

- [ ] **Step 9: Commit**

```bash
git add app/db.py app/models.py app/job_store.py tests/test_job_store.py
git commit -m "feat: add recommendation_json column to jobs table"
```

---

## Task 6: Recommendation service

**Files:**
- Create: `app/recommendation_service.py`
- Test: `tests/test_recommendation_service.py`

**Interfaces:**
- Consumes: `JobStore` (Plan 1 Task 3), `probe_video`/`VideoMetadataError` (Task 1), `parse_srt`/`SrtParseError` (Task 2), `detect_hardware` (Task 3), `recommend`/`Recommendation` (Task 4).
- Produces: `compute_recommendation(store: JobStore, job_id: str, srt_path: Path | None = None) -> Recommendation` and `RecommendationError`. Sets `job.recommendation_json` (JSON-serialized `Recommendation`) and the `recommendation` stage to `done` (or `failed`, per the binding pre-flight rule — except when `job_id` doesn't exist at all). Task 7 (API) calls this directly.

- [ ] **Step 1: Write the failing tests**

`tests/test_recommendation_service.py`:
```python
import json
from unittest.mock import MagicMock

import pytest

from app import recommendation_service
from app.drone_telemetry import SrtParseError
from app.hardware import HardwareInfo
from app.recommendation import LichtFeldSettings, Recommendation, RealityScanSettings, Verdict
from app.video_metadata import VideoMetadata, VideoMetadataError

FAKE_RECOMMENDATION = Recommendation(
    realityscan=RealityScanSettings(downscale=1, rolling_shutter_correction=False),
    lichtfeld=LichtFeldSettings(max_gaussians=1_000_000, bf16=True),
    verdict=Verdict(use_cloud=False, reason="Local GPU meets LichtFeld's minimum requirements"),
)


def test_compute_recommendation_updates_job_and_stage(job_store, tmp_path, monkeypatch):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    monkeypatch.setattr(
        recommendation_service, "probe_video", MagicMock(return_value=VideoMetadata(
            width=1920, height=1080, fps=30.0, duration_seconds=10.0, codec="h264"
        ))
    )
    monkeypatch.setattr(recommendation_service, "detect_hardware", MagicMock(
        return_value=HardwareInfo(gpu=None, cpu_cores=8, ram_gb=16.0)
    ))
    monkeypatch.setattr(
        recommendation_service, "recommend", MagicMock(return_value=FAKE_RECOMMENDATION)
    )

    result = recommendation_service.compute_recommendation(job_store, job.id)

    assert result == FAKE_RECOMMENDATION

    fetched = job_store.get_job(job.id)
    assert fetched.recommendation_json is not None
    payload = json.loads(fetched.recommendation_json)
    assert payload["verdict"]["use_cloud"] is False
    assert payload["lichtfeld"]["max_gaussians"] == 1_000_000

    stage = job_store.get_stage(job.id, "recommendation")
    assert stage.status == "done"


def test_compute_recommendation_without_ingested_video_records_failed_stage(job_store):
    job = job_store.create_job("Test")

    with pytest.raises(recommendation_service.RecommendationError, match="no ingested video"):
        recommendation_service.compute_recommendation(job_store, job.id)

    stage = job_store.get_stage(job.id, "recommendation")
    assert stage is not None
    assert stage.status == "failed"
    assert stage.error is not None


def test_compute_recommendation_unknown_job_raises_without_stage_write(job_store):
    with pytest.raises(recommendation_service.RecommendationError, match="Unknown job"):
        recommendation_service.compute_recommendation(job_store, "does-not-exist")

    assert job_store.get_stage("does-not-exist", "recommendation") is None


def test_compute_recommendation_marks_stage_failed_on_video_metadata_error(
    job_store, tmp_path, monkeypatch
):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    monkeypatch.setattr(
        recommendation_service,
        "probe_video",
        MagicMock(side_effect=VideoMetadataError("ffprobe not found on PATH")),
    )

    with pytest.raises(recommendation_service.RecommendationError, match="ffprobe not found"):
        recommendation_service.compute_recommendation(job_store, job.id)

    stage = job_store.get_stage(job.id, "recommendation")
    assert stage.status == "failed"
    assert stage.error == "ffprobe not found on PATH"


def test_compute_recommendation_marks_stage_failed_on_srt_parse_error(
    job_store, tmp_path, monkeypatch
):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path=str(tmp_path / "video.mp4"))

    monkeypatch.setattr(
        recommendation_service, "probe_video", MagicMock(return_value=VideoMetadata(
            width=1920, height=1080, fps=30.0, duration_seconds=10.0, codec="h264"
        ))
    )
    monkeypatch.setattr(
        recommendation_service,
        "parse_srt",
        MagicMock(side_effect=SrtParseError("SRT file not found: bad.srt")),
    )

    with pytest.raises(recommendation_service.RecommendationError, match="SRT file not found"):
        recommendation_service.compute_recommendation(job_store, job.id, srt_path=tmp_path / "bad.srt")

    stage = job_store.get_stage(job.id, "recommendation")
    assert stage.status == "failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_recommendation_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.recommendation_service'`

- [ ] **Step 3: Write `app/recommendation_service.py`**

```python
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from app.drone_telemetry import SrtParseError, parse_srt
from app.hardware import detect_hardware
from app.job_store import JobStore
from app.recommendation import Recommendation, recommend
from app.video_metadata import VideoMetadataError, probe_video


class RecommendationError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_recommendation(
    store: JobStore, job_id: str, srt_path: Path | None = None
) -> Recommendation:
    job = store.get_job(job_id)
    if job is None:
        # No stage write: job_stages.job_id has a foreign-key reference to
        # jobs.id (PRAGMA foreign_keys = ON in app/db.py), so there is no row
        # to attach a stage to for a job_id that doesn't exist.
        raise RecommendationError(f"Unknown job: {job_id}")
    if job.raw_video_path is None:
        message = "Job has no ingested video yet"
        now = _now()
        store.upsert_stage(
            job_id, "recommendation", "failed", started_at=now, finished_at=now, error=message
        )
        raise RecommendationError(message)

    store.upsert_stage(job_id, "recommendation", "running", started_at=_now())
    try:
        video_meta = probe_video(Path(job.raw_video_path))
        drone = parse_srt(srt_path) if srt_path is not None else None
        hardware = detect_hardware()
        recommendation = recommend(video_meta, drone, hardware)
    except (VideoMetadataError, SrtParseError) as exc:
        store.upsert_stage(job_id, "recommendation", "failed", finished_at=_now(), error=str(exc))
        raise RecommendationError(str(exc)) from exc

    payload = json.dumps(dataclasses.asdict(recommendation))
    store.update_job(job_id, recommendation_json=payload)
    store.upsert_stage(job_id, "recommendation", "done", finished_at=_now())
    return recommendation
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_recommendation_service.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/recommendation_service.py tests/test_recommendation_service.py
git commit -m "feat: add recommendation service orchestrating metadata, hardware, and engine"
```

---

## Task 7: Hardware and recommendation API endpoints

**Files:**
- Create: `app/api/hardware.py`
- Create: `app/api/recommendation.py`
- Modify: `app/main.py`
- Test: `tests/test_api_hardware.py`
- Test: `tests/test_api_recommendation.py`

**Interfaces:**
- Consumes: `detect_hardware` (Task 3), `get_job_store` (Plan 1 Task 4), `compute_recommendation`/`RecommendationError` (Task 6).
- Produces: `GET /hardware` (returns `HardwareInfo` as JSON), `POST /jobs/{job_id}/recommend` (body `{"srt_path": str | null}`, returns the computed `Recommendation` as JSON; 404 if job doesn't exist, 400 on `RecommendationError`), `GET /jobs/{job_id}/recommendation` (returns the stored `Recommendation` as JSON; 404 if job doesn't exist or no recommendation computed yet).

- [ ] **Step 1: Write the failing tests**

`tests/test_api_hardware.py`:
```python
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
```

`tests/test_api_recommendation.py`:
```python
from app.api import recommendation as recommendation_api
from app.recommendation import LichtFeldSettings, Recommendation, RealityScanSettings, Verdict

FAKE_RECOMMENDATION = Recommendation(
    realityscan=RealityScanSettings(downscale=1, rolling_shutter_correction=False),
    lichtfeld=LichtFeldSettings(max_gaussians=1_000_000, bf16=True),
    verdict=Verdict(use_cloud=False, reason="Local GPU meets LichtFeld's minimum requirements"),
)


def test_recommend_endpoint_returns_computed_recommendation(client, monkeypatch):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    monkeypatch.setattr(
        recommendation_api,
        "compute_recommendation",
        lambda store, job_id, srt_path=None: FAKE_RECOMMENDATION,
    )

    response = client.post(f"/jobs/{job_id}/recommend", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["realityscan"]["downscale"] == 1
    assert body["lichtfeld"]["max_gaussians"] == 1_000_000
    assert body["verdict"]["use_cloud"] is False


def test_recommend_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/recommend", json={})
    assert response.status_code == 404


def test_recommend_endpoint_returns_400_on_error(client, monkeypatch):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    def raise_error(store, job_id, srt_path=None):
        raise recommendation_api.RecommendationError("Job has no ingested video yet")

    monkeypatch.setattr(recommendation_api, "compute_recommendation", raise_error)

    response = client.post(f"/jobs/{job_id}/recommend", json={})
    assert response.status_code == 400


def test_get_recommendation_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/recommendation")
    assert response.status_code == 404


def test_get_recommendation_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/recommendation")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_hardware.py tests/test_api_recommendation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.hardware'`

- [ ] **Step 3: Write `app/api/hardware.py`**

```python
from __future__ import annotations

import dataclasses

from fastapi import APIRouter

from app.hardware import detect_hardware

router = APIRouter(tags=["hardware"])


@router.get("/hardware")
def get_hardware() -> dict:
    return dataclasses.asdict(detect_hardware())
```

- [ ] **Step 4: Write `app/api/recommendation.py`**

```python
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_job_store
from app.job_store import JobStore
from app.recommendation_service import RecommendationError, compute_recommendation

router = APIRouter(prefix="/jobs", tags=["recommendation"])


class RecommendRequest(BaseModel):
    srt_path: str | None = None


@router.post("/{job_id}/recommend")
def recommend_endpoint(
    job_id: str, body: RecommendRequest, store: JobStore = Depends(get_job_store)
) -> dict:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    srt_path = Path(body.srt_path) if body.srt_path else None
    try:
        recommendation = compute_recommendation(store, job_id, srt_path)
    except RecommendationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dataclasses.asdict(recommendation)


@router.get("/{job_id}/recommendation")
def get_recommendation(job_id: str, store: JobStore = Depends(get_job_store)) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.recommendation_json is None:
        raise HTTPException(status_code=404, detail="Recommendation not computed yet")
    return json.loads(job.recommendation_json)
```

- [ ] **Step 5: Wire both routers into the app**

Replace the full contents of `app/main.py`:

```python
from fastapi import FastAPI

from app.api.frames import router as frames_router
from app.api.hardware import router as hardware_router
from app.api.ingestion import router as ingestion_router
from app.api.jobs import router as jobs_router
from app.api.recommendation import router as recommendation_router

app = FastAPI(title="Splat Prep")
app.include_router(jobs_router)
app.include_router(ingestion_router)
app.include_router(frames_router)
app.include_router(hardware_router)
app.include_router(recommendation_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_hardware.py tests/test_api_recommendation.py -v`
Expected: PASS (7 passed)

- [ ] **Step 7: Run the full suite to check for regressions**

Run: `python -m pytest -v`
Expected: all tests pass, 69 passed (57 from Tasks 1-5 + 5 from Task 6 + 7 new in this task — exact count isn't load-bearing, what matters is zero failures)

- [ ] **Step 8: Commit**

```bash
git add app/api/hardware.py app/api/recommendation.py app/main.py tests/test_api_hardware.py tests/test_api_recommendation.py
git commit -m "feat: add hardware and recommendation API endpoints"
```

---

## Task 8: End-to-end integration test

**Files:**
- Test: `tests/test_recommendation_end_to_end.py`

**Interfaces:**
- Consumes: the full app (`app.main.app`) and every module built in Tasks 1-7. No new production interfaces — this task only verifies the pieces work together with a real `ffprobe` call and real (not mocked) hardware detection.

- [ ] **Step 1: Write the end-to-end test**

`tests/test_recommendation_end_to_end.py`:
```python
import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="requires ffmpeg/ffprobe installed on PATH"
)
def test_full_ingest_and_recommend_flow(client, tmp_path):
    video_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=10",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    job_id = client.post("/jobs", json={"name": "E2E recommendation"}).json()["id"]

    ingest_response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(video_path)}
    )
    assert ingest_response.status_code == 200

    recommend_response = client.post(f"/jobs/{job_id}/recommend", json={})
    assert recommend_response.status_code == 200

    body = recommend_response.json()
    assert body["realityscan"]["downscale"] in (1, 2)
    assert isinstance(body["verdict"]["use_cloud"], bool)
    assert isinstance(body["verdict"]["reason"], str) and body["verdict"]["reason"]

    stored = client.get(f"/jobs/{job_id}/recommendation").json()
    assert stored == body


def test_hardware_endpoint_returns_real_detection(client):
    response = client.get("/hardware")
    assert response.status_code == 200
    body = response.json()
    assert body["cpu_cores"] >= 1
    assert body["ram_gb"] > 0
    # gpu may legitimately be null on a machine with no NVIDIA GPU/driver —
    # both null and a populated dict are valid real-world results.
    assert body["gpu"] is None or "name" in body["gpu"]
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_recommendation_end_to_end.py -v`
Expected: `test_hardware_endpoint_returns_real_detection` PASSES unconditionally. `test_full_ingest_and_recommend_flow` PASSES if `ffmpeg`/`ffprobe` are on PATH (they are, on this machine — confirmed during planning), SKIPPED otherwise. Never a failure either way.

- [ ] **Step 3: Run the full suite one more time**

Run: `python -m pytest -v`
Expected: all tests pass, 71 passed (69 from Tasks 1-7 + 2 new in this task), or 70 passed + 1 skipped if `ffmpeg`/`ffprobe` are missing

- [ ] **Step 4: Commit**

```bash
git add tests/test_recommendation_end_to_end.py
git commit -m "test: add end-to-end ingest-and-recommend integration test"
```
