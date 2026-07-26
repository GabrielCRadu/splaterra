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

EXTRACTION_MIN_FPS = 2
EXTRACTION_MAX_FPS = 10
EXTRACTION_MIN_FRAMES = 80
EXTRACTION_MAX_FRAMES = 300
EXTRACTION_SELECTION_RATIO = 0.5
EXTRACTION_TARGET_SAMPLE_SECONDS = 300  # tuned so a 10s clip lands at EXTRACTION_MAX_FPS


@dataclass
class ExtractionSettings:
    fps: int
    num_frames: int
    selection_method: str


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
    extraction: ExtractionSettings
    realityscan: RealityScanSettings
    lichtfeld: LichtFeldSettings
    verdict: Verdict


def recommend_extraction(video: VideoMetadata) -> ExtractionSettings:
    duration = max(video.duration_seconds, 1.0)
    # Sample richly for short clips, more sparsely for long ones, within a
    # practical fps range for sharp-frames. Then select roughly half the
    # sampled pool as "sharp enough", clamped to a sane frame-count range,
    # and never request more frames than were actually sampled.
    fps = int(min(EXTRACTION_MAX_FPS, max(EXTRACTION_MIN_FPS, round(EXTRACTION_TARGET_SAMPLE_SECONDS / duration))))
    total_sampled = duration * fps
    target = round(total_sampled * EXTRACTION_SELECTION_RATIO)
    num_frames = int(min(EXTRACTION_MAX_FRAMES, max(EXTRACTION_MIN_FRAMES, target)))
    num_frames = min(num_frames, int(total_sampled))
    # "batched" (sharp-frames' own default batch_size=5/batch_buffer=2) matches
    # reflct.app's own capture guide for 3DGS: extract at >=10fps, pick the
    # sharpest frame per batch of 5 - one frame per ~0.5s of footage, ~80%
    # overlap between selections. num_frames stays computed for display/
    # informational purposes but is unused by run_extraction for this method.
    return ExtractionSettings(fps=fps, num_frames=num_frames, selection_method="batched")


def recommend(
    video: VideoMetadata,
    drone: DroneTelemetry | None,
    hardware: HardwareInfo,
) -> Recommendation:
    extraction = recommend_extraction(video)

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

    return Recommendation(
        extraction=extraction, realityscan=realityscan, lichtfeld=lichtfeld, verdict=verdict
    )


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
