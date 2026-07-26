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
MIN_MAX_GAUSSIANS = 500_000
HIGH_RES_PIXEL_THRESHOLD = 3840 * 2160

# LichtFeld's MCMC strategy pre-allocates VRAM for the full max_gaussians
# CAPACITY upfront (not just the currently-active gaussian count), and the
# pipelined image loader/nvJPEG decoder pool separately consumes VRAM that
# scales with how many training images there are. A 12GB RTX 3060 trained
# successfully with max_gaussians=4_800_000 on a 78-image job, but hit CUDA
# out-of-memory with the identical cap on an 830-image job - the GPU had
# only ~565MB free when a routine 79MB image-decode allocation failed.
# TRAINING_IMAGE_OVERHEAD_MB reserves a per-image VRAM budget before
# computing gaussians-per-GB from what's left. This is a conservative
# first-pass guess from a single failure, not a measured constant -
# recalibrate once more multi-clip training runs are logged (see
# data/training_runs.jsonl's num_images/final_gaussians columns).
TRAINING_IMAGE_OVERHEAD_MB = 8

EXTRACTION_MIN_FPS = 2
EXTRACTION_MAX_FPS = 10
EXTRACTION_MAX_FRAMES = 300
EXTRACTION_TARGET_SAMPLE_SECONDS = 300  # tuned so a 10s clip lands at EXTRACTION_MAX_FPS

# Matches app/sharp_frames.py's run_extraction() defaults, which now pass these
# same values as explicit --batch-size/--batch-buffer flags instead of relying
# on sharp-frames' own CLI defaults - kept as separate constants (like
# app/extraction.py's _IMAGE_SUFFIXES) since this module is pure/dependency-free
# and shouldn't import a subprocess-wrapping module just for two ints.
EXTRACTION_BATCH_SIZE = 5
EXTRACTION_BATCH_BUFFER = 2


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
    # practical fps range for sharp-frames.
    fps = int(min(EXTRACTION_MAX_FPS, max(EXTRACTION_MIN_FPS, round(EXTRACTION_TARGET_SAMPLE_SECONDS / duration))))
    total_sampled = duration * fps
    # "batched" (matches reflct.app's own capture guide for 3DGS: extract at
    # >=10fps, pick the sharpest frame per batch of 5) keeps 1 frame per
    # (batch_size + batch_buffer) sampled frames - see app/sharp_frames.py's
    # run_extraction(), which pins these exact values as explicit CLI flags.
    # num_frames must track that real yield, not an arbitrary target ratio,
    # or the UI promises a frame count "batched" was never going to produce.
    step = EXTRACTION_BATCH_SIZE + EXTRACTION_BATCH_BUFFER
    target = round(total_sampled / step)
    num_frames = min(EXTRACTION_MAX_FRAMES, target, int(total_sampled))
    return ExtractionSettings(fps=fps, num_frames=num_frames, selection_method="batched")


def max_gaussians_for_vram(vram_gb: float, frame_count: int | None = None) -> int:
    if vram_gb <= 0:
        return 0
    reserved_gb = (frame_count * TRAINING_IMAGE_OVERHEAD_MB / 1024) if frame_count else 0.0
    available_gb = max(0.0, vram_gb - reserved_gb)
    return max(MIN_MAX_GAUSSIANS, min(MAX_GAUSSIANS_CAP, int(available_gb * GAUSSIANS_PER_VRAM_GB)))


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
    # frame_count isn't known yet at recommendation time (extraction hasn't
    # run) - this is a VRAM-only estimate; app/training_service.py recomputes
    # a frame-count-aware cap once the real count is known.
    max_gaussians = max_gaussians_for_vram(vram_gb)
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
