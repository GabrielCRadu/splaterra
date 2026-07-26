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

# A 12GB RTX 3060 trained successfully with max_gaussians=4_800_000 on a
# 78-image job, but hit CUDA out-of-memory with the identical cap on an
# 830-image job - the GPU had only ~565MB free when a routine 79MB
# image-decode allocation failed (not a gaussian-tensor allocation), during
# image loading, before training had even started iterating. That points at
# the pipelined image loader/nvJPEG decoder pool - VRAM that scales with how
# many training images there are and their resolution - not at the gaussian
# model itself, which is sized by max_gaussians independent of frame_count.
# So the fix is downscaling training images (LichtFeld's own --resize_factor,
# which defaulted to "auto" and wasn't aggressive enough here) rather than
# shrinking max_gaussians, which would cost detail on exactly the jobs that
# have the most of it to capture. See training_resize_factor() below.

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


def max_gaussians_for_vram(vram_gb: float) -> int:
    if vram_gb <= 0:
        return 0
    return max(MIN_MAX_GAUSSIANS, min(MAX_GAUSSIANS_CAP, int(vram_gb * GAUSSIANS_PER_VRAM_GB)))


# LichtFeld's --resize_factor halves width/height per step (2 -> 1/4 the
# pixels, 4 -> 1/16), directly shrinking the per-image decode/pipeline VRAM
# footprint that caused the real crash above, without touching max_gaussians.
# Thresholds are calibrated against a *total dataset megapixel* budget rather
# than a bare frame count, assuming ~4K frames (8.3 Mpx each - this app's
# typical GoPro/phone footage, also LichtFeld's own --max-width=3840 default
# cap): a LichtFeld GitHub issue (MrNeRF/LichtFeld-Studio#65) documents an
# unrelated OOM crash on datasets past ~2500 total megapixels, which at 8.3
# Mpx/frame is ~300 frames untouched, or ~1200 frames once resize_factor=2
# cuts pixels to a quarter. The real crash this fixes (830 frames, 12GB RTX
# 3060) sits comfortably under the 1200-frame ceiling at factor=2 (~1721
# Mpx), so that's the first rung, not factor=4. Still a first-pass estimate
# extrapolated from a different project's RAM-pressure issue, not a direct
# LichtFeld VRAM benchmark - recalibrate as more multi-clip runs are logged.
TRAINING_RESIZE_FRAME_THRESHOLDS = ((1200, 4), (300, 2))


def training_resize_factor(frame_count: int | None) -> int:
    if not frame_count:
        return 1
    for threshold, factor in TRAINING_RESIZE_FRAME_THRESHOLDS:
        if frame_count >= threshold:
            return factor
    return 1


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
