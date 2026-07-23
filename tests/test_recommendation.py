from app.drone_telemetry import DroneTelemetry
from app.hardware import GpuInfo, HardwareInfo
from app.recommendation import recommend, recommend_extraction
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


def test_recommend_extraction_short_clip_uses_max_fps_and_min_frames():
    video = VideoMetadata(width=1920, height=1080, fps=30.0, duration_seconds=10.0, codec="h264")
    settings = recommend_extraction(video)
    assert settings.fps == 10
    assert settings.num_frames == 80
    assert settings.selection_method == "best-n"


def test_recommend_extraction_very_short_clip_caps_frames_to_what_exists():
    video = VideoMetadata(width=1920, height=1080, fps=30.0, duration_seconds=2.0, codec="h264")
    settings = recommend_extraction(video)
    assert settings.fps == 10
    # only ~20 frames actually get sampled at 10fps over 2s - never ask for more than that
    assert settings.num_frames == 20


def test_recommend_extraction_medium_clip_scales_down_fps():
    video = VideoMetadata(width=1920, height=1080, fps=30.0, duration_seconds=60.0, codec="h264")
    settings = recommend_extraction(video)
    assert settings.fps == 5
    assert settings.num_frames == 150


def test_recommend_extraction_long_clip_uses_min_fps_and_caps_frames():
    video = VideoMetadata(width=1920, height=1080, fps=30.0, duration_seconds=300.0, codec="h264")
    settings = recommend_extraction(video)
    assert settings.fps == 2
    assert settings.num_frames == 300


def test_recommend_included_in_full_recommendation():
    rec = recommend(VIDEO_1080P, NO_DRONE, STRONG_GPU_HW)
    assert rec.extraction == recommend_extraction(VIDEO_1080P)
