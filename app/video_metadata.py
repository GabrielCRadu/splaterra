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
