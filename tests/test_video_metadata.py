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
