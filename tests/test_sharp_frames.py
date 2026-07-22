from unittest.mock import MagicMock

import pytest

from app import sharp_frames


def test_run_extraction_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: None)
    with pytest.raises(sharp_frames.SharpFramesError, match="not found on PATH"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_extraction_returns_frames_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    output_dir = tmp_path / "out"

    def fake_run(command, capture_output, text, check):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        (output_dir / "frame_0002.jpg").write_bytes(b"fake")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "extracted 2 frames"
        result.stderr = ""
        return result

    monkeypatch.setattr(sharp_frames.subprocess, "run", fake_run)

    result = sharp_frames.run_extraction(tmp_path / "video.mp4", output_dir)

    assert len(result.frames) == 2
    assert "extracted 2 frames" in result.log


def test_run_extraction_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    def fake_run(command, capture_output, text, check):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "ffmpeg not found"
        return result

    monkeypatch.setattr(sharp_frames.subprocess, "run", fake_run)

    with pytest.raises(sharp_frames.SharpFramesError, match="ffmpeg not found"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")
