import subprocess
from unittest.mock import MagicMock

import pytest

from app import realityscan


def test_run_alignment_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: None)
    monkeypatch.setattr(realityscan, "_EPIC_GAMES_DIR", tmp_path)  # empty dir, nothing to glob
    with pytest.raises(realityscan.RealityScanError, match="not found"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_realityscan_binary_falls_back_to_epic_games_install(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: None)
    fake_exe = tmp_path / "RealityScan_2.2" / "RealityScan.exe"
    fake_exe.parent.mkdir(parents=True)
    fake_exe.write_text("")
    monkeypatch.setattr(realityscan, "_EPIC_GAMES_DIR", tmp_path)

    binary = realityscan._realityscan_binary()

    assert binary == str(fake_exe)


def test_run_alignment_returns_result_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    output_dir = tmp_path / "out"

    def fake_run(command, capture_output, text, check, timeout):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "registration.csv").write_text("frame_001.jpg\n")
        (output_dir / "points.ply").write_bytes(b"fake ply")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "aligned ok"
        result.stderr = ""
        return result

    monkeypatch.setattr(realityscan.subprocess, "run", fake_run)

    result = realityscan.run_alignment(tmp_path / "frames", output_dir)

    assert result.registration_path == output_dir / "registration.csv"
    assert result.point_cloud_path == output_dir / "points.ply"
    assert "aligned ok" in result.log


def test_run_alignment_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 5
        result.stdout = ""
        result.stderr = "export failed: no valid reconstruction"
        return result

    monkeypatch.setattr(realityscan.subprocess, "run", fake_run)

    with pytest.raises(realityscan.RealityScanError, match="export failed"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_when_output_files_missing_despite_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(realityscan.subprocess, "run", fake_run)

    with pytest.raises(realityscan.RealityScanError, match="did not produce"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    def fake_run(command, capture_output, text, check, timeout):
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr(realityscan.subprocess, "run", fake_run)

    with pytest.raises(realityscan.RealityScanError, match="did not finish within"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out", timeout_seconds=5)
