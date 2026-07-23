import subprocess
from unittest.mock import MagicMock

import pytest

from app import realityscan


def _fake_process(returncode: int, stdout_text: str = "") -> MagicMock:
    process = MagicMock()
    process.poll = MagicMock(side_effect=[None, None, returncode])
    process.returncode = returncode
    process.stdout = MagicMock()
    process.stdout.read = MagicMock(return_value=stdout_text)
    process.kill = MagicMock()
    process.wait = MagicMock(return_value=returncode)
    return process


def _write_fake_colmap_output(output_dir):
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras.txt").write_text("1 PINHOLE 100 100 50 50 50 50\n")
    (sparse_dir / "images.txt").write_text("1 1 0 0 0 0 0 0 1 00000.png\n\n")
    (sparse_dir / "points3D.txt").write_text("1 0 0 0 255 255 255 0.1\n")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "00000.png").write_bytes(b"fake")


def test_run_alignment_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: None)
    monkeypatch.setattr(realityscan, "_EPIC_GAMES_DIR", tmp_path)
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


def test_run_alignment_uses_bundled_colmap_export_params(tmp_path, monkeypatch):
    # The exact command shape (-exportRegistration <target> <params.xml>) was
    # verified empirically against the real installed binary: with this
    # bundled params.xml (calexFileFormat=COLMAP), RealityScan writes
    # sparse/0/{cameras,images,points3D}.txt and images/ as siblings of the
    # given target's parent directory, regardless of the target's own name.
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    output_dir = tmp_path / "out"
    captured_command = {}

    def fake_popen(command, stdout, stderr, text):
        captured_command["command"] = command
        _write_fake_colmap_output(output_dir)
        return _fake_process(0, "aligned ok")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    realityscan.run_alignment(tmp_path / "frames", output_dir)

    command = captured_command["command"]
    assert "-exportRegistration" in command
    export_index = command.index("-exportRegistration")
    assert command[export_index + 2] == str(realityscan._COLMAP_EXPORT_PARAMS)
    assert realityscan._COLMAP_EXPORT_PARAMS.is_file()


def test_run_alignment_returns_result_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    output_dir = tmp_path / "out"

    def fake_popen(command, stdout, stderr, text):
        _write_fake_colmap_output(output_dir)
        return _fake_process(0, "aligned ok")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    result = realityscan.run_alignment(tmp_path / "frames", output_dir)

    assert result.sparse_dir == output_dir / "sparse" / "0"
    assert result.images_dir == output_dir / "images"
    assert "aligned ok" in result.log


def test_run_alignment_reports_progress_from_progress_file(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    output_dir = tmp_path / "out"
    progress_path = tmp_path / "progress.txt"
    sleep_calls = {"count": 0}

    def fake_sleep(_seconds):
        # Simulate the progress file gaining a new line between polls. The
        # loop structure is: while poll() is None: [check file] -> sleep().
        # So the Nth sleep() call happens right after the Nth file check -
        # write here what the (N+1)th check should see.
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            progress_path.write_text("65536 0.30 0.02 0.05 #progress\n")
        elif sleep_calls["count"] == 2:
            progress_path.write_text(
                "65536 0.30 0.02 0.05 #progress\n65536 0.90 0.04 0.01 #progress\n"
            )

    monkeypatch.setattr(realityscan.time, "sleep", fake_sleep)

    # 4 poll() calls: the while-condition is checked before each of the 3
    # loop-body entries (None, None, None) and once more to exit (returncode).
    # Body entry 1: file doesn't exist yet -> no report -> sleep #1 writes "0.30".
    # Body entry 2: file has "0.30" -> reports 30.0 -> sleep #2 writes "0.30"+"0.90".
    # Body entry 3: file's latest line is "0.90" -> reports 90.0 -> sleep #3 (no-op write).
    # 4th poll() call returns the returncode, ending the loop.
    process = MagicMock()
    process.poll = MagicMock(side_effect=[None, None, None, 0])
    process.returncode = 0
    process.stdout = MagicMock()
    process.stdout.read = MagicMock(return_value="ok")
    process.kill = MagicMock()
    process.wait = MagicMock(return_value=0)

    def fake_popen(command, stdout, stderr, text):
        _write_fake_colmap_output(output_dir)
        return process

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    seen_percentages = []
    realityscan.run_alignment(
        tmp_path / "frames", output_dir, progress_path=progress_path, on_progress=seen_percentages.append
    )

    assert seen_percentages == [30.0, 90.0]


def test_run_alignment_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    def fake_popen(command, stdout, stderr, text):
        return _fake_process(5, "export failed: no valid reconstruction")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    with pytest.raises(realityscan.RealityScanError, match="export failed"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_when_output_files_missing_despite_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    def fake_popen(command, stdout, stderr, text):
        return _fake_process(0, "")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    with pytest.raises(realityscan.RealityScanError, match="did not produce"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")
    monkeypatch.setattr(realityscan.time, "sleep", lambda _: None)

    def fake_popen(command, stdout, stderr, text):
        process = MagicMock()
        process.poll = MagicMock(return_value=None)  # never finishes
        process.kill = MagicMock()
        process.wait = MagicMock(return_value=1)
        return process

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    fake_clock = [0.0]

    def fake_monotonic():
        fake_clock[0] += 1000.0
        return fake_clock[0]

    monkeypatch.setattr(realityscan.time, "monotonic", fake_monotonic)

    with pytest.raises(realityscan.RealityScanError, match="did not finish within"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out", timeout_seconds=5)
