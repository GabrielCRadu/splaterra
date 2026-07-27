import time
from unittest.mock import MagicMock

import pytest

from app import realityscan

# Captured before any test can monkeypatch time.sleep - see the comment on
# HangingStdout below (and test_lichtfeld.py's identical pattern) for why
# this must be an early-bound reference, not a dynamic `time.sleep` lookup.
_REAL_SLEEP = time.sleep


class FakeStdout:
    """Mimics a real Popen.stdout: yields chars from `text`, then '' forever (EOF)."""

    def __init__(self, text: str):
        self._chars = iter(text)

    def read(self, _n):
        try:
            return next(self._chars)
        except StopIteration:
            return ""


def _fake_process(returncode: int, stdout_text: str = "") -> MagicMock:
    process = MagicMock()
    process.stdout = FakeStdout(stdout_text)
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

    output_dir = tmp_path / "out"

    def fake_popen(command, stdout, stderr, text):
        _write_fake_colmap_output(output_dir)
        return _fake_process(0, "aligned ok")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    result = realityscan.run_alignment(tmp_path / "frames", output_dir)

    assert result.sparse_dir == output_dir / "sparse" / "0"
    assert result.images_dir == output_dir / "images"
    assert "aligned ok" in result.log


def test_run_alignment_calls_on_process_start_with_the_popen_object(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    output_dir = tmp_path / "out"
    fake_process = _fake_process(0, "aligned ok")

    def fake_popen(command, stdout, stderr, text):
        _write_fake_colmap_output(output_dir)
        return fake_process

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)
    seen = {}

    realityscan.run_alignment(
        tmp_path / "frames", output_dir, on_process_start=lambda p: seen.setdefault("process", p)
    )

    assert seen["process"] is fake_process


def test_run_alignment_reports_progress_from_progress_file(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    output_dir = tmp_path / "out"
    progress_path = tmp_path / "progress.txt"
    # Progress is polled once per output_queue iteration now rather than once
    # per time.sleep() call, so precisely interleaving multiple writes with
    # specific loop iterations would race against the reader thread's own
    # timing - write the full progress history upfront instead and assert
    # the latest line is the one that gets reported.
    progress_path.write_text(
        "65536 0.30 0.02 0.05 #progress\n65536 0.90 0.04 0.01 #progress\n"
    )

    def fake_popen(command, stdout, stderr, text):
        _write_fake_colmap_output(output_dir)
        return _fake_process(0, "aligned ok")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    seen_percentages = []
    realityscan.run_alignment(
        tmp_path / "frames", output_dir, progress_path=progress_path, on_progress=seen_percentages.append
    )

    assert seen_percentages == [90.0]


def test_run_alignment_calls_on_log_with_accumulated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    output_dir = tmp_path / "out"

    def fake_popen(command, stdout, stderr, text):
        _write_fake_colmap_output(output_dir)
        return _fake_process(0, "line one\nline two")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    log_updates = []
    # Force every iteration to count as "throttle window elapsed" so on_log
    # fires deterministically without depending on real wall-clock timing.
    call_count = [0]

    def fake_monotonic():
        call_count[0] += 1
        return call_count[0] * 10.0

    monkeypatch.setattr(realityscan.time, "monotonic", fake_monotonic)

    realityscan.run_alignment(tmp_path / "frames", output_dir, on_log=log_updates.append)

    assert log_updates
    assert "line one" in log_updates[-1]
    assert "line two" in log_updates[-1]


def test_run_alignment_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    def fake_popen(command, stdout, stderr, text):
        return _fake_process(5, "export failed: no valid reconstruction")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    with pytest.raises(realityscan.RealityScanError, match="export failed"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_when_output_files_missing_despite_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    def fake_popen(command, stdout, stderr, text):
        return _fake_process(0, "")

    monkeypatch.setattr(realityscan.subprocess, "Popen", fake_popen)

    with pytest.raises(realityscan.RealityScanError, match="did not produce"):
        realityscan.run_alignment(tmp_path / "frames", tmp_path / "out")


def test_run_alignment_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(realityscan.shutil, "which", lambda name: "C:/fake/RealityScan.exe")

    # This reader thread outlives this test (HangingStdout never reaches EOF,
    # so it's never joined) - see docs/ISSUES.md and test_lichtfeld.py's
    # identical pattern for why _REAL_SLEEP (not the monkeypatched
    # time.sleep) must be used here.
    class HangingStdout:
        def read(self, _n):
            _REAL_SLEEP(0.01)
            return " "  # never EOF, never \r or \n

    def fake_popen(command, stdout, stderr, text):
        process = MagicMock()
        process.stdout = HangingStdout()
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
