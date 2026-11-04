import io
import subprocess
import time
from unittest.mock import MagicMock

import pytest

from app import sharp_frames

# Captured before any test can monkeypatch time.sleep - see the comment on
# HangingStdout below for why this must be an early-bound reference, not a
# dynamic `time.sleep` attribute lookup.
_REAL_SLEEP = time.sleep


class FakeStdout(io.StringIO):
    """A StringIO that, once exhausted, keeps returning "" like a real closed pipe."""


def _fake_popen(stdout_text: str, returncode: int = 0) -> MagicMock:
    process = MagicMock()
    process.stdout = FakeStdout(stdout_text)
    process.wait = MagicMock(return_value=returncode)
    process.kill = MagicMock()
    return process


def test_run_extraction_forces_utf8_decoding(tmp_path, monkeypatch):
    # Windows' text=True otherwise falls back to the console codepage
    # (cp1252 etc.), which can raise UnicodeDecodeError inside the reader
    # thread on non-ASCII subprocess output.
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_popen(command, **kwargs):
        captured["kwargs"] = kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    sharp_frames.run_extraction(tmp_path / "video.mp4", output_dir)

    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"


def test_run_extraction_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: None)
    with pytest.raises(sharp_frames.SharpFramesError, match="not found on PATH"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_extraction_returns_frames_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    output_dir = tmp_path / "out"

    def fake_popen(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        (output_dir / "frame_0002.jpg").write_bytes(b"fake")
        return _fake_popen("extracted 2 frames\n", returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    result = sharp_frames.run_extraction(tmp_path / "video.mp4", output_dir)

    assert len(result.frames) == 2
    assert "extracted 2 frames" in result.log


def test_run_extraction_calls_on_process_start_with_the_popen_object(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    output_dir = tmp_path / "out"
    fake_process = _fake_popen("done\n", returncode=0)

    def fake_popen(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return fake_process

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)
    seen = {}

    sharp_frames.run_extraction(
        tmp_path / "video.mp4", output_dir, on_process_start=lambda p: seen.setdefault("process", p)
    )

    assert seen["process"] is fake_process


def test_run_extraction_calls_on_log_with_accumulated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    output_dir = tmp_path / "out"

    def fake_popen(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("line one\nline two", returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    log_updates = []
    # Force every iteration to count as "throttle window elapsed" so on_log
    # fires deterministically without depending on real wall-clock timing.
    call_count = [0]

    def fake_monotonic():
        call_count[0] += 1
        return call_count[0] * 10.0

    monkeypatch.setattr(sharp_frames.time, "monotonic", fake_monotonic)

    sharp_frames.run_extraction(tmp_path / "video.mp4", output_dir, on_log=log_updates.append)

    assert log_updates
    assert "line one" in log_updates[-1]
    assert "line two" in log_updates[-1]


def test_run_extraction_pins_batch_flags_for_batched_method(tmp_path, monkeypatch):
    # app/recommendation.py's num_frames math assumes batch_size=5/batch_buffer=2 -
    # these must be passed explicitly, not left to sharp-frames' own CLI defaults,
    # or an upgraded sharp-frames could silently break that assumption.
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    sharp_frames.run_extraction(tmp_path / "video.mp4", output_dir, selection_method="batched")

    command = captured["command"]
    assert command[command.index("--batch-size") + 1] == "5"
    assert command[command.index("--batch-buffer") + 1] == "2"
    assert "--num-frames" not in command


def test_run_extraction_omits_batch_flags_for_best_n_method(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    sharp_frames.run_extraction(
        tmp_path / "video.mp4", output_dir, selection_method="best-n", num_frames=50
    )

    command = captured["command"]
    assert command[command.index("--num-frames") + 1] == "50"
    assert "--batch-size" not in command
    assert "--batch-buffer" not in command


def test_run_extraction_reports_progress_from_tqdm_style_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    output_dir = tmp_path / "out"
    # tqdm updates with \r between percentages, only a real \n at the end
    fake_output = (
        "Calculating sharpness for frames:  10%|1  | 1/10\r"
        "Calculating sharpness for frames:  50%|5  | 5/10\r"
        "Calculating sharpness for frames: 100%|##| 10/10\n"
    )

    def fake_popen(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen(fake_output, returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    seen_percentages = []
    sharp_frames.run_extraction(
        tmp_path / "video.mp4", output_dir, on_progress=seen_percentages.append
    )

    assert seen_percentages == [10.0, 50.0, 100.0]


def test_run_extraction_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    def fake_popen(command, **kwargs):
        return _fake_popen("ffmpeg not found\n", returncode=1)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    with pytest.raises(sharp_frames.SharpFramesError, match="ffmpeg not found"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_extraction_raises_when_no_frames_produced(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")

    def fake_popen(command, **kwargs):
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    with pytest.raises(sharp_frames.SharpFramesError, match="produced no frames"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_extraction_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp_frames.shutil, "which", lambda name: "/usr/bin/sharp-frames")
    # kill_process_tree talks to the real OS via psutil.Process(pid) - not
    # meaningful against a MagicMock's fake pid, and already covered for real
    # in tests/test_process_kill.py.
    monkeypatch.setattr(sharp_frames, "kill_process_tree", MagicMock())

    # A stream that never reaches EOF, simulating a hung process. The reader
    # thread is daemon=True so it won't block the test process from exiting,
    # but without a small real sleep here it would busy-loop at 100% CPU on
    # one core for the rest of the test session (harmless but wasteful) -
    # this uses the real time.sleep (not app.sharp_frames.time, which the
    # test mocks separately for the main-thread deadline check below).
    #
    # _REAL_SLEEP is captured at module import time, before any test's
    # monkeypatch can run - `time` is a process-wide singleton module, so a
    # dynamic `import time; time.sleep(...)` inside read() would pick up
    # whatever another test currently has `time.sleep` monkeypatched to. This
    # thread outlives this test (never reaches EOF, so it's never joined) and
    # was observed corrupting an unrelated later test in the same session
    # (test_realityscan.py's progress-file test) that monkeypatches
    # `realityscan.time.sleep` - the same shared attribute. See docs/ISSUES.md.
    class HangingStdout:
        def read(self, _n):
            _REAL_SLEEP(0.01)
            return " "  # never "" (EOF), never \r or \n - buffers forever

    def fake_popen(command, **kwargs):
        process = MagicMock()
        process.stdout = HangingStdout()
        process.kill = MagicMock()
        process.wait = MagicMock(return_value=1)
        return process

    monkeypatch.setattr(sharp_frames.subprocess, "Popen", fake_popen)

    fake_clock = [0.0]

    def fake_monotonic():
        fake_clock[0] += 1000.0  # jump straight past any deadline
        return fake_clock[0]

    monkeypatch.setattr(sharp_frames.time, "monotonic", fake_monotonic)

    with pytest.raises(sharp_frames.SharpFramesError, match="did not finish within"):
        sharp_frames.run_extraction(tmp_path / "video.mp4", tmp_path / "out", timeout_seconds=5)
