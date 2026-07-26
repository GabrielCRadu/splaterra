import time
from unittest.mock import MagicMock

import pytest

from app import lichtfeld

# Captured before any test can monkeypatch time.sleep - see the comment on
# HangingStdout in test_run_training_raises_on_timeout for why this must be
# an early-bound reference, not a dynamic `time.sleep` attribute lookup.
_REAL_SLEEP = time.sleep


def _fake_popen_result(returncode: int = 0):
    process = MagicMock()
    process.wait = MagicMock(return_value=returncode)
    process.kill = MagicMock()
    return process


class FakeStdout:
    """Mimics a real Popen.stdout: yields chars from `text`, then '' forever (EOF)."""

    def __init__(self, text: str):
        self._chars = iter(text)

    def read(self, _n):
        try:
            return next(self._chars)
        except StopIteration:
            return ""


def _make_process_with_output(process, text: str):
    process.stdout = FakeStdout(text)
    return process


def test_run_training_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: None)
    monkeypatch.delenv("SPLATPREP_LICHTFELD_DIR", raising=False)
    with pytest.raises(lichtfeld.LichtFeldError, match="not found"):
        lichtfeld.run_training(tmp_path / "data", tmp_path / "out")


def test_lichtfeld_binary_uses_splatprep_lichtfeld_dir_env_var(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: None)
    fake_exe = tmp_path / "bin" / "LichtFeld-Studio.exe"
    fake_exe.parent.mkdir(parents=True)
    fake_exe.write_text("")
    monkeypatch.setenv("SPLATPREP_LICHTFELD_DIR", str(tmp_path))

    assert lichtfeld._lichtfeld_binary() == str(fake_exe)


def test_run_training_omits_max_cap_flag_when_not_given(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_popen(command, stdout, stderr, text, bufsize):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake ply")
        return _make_process_with_output(process=_fake_popen_result(0), text="")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    lichtfeld.run_training(tmp_path / "data", output_dir)

    assert "--max-cap" not in captured["command"]
    assert "--headless" in captured["command"]
    assert "--train" in captured["command"]
    assert "--no-splash" in captured["command"]


def test_run_training_includes_max_cap_flag_when_given(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_popen(command, stdout, stderr, text, bufsize):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake ply")
        return _make_process_with_output(process=_fake_popen_result(0), text="")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    lichtfeld.run_training(tmp_path / "data", output_dir, max_gaussians=2_000_000)

    index = captured["command"].index("--max-cap")
    assert captured["command"][index + 1] == "2000000"


def test_run_training_calls_on_process_start_with_the_popen_object(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"
    fake_process = _make_process_with_output(process=_fake_popen_result(0), text="")

    def fake_popen(command, stdout, stderr, text, bufsize):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake ply")
        return fake_process

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)
    seen = {}

    lichtfeld.run_training(
        tmp_path / "data", output_dir, on_process_start=lambda p: seen.setdefault("process", p)
    )

    assert seen["process"] is fake_process


def test_run_training_omits_bilateral_grid_flag_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_popen(command, stdout, stderr, text, bufsize):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake ply")
        return _make_process_with_output(process=_fake_popen_result(0), text="")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    lichtfeld.run_training(tmp_path / "data", output_dir)

    assert "--bilateral-grid" not in captured["command"]


def test_run_training_includes_bilateral_grid_flag_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_popen(command, stdout, stderr, text, bufsize):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake ply")
        return _make_process_with_output(process=_fake_popen_result(0), text="")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    lichtfeld.run_training(tmp_path / "data", output_dir, bilateral_grid=True)

    assert "--bilateral-grid" in captured["command"]


def test_run_training_omits_resize_factor_flag_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_popen(command, stdout, stderr, text, bufsize):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake ply")
        return _make_process_with_output(process=_fake_popen_result(0), text="")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    lichtfeld.run_training(tmp_path / "data", output_dir)

    assert "--resize_factor" not in captured["command"]


def test_run_training_includes_resize_factor_flag_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_popen(command, stdout, stderr, text, bufsize):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake ply")
        return _make_process_with_output(process=_fake_popen_result(0), text="")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    lichtfeld.run_training(tmp_path / "data", output_dir, resize_factor=2)

    index = captured["command"].index("--resize_factor")
    assert captured["command"][index + 1] == "2"


def test_run_training_picks_most_recently_modified_ply_or_sog(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"

    def fake_popen(command, stdout, stderr, text, bufsize):
        nested = output_dir / "checkpoints"
        nested.mkdir(parents=True, exist_ok=True)
        older = nested / "iter_1000.ply"
        older.write_bytes(b"old")
        newer = nested / "iter_5000.sog"
        newer.write_bytes(b"new")
        import os
        import time as real_time

        os.utime(older, (real_time.time() - 100, real_time.time() - 100))
        return _make_process_with_output(process=_fake_popen_result(0), text="")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    result = lichtfeld.run_training(tmp_path / "data", output_dir)

    assert result.splat_path == output_dir / "checkpoints" / "iter_5000.sog"


def test_run_training_raises_when_no_output_files_found(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")

    def fake_popen(command, stdout, stderr, text, bufsize):
        return _make_process_with_output(process=_fake_popen_result(0), text="")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    with pytest.raises(lichtfeld.LichtFeldError, match="did not produce"):
        lichtfeld.run_training(tmp_path / "data", tmp_path / "out")


def test_run_training_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")

    def fake_popen(command, stdout, stderr, text, bufsize):
        return _make_process_with_output(process=_fake_popen_result(1), text="CUDA error: out of memory\n")

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    with pytest.raises(lichtfeld.LichtFeldError, match="out of memory"):
        lichtfeld.run_training(tmp_path / "data", tmp_path / "out")


def test_run_training_reports_progress_from_tqdm_style_output(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"
    fake_output = "training:  10%|1  | 1000/10000\rtraining:  50%|5  | 5000/10000\rtraining: 100%|##| 10000/10000\n"

    def fake_popen(command, stdout, stderr, text, bufsize):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake")
        return _make_process_with_output(process=_fake_popen_result(0), text=fake_output)

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    seen = []
    lichtfeld.run_training(tmp_path / "data", output_dir, on_progress=seen.append)

    assert seen == [10.0, 50.0, 100.0]


def test_run_training_does_not_crash_when_no_progress_pattern_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_dir = tmp_path / "out"

    def fake_popen(command, stdout, stderr, text, bufsize):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"fake")
        return _make_process_with_output(
            process=_fake_popen_result(0), text="loading dataset...\nsetting up optimizer...\n"
        )

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    seen = []
    lichtfeld.run_training(tmp_path / "data", output_dir, on_progress=seen.append)

    assert seen == []


def test_run_training_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")

    # This reader thread outlives this test (HangingStdout never reaches EOF,
    # so it's never joined) and was observed corrupting an unrelated later
    # test in the same session (test_realityscan.py's progress-file test)
    # that monkeypatches the same shared `time.sleep` attribute. See
    # docs/ISSUES.md and _REAL_SLEEP's definition above.
    class HangingStdout:
        def read(self, _n):
            _REAL_SLEEP(0.01)
            return " "  # never EOF, never \r or \n

    def fake_popen(command, stdout, stderr, text, bufsize):
        process = MagicMock()
        process.stdout = HangingStdout()
        process.kill = MagicMock()
        process.wait = MagicMock(return_value=1)
        return process

    monkeypatch.setattr(lichtfeld.subprocess, "Popen", fake_popen)

    fake_clock = [0.0]

    def fake_monotonic():
        fake_clock[0] += 1000.0
        return fake_clock[0]

    monkeypatch.setattr(lichtfeld.time, "monotonic", fake_monotonic)

    with pytest.raises(lichtfeld.LichtFeldError, match="did not finish within"):
        lichtfeld.run_training(tmp_path / "data", tmp_path / "out", timeout_seconds=5)


def test_convert_splat_raises_on_unsupported_format(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    with pytest.raises(lichtfeld.LichtFeldError, match="Unsupported export format"):
        lichtfeld.convert_splat(tmp_path / "in.ply", tmp_path / "out.xyz", format="xyz")


def test_convert_splat_builds_correct_command(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    input_path = tmp_path / "in.ply"
    output_path = tmp_path / "out.spz"
    captured = {}

    def fake_run(command, capture_output, text, check, timeout):
        captured["command"] = command
        output_path.write_bytes(b"fake spz")
        result = MagicMock()
        result.returncode = 0
        result.stdout = "Done: 1 succeeded, 0 skipped, 0 failed\n"
        result.stderr = ""
        return result

    monkeypatch.setattr(lichtfeld.subprocess, "run", fake_run)

    result = lichtfeld.convert_splat(input_path, output_path, format="spz")

    assert result == output_path
    command = captured["command"]
    assert command[1] == "convert"
    assert str(input_path) in command
    assert str(output_path) in command
    assert "-f" in command
    assert command[command.index("-f") + 1] == "spz"
    assert "-y" in command


def test_convert_splat_passes_sh_degree(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")
    output_path = tmp_path / "out.spz"
    captured = {}

    def fake_run(command, capture_output, text, check, timeout):
        captured["command"] = command
        output_path.write_bytes(b"fake")
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(lichtfeld.subprocess, "run", fake_run)

    lichtfeld.convert_splat(tmp_path / "in.ply", output_path, format="spz", sh_degree=0)

    command = captured["command"]
    index = command.index("--sh-degree")
    assert command[index + 1] == "0"


def test_convert_splat_raises_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "Error: unsupported input format\n"
        return result

    monkeypatch.setattr(lichtfeld.subprocess, "run", fake_run)

    with pytest.raises(lichtfeld.LichtFeldError, match="unsupported input format"):
        lichtfeld.convert_splat(tmp_path / "in.ply", tmp_path / "out.spz", format="spz")


def test_convert_splat_raises_when_output_missing_despite_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")

    def fake_run(command, capture_output, text, check, timeout):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(lichtfeld.subprocess, "run", fake_run)

    with pytest.raises(lichtfeld.LichtFeldError, match="did not produce"):
        lichtfeld.convert_splat(tmp_path / "in.ply", tmp_path / "out.spz", format="spz")


def test_convert_splat_raises_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(lichtfeld.shutil, "which", lambda name: "C:/fake/LichtFeld-Studio.exe")

    def fake_run(command, capture_output, text, check, timeout):
        raise lichtfeld.subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr(lichtfeld.subprocess, "run", fake_run)

    with pytest.raises(lichtfeld.LichtFeldError, match="did not finish within"):
        lichtfeld.convert_splat(
            tmp_path / "in.ply", tmp_path / "out.spz", format="spz", timeout_seconds=5
        )
