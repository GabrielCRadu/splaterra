import io
import time
from unittest.mock import MagicMock

import pytest

from app import three_sixty_extractor

# Captured before any test can monkeypatch time.sleep/time.monotonic - see
# tests/test_sharp_frames.py's identical HangingStdout comment for why this
# must be an early-bound reference, not a dynamic `time.sleep` attribute
# lookup (a leaked reader thread from a timeout test can otherwise corrupt an
# unrelated later test that monkeypatches the same shared `time` module).
_REAL_SLEEP = time.sleep


class FakeStdout(io.StringIO):
    """A StringIO that, once exhausted, keeps returning "" like a real closed pipe."""


def _fake_popen(stdout_text: str, returncode: int = 0) -> MagicMock:
    process = MagicMock()
    process.stdout = FakeStdout(stdout_text)
    process.wait = MagicMock(return_value=returncode)
    process.kill = MagicMock()
    return process


def _setup_fake_install(tmp_path, monkeypatch):
    install_dir = tmp_path / "360Extractor"
    python_exe = install_dir / "venv" / "Scripts" / "python.exe"
    main_script = install_dir / "src" / "main.py"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("")
    main_script.parent.mkdir(parents=True)
    main_script.write_text("")
    monkeypatch.setenv("SPLATERRA_360EXTRACTOR_DIR", str(install_dir))
    return python_exe, main_script


def test_run_360_extraction_forces_utf8_decoding(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured = {}

    def fake_popen(command, **kwargs):
        captured["kwargs"] = kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir)

    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"


def test_run_360_extraction_raises_when_env_var_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("SPLATERRA_360EXTRACTOR_DIR", raising=False)
    with pytest.raises(three_sixty_extractor.ThreeSixtyExtractorError, match="not configured"):
        three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_360_extraction_raises_when_install_paths_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATERRA_360EXTRACTOR_DIR", str(tmp_path / "does-not-exist"))
    with pytest.raises(three_sixty_extractor.ThreeSixtyExtractorError, match="was not found"):
        three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_360_extraction_returns_frames_on_success(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"

    def fake_popen(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cam0_0001.jpg").write_bytes(b"fake")
        (output_dir / "cam1_0001.jpg").write_bytes(b"fake")
        return _fake_popen("extracted 2 frames\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    result = three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir)

    assert len(result.frames) == 2
    assert "extracted 2 frames" in result.log


def test_run_360_extraction_flattens_frames_from_processed_subfolder(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"

    def fake_popen(command, **kwargs):
        # Mirrors the real tool: writes into "<video_stem>_processed/" under
        # --output rather than flat into it.
        processed_dir = output_dir / "video_processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / "video_frame000000_Front.jpg").write_bytes(b"fake")
        (processed_dir / "video_frame000000_Back.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    result = three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir)

    assert len(result.frames) == 2
    for frame in result.frames:
        assert frame.parent == output_dir
        assert frame.is_file()
    assert not (output_dir / "video_processed").exists()


def test_run_360_extraction_pins_fixed_layout_flags(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir, interval_seconds=2.5)

    command = captured["command"]
    assert command[command.index("--layout") + 1] == "cube"
    assert command[command.index("--camera-count") + 1] == "6"
    assert command[command.index("--format") + 1] == "jpg"
    assert command[command.index("--interval") + 1] == "2.5"
    assert "--ai-mask" not in command


def test_run_360_extraction_includes_ai_mask_flag_when_requested(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir, ai_mask=True)

    assert "--ai-mask" in captured["command"]


def test_run_360_extraction_includes_active_cameras_flag_when_skipping_up(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir, skip_up_frames=True)

    command = captured["command"]
    assert command[command.index("--active-cameras") + 1] == "0,1,2,3,5"


def test_run_360_extraction_omits_active_cameras_flag_by_default(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir)

    assert "--active-cameras" not in captured["command"]


def test_run_360_extraction_ignores_skip_up_frames_for_non_cube_layout(tmp_path, monkeypatch):
    # _ACTIVE_CAMERAS_EXCLUDING_UP is calibrated specifically for the cube
    # layout's fixed index 4 = Up - must not be sent for ring/fibonacci,
    # which have no equivalent fixed "Up" index.
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(
        tmp_path / "video.mp4", output_dir, skip_up_frames=True, layout="ring"
    )

    assert "--active-cameras" not in captured["command"]


def test_run_360_extraction_passes_layout_and_camera_count(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(
        tmp_path / "video.mp4", output_dir, layout="fibonacci", camera_count=12
    )

    command = captured["command"]
    assert command[command.index("--layout") + 1] == "fibonacci"
    assert command[command.index("--camera-count") + 1] == "12"


def test_run_360_extraction_omits_resolution_flag_by_default(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir)

    assert "--resolution" not in captured["command"]


def test_run_360_extraction_includes_resolution_flag_when_set(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir, resolution=3072)

    command = captured["command"]
    assert command[command.index("--resolution") + 1] == "3072"


def test_run_360_extraction_includes_nadir_mask_flag_when_enabled(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir, nadir_mask=True)

    assert "--nadir-mask" in captured["command"]


def test_run_360_extraction_omits_nadir_mask_flag_by_default(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir)

    assert "--nadir-mask" not in captured["command"]


def test_run_360_extraction_calls_on_process_start_with_the_popen_object(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"
    fake_process = _fake_popen("done\n", returncode=0)

    def fake_popen(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return fake_process

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)
    seen = {}

    three_sixty_extractor.run_360_extraction(
        tmp_path / "video.mp4", output_dir, on_process_start=lambda p: seen.setdefault("process", p)
    )

    assert seen["process"] is fake_process


def test_run_360_extraction_calls_on_log_with_accumulated_output(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"

    def fake_popen(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("line one\nline two", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    log_updates = []
    call_count = [0]

    def fake_monotonic():
        call_count[0] += 1
        return call_count[0] * 10.0

    monkeypatch.setattr(three_sixty_extractor.time, "monotonic", fake_monotonic)

    three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", output_dir, on_log=log_updates.append)

    assert log_updates
    assert "line one" in log_updates[-1]
    assert "line two" in log_updates[-1]


def test_run_360_extraction_reports_progress_from_percent_output(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    output_dir = tmp_path / "out"

    def fake_popen(command, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_0001.jpg").write_bytes(b"fake")
        return _fake_popen("Processing: 10%\nProcessing: 50%\nProcessing: 100%\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    seen_percentages = []
    three_sixty_extractor.run_360_extraction(
        tmp_path / "video.mp4", output_dir, on_progress=seen_percentages.append
    )

    assert seen_percentages == [10.0, 50.0, 100.0]


def test_run_360_extraction_raises_on_nonzero_exit(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)

    def fake_popen(command, **kwargs):
        return _fake_popen("CUDA out of memory\n", returncode=1)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    with pytest.raises(three_sixty_extractor.ThreeSixtyExtractorError, match="CUDA out of memory"):
        three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_360_extraction_raises_when_no_frames_produced(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)

    def fake_popen(command, **kwargs):
        return _fake_popen("done\n", returncode=0)

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    with pytest.raises(three_sixty_extractor.ThreeSixtyExtractorError, match="produced no frames"):
        three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", tmp_path / "out")


def test_run_360_extraction_raises_on_timeout(tmp_path, monkeypatch):
    _setup_fake_install(tmp_path, monkeypatch)
    # kill_process_tree talks to the real OS via psutil.Process(pid) - not
    # meaningful against a MagicMock's fake pid, and already covered for real
    # in tests/test_process_kill.py.
    monkeypatch.setattr(three_sixty_extractor, "kill_process_tree", MagicMock())

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

    monkeypatch.setattr(three_sixty_extractor.subprocess, "Popen", fake_popen)

    fake_clock = [0.0]

    def fake_monotonic():
        fake_clock[0] += 1000.0
        return fake_clock[0]

    monkeypatch.setattr(three_sixty_extractor.time, "monotonic", fake_monotonic)

    with pytest.raises(three_sixty_extractor.ThreeSixtyExtractorError, match="did not finish within"):
        three_sixty_extractor.run_360_extraction(tmp_path / "video.mp4", tmp_path / "out", timeout_seconds=5)
