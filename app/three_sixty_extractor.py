from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# 360Extractor's real stdout progress format has never been observed against
# a real run - this is a generic best-effort guess, same honesty as
# app/lichtfeld.py's _ITERATION_RE/_PERCENT_RE comment. If it never matches,
# on_progress is simply never called and the frontend renders an
# indeterminate bar - graceful degradation, not an error.
_PERCENT_RE = re.compile(r"(\d{1,3})\s*%")


class ThreeSixtyExtractorError(Exception):
    pass


@dataclass
class ThreeSixtyExtractionResult:
    frames: list[Path]
    log: str


# Fixed for Phase 1, not exposed as UI settings - first-pass choices with no
# real 360Extractor run behind them yet, not measured constants:
#   - cube/6: the most predictable full-sphere coverage (6 cube faces),
#     simplest to reason about without real-run data to compare layouts.
#   - jpg: matches sharp-frames' own output format, keeps downstream code
#     (frames_dir globbing, static/frames.html's image-suffix handling)
#     unchanged regardless of which extractor produced the frames.
#   - no --resolution flag at all: leaving 360Extractor's own default is a
#     smaller commitment than guessing a number with nothing to anchor it.
# Recalibrate once real Avata/Osmo footage has gone through this.
_LAYOUT = "cube"
_CAMERA_COUNT = 6
_FORMAT = "jpg"


def _three_sixty_extractor_paths() -> tuple[Path, Path]:
    # 360Extractor is a cloned Python project (python3 src/main.py), not a
    # standalone binary - unlike app/lichtfeld.py/app/realityscan.py, there's
    # deliberately no shutil.which() fallback here, since it has its own
    # heavy/isolated dependency set (PyTorch, OpenCV, PySide6, optional CUDA)
    # that must not touch this app's own venv. The owner sets up the clone +
    # its dedicated venv themselves, outside this app.
    install_dir = os.environ.get("SPLATPREP_360EXTRACTOR_DIR")
    if install_dir is None:
        raise ThreeSixtyExtractorError(
            "360Extractor not configured. Clone "
            "https://github.com/nicolasdiolez/360Extractor, set up its own venv per the "
            "repo's README, then set SPLATPREP_360EXTRACTOR_DIR to the clone root "
            "(the directory containing src/main.py and venv/)."
        )
    root = Path(install_dir)
    python_exe = root / "venv" / "Scripts" / "python.exe"
    main_script = root / "src" / "main.py"
    if not python_exe.is_file() or not main_script.is_file():
        raise ThreeSixtyExtractorError(
            f"SPLATPREP_360EXTRACTOR_DIR is set to {root}, but {python_exe} or {main_script} "
            "was not found. Verify the clone and its venv are set up per the repo's README."
        )
    return python_exe, main_script


def _reader_thread(stream, output_queue: queue.Queue) -> None:
    buffer = ""
    for char in iter(lambda: stream.read(1), ""):
        if char in ("\r", "\n"):
            if buffer:
                output_queue.put(buffer)
                buffer = ""
        else:
            buffer += char
    if buffer:
        output_queue.put(buffer)
    output_queue.put(None)  # sentinel: stream closed


def run_360_extraction(
    input_path: Path,
    output_dir: Path,
    *,
    interval_seconds: float = 1.0,
    ai_mask: bool = False,
    on_progress: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    on_process_start: Callable[[subprocess.Popen], None] | None = None,
    timeout_seconds: int = 3600,
) -> ThreeSixtyExtractionResult:
    # --ai-mask runs a segmentation model per frame per virtual camera, on top
    # of the reprojection itself - meaningfully slower than sharp-frames'
    # plain extraction, hence the longer default timeout than
    # app/sharp_frames.py's 1800s. No real-run timing data exists yet either.
    python_exe, main_script = _three_sixty_extractor_paths()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_exe),
        str(main_script),
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--interval",
        str(interval_seconds),
        "--layout",
        _LAYOUT,
        "--camera-count",
        str(_CAMERA_COUNT),
        "--format",
        _FORMAT,
    ]
    if ai_mask:
        command.append("--ai-mask")

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    if on_process_start is not None:
        on_process_start(process)
    output_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_reader_thread, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    log_lines: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    last_reported: float | None = None
    last_log_update = time.monotonic()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            process.wait()
            raise ThreeSixtyExtractorError(
                f"360Extractor did not finish within {timeout_seconds}s and was terminated"
            )
        try:
            item = output_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if item is None:
            break
        log_lines.append(item)
        match = _PERCENT_RE.search(item)
        if match and on_progress is not None:
            percent = float(match.group(1))
            if percent != last_reported:
                on_progress(percent)
                last_reported = percent
        # Throttled to ~1/s (matches the poll cadence elsewhere) so verbose
        # CLI output doesn't turn into a SQLite write per line.
        if on_log is not None and time.monotonic() - last_log_update >= 1.0:
            on_log("\n".join(log_lines))
            last_log_update = time.monotonic()

    returncode = process.wait()
    log = "\n".join(log_lines)

    if returncode != 0:
        raise ThreeSixtyExtractorError(f"360Extractor exited with {returncode}:\n{log}")

    frames = sorted(output_dir.glob(f"*.{_FORMAT}"))
    if not frames:
        raise ThreeSixtyExtractorError(f"360Extractor produced no frames:\n{log}")

    return ThreeSixtyExtractionResult(frames=frames, log=log)
