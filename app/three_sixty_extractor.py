from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.process_kill import kill_process_tree

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


# jpg matches sharp-frames' own output format, keeps downstream code
# (frames_dir globbing, static/frames.html's image-suffix handling) unchanged
# regardless of which extractor produced the frames - not exposed as a UI
# setting, no reason to ever change it.
_FORMAT = "jpg"

# --active-cameras takes comma-separated camera *indices*, not face names
# (confirmed against the installed tool's --help - only --ai-mask-cameras
# takes names). Index 4 = Up for the cube layout - verified directly against
# a real run (`--active-cameras 4` alone produced only a "..._Up.jpg" file),
# not just inferred from --ai-mask-cameras' documented face order. Only valid
# for the cube layout (docs/SETTINGS.md documents cube as the only layout
# with a fixed "Up" face; ring/fibonacci distribute cameras differently with
# no equivalent fixed index) - callers must not combine this with a non-cube
# layout, see the skip_up_frames handling below.
_ACTIVE_CAMERAS_EXCLUDING_UP = "0,1,2,3,5"


def _three_sixty_extractor_paths() -> tuple[Path, Path]:
    # 360Extractor is a cloned Python project (python3 src/main.py), not a
    # standalone binary - unlike app/lichtfeld.py/app/realityscan.py, there's
    # deliberately no shutil.which() fallback here, since it has its own
    # heavy/isolated dependency set (PyTorch, OpenCV, PySide6, optional CUDA)
    # that must not touch this app's own venv. The owner sets up the clone +
    # its dedicated venv themselves, outside this app.
    install_dir = os.environ.get("SPLATERRA_360EXTRACTOR_DIR")
    if install_dir is None:
        raise ThreeSixtyExtractorError(
            "360Extractor not configured. Clone "
            "https://github.com/nicolasdiolez/360Extractor, set up its own venv per the "
            "repo's README, then set SPLATERRA_360EXTRACTOR_DIR to the clone root "
            "(the directory containing src/main.py and venv/)."
        )
    root = Path(install_dir)
    python_exe = root / "venv" / "Scripts" / "python.exe"
    main_script = root / "src" / "main.py"
    if not python_exe.is_file() or not main_script.is_file():
        raise ThreeSixtyExtractorError(
            f"SPLATERRA_360EXTRACTOR_DIR is set to {root}, but {python_exe} or {main_script} "
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
    skip_up_frames: bool = False,
    layout: str = "cube",
    camera_count: int = 6,
    resolution: int | None = None,
    nadir_mask: bool = False,
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
        layout,
        # The real tool ignores --camera-count for the cube layout (forces 6
        # regardless - confirmed in docs/SETTINGS.md) - passed unconditionally
        # anyway since that's harmless and keeps this call site simple; the
        # UI is what actually disables the control for cube.
        "--camera-count",
        str(camera_count),
        "--format",
        _FORMAT,
    ]
    if resolution is not None:
        command += ["--resolution", str(resolution)]
    if nadir_mask:
        command.append("--nadir-mask")
    if ai_mask:
        command.append("--ai-mask")
    if skip_up_frames and layout == "cube":
        # _ACTIVE_CAMERAS_EXCLUDING_UP is calibrated specifically for the cube
        # layout's fixed index 4 = Up - silently ignored for ring/fibonacci
        # rather than sending a meaningless index for those layouts (the UI
        # already only offers this checkbox when layout is cube).
        command += ["--active-cameras", _ACTIVE_CAMERAS_EXCLUDING_UP]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # See app/lichtfeld.py's identical Popen call for why: Windows'
        # text=True defaults to the console codepage, not UTF-8. This tool's
        # own tqdm progress bars are exactly the kind of output that can
        # trigger it, given the emoji/UnicodeEncodeError already seen from
        # this same tool's setup script under the default codepage.
        encoding="utf-8",
        errors="replace",
        bufsize=1,
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
            kill_process_tree(process)
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

    # 360Extractor writes into a "<video_stem>_processed/" subfolder of
    # --output, not flat into it (confirmed against a real run) - rglob to
    # find frames regardless of nesting depth.
    frames = sorted(output_dir.rglob(f"*.{_FORMAT}"))
    if not frames:
        raise ThreeSixtyExtractorError(f"360Extractor produced no frames:\n{log}")

    # Flatten into output_dir itself: every other frame producer (sharp-frames,
    # import_local_frames) writes flat, and RealityScan/the frames API/frames.html
    # all assume frames_dir(job_id) is flat with no subfolders - keeping that
    # invariant here means the rest of the pipeline never has to know a 360
    # extraction happened.
    flattened: list[Path] = []
    for frame in frames:
        if frame.parent != output_dir:
            target = output_dir / frame.name
            frame.rename(target)
            flattened.append(target)
        else:
            flattened.append(frame)
    frames = sorted(flattened)

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)

    return ThreeSixtyExtractionResult(frames=frames, log=log)
