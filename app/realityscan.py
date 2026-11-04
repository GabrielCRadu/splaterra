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

_EPIC_GAMES_DIR = Path(os.environ.get("SPLATERRA_REALITYSCAN_DIR", r"C:\Program Files\Epic Games"))
_PROGRESS_LINE_RE = re.compile(r"^\d+\s+([\d.]+)\s+")
_COLMAP_EXPORT_PARAMS = Path(__file__).resolve().parent / "resources" / "realityscan_colmap_export.xml"


class RealityScanError(Exception):
    pass


@dataclass
class AlignmentResult:
    sparse_dir: Path
    images_dir: Path
    log: str


def _realityscan_binary() -> str:
    binary = shutil.which("RealityScan")
    if binary is not None:
        return binary
    candidates = sorted(_EPIC_GAMES_DIR.glob("RealityScan_*/RealityScan.exe"), reverse=True)
    if candidates:
        return str(candidates[0])
    raise RealityScanError(
        "RealityScan not found on PATH or in the default Epic Games install location. "
        "Install RealityScan, or add its install folder to PATH."
    )


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


def _latest_progress_percent(progress_path: Path) -> float | None:
    try:
        lines = progress_path.read_text(errors="ignore").strip().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        match = _PROGRESS_LINE_RE.match(line)
        if match:
            return float(match.group(1)) * 100
    return None


def run_alignment(
    frames_dir: Path,
    output_dir: Path,
    *,
    progress_path: Path | None = None,
    timeout_seconds: int = 600,
    on_progress: Callable[[float], None] | None = None,
    on_process_start: Callable[[subprocess.Popen], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> AlignmentResult:
    binary = _realityscan_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    crash_dir = output_dir / "crash_reports"
    crash_dir.mkdir(parents=True, exist_ok=True)

    # calexFileFormat=COLMAP in the bundled params writes sparse/0/*.txt and
    # images/ as siblings of the export target's parent dir, regardless of
    # the target's own filename - verified against the real binary.
    export_target = output_dir / "project.txt"
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    project_path = output_dir / "project.rsproj"

    command = [
        binary,
        "-headless",
        "-silent",
        str(crash_dir),
        "-set",
        "appQuitOnError=true",
        "-newScene",
    ]
    if progress_path is not None:
        command += ["-writeProgress", str(progress_path)]
    command += [
        "-addFolder",
        str(frames_dir),
        "-align",
        "-exportRegistration",
        str(export_target),
        str(_COLMAP_EXPORT_PARAMS),
        "-save",
        str(project_path),
        "-quit",
    ]

    # See app/lichtfeld.py's identical Popen call for why encoding/errors are
    # explicit: Windows' text=True defaults to the console codepage, not
    # UTF-8, which can raise UnicodeDecodeError inside the reader thread.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
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
            raise RealityScanError(
                f"RealityScan did not finish within {timeout_seconds}s and was terminated"
            )
        try:
            item = output_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            pass
        else:
            if item is None:
                break
            log_lines.append(item)

        # Progress comes from a separate file RealityScan writes to
        # (-writeProgress), not from stdout - checked every iteration
        # (roughly once/second) regardless of whether new stdout arrived.
        if progress_path is not None and on_progress is not None and progress_path.is_file():
            percent = _latest_progress_percent(progress_path)
            if percent is not None and percent != last_reported:
                on_progress(percent)
                last_reported = percent

        # Throttled to ~1/s so verbose CLI output doesn't turn into a SQLite
        # write per line.
        if on_log is not None and time.monotonic() - last_log_update >= 1.0:
            on_log("\n".join(log_lines))
            last_log_update = time.monotonic()

    returncode = process.wait()
    log = "\n".join(log_lines)
    if progress_path is not None and progress_path.is_file():
        log += "\n--- progress ---\n" + progress_path.read_text(errors="ignore")

    if returncode != 0:
        raise RealityScanError(f"RealityScan exited with {returncode}:\n{log}")

    required_files = ["cameras.txt", "images.txt", "points3D.txt"]
    if not all((sparse_dir / name).is_file() for name in required_files) or not images_dir.is_dir():
        raise RealityScanError(f"RealityScan did not produce expected output files:\n{log}")

    return AlignmentResult(
        sparse_dir=sparse_dir,
        images_dir=images_dir,
        log=log,
    )
