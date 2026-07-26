from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_EPIC_GAMES_DIR = Path(os.environ.get("SPLATPREP_REALITYSCAN_DIR", r"C:\Program Files\Epic Games"))
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

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if on_process_start is not None:
        on_process_start(process)
    deadline = time.monotonic() + timeout_seconds
    last_reported: float | None = None

    while process.poll() is None:
        if time.monotonic() > deadline:
            process.kill()
            process.wait()
            raise RealityScanError(
                f"RealityScan did not finish within {timeout_seconds}s and was terminated"
            )
        if progress_path is not None and on_progress is not None and progress_path.is_file():
            percent = _latest_progress_percent(progress_path)
            if percent is not None and percent != last_reported:
                on_progress(percent)
                last_reported = percent
        time.sleep(1.0)

    returncode = process.returncode
    stdout_text = process.stdout.read() if process.stdout is not None else ""
    log = stdout_text
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
