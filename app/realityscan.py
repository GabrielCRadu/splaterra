from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_EPIC_GAMES_DIR = Path(r"C:\Program Files\Epic Games")


class RealityScanError(Exception):
    pass


@dataclass
class AlignmentResult:
    registration_path: Path
    point_cloud_path: Path
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


def run_alignment(
    frames_dir: Path,
    output_dir: Path,
    *,
    progress_path: Path | None = None,
    timeout_seconds: int = 600,
) -> AlignmentResult:
    binary = _realityscan_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    crash_dir = output_dir / "crash_reports"
    crash_dir.mkdir(parents=True, exist_ok=True)

    registration_path = output_dir / "registration.csv"
    point_cloud_path = output_dir / "points.ply"
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
        str(registration_path),
        "-exportSparsePointCloud",
        str(point_cloud_path),
        "-save",
        str(project_path),
        "-quit",
    ]

    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        raise RealityScanError(
            f"RealityScan did not finish within {timeout_seconds}s and was terminated"
        ) from exc

    log = process.stdout + process.stderr
    if progress_path is not None and progress_path.is_file():
        log += "\n--- progress ---\n" + progress_path.read_text(errors="ignore")

    if process.returncode != 0:
        raise RealityScanError(f"RealityScan exited with {process.returncode}:\n{log}")

    if not registration_path.is_file() or not point_cloud_path.is_file():
        raise RealityScanError(f"RealityScan did not produce expected output files:\n{log}")

    return AlignmentResult(
        registration_path=registration_path,
        point_cloud_path=point_cloud_path,
        log=log,
    )
