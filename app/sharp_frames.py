from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SharpFramesError(Exception):
    pass


@dataclass
class ExtractionResult:
    frames: list[Path]
    log: str


def _sharp_frames_binary() -> str:
    binary = shutil.which("sharp-frames")
    if binary is None:
        raise SharpFramesError(
            "sharp-frames CLI not found on PATH. Install it with 'pip install sharp-frames'."
        )
    return binary


def run_extraction(
    video_path: Path,
    output_dir: Path,
    *,
    selection_method: str = "best-n",
    num_frames: int = 300,
    fps: int = 10,
) -> ExtractionResult:
    binary = _sharp_frames_binary()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        binary,
        str(video_path),
        str(output_dir),
        "--fps",
        str(fps),
        "--selection-method",
        selection_method,
        "--force-overwrite",
    ]
    if selection_method == "best-n":
        command += ["--num-frames", str(num_frames)]

    process = subprocess.run(command, capture_output=True, text=True, check=False)
    log = process.stdout + process.stderr

    if process.returncode != 0:
        raise SharpFramesError(f"sharp-frames exited with {process.returncode}:\n{log}")

    frames = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))
    if not frames:
        raise SharpFramesError(f"sharp-frames produced no frames:\n{log}")

    return ExtractionResult(frames=frames, log=log)
