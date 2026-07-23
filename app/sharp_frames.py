from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_PERCENT_RE = re.compile(r"(\d+)%\|")


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


def run_extraction(
    video_path: Path,
    output_dir: Path,
    *,
    selection_method: str = "best-n",
    num_frames: int = 300,
    fps: int = 10,
    on_progress: Callable[[float], None] | None = None,
    timeout_seconds: int = 1800,
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

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    output_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_reader_thread, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    log_lines: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            process.wait()
            raise SharpFramesError(
                f"sharp-frames did not finish within {timeout_seconds}s and was terminated"
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
            on_progress(float(match.group(1)))

    returncode = process.wait()
    log = "\n".join(log_lines)

    if returncode != 0:
        raise SharpFramesError(f"sharp-frames exited with {returncode}:\n{log}")

    frames = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))
    if not frames:
        raise SharpFramesError(f"sharp-frames produced no frames:\n{log}")

    return ExtractionResult(frames=frames, log=log)
