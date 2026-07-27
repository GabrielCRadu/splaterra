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


# sharp-frames' own CLI defaults (verified against the installed package) -
# pinned here as explicit flags rather than left implicit, so an upgraded
# sharp-frames version can't silently change "batched"'s real yield out from
# under app/recommendation.py's num_frames math, which assumes this exact ratio.
_BATCH_SIZE = 5
_BATCH_BUFFER = 2


def run_extraction(
    input_path: Path,
    output_dir: Path,
    *,
    selection_method: str = "best-n",
    num_frames: int = 300,
    fps: int = 10,
    on_progress: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    timeout_seconds: int = 1800,
) -> ExtractionResult:
    # input_path may be a single video file or a directory containing
    # multiple video files - sharp-frames' own detect_input_type() auto-detects
    # which one it is (verified directly against the installed package) and
    # handles multi-video extraction/attribution internally, no extra flag needed.
    binary = _sharp_frames_binary()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        binary,
        str(input_path),
        str(output_dir),
        "--fps",
        str(fps),
        "--selection-method",
        selection_method,
        "--force-overwrite",
    ]
    if selection_method == "best-n":
        command += ["--num-frames", str(num_frames)]
    elif selection_method == "batched":
        command += ["--batch-size", str(_BATCH_SIZE), "--batch-buffer", str(_BATCH_BUFFER)]

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    output_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_reader_thread, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    log_lines: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    last_log_update = time.monotonic()
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
        # Throttled to ~1/s (matches the poll cadence elsewhere) so verbose
        # CLI output doesn't turn into a SQLite write per line.
        if on_log is not None and time.monotonic() - last_log_update >= 1.0:
            on_log("\n".join(log_lines))
            last_log_update = time.monotonic()

    returncode = process.wait()
    log = "\n".join(log_lines)

    if returncode != 0:
        raise SharpFramesError(f"sharp-frames exited with {returncode}:\n{log}")

    frames = sorted(output_dir.glob("*.jpg")) + sorted(output_dir.glob("*.png"))
    if not frames:
        raise SharpFramesError(f"sharp-frames produced no frames:\n{log}")

    return ExtractionResult(frames=frames, log=log)
