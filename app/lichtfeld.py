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

# Best-effort progress patterns, tried in order against each line/chunk read
# from stdout. The exact format LichtFeld Studio prints during headless
# training is unknown (never observed against a real run) - these are common
# shapes seen in similar CUDA training CLIs. If none match, on_progress is
# simply never called - the caller then never sets progress_percent, and the
# frontend already renders an indeterminate bar for that case. Graceful
# degradation, not an error.
_ITERATION_RE = re.compile(r"[Ii]teration[:\s]+(\d+)\s*/\s*(\d+)")
_PERCENT_RE = re.compile(r"(\d{1,3})\s*%\s*\|")
_FRACTION_RE = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")


class LichtFeldError(Exception):
    pass


@dataclass
class TrainingResult:
    splat_path: Path
    log: str


def _lichtfeld_binary() -> str:
    binary = shutil.which("LichtFeld-Studio")
    if binary is not None:
        return binary
    install_dir = os.environ.get("SPLATPREP_LICHTFELD_DIR")
    if install_dir is not None:
        candidate = Path(install_dir) / "bin" / "LichtFeld-Studio.exe"
        if candidate.is_file():
            return str(candidate)
    raise LichtFeldError(
        "LichtFeld Studio not found on PATH or via SPLATPREP_LICHTFELD_DIR. "
        "Add its install folder to PATH, or set SPLATPREP_LICHTFELD_DIR to the "
        "LichtFeld Studio install root (the directory containing bin/LichtFeld-Studio.exe)."
    )


def _parse_progress_percent(chunk: str) -> float | None:
    match = _ITERATION_RE.search(chunk)
    if match:
        current, total = int(match.group(1)), int(match.group(2))
        return (current / total) * 100 if total else None
    match = _PERCENT_RE.search(chunk)
    if match:
        return float(match.group(1))
    match = _FRACTION_RE.search(chunk)
    if match:
        current, total = int(match.group(1)), int(match.group(2))
        return (current / total) * 100 if total else None
    return None


def _reader_thread(stream, output_queue: queue.Queue) -> None:
    # Mirrors app/sharp_frames.py's reader thread: a background thread does
    # blocking reads and pushes complete chunks onto a queue, because Windows
    # can't reliably do non-blocking reads on a live subprocess pipe. Splits
    # on both \r and \n since progress output of unknown shape may use
    # tqdm-style carriage returns instead of newlines.
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


def run_training(
    data_path: Path,
    output_dir: Path,
    *,
    max_gaussians: int | None = None,
    timeout_seconds: int = 7200,
    on_progress: Callable[[float], None] | None = None,
) -> TrainingResult:
    binary = _lichtfeld_binary()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --headless --train together start training immediately with no GUI and
    # exit when done, per the CLI's own --help semantics - not yet confirmed
    # against a real training run.
    command = [
        binary,
        "--headless",
        "--train",
        "--no-splash",
        "-d", str(data_path),
        "-o", str(output_dir),
        "--log-level", "info",
    ]
    if max_gaussians is not None:
        command += ["--max-cap", str(max_gaussians)]

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    output_queue: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_reader_thread, args=(process.stdout, output_queue), daemon=True)
    reader.start()

    log_lines: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    last_reported: float | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            process.wait()
            raise LichtFeldError(
                f"LichtFeld Studio did not finish within {timeout_seconds}s and was terminated"
            )
        try:
            item = output_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if item is None:
            break
        log_lines.append(item)
        if on_progress is not None:
            percent = _parse_progress_percent(item)
            if percent is not None and percent != last_reported:
                on_progress(percent)
                last_reported = percent

    returncode = process.wait()
    log = "\n".join(log_lines)

    if returncode != 0:
        raise LichtFeldError(f"LichtFeld Studio exited with {returncode}:\n{log}")

    candidates = list(output_dir.rglob("*.ply")) + list(output_dir.rglob("*.sog"))
    if not candidates:
        raise LichtFeldError(f"LichtFeld Studio did not produce a .ply or .sog output file:\n{log}")
    splat_path = max(candidates, key=lambda p: p.stat().st_mtime)

    return TrainingResult(splat_path=splat_path, log=log)
