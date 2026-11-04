from __future__ import annotations

import csv
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

# Formats accepted by LichtFeld Studio's "convert" subcommand (verified via
# `LichtFeld-Studio.exe convert --help` - this subcommand is not listed in the
# top-level --help output). .sog is PlayCanvas SuperSplat's own compressed
# format (~18x smaller than .ply on a real 4.8M-Gaussian export), .spz is
# Niantic's compressed format (~9x smaller), .html is a standalone viewer.
CONVERT_FORMATS = frozenset({"ply", "sog", "spz", "html"})


class LichtFeldError(Exception):
    pass


@dataclass
class TrainingResult:
    splat_path: Path
    log: str
    eval_metrics: list[dict[str, float | int]] | None = None


def parse_eval_metrics(output_dir: Path) -> list[dict[str, float | int]] | None:
    # With --eval, LichtFeld writes its own structured metrics.csv into
    # --output-path (header: iteration,psnr,ssim,time_per_image,num_gaussians
    # - confirmed against MetricsReporter in the real LichtFeld-Studio source,
    # src/training/metrics/metrics.cpp). Reading this real file is far more
    # reliable than regexing the LOG_INFO-prefixed stdout line it also prints
    # (metrics.to_string() goes through the logger, whose exact timestamp/level
    # prefix format was never confirmed against a real run).
    csv_path = output_dir / "metrics.csv"
    if not csv_path.is_file():
        return None
    rows: list[dict[str, float | int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "iteration": int(row["iteration"]),
                        "psnr": float(row["psnr"]),
                        "ssim": float(row["ssim"]),
                        "time_per_image": float(row["time_per_image"]),
                        "num_gaussians": int(row["num_gaussians"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows or None


def _lichtfeld_binary() -> str:
    binary = shutil.which("LichtFeld-Studio")
    if binary is not None:
        return binary
    install_dir = os.environ.get("SPLATERRA_LICHTFELD_DIR")
    if install_dir is not None:
        candidate = Path(install_dir) / "bin" / "LichtFeld-Studio.exe"
        if candidate.is_file():
            return str(candidate)
    raise LichtFeldError(
        "LichtFeld Studio not found on PATH or via SPLATERRA_LICHTFELD_DIR. "
        "Add its install folder to PATH, or set SPLATERRA_LICHTFELD_DIR to the "
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


def build_training_command(
    binary: str,
    data_path: str,
    output_dir: str,
    *,
    max_gaussians: int | None = None,
    iterations: int | None = None,
    bilateral_grid: bool = False,
    resize_factor: int | None = None,
    eval: bool = False,
    test_every: int | None = None,
) -> list[str]:
    # Shared by run_training() below (binary = the resolved local .exe path)
    # and app/cloud_training.py's cloud path (binary = just "LichtFeld-Studio",
    # assumed on PATH inside whatever Docker image the pod was created from -
    # there's no local-install-dir discovery to do on a remote container) -
    # every flag past the binary/data/output paths is identical either way.
    #
    # --headless --train together start training immediately with no GUI and
    # exit when done - confirmed against a real 30000-iteration training run
    # (see docs/ISSUES.md).
    command = [
        binary,
        "--headless",
        "--train",
        "--no-splash",
        "-d", data_path,
        "-o", output_dir,
        "--log-level", "info",
    ]
    if max_gaussians is not None:
        command += ["--max-cap", str(max_gaussians)]
    if iterations is not None:
        command += ["--iter", str(iterations)]
    if bilateral_grid:
        # Compensates for per-image exposure/white-balance variation during
        # training (auto-exposure cameras, clips shot under different
        # settings) - the bilateral grid appearance-correction technique from
        # recent Gaussian Splatting/NeRF literature.
        command.append("--bilateral-grid")
    if resize_factor is not None:
        # LichtFeld's own default is "auto" (flag omitted), which wasn't
        # aggressive enough to avoid a real CUDA OOM on a large multi-clip
        # job (see app/recommendation.py's training_resize_factor) - pass an
        # explicit value instead of trusting auto-detection for large
        # datasets. None (not "== 1") is the auto/omit sentinel so an
        # explicit resize_factor=1 (force full resolution) is still passed
        # through literally rather than silently falling back to auto.
        command += ["--resize_factor", str(resize_factor)]
    if eval:
        # Holds out every Nth image (--test-every, LichtFeld's own default 8)
        # as a validation set and scores PSNR/SSIM against it at fixed
        # iterations (LichtFeld's own default eval_steps: 7000 and 30000 -
        # not exposed as a CLI flag, confirmed against the real source,
        # src/core/include/core/parameters.hpp) instead of only ever
        # comparing renders by eye - see parse_eval_metrics() below for
        # where the actual numbers get read back after the run.
        command.append("--eval")
    if test_every is not None:
        command += ["--test-every", str(test_every)]
    return command


def run_training(
    data_path: Path,
    output_dir: Path,
    *,
    max_gaussians: int | None = None,
    iterations: int | None = None,
    bilateral_grid: bool = False,
    resize_factor: int | None = None,
    eval: bool = False,
    test_every: int | None = None,
    timeout_seconds: int = 7200,
    on_progress: Callable[[float], None] | None = None,
    on_process_start: Callable[[subprocess.Popen], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> TrainingResult:
    binary = _lichtfeld_binary()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_training_command(
        binary, str(data_path), str(output_dir),
        max_gaussians=max_gaussians, iterations=iterations, bilateral_grid=bilateral_grid,
        resize_factor=resize_factor, eval=eval, test_every=test_every,
    )

    # Wall-clock (not time.monotonic(), which isn't comparable to file mtimes)
    # timestamp taken just before the run starts, so a stale .ply/.sog left
    # in output_dir by an earlier run against the same job can't be picked
    # up as if it were this run's result.
    run_started_at = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Windows' text=True falls back to the console codepage (cp1252 etc.)
        # via locale.getpreferredencoding(), not UTF-8 - a tool that ever
        # prints Unicode (tqdm bars, non-ASCII log text) can raise
        # UnicodeDecodeError inside the reader thread below, silently killing
        # it with no sentinel ever reaching output_queue (same bug class as
        # the cp1252 UnicodeEncodeError that broke 360Extractor's setup script).
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
        # Throttled to ~1/s (matches the poll cadence elsewhere) so verbose
        # CLI output doesn't turn into a SQLite write per line.
        if on_log is not None and time.monotonic() - last_log_update >= 1.0:
            on_log("\n".join(log_lines))
            last_log_update = time.monotonic()

    returncode = process.wait()
    log = "\n".join(log_lines)

    if returncode != 0:
        raise LichtFeldError(f"LichtFeld Studio exited with {returncode}:\n{log}")

    splat_path = find_result_splat(output_dir, run_started_at, log)

    return TrainingResult(splat_path=splat_path, log=log, eval_metrics=parse_eval_metrics(output_dir))


def find_result_splat(output_dir: Path, run_started_at: float, log: str) -> Path:
    # Shared by run_training() above and app/cloud_training.py's cloud path,
    # which downloads into an equivalent local output_dir over scp and then
    # needs the exact same freshest-.ply/.sog selection logic.
    candidates = list(output_dir.rglob("*.ply")) + list(output_dir.rglob("*.sog"))
    # A 2s tolerance absorbs coarse filesystem mtime resolution (e.g. FAT32's
    # 2s granularity) without reopening the door to picking up a genuinely
    # stale file from a previous run against the same output_dir.
    fresh_candidates = [p for p in candidates if p.stat().st_mtime >= run_started_at - 2]
    if not fresh_candidates:
        raise LichtFeldError(
            f"LichtFeld Studio did not produce a .ply or .sog output file:\n{log}"
        )
    return max(fresh_candidates, key=lambda p: p.stat().st_mtime)


def convert_splat(
    input_path: Path,
    output_path: Path,
    *,
    format: str,
    sh_degree: int = -1,
    timeout_seconds: int = 1800,
) -> Path:
    if format not in CONVERT_FORMATS:
        raise LichtFeldError(
            f"Unsupported export format: {format!r}. Expected one of {sorted(CONVERT_FORMATS)}"
        )
    binary = _lichtfeld_binary()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Unlike training, `convert` prints only start/end messages, no live
    # progress - a plain blocking subprocess.run (like _run_nvidia_smi in
    # app/hardware.py) is enough, no Popen/reader-thread machinery needed.
    command = [
        binary, "convert", str(input_path), str(output_path),
        "-f", format,
        "--sh-degree", str(sh_degree),
        "-y",
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout_seconds,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise LichtFeldError(
            f"LichtFeld Studio conversion did not finish within {timeout_seconds}s and was terminated"
        ) from exc

    log = process.stdout + process.stderr
    if process.returncode != 0:
        raise LichtFeldError(f"LichtFeld Studio conversion exited with {process.returncode}:\n{log}")
    if not output_path.is_file():
        raise LichtFeldError(f"LichtFeld Studio conversion did not produce the expected output file:\n{log}")

    return output_path
