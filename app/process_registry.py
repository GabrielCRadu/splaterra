from __future__ import annotations

import subprocess
import threading

# Background tasks (alignment, training) run in-process via FastAPI's thread
# pool, not as separate worker processes, so a plain in-memory dict shared
# across threads is enough to let a cancel request reach the right
# subprocess.Popen - no IPC needed. Keyed by (job_id, stage) since a job only
# ever has one of each stage running at a time.
_lock = threading.Lock()
_processes: dict[tuple[str, str], subprocess.Popen] = {}
_cancelled: set[tuple[str, str]] = set()


def register(job_id: str, stage: str, process: subprocess.Popen) -> None:
    with _lock:
        _processes[(job_id, stage)] = process
        _cancelled.discard((job_id, stage))


def unregister(job_id: str, stage: str) -> None:
    with _lock:
        _processes.pop((job_id, stage), None)
        _cancelled.discard((job_id, stage))


def cancel(job_id: str, stage: str) -> bool:
    with _lock:
        process = _processes.get((job_id, stage))
        if process is None or process.poll() is not None:
            return False
        _cancelled.add((job_id, stage))
        process.kill()
        return True


def was_cancelled(job_id: str, stage: str) -> bool:
    with _lock:
        return (job_id, stage) in _cancelled
