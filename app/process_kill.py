from __future__ import annotations

import subprocess

import psutil


def kill_process_tree(process: subprocess.Popen) -> None:
    # Popen.kill() only terminates the direct child handle - on Windows that's
    # TerminateProcess on a single PID, not a process-tree kill. Any
    # grandchild the tool spawned (sharp-frames shelling out to ffmpeg,
    # 360Extractor launching its own python.exe, a forked CUDA worker)
    # survives and keeps consuming GPU/CPU/VRAM after the app has already
    # reported the job as timed-out/cancelled - a real problem for this
    # remote-access, unattended, single-user app where nobody's watching Task
    # Manager. Walk and kill descendants first (best-effort: a process that
    # already exited between listing and killing is not an error here).
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.kill()
    except psutil.NoSuchProcess:
        pass
