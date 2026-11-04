import subprocess
import sys
import time

import psutil

from app.process_kill import kill_process_tree


def test_kill_process_tree_kills_parent_and_child():
    # Spawns a real Python parent process which itself spawns a real Python
    # child and then sleeps - mirrors an external tool (sharp-frames,
    # 360Extractor) that shells out to its own child. No external binary
    # dependency, just python.exe, so this runs unconditionally rather than
    # skipping like the *_end_to_end.py tests that need RealityScan/LichtFeld.
    child_script = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(p.pid, flush=True); "
        "time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", child_script], stdout=subprocess.PIPE, text=True
    )
    try:
        child_pid = int(parent.stdout.readline().strip())
        assert psutil.pid_exists(parent.pid)
        assert psutil.pid_exists(child_pid)

        kill_process_tree(parent)
        parent.wait(timeout=10)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and psutil.pid_exists(child_pid):
            time.sleep(0.1)

        assert not psutil.pid_exists(parent.pid)
        assert not psutil.pid_exists(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()


def test_kill_process_tree_is_a_noop_for_already_exited_process():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=10)

    # Must not raise even though the PID is long gone.
    kill_process_tree(process)
