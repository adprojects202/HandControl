"""Building the library and training run as processes of their own: the
window stays responsive while they use every core, and a run outlives the
window (closing it does not stop one; Cancel does).

    task = Background()
    task.start("train")        # scripts/train_tracker.py, detached
    status = task.status()     # what the status file says, or None
    task.cancel()

While one runs, the Mac does not idle-sleep (macOS's `caffeinate`, for as
long as that process lives): asleep, a run just stops, and a round that
should take 20 minutes took hours. Closing the lid on battery still sleeps it.

The scripts report as they go into STATUS - also when started from a
terminal, so the Learning panel shows those too:

    task        "library", "hand_library", "train" or "clearml" (training on
                the ClearML agent)
    pid         its process
    state       "starting", "running", "done", "failed", "cancelled" - or
                "stopped": the process ended without saying how
    done, total, text   its progress
    run         the run folder (training)
    error       what went wrong
    log         its output, when started from the panel
    started, ended
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime

from ..paths import ROOT

FOLDER = ROOT / "data" / "learning"
STATUS = FOLDER / "status.json"
SCRIPTS = {"library": "build_library.py", "hand_library": "build_hand_library.py",
           "train": "train_tracker.py", "clearml": "clearml_train.py"}
ACTIVE = ("starting", "running")


def read():
    """The status file as a dict, or None."""
    try:
        return json.loads(STATUS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def report(**fields):
    """Merge `fields` into the status file."""
    status = read() or {}
    status.update(fields)
    _write(status)


def begin(task):
    """Called by a script as it starts: a fresh status for `task`, keeping
    the log path when the panel started it."""
    previous = read() or {}
    started_here = previous.get("state") == "starting" and previous.get("task") == task
    _stay_awake()
    _write({"task": task, "pid": os.getpid(), "state": "running", "started": _now(),
            "done": 0, "total": 0, "text": "starting",
            "log": previous.get("log") if started_here else None})


def finish(**fields):
    """Called by a script as it ends."""
    report(ended=_now(), **fields)


class Background:
    def __init__(self):
        self._process = None

    def start(self, task):
        """Start `task` (a SCRIPTS name); False when one is running."""
        if self.busy():
            return False
        FOLDER.mkdir(parents=True, exist_ok=True)
        log = FOLDER / f"{task}.log"
        _write({"task": task, "pid": None, "state": "starting", "started": _now(),
                "done": 0, "total": 0, "text": "starting", "log": str(log)})
        with open(log, "w") as output:
            self._process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / SCRIPTS[task])], cwd=ROOT,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        report(pid=self._process.pid)
        return True

    def status(self):
        """The status, with a process that ended without saying so marked
        "stopped"."""
        status = read()
        if status and status.get("state") in ACTIVE and not self._alive(status.get("pid")):
            status = dict(status, state="stopped", ended=_now())
            _write(status)
        return status

    def busy(self):
        status = self.status()
        return bool(status) and status.get("state") in ACTIVE

    def cancel(self):
        status = self.status()
        if not status or status.get("state") not in ACTIVE or not status.get("pid"):
            return
        try:
            os.kill(status["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self._process is not None and self._process.pid == status["pid"]:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        report(state="cancelled", ended=_now())

    def _alive(self, pid):
        if pid is None:                          # starting: not yet spawned
            return True
        if self._process is not None and self._process.pid == pid:
            return self._process.poll() is None  # ours: reap it when it ends
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def _stay_awake():
    """Keep the Mac from idle-sleeping until this process ends."""
    caffeinate = shutil.which("caffeinate")
    if caffeinate:
        subprocess.Popen([caffeinate, "-i", "-w", str(os.getpid())],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _write(status):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(STATUS)


def _now():
    return datetime.now().isoformat(timespec="seconds")
