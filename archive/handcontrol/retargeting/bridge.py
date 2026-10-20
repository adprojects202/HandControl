"""The main process's end of the keyvector bridge.

`keyvector_bridge.py` runs the solver in a second interpreter when it has to:
`dex_retargeting` wants numpy >= 2, and a project venv held below that cannot
import it. Where one environment satisfies both - which it does on numpy 2 -
`solver_python` picks THIS interpreter and the split costs nothing but a
subprocess. This is the half
that stays here: it spawns that process, sends it keypoints, and reads back
joint angles. It imports nothing from `dex_retargeting`, by construction - if
it did, the split would be pointless.

ASYNCHRONOUS ON PURPOSE. Send this frame's keypoints, use the newest solution
that has arrived, never wait. A solve is a couple of milliseconds and the
round trip is loopback, so the answer is usually one frame old at worst; but a
blocking read would put the solver's worst case directly into the simulation
loop, and a solver that stalled would stop the hand rather than just date it.

A bridge that will not start, or dies, is reported and stepped over. The caller
falls back to the angle mapping, which needs nothing and always answers.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

from ..paths import ROOT

from ..net import HOST, KEYPOINT_PORT, SOLUTION_PORT

BRIDGE = ROOT / "keyvector_bridge.py"
# Where `python3.11 -m venv .venv-dex` puts it, when there is one.
DEX_VENV = ROOT / ".venv-dex" / "bin" / "python"


def solver_python():
    """The interpreter to run the solver under.

    A dedicated `.venv-dex` if the project has one, and otherwise the
    interpreter already running - which is the right answer whenever the
    project is installed in a single environment, because then
    `dex_retargeting` is importable right here.

    The second environment exists for a version conflict, not for its own
    sake. Nothing in this tree caps numpy below 2 any more, so one venv can
    hold both halves; spawning a second interpreter to reach a package the
    first one already has would be ceremony.
    """
    return DEX_VENV if DEX_VENV.exists() else Path(sys.executable)


class KeyvectorBridge:
    """Spawns the solver, speaks keypoints to it, reads joint angles back."""

    def __init__(self, keypoint_port=KEYPOINT_PORT, solution_port=SOLUTION_PORT,
                 python=None):
        self.keypoint_port = keypoint_port
        self.solution_port = solution_port
        self.python = Path(python) if python else solver_python()
        self.process = None
        self.socket = None
        self.error = ""
        self.seq = 0
        self.answered = 0
        self.ms = None                 # last reported solve time
        self.q = None                  # newest solution
        self.output = deque(maxlen=20)  # what the solver process has said
        self._reader = None

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    @property
    def ready(self):
        return self.alive and self.q is not None

    # ------------------------------------------------------------- process
    def start(self, urdf, joint_names, tip_links, wrist="palm_base_link",
              scaling=1.0, alpha=0.5, distal_links=(), objective="bones", posture=2):
        if self.alive:
            return True
        if not self.python.exists():
            self.error = (f"no solver interpreter at {self.python} - "
                          f"python3.11 -m venv .venv-dex && "
                          f".venv-dex/bin/pip install dex_retargeting torch")
            return False
        # Running in this interpreter means this interpreter's packages, so
        # the answer is already here - worth saying before paying for a
        # process that would only start up in order to fail.
        if self.python == Path(sys.executable) \
                and importlib.util.find_spec("dex_retargeting") is None:
            self.error = ("dex_retargeting is not installed in this "
                          "environment, and there is no .venv-dex - "
                          "pip install dex_retargeting torch")
            return False
        command = [str(self.python), str(BRIDGE),
                   "--urdf", str(urdf),
                   "--joints", ",".join(joint_names),
                   "--tips", ",".join(tip_links),
                   "--distals", ",".join(distal_links),
                   "--wrist", wrist,
                   "--objective", objective,
                   "--posture", str(posture),
                   "--scaling", str(scaling),
                   "--alpha", str(alpha),
                   "--bind", str(self.keypoint_port),
                   "--send-to", str(self.solution_port)]
        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True)
        # Broad: a missing interpreter, a bad path and a permissions problem
        # all land here, and none of them should end the session.
        except OSError as failure:
            self.error = f"cannot start solver: {failure}"
            self.process = None
            return False
        self._reader = threading.Thread(target=self._read_output,
                                        name="keyvector-stdout", daemon=True)
        self._reader.start()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.bind((HOST, self.solution_port))
        except OSError as failure:
            self.error = f"UDP {self.solution_port} in use: {failure}"
            self.stop()
            return False
        self.socket.setblocking(False)
        self.error = ""
        return True

    def _read_output(self):
        """Keep reading the solver's stdout, and keep what it said.

        Two jobs, and the second is the one that was missing. The child
        announces itself on one line - `ready:` with the objective it built,
        or `cannot start:` with the reason before it exits - and that line was
        going into a pipe nobody read, so a solver that refused to start
        looked from here exactly like one that was merely slow.

        Draining it also means the pipe cannot fill. It holds a line or two
        today, but a solver that started logging every frame would block on a
        full buffer, and a wedged solver is a frozen hand.
        """
        stream = self.process.stdout
        if stream is None:
            return
        for line in stream:
            line = line.strip()
            if not line:
                continue
            self.output.append(line)
            if "cannot start" in line:
                self.error = line.split("cannot start:")[-1].strip() or line

    def stop(self):
        if self.process is not None:
            try:
                self.send_raw({"type": "bye"})
                self.process.wait(timeout=1.0)
            except Exception:
                self.process.kill()
            self.process = None
        if self._reader is not None:
            self._reader.join(timeout=0.5)
            self._reader = None
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        self.q = None

    # ---------------------------------------------------------------- wire
    def send_raw(self, message):
        if self.socket is None:
            return
        try:
            self.socket.sendto(json.dumps(message).encode(),
                               (HOST, self.keypoint_port))
        except OSError:
            pass          # the far end is gone; `alive` is the thing to check

    def set_scaling(self, scaling):
        """Retune the human-to-robot size ratio without a restart."""
        self.send_raw({"type": "config", "scaling": float(scaling)})

    def send(self, points):
        self.seq += 1
        self.send_raw({"type": "keypoints", "seq": self.seq,
                       "points": np.asarray(points, dtype=float).tolist()})

    def poll(self):
        """The newest solution, or None. Drains; only the latest is worth having."""
        if self.socket is None:
            return None
        while True:
            try:
                data, _ = self.socket.recvfrom(65535)
            except (BlockingIOError, OSError):
                break
            try:
                message = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                continue
            if message.get("type") == "error":
                self.error = message.get("error", "solver error")
                continue
            if message.get("type") == "solution":
                self.q = np.asarray(message["q"], dtype=float)
                self.ms = message.get("ms")
                self.answered += 1
                self.error = ""
        return self.q

    def status(self):
        return {"alive": self.alive, "ready": self.ready, "ms": self.ms,
                "sent": self.seq, "answered": self.answered,
                "error": self.error,
                "note": self.output[-1] if self.output else ""}
