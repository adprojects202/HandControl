"""The gesture probe, played through the live world's own lanes.

WHY NOT A SECOND WINDOW. `learn.py replay` opens its own viewer, and a second
MuJoCo window beside the live one is two worlds to look at instead of one.
The live world already has every lane the comparison needs - the operator's
hand, the commanded robot, TEACHER, STUDENT, AHEAD and the encoders - so the
probe is fed through those: the authored keypoints stand in for the sensor,
the teacher's labels pose the TEACHER lane, and the student is run live on
the same window of frames it would get from the sensor, so what STUDENT and
AHEAD show is the network on disk now, not the one that ran when the probe
was written.

Real time, looping, at the probe's own rate. The rest of the loop does not
know the difference: it reads `Mimic.alternates` for the copies and
`targets` for the commanded hand, and both are filled in the same shape.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..leap import DIGITS
from ..paths import DATA
from .student import WINDOW

PROBE = DATA / "synthetic" / "gestures"


def available(path=PROBE):
    return (Path(path) / "pairs.npz").exists()


class GesturePlayback:
    """Frames of the authored probe - or of a recorded, solved session - on
    the wall clock.

    A session plays its solved blocks in the order they were recorded, each
    run kept whole so the student's window never straddles a seam. This is
    how the teacher's labels reach the real hand: REPLAY with SIM + HAND on
    and the servos armed sends the offline solve down the bus, block by
    block, with the encoders reading back beside it.
    """

    def __init__(self, path=PROBE, window=WINDOW, blocks=None):
        path = Path(path)
        self.window = int(window)
        if (path / "pairs.npz").exists():
            self._load_probe(path)
        elif (path / "teacher.npz").exists():
            self._load_session(path, blocks)
        else:
            raise FileNotFoundError(f"nothing to play in {path}")
        # Lane 0 draws the OPERATOR's hand, so the per-digit stretch the
        # solver was fed comes off again - see `replay.unscaled`.
        from .replay import unscaled
        self.drawn = unscaled(self.keypoints, self.scale)
        self.started = time.time()
        self.index = 0

    def _load_probe(self, path):
        with np.load(path / "pairs.npz") as handle:
            self.keypoints = handle["keypoints"].astype(np.float32)
            self.raw = handle["keypoints_raw"].astype(np.float32)
            self.q_star = handle["q_star"].astype(np.float64)
            self.block = handle["block"].astype(int)
            self.joint_names = [str(n) for n in handle["joint_names"]]
            self.scale = dict(zip((str(f) for f in handle["fingers"]), handle["digit_scale"]))
        meta = json.loads((path / "pairs.json").read_text())
        self.names = list(meta.get("gestures", []))
        self.hz = float(meta.get("hz", 125.0))
        self.run_start = np.zeros(len(self.q_star), dtype=int)
        self.source = "probe"

    def _load_session(self, path, blocks):
        with np.load(path / "teacher.npz") as handle:
            keypoints = handle["keypoints"].astype(np.float32)
            raw = handle["keypoints_raw"].astype(np.float32)
            q_star = handle["q_star"].astype(np.float64)
            block = handle["block"].astype(int)
            bounds = handle["run_bounds"]
            t = handle["t"]
            self.joint_names = [str(n) for n in handle["joint_names"]]
            self.scale = dict(zip((str(f) for f in handle["fingers"]), handle["digit_scale"]))
        self.names = list(json.loads((path / "meta.json").read_text())["blocks"])
        wanted = None if not blocks else {self.names.index(b) for b in blocks if b in self.names}
        rows, starts, at = [], [], 0
        for start, end in bounds:
            if wanted is not None and int(block[start]) not in wanted:
                continue
            rows.append(np.arange(start, end))
            starts.append(np.full(end - start, at))
            at += end - start
        if not rows:
            raise FileNotFoundError(f"no solved runs for {blocks} in {path}")
        rows = np.concatenate(rows)
        self.keypoints, self.raw = keypoints[rows], raw[rows]
        self.q_star, self.block = q_star[rows], block[rows]
        self.run_start = np.concatenate(starts)
        self.hz = 1.0 / float(np.median(np.diff(t)))
        self.source = "session"

    def __len__(self):
        return len(self.q_star)

    def advance(self, now=None):
        elapsed = (time.time() if now is None else now) - self.started
        self.index = int(elapsed * self.hz) % len(self)
        return self.index

    @property
    def label(self):
        block = int(self.block[self.index])
        return self.names[block] if block < len(self.names) else ""

    def teacher(self):
        return self.q_star[self.index]

    def windows(self):
        """The last `window` frames of both views ending now, the run's first
        frame repeated where the run has not gone that long - the same
        priming `Mimic.prime` gives the live window, and never a frame from
        the run before."""
        floor = int(self.run_start[self.index])
        rows = np.clip(np.arange(self.index - self.window + 1, self.index + 1), floor, None)
        return self.keypoints[rows], self.raw[rows]

    def skeleton(self, origin=(0.0, 0.0, 0.0), offset=None):
        """The drawn hand in the shape `Mimic.skeleton` returns: four points
        per digit, knuckle to tip, placed at `origin` plus `offset`."""
        place = np.asarray(origin, dtype=float) + (
            np.zeros(3) if offset is None else np.asarray(offset, dtype=float))
        points = self.drawn[self.index]
        return {digit: [(points[4 * slot + 1 + k] + place).tolist() for k in range(4)]
                for slot, digit in enumerate(DIGITS)}


def newest_session():
    """The newest recorded session with a solve in it, or None."""
    from .sessions import sessions_under
    solved = [p for p in sessions_under() if (p / "teacher.npz").exists()]
    return solved[-1] if solved else None
