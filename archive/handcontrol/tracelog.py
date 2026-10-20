"""One CSV per session: what your hand did, what the robot did, side by side.

The point is to settle arguments about retargeting with numbers instead of
impressions. Every row is one simulation frame, and every quantity needed to
compute an error is on that row - so no joins, no interpolation between files,
and no chance of comparing a human sample against a robot sample from a
different moment.

The `mode` column is why this is worth having at all: switch mapping mid-run
and the file carries the switch, so ANGLES and KEYVECTOR can be compared on
the same hand doing the same thing rather than on two sessions you have to
trust were similar.

COLUMN GROUPS, all SI, all one frame per row:

    t, frame, mode, tracked, hand, pinch, grab, solve_ms
        when, which mapping, and the Leap's own estimates

    raw_<digit>_{x,y,z}       fingertip, device millimetres, UNFILTERED
    hum_<digit>_{x,y,z}       fingertip, robot world metres, filtered+mapped
        raw against hum measures the filter; hum against rob measures the
        retargeting. Five digits each - the pinky is in there because the
        robot not having one is the interesting part.

    ang_<finger>_<role>       the human joint angle this frame, radians
    cmd_<joint>               what the robot was told, radians
    sim_<joint>               where the simulated joint got to
    enc_<joint>               where the real joint got to, blank with no hardware

    rob_<finger>_{x,y,z}      robot fingertip, world metres
    pinch_human, pinch_robot  thumb-to-index tip distance, metres, both hands

    kp<00..20>_{x,y,z}        the MANO-21 keypoints handed to the solver
        The solver's exact input, so a recording can be replayed through a
        different solver configuration offline. Without these, testing a
        change to the retargeting means asking the operator to perform the
        gesture again, and a recording is minutes of somebody's hand.

Writes are buffered and flushed in blocks. A CSV row of ~120 floats costs
tens of microseconds, which is well under a percent of a 10 ms frame, but a
flush is a syscall and there is no reason to do one per frame.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .robot.model import FINGERS, ROLES
from .retargeting.keyvector import mano_keypoints

from .paths import LOGS

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
AXES = ("x", "y", "z")
FLUSH_ROWS = 200               # about two seconds at 100 Hz


def _columns(joint_names):
    columns = ["t", "frame", "mode", "tracked", "hand", "pinch", "grab",
               "solve_ms"]
    columns += [f"raw_{d}_{a}" for d in DIGITS for a in AXES]
    columns += [f"hum_{d}_{a}" for d in DIGITS for a in AXES]
    columns += [f"ang_{f}_{r}" for f in FINGERS for r in ROLES]
    for prefix in ("cmd", "sim", "enc"):
        columns += [f"{prefix}_{n}" for n in joint_names]
    columns += [f"rob_{f}_{a}" for f in FINGERS for a in AXES]
    columns += ["pinch_human", "pinch_robot"]
    columns += [f"kp{i:02d}_{a}" for i in range(21) for a in AXES]
    return columns


class TraceLog:
    """A running CSV, or nothing. Toggled; never on by default."""

    def __init__(self, directory=LOGS):
        self.directory = Path(directory)
        self.path = None
        self.handle = None
        self.writer = None
        self.rows = 0
        self.started = None
        self.error = ""
        self._pending = 0
        self._joint_names = []

    @property
    def active(self):
        return self.writer is not None

    @property
    def seconds(self):
        return 0.0 if self.started is None else time.monotonic() - self.started

    # ------------------------------------------------------------ lifecycle
    def start(self, joint_names):
        if self.active:
            return self.path
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.path = self.directory / f"trace-{stamp}.csv"
            # "x", not "w": a recording is minutes of somebody's hand and there
            # is no version of this worth silently overwriting. Two sessions
            # started inside the same second get a suffix rather than one file.
            suffix = 0
            while True:
                try:
                    self.handle = self.path.open("x", newline="")
                    break
                except FileExistsError:
                    suffix += 1
                    self.path = self.directory / f"trace-{stamp}-{suffix}.csv"
        except OSError as failure:
            # A log that cannot be written is worth saying out loud and then
            # stepping over; it is a diagnostic, not the session.
            self.error = f"cannot log: {failure}"
            self.path = self.handle = None
            return None
        self._joint_names = list(joint_names)
        self.writer = csv.writer(self.handle)
        self.writer.writerow(_columns(self._joint_names))
        self.rows = 0
        self._pending = 0
        self.started = time.monotonic()
        self.error = ""
        return self.path

    def stop(self):
        if self.handle is not None:
            try:
                self.handle.flush()
                self.handle.close()
            except OSError:
                pass
        self.handle = self.writer = None
        self.started = None
        return self.path

    def toggle(self, joint_names, on=None):
        wanted = (not self.active) if on is None else bool(on)
        if wanted:
            return self.start(joint_names)
        self.stop()
        return None

    # ----------------------------------------------------------------- rows
    def write(self, mimic, sim, hardware, targets):
        """One frame. Silently does nothing when the log is off.

        Everything is read rather than recomputed: the human side comes from
        the same mapped hand the retargeting used this frame, so the file
        records what actually drove the robot and not a second opinion about
        it.
        """
        if not self.active:
            return
        retarget = mimic.retarget
        hand, raw = mimic.hand, mimic.raw_hand
        tracked = bool(mimic.tracked and hand is not None)

        row = [round(self.seconds, 5), mimic.frames, mimic.mode,
               int(tracked), hand.type if tracked else "",
               round(hand.pinch_strength, 4) if tracked else "",
               round(hand.grab_strength, 4) if tracked else "",
               mimic.bridge.ms if mimic.mode == "keyvector" else ""]

        # Raw fingertips, device millimetres, straight off the sensor.
        for digit in DIGITS:
            row += (list(np.round(raw.tip(digit), 3))
                    if raw is not None else ["", "", ""])

        # The same tips filtered and mapped into the robot's world, which is
        # what makes them directly comparable to the rob_ columns below.
        if tracked:
            place = retarget.place(hand, origin=sim.position)
            mapped = {d: place(hand.tip(d))[0] for d in DIGITS}
        else:
            mapped = {}
        for digit in DIGITS:
            row += (list(np.round(mapped[digit], 6)) if digit in mapped
                    else ["", "", ""])

        angles = retarget.measure(hand) if tracked else {}
        row += [round(angles[f, r], 6) if angles else ""
                for f in FINGERS for r in ROLES]

        row += [round(float(v), 6) for v in targets]
        row += [round(float(v), 6) for v in sim.joint_positions()]
        encoders = hardware.positions
        row += ([round(float(v), 6) for v in encoders] if encoders is not None
                else [""] * len(self._joint_names))

        tips = sim.tip_positions()
        for finger in FINGERS:
            row += list(np.round(tips[finger], 6))

        # The single most useful curve in the file: how far apart the thumb
        # and index tips are, on each hand, frame by frame. A retargeting that
        # tracks will have these two rise and fall together.
        row.append(round(float(np.linalg.norm(mapped["thumb"] - mapped["index"])), 6)
                   if tracked else "")
        row.append(round(float(np.linalg.norm(tips["thumb"] - tips["index"])), 6))

        # The solver's own input, verbatim, so this file can be replayed
        # through a different configuration without the operator repeating
        # the gesture.
        if tracked:
            points = mano_keypoints(hand, retarget.rotation(hand))
            row += [round(float(v), 6) for v in points.reshape(-1)]
        else:
            row += [""] * 63

        self.writer.writerow(row)
        self.rows += 1
        self._pending += 1
        if self._pending >= FLUSH_ROWS:
            self._pending = 0
            try:
                self.handle.flush()
            except OSError as failure:
                self.error = f"log write failed: {failure}"
                self.stop()

    def status(self):
        return {"active": self.active, "rows": self.rows,
                "seconds": round(self.seconds, 1),
                "path": self.path.name if self.path else "",
                "error": self.error}
