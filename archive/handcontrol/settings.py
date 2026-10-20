"""What the panel remembers between runs.

`config/panel_settings.json`, alongside the other two files that describe one
particular setup: `hand_calibration.json` is one physical robot, and
`leap_calibration.json` is one operator's hand. This one is neither - it is
just where the operator left the controls.

WHAT IS DELIBERATELY NOT REMEMBERED is the output switch and the arm switch.
The panel has three hardware states on purpose - available, enabled, armed -
and only the first is safe to restore. A port is remembered, because naming a
port makes the hand *available* and nothing more; that is the state the rest of
this project already calls safe, and re-picking it every launch is the friction
this file exists to remove. Sending commands and energising sixteen servos stay
off at startup, every startup, and no file can change that.

The file is plain JSON and meant to be readable, so it is also editable, so it
is treated as untrusted: every value is checked against the defaults before it
is used, unknown keys are dropped, and anything that fails validation falls
back rather than propagating. A settings file should never be able to stop the
panel from opening.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .robot.collide import MARGIN
from .hardware import COMMAND_HZ, MAX_SPEED
from .robot.model import EFFORT, IK, KP, KV

from .paths import CONFIG

SETTINGS = CONFIG / "panel_settings.json"

# Writes are debounced: a slider drag fires on every mouse-motion event, and
# none of those is worth a syscall. Dirty state is flushed on this interval and
# again on the way out.
WRITE_INTERVAL = 2.0

NOTE = ("the panel's controls, as you left them. Safe to edit or delete. The "
        "output and arm switches are deliberately not in here - they start off "
        "every time, whatever this file says.")

# Bumped whenever the shipped tuning defaults change. A stored `tuning` block
# written against different defaults is DISCARDED rather than restored.
#
# This exists because the alternative already happened: the defaults were
# retuned for responsiveness in the code, and every launch quietly put the old
# slow numbers back from this file - a change that appeared to have been made
# and had not been, with nothing on screen to say so. A preference file may
# remember what you chose; it must not outvote the program about what it ships.
TUNING_VERSION = 3

# The tuning fields that are safe to put back automatically. They are all
# software: simulator gains, and how fast the bus thread is allowed to move a
# target. The SERVO fields on that tab are deliberately absent - those are RAM
# registers on sixteen motors, and rewriting them because a file said so is a
# hardware action nobody asked for at launch. The panel shows what the
# simulator reports for those, which is the truth.
DEFAULTS = {
    "tab": "joints",
    "finger": "index",
    "port": "",
    "mimic": {"mode": "keyvector", "thumb": "angles", "mirror": False,
              "smoothing": 0.35, "spread_gain": 0.6, "scale": 1.0},
    # Taken from the modules that own them, never copied. A copy here is a
    # second set of defaults that silently wins: this file is READ at startup
    # and applied over whatever the code shipped, so a stale number here undoes
    # a retune everywhere else and leaves no trace of having done it.
    "tuning": {"kp": KP, "kv": KV, "effort": EFFORT,
               "collision_margin": MARGIN, "max_speed": MAX_SPEED,
               "command_hz": COMMAND_HZ,
               "ik_damping": IK["damping"], "ik_iters": IK["max_solver_iters"],
               "ik_tol": IK["pos_tol"]},
}

# name -> what it must be. A choice list, or (low, high) for a number, or a
# type. Anything failing its check is replaced by the default rather than
# rejected, so one bad value costs one setting and not the file.
SCHEMA = {
    "tab": ("joints", "tip", "mimic", "learning", "tune"),
    "finger": ("index", "middle", "ring", "thumb"),
    "port": str,
    "mimic": {"mode": ("raw", "angles", "tip", "keyvector", "posebank", "student"),
              "thumb": ("angles", "pinch"),
              "mirror": bool, "smoothing": (0.05, 1.0),
              "spread_gain": (0.0, 1.5), "scale": (0.5, 1.6)},
    # Deliberately wider than the panel's own sliders. These bounds are here
    # to reject nonsense, not to define the control range - the panel clamps
    # every value to its slider when it applies it, so the two cannot drift
    # apart as the TUNING table changes.
    "tuning": {"kp": (0.1, 1000.0), "kv": (0.0, 100.0), "effort": (0.01, 50.0),
               "collision_margin": (0.0, 0.1), "max_speed": (0.1, 50.0),
               "command_hz": (1.0, 500.0), "ik_damping": (0.0001, 1.0),
               "ik_iters": (1, 200), "ik_tol": (1e-5, 0.02)},
}


def _clean(value, rule, fallback):
    """One value against one rule. Never raises; returns `fallback` instead."""
    if isinstance(rule, dict):
        if not isinstance(value, dict):
            return dict(fallback)
        return {key: _clean(value.get(key, fallback[key]), sub, fallback[key])
                for key, sub in rule.items()}
    if isinstance(rule, tuple) and rule and isinstance(rule[0], str):
        return value if value in rule else fallback
    if isinstance(rule, tuple):                       # a numeric range
        try:
            # bool is an int in Python and would sail through this; a range
            # rule always means a number, so reject it explicitly.
            if isinstance(value, bool):
                raise TypeError
            return min(max(float(value), rule[0]), rule[1])
        except (TypeError, ValueError):
            return fallback
    if rule is bool:
        return bool(value) if isinstance(value, bool) else fallback
    if rule is str:
        return value if isinstance(value, str) else fallback
    return fallback


class Settings:
    """A validated dict that knows when it needs writing.

    Reads never fail: a missing file is the first run, and a corrupt one is a
    first run with a warning. Writes never fail loudly either - a read-only
    checkout should cost you the memory of where the sliders were, not the
    session.
    """

    def __init__(self, path=None):
        # Resolved here rather than as a default argument, which would bind
        # the module constant once at import and ignore any later change to
        # it - including the one a test makes to keep out of config/.
        self.path = Path(path or SETTINGS)
        self.values = {key: _clean(value, SCHEMA[key], value)
                       for key, value in DEFAULTS.items()}
        self.restored = False
        self.error = ""
        self._dirty = False
        self._written = time.time()
        self.load()

    # ------------------------------------------------------------- access
    def __getitem__(self, key):
        return self.values[key]

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        """Store one top-level setting, if it is different and it is valid."""
        cleaned = _clean(value, SCHEMA[key], DEFAULTS[key])
        if self.values.get(key) != cleaned:
            self.values[key] = cleaned
            self._dirty = True
        return cleaned

    def update(self, key, **changes):
        """Store part of a nested setting - `update("mimic", scale=1.2)`."""
        merged = dict(self.values.get(key) or {})
        merged.update(changes)
        return self.set(key, merged)

    # --------------------------------------------------------------- file
    def load(self):
        if not self.path.exists():
            return False
        try:
            stored = json.loads(self.path.read_text())
        except (ValueError, OSError) as failure:
            # Keep the defaults and say so. Deleting the file for the operator
            # would be worse: it may be theirs and hand-edited.
            self.error = f"{self.path.name}: {failure}"
            return False
        if not isinstance(stored, dict):
            self.error = f"{self.path.name}: not an object"
            return False
        stale = stored.get("tuning_version") != TUNING_VERSION
        for key, rule in SCHEMA.items():
            if key not in stored:
                continue
            if key == "tuning" and stale:
                continue          # written against different defaults
            self.values[key] = _clean(stored[key], rule, DEFAULTS[key])
        if stale and "tuning" in stored:
            self.error = ("tuning defaults changed - the saved gains were "
                          "dropped and the shipped ones are in use")
            self._dirty = True
        self.restored = True
        return True

    def save(self, force=False):
        """Write if anything changed, at most every `WRITE_INTERVAL`."""
        now = time.time()
        if not self._dirty:
            return False
        if not force and now - self._written < WRITE_INTERVAL:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"_note": NOTE, "tuning_version": TUNING_VERSION, **self.values},
                indent=2, sort_keys=True) + "\n")
        except OSError as failure:
            self.error = f"{self.path.name}: {failure}"
            self._dirty = False          # do not retry every frame
            return False
        self._dirty = False
        self._written = now
        return True

    def close(self):
        return self.save(force=True)
