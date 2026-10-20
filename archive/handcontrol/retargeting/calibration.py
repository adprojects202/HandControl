"""Fitting the mapping to one operator's hand, and keeping it on disk.

Two fits, captured from the same three poses:

  RANGES - per joint, the human angles that span that joint's travel. This is
  the answer to "it moves, but it does not look like my hand": a fixed gain
  assumes the operator curls through the same angles the robot has, and nobody
  does. The Leap under-reports a DIP it cannot see, people differ by tens of
  degrees at full curl, and human finger spread is a sixth of the robot's.

  PLACEMENT - one scale and one translation for the fingertip mapping, so a
  flat hand lands on the robot's own flat pose rather than short and inside
  the palm.

Free functions rather than methods, and no robot in sight: they take the joint
names and travel spans they need. That keeps the file format decided in one
place and lets the fit be tested against numbers typed by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..paths import CONFIG


# One operator's hand, the way config/hand_calibration.json is one physical
# robot's. Separate files because they describe different objects and go wrong
# at different times: a new person needs a new fit, a rebuilt hand does not.
CALIBRATION = CONFIG / "leap_calibration.json"

# Below this the two poses did not differ enough to call it a range, and the
# joint is left on its fixed gain rather than given a fit built on noise.
MIN_SPAN = math.radians(4.0)

NOTE = ("one operator's hand, fitted by the MIMIC tab's CALIBRATE button. "
        "ranges are the human angles in radians that span each joint's travel; "
        "scale and offset place the fingertips for tip mode. Delete this file "
        "to start over.")


def fit_ranges(poses, keys, spread_keys, min_span=MIN_SPAN):
    """Human angle spans per joint, from the open / fist / spread captures.

    Flexion is fitted open -> fist. Spread is fitted as a band around the
    operator's own neutral, symmetric by construction, because "fingers
    together" is the zero and a fist cannot report a spread - the fingers are
    touching.

    Returns {} rather than a partial fit if the captures are unusable, so the
    caller can leave the fixed gains alone and say so.
    """
    open_pose, fist, spread = (poses.get("open"), poses.get("fist"),
                               poses.get("spread"))
    if not open_pose or not fist:
        return {}
    ranges = {}
    for key in keys:
        if key in spread_keys:
            if spread is None:
                continue
            neutral = open_pose[key]
            extent = abs(spread[key] - neutral)
            low, high = neutral - extent, neutral + extent
        else:
            low, high = open_pose[key], fist[key]
        if abs(high - low) > min_span:
            ranges[key] = (float(low), float(high))
    return ranges


def save_fit(path, ranges_by_name, hand_scale, offset, calibrated):
    """Write both fits. Keyed by joint NAME rather than by (finger, role) tuple,
    so the file stays readable and survives a change to the role tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_note": NOTE,
        "ranges": {name: list(span) for name, span in ranges_by_name.items()},
        "hand_scale": hand_scale,
        "offset": [float(v) for v in offset],
        "calibrated": bool(calibrated),
    }, indent=2, sort_keys=True) + "\n")
    return path


def load_fit(path, key_for_name):
    """Read a previous fit back.

    Returns (ranges, hand_scale, offset, calibrated). A missing file is the
    normal first run; a half-written one is a re-fit away and must not be
    allowed to stop the session starting, so it reads as "no fit" too.
    """
    path = Path(path)
    empty = ({}, None, np.zeros(3), False)
    if not path.exists():
        return empty
    try:
        stored = json.loads(path.read_text())
    except (ValueError, OSError):
        return empty
    if not isinstance(stored, dict):
        return empty

    ranges = {}
    for name, span in (stored.get("ranges") or {}).items():
        key = key_for_name.get(name)
        if key is not None and isinstance(span, list) and len(span) == 2:
            try:
                ranges[key] = (float(span[0]), float(span[1]))
            except (TypeError, ValueError):
                continue
    calibrated = bool(stored.get("calibrated")) and stored.get("hand_scale")
    if not calibrated:
        return ranges, None, np.zeros(3), False
    try:
        offset = np.array(stored.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        if offset.shape != (3,):
            raise ValueError
        return ranges, float(stored["hand_scale"]), offset, True
    except (TypeError, ValueError):
        return ranges, None, np.zeros(3), False
