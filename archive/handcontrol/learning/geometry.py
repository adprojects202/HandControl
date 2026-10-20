"""The hand as distances: every fingertip pair, every digit's curl, and how
fast each is changing.

WHY DISTANCES. Everything else in this package is expressed in a frame -
wrist at the origin, the robot's axes, a rest rotation per bone - and every
one of those choices has cost something. A distance between two points has no
frame. It survives the rotation into robot axes untouched, it never sees the
rest offsets, it does not care which chirality the robot is. On the raw view it
is exactly what the sensor reported; on the robot it is exactly what the
kinematics produce; and the two are comparable up to one scale per pair.

WHAT THE CELLS MEAN. A pinch is one thumb cell under a threshold. A fist is
every thumb cell small at once. A spread is the adjacent-finger cells large.
How closed a digit is, is its tip-to-knuckle cell. The teacher's contact,
tracking and fist terms were already reading the thumb row of this matrix
before it had a name; this file gives it one, so the objective, the student's
input features and the labels all index the same cell the same way.

THE DERIVATIVES ARE THE PART THE OPTIMISER CANNOT USE. A per-frame solve sees
a distance. A student sees the distance closing at -13 mm/s while decelerating
at -867 mm/s^2, and that is a strike arriving. `time_to_contact` is the label
for it: seconds until this cell next crosses the threshold, known exactly here
because the whole run is present, and something a causal network can only
predict. Measured on a real tap, the linear guess d/|v| said 126 ms and the
truth was 80 - the acceleration profile is the difference, and it is what a
student learns.

WHAT THIS IS NOT. Distances lose orientation and handedness: a mirrored hand
has the same matrix, and a finger bent forward can look like one bent back.
They carry the hand's relations, not its pose. The aligned keypoints carry the
pose; this sits beside them, never instead of them.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from ..leap import DIGITS

# MANO slots. A digit's first joint is its knuckle; its last is its tip.
TIP_SLOT = {digit: 4 * index + 4 for index, digit in enumerate(DIGITS)}
KNUCKLE_SLOT = {digit: 4 * index + 1 for index, digit in enumerate(DIGITS)}

# THE TABLE. Ten tip-to-tip pairs in `combinations` order, then five curls.
# Everything that indexes a cell - the objective's thumb row, a stored label, a
# student feature - goes through this, so the cell called "thumb-middle" is
# the same number everywhere.
PAIRS = tuple((f"{a}-{b}", "tip", a, b) for a, b in combinations(DIGITS, 2)) \
    + tuple((f"{d}-curl", "curl", d, d) for d in DIGITS)
NAMES = tuple(name for name, _, _, _ in PAIRS)
INDEX = {name: k for k, name in enumerate(NAMES)}
CELLS = len(PAIRS)

# The three cells the contact objective reads, in the order it reads them:
# thumb against index, middle, ring. The robot has no pinky.
THUMB_ROW = (INDEX["thumb-index"], INDEX["thumb-middle"], INDEX["thumb-ring"])


def _slots():
    """(a_slot, b_slot) per cell into the 21-point layout."""
    out = []
    for _, kind, a, b in PAIRS:
        if kind == "tip":
            out.append((TIP_SLOT[a], TIP_SLOT[b]))
        else:
            out.append((TIP_SLOT[a], KNUCKLE_SLOT[a]))
    return np.array(out)


SLOTS = _slots()


def pairwise(points):
    """(..., 21, 3) metres -> (..., CELLS) millimetres. numpy or torch."""
    a, b = SLOTS[:, 0], SLOTS[:, 1]
    if isinstance(points, torch.Tensor):
        return (points[..., a, :] - points[..., b, :]).norm(dim=-1) * 1000.0
    points = np.asarray(points, dtype=float)
    return np.linalg.norm(points[..., a, :] - points[..., b, :], axis=-1) * 1000.0


def time_to_contact(distances, t, bounds, threshold):
    """Seconds until each cell next drops below `threshold`, per frame, per
    run. Zero while it is below; infinite if it never does again in the run.

    The label a causal student is asked to predict, computed the one way it
    can be computed exactly: with the future in hand.
    """
    distances = np.asarray(distances, dtype=float)
    out = np.full(distances.shape, np.inf)
    for start, end in bounds:
        run_t = np.asarray(t[start:end], dtype=float)
        below = distances[start:end] < threshold
        for cell in range(distances.shape[1]):
            hits = np.flatnonzero(below[:, cell])
            if not len(hits):
                continue
            # for every frame, the first hit at or after it
            where = np.searchsorted(hits, np.arange(end - start))
            has = where < len(hits)
            rows = np.flatnonzero(has)
            out[start + rows, cell] = run_t[hits[where[has]]] - run_t[rows]
    return out
