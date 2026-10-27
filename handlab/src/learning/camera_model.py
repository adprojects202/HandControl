"""What the camera makes of the robot's poses: a model, fitted to the Mimic
sessions (mimic.py), of the hand shape the camera gives
(handmotion.human_features) when you copy a robot pose - its mean and its
spread.

    report = fit()                     # -> data/learning/camera_model.npz
    model = CameraModel(load())        # on any torch device
    mean, std = model(shapes)          # (n, FEATURES) -> both (n, FEATURES)

The hand-motion training uses it to show the policy goals as the camera
would give them (gpu_hand.py): so it learns, say, that the camera's 0.22
palm lengths between thumb and index means the fingertips touch.

Pairs: every frame of every session's holds (the last HOLD_S of each pose)
with your hand seen - your shape from the camera, and the shape of the pose
shown. The model: each number on its own, a straight line from the robot
pose's to the camera's (mean = slope x + offset, drawn a little towards
slope 1 by RIDGE), and a spread per number - the root mean square of its
errors on poses the line was not fitted to (every HELD_OUT-th pose; at least
MIN_SPREAD), then refitted to every pose. A spread that wide where the
camera is unsure (a finger's direction it cannot see well) and narrow where
it is sure (a pinch) tells the policy which numbers to trust. A network
(robot shape plus a learnt correction) was tried first: on the first real
session, 40 poses, its held-out error was 0.245 palm lengths and its pinches
0.151, where the lines' were 0.149 and 0.130 - too few poses for it.

The report gives the held-out error next to taking the camera's shape to be
the robot's own (identity) - overall and for the pinches - and the spread.

    camera_model.npz   slope, offset, spread (FEATURES,); report (JSON),
                       sessions, made
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import torch

from ..paths import ROOT
from . import mimic
from .handmotion import FEATURES, human_features

MODEL = ROOT / "data" / "learning" / "camera_model.npz"
HELD_OUT = 5                  # every 5th pose is not fitted, only measured
RIDGE = 0.05                  # pull towards slope 1, in squared palm lengths per pose
MIN_SPREAD = 0.01             # palm lengths
PINCH = slice(FEATURES - 3, FEATURES)


def pairs(paths=None):
    """(robot shapes X, camera shapes Y, pose ids) over every session's holds
    with the hand seen; pose ids are unique across sessions."""
    xs, ys, ids, offset = [], [], [], 0
    for path in paths if paths is not None else mimic.sessions():
        session = mimic.load(path)
        keep = (session["phase"] == mimic.HOLD) & session["present"]
        if keep.any():
            world = session["world"][keep].astype(float)
            sides = session["side"][keep]
            for side in np.unique(sides):
                these = sides == side
                ys.append(human_features(world[these], str(side)))
                xs.append(session["target_shape"][keep][these])
                ids.append(session["prompt"][keep][these].astype(int) + offset)
        offset += len(session["prompt_kind"])
    if not xs:
        return np.zeros((0, FEATURES)), np.zeros((0, FEATURES)), np.zeros(0, int)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def fit(progress=lambda done, total, text="": None, path=MODEL, paths=None):
    """Fit the model to every session; saved to `path`. Returns the report."""
    x, y, ids = pairs(paths)
    if len(np.unique(ids)) < 2 * HELD_OUT:
        raise RuntimeError("too few Mimic poses to fit the camera model - record a Mimic "
                           "session (Vision panel, or scripts/record_mimic.py)")
    progress(0, 1, "fitting the camera model")
    held = ids % HELD_OUT == 0
    slope, offset = _lines(x[~held], y[~held], len(np.unique(ids[~held])))
    guess = x[held] * slope + offset
    spread = np.maximum(MIN_SPREAD, np.sqrt(((y[held] - guess) ** 2).mean(axis=0)))
    report = {
        "frames": int(len(x)), "poses": int(len(np.unique(ids))),
        "held_out_poses": int(len(np.unique(ids[held]))),
        "error": float(np.abs(guess - y[held]).mean()),
        "identity_error": float(np.abs(x[held] - y[held]).mean()),
        "pinch_error": float(np.abs(guess - y[held])[:, PINCH].mean()),
        "identity_pinch_error": float(np.abs(x[held] - y[held])[:, PINCH].mean()),
        "spread": float(spread.mean()), "pinch_spread": float(spread[PINCH].mean())}
    slope, offset = _lines(x, y, len(np.unique(ids)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, slope=slope.astype(np.float32), offset=offset.astype(np.float32),
             spread=spread.astype(np.float32), report=np.array(json.dumps(report)),
             sessions=np.array([p.name for p in (paths or mimic.sessions())]),
             made=np.array(datetime.now().isoformat(timespec="seconds")))
    progress(1, 1, "camera model fitted")
    return report


def load(path=MODEL):
    if not path.exists():
        return {}
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def report(arrays):
    return json.loads(str(arrays["report"])) if arrays else {}


class CameraModel:
    """The fitted model on a torch device: shapes in, (mean, std) out."""

    def __init__(self, arrays, device="cpu"):
        self.slope, self.offset, self.spread = (
            torch.as_tensor(arrays[key], device=device) for key in ("slope", "offset", "spread"))

    def __call__(self, shapes):
        return shapes * self.slope + self.offset, self.spread.expand_as(shapes)


def _lines(x, y, poses):
    """Each number's line, y = slope x + offset, least squares with the slope
    drawn towards 1 (RIDGE per pose: frames of one pose are not separate
    evidence)."""
    dx, dy = x - x.mean(axis=0), y - y.mean(axis=0)
    ridge = RIDGE * poses
    slope = ((dx * dy).sum(axis=0) + ridge) / ((dx * dx).sum(axis=0) + ridge)
    return slope, y.mean(axis=0) - slope * x.mean(axis=0)
