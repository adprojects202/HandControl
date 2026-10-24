"""What the camera makes of the robot's poses: a model, fitted to the Mimic
sessions (mimic.py), of the hand shape the camera gives
(handmotion.human_features) when you copy a robot pose - its mean and its
spread, each a function of the robot pose's shape.

    report = fit()                     # -> data/learning/camera_model.npz
    model = CameraModel(load())        # on any torch device
    mean, std = model(shapes)          # (n, FEATURES) -> both (n, FEATURES)

The hand-motion training uses it to show the policy goals as the camera
would give them (gpu_hand.py): so it learns, say, that the camera's 0.22
palm lengths between thumb and index means the fingertips touch.

Pairs: every frame of every session's holds (the last HOLD_S of each pose)
with your hand seen - your shape from the camera, and the shape of the pose
shown. The model: the pose's shape plus a correction, mean = x + f(x), f a
small network (HIDDEN), and a spread per number, log std = g(x); fitted by
Gaussian likelihood, with weight decay, on all but every HELD_OUT-th pose,
which it is measured on. The report gives the held-out error next to taking
the camera's shape to be the robot's own (identity) - overall and for the
pinches.

    camera_model.npz   weights (W0, b0, W1, b1, ...: the shared trunk, then
                       the mean head and the log-std head), report (JSON),
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
HIDDEN = (128, 128)
HELD_OUT = 5                  # every 5th pose is not fitted, only measured
STEPS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_LOG_STD, MAX_LOG_STD = np.log(0.005), np.log(0.5)
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


def fit(progress=lambda done, total, text="": None, path=MODEL, paths=None, seed=0):
    """Fit the model to every session; saved to `path`. Returns the report."""
    x, y, ids = pairs(paths)
    if len(np.unique(ids)) < 2 * HELD_OUT:
        raise RuntimeError("too few Mimic poses to fit the camera model - record a Mimic "
                           "session (Vision panel)")
    torch.manual_seed(seed)
    held = ids % HELD_OUT == 0
    tx, ty = (torch.as_tensor(a[~held], dtype=torch.float32) for a in (x, y))
    layers = _layers()
    optimiser = torch.optim.AdamW([p for layer in layers for p in layer],
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for step in range(STEPS):
        mean, log_std = _forward(layers, tx)
        loss = (((ty - mean) / log_std.exp()) ** 2 / 2 + log_std).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 500 == 0:
            progress(step, STEPS, f"fitting the camera model: loss {loss.item():.3f}")
    with torch.no_grad():
        mean, log_std = _forward(layers, torch.as_tensor(x[held], dtype=torch.float32))
    mean = mean.numpy()
    report = {
        "frames": int(len(x)), "poses": int(len(np.unique(ids))),
        "held_out_poses": int(len(np.unique(ids[held]))),
        "error": float(np.abs(mean - y[held]).mean()),
        "identity_error": float(np.abs(x[held] - y[held]).mean()),
        "pinch_error": float(np.abs(mean - y[held])[:, PINCH].mean()),
        "identity_pinch_error": float(np.abs(x[held] - y[held])[:, PINCH].mean()),
        "spread": float(np.exp(log_std.numpy()).mean())}
    weights = {f"W{i}": w.detach().numpy() for i, (w, _) in enumerate(layers)}
    weights.update({f"b{i}": b.detach().numpy() for i, (_, b) in enumerate(layers)})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights, report=np.array(json.dumps(report)),
             sessions=np.array([p.name for p in (paths or mimic.sessions())]),
             made=np.array(datetime.now().isoformat(timespec="seconds")))
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
        count = len([key for key in arrays if key.startswith("W")])
        self.layers = [(torch.as_tensor(arrays[f"W{i}"], device=device),
                        torch.as_tensor(arrays[f"b{i}"], device=device)) for i in range(count)]

    def __call__(self, shapes):
        with torch.no_grad():
            mean, log_std = _forward(self.layers, shapes)
        return mean, log_std.exp()


def _layers():
    sizes = (FEATURES, *HIDDEN)
    layers = []
    for size_in, size_out in [*zip(sizes, sizes[1:]), (sizes[-1], FEATURES),
                              (sizes[-1], FEATURES)]:
        weight = torch.randn(size_out, size_in) * np.sqrt(1.0 / size_in)
        layers.append((weight.requires_grad_(), torch.zeros(size_out, requires_grad=True)))
    for weight, bias in layers[-2:]:                 # start as the identity, spread 0.05
        weight.data.zero_()
    layers[-1][1].data.fill_(float(np.log(0.05)))
    return layers


def _forward(layers, x):
    hidden = x
    for weight, bias in layers[:-2]:
        hidden = torch.nn.functional.elu(hidden @ weight.T + bias)
    (mean_w, mean_b), (std_w, std_b) = layers[-2:]
    mean = x + hidden @ mean_w.T + mean_b
    log_std = torch.clamp(hidden @ std_w.T + std_b, float(MIN_LOG_STD), float(MAX_LOG_STD))
    return mean, log_std
