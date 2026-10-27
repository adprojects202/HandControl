"""Retargeting: your hand as the camera sees it -> the robot hand's joints -
what you mean, not your hand's shape - learnt from the Mimic sessions.

    report = fit(progress)                  # -> data/learning/retarget.npz
    retarget = Retarget(load())
    q, touching = retarget.map(world, side)  # (J,) rad, (3,) bool: index, middle, ring
    q, fingers = retarget.pose(world, side, before)   # with the touch made

Why: your hand and the robot's are built differently - its thumb sits much
lower and its fingers are longer for its palm - so no robot pose has a
pinching person's hand shape, and following the camera's hand shape never
pinches (0% of your held-out pinches through the hand-motion policy). The
mapping goes to the robot's joints instead, trained on what you meant.

Pairs: every Mimic frame with your hand seen - in a follow session every
frame, the pose shown your lag earlier (mimic.clips finds the lag); in a
pose session its holds. In: the 21 landmarks in the palm's frame, in palm
lengths (handmotion.in_palm; a right hand mirrored), 63 numbers. Out: the
robot pose's 16 joint angles, and for the index, middle and ring fingers
whether the thumb touches it. A small network - a shared trunk (HIDDEN), a
joints head and a touch head - fitted with input noise (NOISE), touches
counted as much as they are rare. Measured on what it was not fitted to:
every HELD_OUT-th 6 s stretch of a follow session, every HELD_OUT-th pose
of a pose session; then fitted again to all of it.

Joint angles average where touching and open poses meet, so a pinch comes
out short of touching; the touch head says when you mean one, and then the
thumb tip is put on those fingertips by inverse kinematics
(hand_library.HandModel.reach) - the robot's own touch, whatever your
thumb's shape. On the first follow session, end to end through the
hand-motion policy on held-out stretches: the robot touched in 50% of the
frames you meant a pinch from the joints alone, 68% with the touch made
(74% half a second into it).

    retarget.npz   W0, b0, ... (trunk, then joints head, then touch head),
                   mean, std (63,): the input's normalisation; joints (J,):
                   the joint names out; report (JSON); sessions; made
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import torch

from ..paths import ROOT
from . import mimic
from .hand_library import HandModel
from .handmotion import TOUCH_PL, in_palm

MODEL = ROOT / "data" / "learning" / "retarget.npz"
HIDDEN = (128, 128)
NOISE = 0.02                  # palm lengths, on the landmarks while fitting
STEPS = 6000
BATCH = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
TOUCH_LOSS = 0.2
STRETCH_S = 6.0
HELD_OUT = 5
FINGERS = ("index", "middle", "ring")


def pairs(paths=None, hand=None):
    """(landmarks (N, 63), joints (N, J), touching (N, 3), held out (N,))
    over every session."""
    hand = hand or HandModel()
    inputs, joints, held = [], [], []
    for path in paths if paths is not None else mimic.sessions():
        session = mimic.load(path)
        times = session["time"].astype(float)
        present = session["present"]
        if (session["phase"] == mimic.FOLLOW).any():
            lag = mimic.clips([path])["lag"][0]
            target = np.stack([np.interp(times - lag, times, session["target"][:, j])
                               for j in range(session["target"].shape[1])], axis=1)
            keep = present & (times > lag)
            out = (times // STRETCH_S).astype(int) % HELD_OUT == HELD_OUT - 1
        else:
            target = session["target"]
            keep = present & (session["phase"] == mimic.HOLD)
            out = session["prompt"] % HELD_OUT == 0
        for side in np.unique(session["side"][keep]):
            these = keep & (session["side"] == side)
            landmarks = in_palm(session["world"][these].astype(float), str(side))
            inputs.append(landmarks.reshape(len(landmarks), -1))
            joints.append(target[these])
            held.append(out[these])
    if not inputs:
        raise RuntimeError("no Mimic sessions - record one (Vision panel, or "
                           "scripts/record_mimic.py)")
    joints = np.concatenate(joints)
    touching = hand.kinematics.shapes(joints)[:, -3:] < TOUCH_PL
    return np.concatenate(inputs), joints, touching, np.concatenate(held)


def fit(progress=lambda done, total, text="": None, path=MODEL, paths=None, seed=0,
        steps=STEPS):
    """Fit to every session, measured on the held-out part first; saved to
    `path`. Returns the report."""
    hand = HandModel()
    x, q, touching, held = pairs(paths, hand)
    progress(0, 2, "fitting the mapping on all but the held-out part")
    trial = _train(x[~held], q[~held], touching[~held], seed, steps)
    report = _measure(Retarget(trial, hand), x[held], q[held], touching[held])
    report.update(pairs=int(len(x)), held_out=int(held.sum()))
    progress(1, 2, "fitting the mapping on all of it")
    arrays = _train(x, q, touching, seed, steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, joints=np.array([hand.model.joint(j).name
                                              for j in range(hand.model.njnt)]),
             report=np.array(json.dumps(report)),
             sessions=np.array([p.name for p in (paths or mimic.sessions())]),
             made=np.array(datetime.now().isoformat(timespec="seconds")))
    progress(2, 2, "mapping fitted")
    return report


def load(path=MODEL):
    if not path.exists():
        return {}
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def report(arrays):
    return json.loads(str(arrays["report"])) if arrays else {}


class Retarget:
    """The fitted mapping, in NumPy: camera hand in, robot joints out."""

    def __init__(self, arrays, hand=None):
        self.hand = hand or HandModel()
        count = len([key for key in arrays if key.startswith("W")])
        self.layers = [(arrays[f"W{i}"], arrays[f"b{i}"]) for i in range(count)]
        self.mean, self.std = arrays["mean"], arrays["std"]

    def map(self, world, side):
        """MediaPipe world landmarks (21, 3) of a `side` hand -> (the robot's
        joints (J,), whether the thumb touches each of the index, middle and
        ring fingers (3,))."""
        return self.map_landmarks(in_palm(np.asarray(world, float), side).reshape(-1))

    def map_landmarks(self, landmarks):
        """As map, from the 21 landmarks already in the palm's frame (63,)."""
        hidden = (landmarks - self.mean) / self.std
        for weight, bias in self.layers[:-2]:
            hidden = _elu(weight @ hidden + bias)
        (joint_w, joint_b), (touch_w, touch_b) = self.layers[-2:]
        return self.hand.clip(joint_w @ hidden + joint_b), (touch_w @ hidden + touch_b) > 0.0

    def pose(self, world, side, before=None):
        """(The robot pose meant - the joints, with the thumb tip put on the
        fingertips it touches - and those fingers.) The search starts from
        `before` (the last pose) when given: quick, for a pose near it."""
        q, touching = self.map(world, side)
        fingers = tuple(f for f, on in zip(FINGERS, touching) if on)
        if not fingers:
            return q, fingers
        start = q if before is None else np.where(np.isin(np.arange(len(q)), self._moved(fingers)),
                                                   before, q)
        return self.hand.reach(start, fingers), fingers

    def _moved(self, fingers):
        return self.hand.joints["thumb"] + [j for f in fingers for j in self.hand.joints[f]]


def _train(x, q, touching, seed, steps=STEPS):
    """Fit the network; its weights as arrays (see the module docstring)."""
    torch.manual_seed(seed)
    # The wrist sits at the palm frame's origin and the middle knuckle on its
    # x axis: numbers that never change. A floor on their spread keeps the
    # noise, given in palm lengths, from being scaled up to nonsense.
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), NOISE)
    tx = torch.as_tensor((x - mean) / std, dtype=torch.float32)
    tq = torch.as_tensor(q, dtype=torch.float32)
    tt = torch.as_tensor(touching, dtype=torch.float32)
    sizes = (x.shape[1], *HIDDEN)
    trunk = [torch.nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])]
    joints_head, touch_head = torch.nn.Linear(sizes[-1], q.shape[1]), torch.nn.Linear(sizes[-1], 3)
    layers = [*trunk, joints_head, touch_head]
    optimiser = torch.optim.AdamW([p for layer in layers for p in layer.parameters()],
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rare = tt.mean(dim=0).clamp(1e-3, 1 - 1e-3)
    noise = torch.as_tensor(NOISE / std, dtype=torch.float32)
    for _ in range(steps):
        pick = torch.randint(0, len(tx), (BATCH,))
        hidden = tx[pick] + noise * torch.randn(BATCH, tx.shape[1])
        for layer in trunk:
            hidden = torch.nn.functional.elu(layer(hidden))
        loss = (((joints_head(hidden) - tq[pick]) ** 2).mean() + TOUCH_LOSS
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    touch_head(hidden), tt[pick], pos_weight=(1 - rare) / rare))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    arrays = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    for i, layer in enumerate(layers):
        arrays[f"W{i}"] = layer.weight.detach().numpy()
        arrays[f"b{i}"] = layer.bias.detach().numpy()
    return arrays


def _measure(retarget, x, q, touching):
    """On held-out pairs: joints off, the touch head's calls, and the poses
    it makes - do they touch where you meant, and where you did not."""
    mapped, said, made = [], [], []
    for row in x:
        joints, touch = retarget.map_landmarks(row)
        mapped.append(joints)
        said.append(touch)
    mapped, said = np.array(mapped), np.array(said)
    for joints, touch in zip(mapped, said):
        fingers = tuple(f for f, on in zip(FINGERS, touch) if on)
        made.append(retarget.hand.reach(joints, fingers) if fingers else joints)
    touches = retarget.hand.kinematics.shapes(np.array(made))[:, -3:] < TOUCH_PL
    return {"joints_deg": float(np.degrees(np.abs(mapped - q)).mean()),
            "touch_said_when_meant": float(said[touching].mean()),
            "touch_said_when_not": float(said[~touching].mean()),
            "touches_made_when_meant": float(touches[touching].mean()),
            "touches_made_when_not": float(touches[~touching].mean())}


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
