"""Mimic sessions: pairs of what the camera sees of your hand and the robot
pose you meant - so the hand-motion policy can learn from camera skeletons,
not only from the robot's own shapes.

    session = MimicSession(prompts(count))
    q, phase, number = session.target(seconds)   # the robot pose to show
    session.add(frame, seconds)                  # each new camera frame
    path = session.save(camera_name)             # -> data/learning/mimic/<stamp>.npz
    pairs = load_all()                           # every session, for training

Why: the camera never sees a pinch as touching. Measured on a recording,
the thumb tip and index tip as MediaPipe gives them come no closer than 0.20
palm lengths (5th percentile 0.26) - where the robot's touch at 0.14 - so a
policy that follows the camera's shape faithfully stays 25-30 mm apart. A
session pairs the camera's skeleton with the pose you were copying.

A session shows `learning.mimic_prompts` key poses of the hand library
(hand_library.py), half of them the thumb touching fingertips, the rest
signs and fingers curled (distinct where the library has enough; there are
some 25 distinct signs and curls, so they come again), in random order. Each: the robot moves there (MOVE_S, minimum-jerk), you get
your hand there (READY_S), and you hold it (HOLD_S). Every camera frame is
kept, with the robot pose shown at its moment and the phase (MOVE, READY,
HOLD); the holds are the pairs to trust, the rest show the way between.

    mimic/<stamp>.npz
        time (N,) s since the start; world (N, 21, 3) m: the hand followed
        (MediaPipe world landmarks, NaN where none); side (N,); present (N,)
        target (N, J) rad: the robot pose shown, its joints in the hand
        model's order; target_shape (N, FEATURES): its shape
        phase (N,) 0 move, 1 ready, 2 hold; prompt (N,): which pose
        prompt_kind (P,): touch / sign / curl; prompt_joints (P, J)
        camera, made
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..paths import ROOT
from . import hand_library
from .handmotion import TOUCH_PL, followed

FOLDER = ROOT / "data" / "learning" / "mimic"
MOVE_S, READY_S, HOLD_S = 0.8, 1.2, 1.5
MOVE, READY, HOLD = 0, 1, 2
TOUCH_SHARE = 0.5
DISTINCT = 0.15               # rad: key poses closer than this (mean per joint) count once


def prompts(count, seed=None):
    """`count` key poses of the hand library to copy: (joints (P, J), shapes
    (P, FEATURES), kinds (P,)) - half touches, the rest signs and curls."""
    library = hand_library.load()
    if "key_first" not in library:
        raise RuntimeError("build the hand library first (Learning, learn: hand motion)")
    rng = np.random.default_rng(seed)
    kind_of_key = np.repeat(library["kind"][~library["recorded"]],
                            np.diff(library["key_first"])[~library["recorded"]])
    keep = np.isin(kind_of_key, ("touch", "gesture", "wave"))
    joints = library["key_joints"][keep]
    shapes = hand_library._Hand().kinematics.shapes(joints)
    touch = shapes[:, -3:].min(axis=1) < TOUCH_PL
    kinds = np.where(touch, "touch", np.where(kind_of_key[keep] == "gesture", "sign", "curl"))
    chosen = []
    for want, share in (("touch", TOUCH_SHARE), (None, 1 - TOUCH_SHARE)):
        pool = np.flatnonzero(kinds == want) if want else np.flatnonzero(kinds != "touch")
        rng.shuffle(pool)
        wanted, picked = round(count * share), []
        for i in pool:                              # distinct poses first
            if len(picked) == wanted:
                break
            if all(np.abs(joints[i] - joints[j]).mean() > DISTINCT for j in picked):
                picked.append(i)
        if picked:                                  # too few distinct: some again
            picked += list(rng.choice(picked, wanted - len(picked)))
        chosen += picked
    chosen = rng.permutation(chosen)
    return joints[chosen].astype(float), shapes[chosen], kinds[chosen]


class MimicSession:
    def __init__(self, prompted):
        """`prompted`: what prompts() gives."""
        self.joints, self.shapes, self.kinds = prompted
        self.period = MOVE_S + READY_S + HOLD_S
        self.rows = []             # (time, world, side, present, prompt, phase)
        self._last_frame = None

    @property
    def length_s(self):
        return len(self.joints) * self.period

    def target(self, seconds):
        """(robot joints, phase, prompt number) at `seconds` into the session;
        None after the last."""
        number = int(seconds // self.period)
        if number >= len(self.joints):
            return None
        into = seconds - number * self.period
        q = self.joints[number]
        if into < MOVE_S:
            before = self.joints[number - 1] if number else np.zeros_like(q)
            s = into / MOVE_S
            return before + (q - before) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5), MOVE, number
        return q, (READY if into < MOVE_S + READY_S else HOLD), number

    def add(self, frame, seconds):
        """Keep a camera frame (once) taken `seconds` into the session."""
        if frame is None or frame.number == self._last_frame:
            return
        self._last_frame = frame.number
        shown = self.target(seconds)
        if shown is None:
            return
        _, phase, number = shown
        hand = followed(frame.hands)
        world = hand.world.astype(np.float32) if hand else np.full((21, 3), np.nan, np.float32)
        self.rows.append((seconds, world, hand.side if hand else "", hand is not None,
                          number, phase))

    def save(self, camera):
        """Write the session; its path (None without frames)."""
        if not self.rows:
            return None
        times, world, side, present, number, phase = (np.array(c) for c in zip(*self.rows))
        target = np.array([self.target(t)[0] for t in times])
        target_shape = hand_library._Hand().kinematics.shapes(target)
        FOLDER.mkdir(parents=True, exist_ok=True)
        path = FOLDER / f"{datetime.now():%Y%m%d-%H%M%S}.npz"
        np.savez_compressed(
            path, time=times.astype(np.float32), world=world.astype(np.float32), side=side,
            present=present, target=target.astype(np.float32),
            target_shape=target_shape.astype(np.float32), phase=phase.astype(np.int8),
            prompt=number.astype(np.int16), prompt_kind=self.kinds,
            prompt_joints=self.joints.astype(np.float32), camera=np.array(camera),
            made=np.array(datetime.now().isoformat(timespec="seconds")))
        return path


def sessions():
    """The saved sessions, oldest first."""
    return sorted(FOLDER.glob("*.npz")) if FOLDER.exists() else []


def load(path):
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}
