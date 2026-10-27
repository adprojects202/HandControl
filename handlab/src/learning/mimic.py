"""Mimic sessions: pairs of what the camera sees of your hand and the robot
pose you meant - so the hand-motion policy can learn from camera skeletons,
not only from the robot's own shapes.

    session = MimicSession(prompts(count))       # poses: copy, then hold
    session = FollowSession(motions(seconds))     # motions: follow along
    q, phase, number = session.target(seconds)   # the robot pose to show
    session.add(frame, seconds)                  # each new camera frame
    path = session.save(camera_name)             # -> data/learning/mimic/<stamp>.npz
    clips = clips()                              # every session, as training clips

Why: the camera never sees a pinch as touching. Measured on a recording,
the thumb tip and index tip as MediaPipe gives them come no closer than 0.20
palm lengths (5th percentile 0.26) - where the robot's touch at 0.14 - so a
policy that follows the camera's shape faithfully stays 25-30 mm apart. A
session pairs the camera's skeleton with the pose you were copying.

POSES (`learning.mimic_mode` "poses"). A session shows
`learning.mimic_prompts` key poses of the hand library
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
        phase (N,) 0 move, 1 ready, 2 hold, 3 follow; prompt (N,): which
        pose (or motion)
        prompt_kind (P,): touch / sign / curl (or the motion's library
        kind); prompt_joints (P, J): each pose (or motion's first)
        camera, made
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..paths import ROOT
from . import hand_library
from .handmotion import RATE, TOUCH_PL, followed, human_features

FOLDER = ROOT / "data" / "learning" / "mimic"
# Slow enough to copy a pose properly (0.8 / 1.2 / 1.5 s was too quick).
MOVE_S, READY_S, HOLD_S = 1.0, 2.5, 2.0
MOVE, READY, HOLD, FOLLOW = 0, 1, 2, 3
SLOW = 3.0                    # motions play this many times slower than the library's
JOIN_S = 1.0                  # from one motion's end to the next's start
FOLLOWED = ("touch", "gesture", "wave")   # library kinds a hand can follow
TOUCH_SHARE = 0.5
DISTINCT = 0.15               # rad: key poses closer than this (mean per joint) count once
CLIP_S = 6.0                  # training clips: pieces of a session
MIN_CLIP_S = 2.0
MAX_LAG_S = 1.0               # how late your hand may follow the robot
HELD_OUT = 5                  # every 5th clip is only tested on (all but the last session's, with one)


def prompts(count, seed=None, library=None):
    """`count` key poses of the hand library (or `library`) to copy: (joints
    (P, J), shapes (P, FEATURES), kinds (P,)) - half touches, the rest signs
    and curls."""
    library = library or hand_library.load()
    if "key_first" not in library:
        raise RuntimeError("build the hand library first (Learning, learn: hand motion)")
    rng = np.random.default_rng(seed)
    kind_of_key = np.repeat(library["kind"][~library["recorded"]],
                            np.diff(library["key_first"])[~library["recorded"]])
    keep = np.isin(kind_of_key, ("touch", "gesture", "wave"))
    joints = library["key_joints"][keep]
    shapes = hand_library.HandModel().kinematics.shapes(joints)
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

    def left(self, seconds):
        """Seconds left in the phase `seconds` into the session."""
        into = seconds % self.period
        for end in (MOVE_S, MOVE_S + READY_S, self.period):
            if into < end:
                return end - into
        return 0.0

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
        target_shape = hand_library.HandModel().kinematics.shapes(target)
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


def motions(seconds, seed=None, library=None):
    """Library motions a hand can follow (FOLLOWED kinds), SLOW times slower,
    one after another for at least `seconds`, joined by JOIN_S minimum-jerk
    moves: (joints (T, J) at RATE, motion number per frame (T,), kinds (M,),
    first joints (M, J))."""
    library = library or hand_library.load()
    if "key_first" not in library:
        raise RuntimeError("build the hand library first (Learning, learn: hand motion)")
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(np.isin(library["kind"], FOLLOWED))
    first = library["key_first"]
    joints, numbers, kinds, starts = [], [], [], []
    while sum(len(j) for j in joints) < seconds * RATE:
        clip = int(rng.choice(pool))
        keys = slice(first[clip], first[clip + 1])
        moving = hand_library.frames(library["key_joints"][keys],
                                     np.round(library["key_hold"][keys] * SLOW),
                                     np.round(library["key_move"][keys] * SLOW))
        before = joints[-1][-1] if joints else np.zeros_like(moving[0])
        steps = round(JOIN_S * RATE)
        s = (np.arange(1, steps + 1) / steps)[:, None]
        join = before + (moving[0] - before) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
        joints.append(np.concatenate([join, moving]))
        numbers.append(np.full(len(joints[-1]), len(kinds)))
        kinds.append(str(library["kind"][clip]))
        starts.append(moving[0])
    return (np.concatenate(joints).astype(float), np.concatenate(numbers), np.array(kinds),
            np.array(starts))


class FollowSession(MimicSession):
    """MOTIONS (`learning.mimic_mode` "motions"): the robot plays library
    motions slowly and without stopping (motions()), and you follow along -
    every camera frame a pair, phase FOLLOW; your lag behind the robot is
    found and taken out when the session becomes training clips (clips())."""

    def __init__(self, moving):
        self.trajectory, self.numbers, self.kinds, self.joints = moving
        self.shapes = None
        self.rows = []
        self._last_frame = None

    @property
    def length_s(self):
        return (len(self.trajectory) - 1) / RATE

    def left(self, seconds):
        return max(0.0, self.length_s - seconds)

    def target(self, seconds):
        if seconds >= self.length_s:
            return None
        at = seconds * RATE
        row = int(at)
        q = self.trajectory[row] + (self.trajectory[row + 1] - self.trajectory[row]) * (at - row)
        return q, FOLLOW, int(self.numbers[row])


def session(seed=None):
    """A new session as `learning.mimic_mode` says: poses or motions."""
    from .. import settings

    if settings.get("learning.mimic_mode") == "motions":
        return FollowSession(motions(settings.get("learning.mimic_motion_s"), seed))
    return MimicSession(prompts(settings.get("learning.mimic_prompts"), seed))


def sessions():
    """The saved sessions, oldest first."""
    return sorted(FOLDER.glob("*.npz")) if FOLDER.exists() else []


def load(path):
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def clips(paths=None):
    """Every session as training clips at RATE: {target (M, FEATURES) - the
    pose shown, moved later by how late your hand followed it (`lag`); seen
    (M, FEATURES) - your hand's shape from the camera; first (K + 1,) each
    clip's first row; held_out (K,): only tested on - the last session's
    clips when there are more, else every HELD_OUT-th; lag (S,) s per
    session}. Only where your hand was seen throughout."""
    from .hand_library import HandModel

    kinematics = HandModel().kinematics
    paths = list(paths if paths is not None else sessions())
    target, seen, first, held, lags = [], [], [0], [], []
    for number, path in enumerate(paths):
        session = load(path)
        times = session["time"].astype(float)
        grid = np.arange(times[0], times[-1], 1.0 / RATE)
        present = _resampled(session["present"].astype(float), times, grid) > 0.99
        world = np.stack([np.interp(grid, times[session["present"]],
                                    session["world"][session["present"], p, k])
                          for p in range(21) for k in range(3)], axis=-1).reshape(-1, 21, 3)
        side = str(session["side"][session["present"]][0]) if session["present"].any() else "Left"
        shown = np.stack([_resampled(session["target"][:, j], times, grid)
                          for j in range(session["target"].shape[1])], axis=-1)
        camera = human_features(world, side)
        lag = _lag(kinematics.shapes(shown), camera)
        lags.append(lag / RATE)
        shown = np.concatenate([np.repeat(shown[:1], lag, axis=0), shown[:len(shown) - lag]])
        shapes = kinematics.shapes(shown)
        for begin, end in _runs(present):
            for start in range(begin, end, round(CLIP_S * RATE)):
                stop = min(end, start + round(CLIP_S * RATE))
                if stop - start < MIN_CLIP_S * RATE:
                    continue
                target.append(shapes[start:stop])
                seen.append(camera[start:stop])
                first.append(first[-1] + stop - start)
                held.append(number == len(paths) - 1 if len(paths) > 1
                            else len(held) % HELD_OUT == HELD_OUT - 1)
    if not target:
        return {}
    return {"target": np.concatenate(target).astype(np.float32),
            "seen": np.concatenate(seen).astype(np.float32), "first": np.array(first),
            "held_out": np.array(held), "lag": np.array(lags)}


def _resampled(values, times, grid):
    return np.interp(grid, times, values)


def _runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _lag(shown, camera):
    """How many frames your hand followed the robot late: the shift that
    best lines up the pinches' changes (most correlated)."""
    pinch_shown, pinch_seen = np.diff(shown[:, -3:], axis=0), np.diff(camera[:, -3:], axis=0)
    best, lag_best = -np.inf, 0
    for lag in range(round(MAX_LAG_S * RATE) + 1):
        a = pinch_shown[:len(pinch_shown) - lag]
        b = pinch_seen[lag:]
        score = (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        if score > best:
            best, lag_best = score, lag
    return lag_best
