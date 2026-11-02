"""GRAB's people handling objects, put onto the robot hand: real motion -
people's own timing, grasps, holds and rests - for the motion space
(motion_space.py); the objects themselves come later.

    ../.venv-wilor/bin/python wilor/grab.py   # GRAB's hands, their 21 points (~1 min)
    python scripts/build_grab.py              # label(): onto the robot (~20 min)
    labels = load()
    takes = clips(labels, ("s1", "s2"))       # [(T, J) rad] a hand of a take each

Each hand of each take is retargeted frame by frame (human_retarget.
HumanRetarget), each frame solved from the last and held towards it - one
motion, not poses one by one. Where the thumb and a fingertip both touch the
object (GRAB's contact labels on their last segments) the hand is holding:
the object is between them, so no touch is made. Right hands go as
FreiHAND's do (mirrored onto the robot's left hand), left hands as they are.

    data/learning/intent/grab_robot.npz
        joints (N, 2, J) rad: right then left hand, the robot's settled pose;
        touching (N, 2, 3) bool: the robot's thumb on the index, middle,
        ring; holding (N, 2, 3) bool; take (N,) as grab_hands.npz; subject,
        object, intent, name (takes,); rate; made
"""

from __future__ import annotations

from datetime import datetime
from multiprocessing import get_context

import numpy as np

from .handmotion import RATE
from .human_retarget import FINGERS, HumanRetarget
from .intent_pairs import FOLDER, WORKERS

HANDS = FOLDER / "grab_hands.npz"
LABELS = FOLDER / "grab_robot.npz"
SIDES = ("Right", "Left")
THUMB_TIP, FINGER_TIPS = 3, (6, 9, 12)       # in wilor/grab.py's PARTS: the last segments
JUMP = 20                                    # degrees in a frame: a jump, reported


def label(progress=lambda done, total, text="": None):
    """Retarget every take's hands; writes grab_robot.npz, returns a summary."""
    if not HANDS.exists():
        raise RuntimeError("no GRAB hands - run ../.venv-wilor/bin/python wilor/grab.py")
    hands = dict(np.load(HANDS))         # read once: an npz reads an array at every lookup
    if int(hands["rate"]) != RATE:
        raise RuntimeError(f"GRAB hands at {int(hands['rate'])} Hz, the motion space's {RATE}")
    take = hands["take"]
    edges = np.r_[0, np.flatnonzero(np.diff(take)) + 1, len(take)]
    jobs = [(hands["joints"][a:b], hands["contact"][a:b]) for a, b in zip(edges[:-1], edges[1:])]
    out = {"joints": [], "touching": [], "holding": []}
    with get_context("spawn").Pool(WORKERS, initializer=_start) as pool:
        for k, done in enumerate(pool.imap(_retarget, jobs)):
            for key, value in zip(out, done):
                out[key].append(value)
            progress(k + 1, len(jobs), str(hands["name"][take[edges[k]]]))
    arrays = {key: np.concatenate(value) for key, value in out.items()}
    np.savez(LABELS, **arrays, take=take, **{k: hands[k] for k in ("subject", "object", "intent", "name")},
             rate=hands["rate"], made=np.array(datetime.now().isoformat(timespec="seconds")))
    return report(arrays["joints"], arrays["touching"], arrays["holding"], take)


def load():
    return dict(np.load(LABELS)) if LABELS.exists() else {}


def clips(labels, subjects):
    """Each hand of each take of `subjects`, its joints (T, J) float32."""
    take = labels["take"]
    edges = np.r_[0, np.flatnonzero(np.diff(take)) + 1, len(take)]
    return [labels["joints"][a:b, hand].astype(np.float32)
            for a, b in zip(edges[:-1], edges[1:]) if labels["subject"][take[a]] in subjects
            for hand in range(len(SIDES))]


def report(joints, touching, holding, take):
    """How often each hand touches and holds, and how often it jumps."""
    lines = [f"{len(np.unique(take))} takes, {len(take)} frames at {RATE} Hz"]
    same = take[1:] == take[:-1]
    for hand, side in enumerate(SIDES):
        step = np.degrees(np.abs(np.diff(joints[:, hand], axis=0))).max(1)[same]
        lines.append(f"{side}: touching " + ", ".join(
            f"{f} {touching[:, hand, k].mean():.1%}" for k, f in enumerate(FINGERS))
            + "; holding " + ", ".join(f"{f} {holding[:, hand, k].mean():.1%}" for k, f in enumerate(FINGERS))
            + f"; steps over {JUMP} degrees {np.mean(step > JUMP):.1%}")
    return "\n".join(lines)


_worker = None


def _start():
    global _worker
    _worker = HumanRetarget()


def _retarget(job):
    """One take's hands (T, 2, 21, 3), contact (T, 2, 16) -> joints, touching, holding."""
    world, contact = job
    holding = contact[:, :, THUMB_TIP:THUMB_TIP + 1] & contact[:, :, FINGER_TIPS]
    joints = np.zeros((len(world), len(SIDES), _worker.hand.model.nq), np.float32)
    touching = np.zeros((len(world), len(SIDES), len(FINGERS)), bool)
    for hand, side in enumerate(SIDES):
        last = None
        for frame in range(len(world)):
            last, touching[frame, hand] = _worker(world[frame, hand], side, holding[frame, hand], last)
            joints[frame, hand] = last
    return joints, touching, holding
