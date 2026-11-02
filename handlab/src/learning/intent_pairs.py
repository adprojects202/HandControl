"""Intent pairs: what a camera's tracker sees of a hand -> the robot pose that
hand means - what a model learns to read intent from (step 1 of the
plan). From FreiHAND, whose hands are measured in 3D:

    the robot side    its true hands retargeted (human_retarget.
                      HumanRetarget): the robot pose each means, its touches
                      made contacts - label(), here
    the camera side   WiLoR on its photos (wilor/freihand.py, in its own
                      environment): the hand a tracker makes of each

    python scripts/build_intent_pairs.py      # label(); about 5 min
    pairs = load("training")                  # or "evaluation"

Two sets, as wilor/freihand.py runs them: training (32,560 poses, people in
front of a green screen given other backgrounds) and evaluation (3,960, real
backgrounds, a different setting - held out, for testing). FreiHAND's hands
are all right hands.

    data/learning/intent/freihand_labels_<set>.npz
        person (N, 21, 3) m: the true hand, camera axes; joints (N, J) rad:
        the robot's settled pose; touching (N, 3) bool: the thumb on the
        index, middle, ring; made
    load(set) -> dict: those, and from freihand_wilor_<set>.npz seen
        (N, 21, 3) m: WiLoR's hand (NaN where none was found), found (N,)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from multiprocessing import get_context

import numpy as np

from ..paths import ROOT
from .human_retarget import TOUCH_HUMAN, HumanRetarget
from .handmotion import in_palm

FOLDER = ROOT / "data" / "learning" / "intent"
SOURCES = {"training": ROOT / "data" / "external" / "freihand_full" / "training_xyz.json",
           "evaluation": ROOT / "data" / "external" / "freihand" / "evaluation_xyz.json"}
SIDE = "Right"
CHUNK = 250
WORKERS = max(1, (os.cpu_count() or 2) // 2)
FINGERS = ("index", "middle", "ring")


def label(progress=lambda done, total, text="": None):
    """Retarget every true hand of both sets; writes freihand_labels_<set>.npz."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, source in SOURCES.items():
        person = np.asarray(json.loads(source.read_text()), np.float32)
        chunks = [person[i:i + CHUNK] for i in range(0, len(person), CHUNK)]
        joints, touching = [], []
        with get_context("spawn").Pool(WORKERS, initializer=_start) as pool:
            for k, (q, touch) in enumerate(pool.imap(_retarget, chunks)):
                joints.append(q)
                touching.append(touch)
                progress(min((k + 1) * CHUNK, len(person)), len(person), name)
        joints, touching = np.concatenate(joints), np.concatenate(touching)
        np.savez(FOLDER / f"freihand_labels_{name}.npz", person=person, joints=joints,
                 touching=touching, made=np.array(datetime.now().isoformat(timespec="seconds")))
        summary[name] = report(person, touching)
    return summary


def load(name):
    pairs = dict(np.load(FOLDER / f"freihand_labels_{name}.npz"))
    seen = np.load(FOLDER / f"freihand_wilor_{name}.npz")
    if not seen["tried"].all():
        raise RuntimeError(f"wilor/freihand.py has not finished the {name} photos")
    pairs["seen"], pairs["found"] = seen["joints"], seen["found"]
    return pairs


def report(person, touching, seen=None):
    """How often each finger's touch is labelled; with the tracker's hands
    `seen`, how its own thumb-tip gaps (under TOUCH_HUMAN) read those touches."""
    true = _gaps(person) < TOUCH_HUMAN
    lines = [f"{len(person)} hands, a touch in {touching.any(1).mean():.0%}: " + ", ".join(
        f"{f} {touching[:, k].mean():.1%} ({_share(touching[true[:, k], k])} of the person's)"
        for k, f in enumerate(FINGERS))]
    if seen is not None:
        found = ~np.isnan(seen).any((1, 2))
        read = _gaps(seen[found]) < TOUCH_HUMAN
        label = touching[found]
        lines.append(f"tracker found {found.mean():.1%}; its gaps read the touches: " + ", ".join(
            f"{f} {_share(read[label[:, k], k])} caught, {_share(read[~label[:, k], k])} false"
            for k, f in enumerate(FINGERS)))
    return "\n".join(lines)


def _gaps(world):
    """The thumb tip's distance to the index, middle and ring tips (N, 3), palm lengths."""
    local = in_palm(np.asarray(world, float), SIDE)
    return np.linalg.norm(local[:, [8, 12, 16]] - local[:, 4:5], axis=-1)


def _share(flags):
    return f"{flags.mean():.0%}" if len(flags) else "-"


_worker = None


def _start():
    global _worker
    _worker = HumanRetarget()


def _retarget(person):
    out = [_worker(hand, SIDE) for hand in person.astype(float)]
    return np.array([q for q, _ in out]), np.array([t for _, t in out])
