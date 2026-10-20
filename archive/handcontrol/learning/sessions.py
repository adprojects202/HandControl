"""Recorded Leap sessions: pooling them, and cutting them into runs.

Nothing here knows about robots. It takes what `learn.py survey` wrote and gives
back arrays, hands and keypoints.

WHAT A RUN IS, AND WHY EVERYTHING IS CUT INTO THEM. Every estimate downstream
is temporal - a smoothing window, a warm start, a frame-to-frame step - and all
of them assume the previous row happened a hundredth of a second ago. Three
things break that assumption while leaving the rows adjacent in the array: a
tracking loss, the gap between two blocks while the operator reads a prompt,
and the join between two sessions recorded on different days.

Measured, this is not pedantry. Solving across block seams warm-started a fist
from a flat hand; measuring jerk across them counted the seam as a 108 degree
single-frame joint step and reported the offline teacher as WORSE than the live
mapping it improves on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..leap import Hand
from ..leap import DIGITS
from ..retargeting.keyvector import (MM_TO_M, apply_bone_offsets, digit_lengths,
                         mano_keypoints, rest_offsets)

from ..paths import DATA as DATA_ROOT

DATA = DATA_ROOT / "leap"


def sessions_under(root=DATA):
    """Every recorded session, oldest first. Directory names are timestamps."""
    return sorted(p for p in Path(root).glob("*") if (p / "session.npz").exists())


def load(paths):
    """Pool any number of sessions into one set of arrays, plus their metadata.

    Pooled rather than summarised-then-averaged because most of what is
    computed from these is not a mean: a snap rate, a robust sigma and a
    spectrum each come from the samples, and averaging twelve per-session
    summaries would weight a thirty-second block like a five-minute one.

    A `session` column is added, and it is not bookkeeping - `tracked_runs`
    breaks on it.
    """
    paths = [Path(p) for p in paths]
    pooled, metas = None, []
    for index, path in enumerate(paths):
        with np.load(path / "session.npz") as handle:
            data = {name: handle[name] for name in handle.files}
        meta = json.loads((path / "meta.json").read_text())
        meta["path"] = str(path)
        metas.append(meta)
        data["session"] = np.full(len(data["t"]), index, dtype=np.int16)
        pooled = data if pooled is None else _pool(pooled, data)
    if pooled is None:
        raise SystemExit("no sessions found")
    return pooled, metas


def _pool(older, newer):
    """Concatenate two sessions' columns, over the UNION of their names.

    Keyed on the first session's names, a column the recorder gained later
    was silently dropped from the pool whenever the older session came first
    - and `sessions_under` sorts oldest first - so `confidence` would have
    read NaN everywhere with nothing said; listed the other way round it
    raised `KeyError` instead. Either way the answer depended on the order.
    A column one side lacks is NaN for that side's rows, which is what a
    field nobody recorded actually means.
    """
    out = {}
    for name in set(older) | set(newer):
        a, b = older.get(name), newer.get(name)
        if a is None:
            a = np.full((len(older["t"]),) + b.shape[1:], np.nan, dtype=b.dtype)
        if b is None:
            b = np.full((len(newer["t"]),) + a.shape[1:], np.nan, dtype=a.dtype)
        out[name] = np.concatenate([a, b])
    return out


def tracked_runs(pick, tracked, session, minimum=42):
    """Contiguous tracked stretches inside one block of one session.

    Runs shorter than `minimum` are dropped rather than analysed on a smoothing
    window that would be mostly edge extrapolation.
    """
    index = np.flatnonzero(pick & tracked)
    if len(index) == 0:
        return []
    cut = (np.diff(index) != 1) | (np.diff(session[index]) != 0)
    return [run for run in np.split(index, np.flatnonzero(cut) + 1)
            if len(run) >= minimum]


def runs_for(data, minimum=42, blocks=None):
    """Every run in a pooled session, block by block.

    `blocks` selects by key, and needs the block names - pass them from the
    metadata rather than guessing, because the index is into `BLOCKS` as it was
    when the session was recorded.
    """
    session = data.get("session")
    if session is None:
        session = np.zeros(len(data["t"]), dtype=np.int16)
    out = []
    for block in np.unique(data["block"]):
        if blocks is not None and int(block) not in blocks:
            continue
        out += tracked_runs(data["block"] == block, data["tracked"], session,
                            minimum)
    return out


def hands_from(data, index):
    """One `Hand` back out of the recorded arrays.

    Rebuilt rather than carried, because everything downstream - the hand
    frame, the rotation into the robot's axes, the MANO layout - is written
    against `Hand`, and re-deriving those against raw arrays would be a second
    implementation of the frame convention free to disagree with the first.
    """
    return Hand(
        id=int(data["hand_id"][index]),
        type=str(data.get("hand_type", "")),
        palm=data["palm"][index].astype(float),
        normal=data["normal"][index].astype(float),
        direction=data["direction"][index].astype(float),
        width=float(data["width"][index]),
        pinch_strength=float(data["pinch_strength"][index]),
        grab_strength=float(data["grab_strength"][index]),
        pinch_distance=float(data["pinch_distance"][index]),
        bones={digit: data["bones"][index][slot].astype(float)
               for slot, digit in enumerate(DIGITS)},
        confidence=(float(data["confidence"][index])
                    if "confidence" in data else float("nan")),
    )


def fit_proportions(retarget, hands, fingers):
    """How much longer each robot digit is than this operator's, per digit.

    Fitted over many open frames rather than the first hand the solver happens
    to see, which is what the live path has to do. A single frame carries that
    frame's noise into a constant every keypoint is then multiplied by.
    """
    lengths = {}
    for hand in hands:
        for digit, span in digit_lengths(hand).items():
            lengths.setdefault(digit, []).append(span)
    scale = {}
    for finger in fingers:
        reach = float(np.linalg.norm(retarget.rest[finger]))
        median = float(np.median(lengths.get(finger, [0.0])))
        if median > 1e-6:
            scale[finger] = reach / median
    return scale


def keypoints_for(retarget, data, index, digit_scale, bone_offsets=None):
    """One frame of recorded bones as 21 MANO keypoints in the robot's axes,
    per-digit scaled and rest-aligned: the view that carries DIRECTIONS."""
    hand = hands_from(data, index)
    return mano_keypoints(hand, retarget.rotation(hand),
                          digit_scale=digit_scale, bone_offsets=bone_offsets)


def raw_keypoints_for(retarget, data, index):
    """The same frame rotated into the robot's axes and nothing else: the view
    that carries DISTANCES.

    A pinch does not survive the aligned view. Scaling the thumb 1.98x and the
    index 1.27x radially from the wrist sends two coincident tips to different
    places, and the per-bone rest rotation does the same by another route - on
    `h4nd_left` a real pinch measures 22 mm raw and 189 mm aligned, which
    `selftest.test_a_pinch_survives_into_the_raw_view` re-measures every run.
    A rotation preserves every distance, so anything that IS a distance is
    measured here.
    """
    hand = hands_from(data, index)
    return mano_keypoints(hand, retarget.rotation(hand))


def robot_rest_bones(model, joint_names, fingers):
    """Each robot finger's bone directions at q = 0, wrist-relative unit vectors.

    What a flat operator hand is going to be aligned ONTO. Measured off the
    model rather than written down, so a re-export or a change of chirality
    carries through.
    """
    zero = np.zeros(len(joint_names))
    out = {}
    for finger in fingers:
        links = [f"{finger}_proximal_link", f"{finger}_middle_link",
                 f"{finger}_distal_link", f"{finger}_tip_link"]
        points = [model.link_pose(link, zero, joint_names)[0] for link in links]
        out[finger] = [(b - a) / np.linalg.norm(b - a)
                       for a, b in zip(points[:-1], points[1:])]
    return out


def fit_bone_offsets(retarget, model, hands, fingers):
    """Per bone, the rotation making this operator's flat hand the robot's zero.

    WHAT THIS FIXES, measured. The two rest poses are not the same shape - an
    operator holding a hand flat has bones sitting 9 to 23 degrees off where the
    robot's are at zero, and most of that sits in the SPREAD plane. Uncorrected,
    the solver reproduces it faithfully every frame: on this operator's
    flat-hold block it commanded -17, -18 and -13 degrees of adduction on the
    three fingers, all the same sign, which swings the whole finger set off to
    one side. A flat hand did not give a straight robot.

    Corrected, the same block commands -3.1, -1.5 and -1.3 degrees, and the
    thumb's proximal joint drops from 92 degrees to 3.

    AVERAGED OVER MANY FRAMES, and through the nearest rotation rather than the
    mean of matrices - the elementwise mean of rotations is not a rotation, and
    feeding one to the chain would scale the bones as well as turning them.
    """
    robot = robot_rest_bones(model, retarget.joint_names, fingers)
    totals = {f: [np.zeros((3, 3)) for _ in range(3)] for f in fingers}
    count = 0
    for hand in hands:
        found = rest_offsets(hand, retarget.rotation(hand), robot)
        for finger in fingers:
            for k in range(3):
                totals[finger][k] += found[finger][k]
        count += 1
    if not count:
        return None

    offsets = {}
    for finger in fingers:
        turns = []
        for k in range(3):
            left, _, right = np.linalg.svd(totals[finger][k] / count)
            turn = left @ right
            if np.linalg.det(turn) < 0:
                left[:, -1] *= -1
                turn = left @ right
            turns.append(turn)
        offsets[finger] = turns
    return offsets


def robot_basis(retarget):
    """The robot's axes as rows: (forward, normal, thumbward).

    ONE DEFINITION, because writing it out at each use is what went wrong. It
    was spelled out in three separate places and two of them were wrong in
    different ways - once a whole synthetic hand was left unrotated and solved
    ninety degrees out for 150,000 frames, and once the robot's own rest bones
    were dropped into the operator's basis without conversion, collapsing the
    sampled hand's reach from 125 mm to 82.

    Neither failure is visible in a shape check: a rigid rotation preserves
    every distance a distribution test measures. They are only visible on
    screen, or here, by there being one place to get right.

    ITS DETERMINANT IS -1, AND THAT IS CORRECT. (forward, normal, thumbward)
    is a left-handed triple for a LEFT hand - the thumb sits on the opposite
    side from where normal x forward puts it - so this basis is a reflection,
    not a rotation. Do not "fix" it: `robot_rest_bones` and the operator's
    keypoints are both carried through it, so the flip cancels, which is why
    `test_rest_offsets_align_every_bone` reads 0.00 degrees rather than the
    180 a one-sided reflection would give.

    `selftest.test_rotation_is_not_a_reflection` is about a different matrix -
    `retarget.rotation(hand)`, the per-frame hand pose, which must be proper
    because it is a pose and not a basis. Anything comparing points ACROSS
    this boundary (robot FK on one side, keypoints on the other) has to apply
    the basis first or it will read a chirality difference that is not there.
    """
    return np.array([retarget.forward, retarget.normal, retarget.thumbward])


def to_robot_axes(points, retarget):
    """Points expressed on the robot's axes. Accepts (..., 3)."""
    return np.asarray(points, dtype=float) @ robot_basis(retarget)


def from_robot_axes(points, retarget):
    """The inverse of `to_robot_axes`."""
    return np.asarray(points, dtype=float) @ robot_basis(retarget).T


def raw_synthetic(points, retarget):
    """Raw sampler millimetres -> the teacher's RAW view. Metres, robot axes,
    nothing else. The sibling of `finish_keypoints` for distances."""
    return to_robot_axes(np.asarray(points, dtype=float) * MM_TO_M, retarget)


def finish_keypoints(points, scale, retarget, offsets=None):
    """Raw sampler millimetres -> exactly what the teacher receives.

    Metres, per-digit scaled, on the robot's axes, rest-aligned. All four
    steps, in one place, because a recorded hand gets all four inside
    `mano_keypoints` and a generated one has to match it step for step - and
    every time these were spelled out separately, one was skipped or done
    twice.

    THE REST ALIGNMENT IS PER FRAME, exactly as it is for a recording.
    Shifting the sampler's rest pose instead sounds equivalent and measures
    worse on every digit - synthetic mean fingertips 44/19/23/27 mm from the
    recorded ones against 30/16/18/20 - because it leaves the thumb's
    opposition axis, which is fitted relative to the OPERATOR's rest, turning
    about a rest that has been replaced by the robot's.
    """
    out = np.asarray(points, dtype=float) * MM_TO_M
    for slot, digit in enumerate(DIGITS):
        out[..., 4 * slot + 1:4 * slot + 5, :] *= scale.get(digit, 1.0)
    out = to_robot_axes(out, retarget)
    if offsets:
        flat = out.reshape(-1, 21, 3)
        for frame in flat:
            apply_bone_offsets(frame, offsets)
        out = flat.reshape(out.shape)
    return out
