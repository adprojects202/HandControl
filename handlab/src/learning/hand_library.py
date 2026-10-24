"""The hand-motion library: clips of hand shapes for the robot hand to learn
to follow (gpu_hand.py), each a few seconds at handmotion.RATE.

    summary = build(progress)                 # -> data/learning/hand_library.npz
    library = load()
    shapes = features(library, kinematics)    # (C, L, FEATURES): what is followed
    export_bundle(folder)                     # what the GPU trains on

Generated clips move the robot hand through key poses - still for a while at
each, minimum-jerk between them - in four kinds, and the recordings add a
fifth:

    random    RANDOM_CLIPS: poses anywhere the hand reaches (each finger's
              joints curling together, or - one pose in UNIFORM_SHARE -
              every joint anywhere in its range), fingers clear of each other
    touch     TOUCH_CLIPS: the thumb tip on one fingertip after another, or
              on two at once (TOUCH_GROUPS), sometimes opening in between,
              quickly (FAST_MOVE_S) - the dexterous part
    gesture   GESTURE_CLIPS: hand signs one after another (GESTURES)
    wave      WAVE_CLIPS: the fingers and the thumb curling in turn - one at
              a time, or counting up and back down
    recorded  every hand seen in data/tasks (hands_clean.npz, else
              hands.npz), in pieces of up to CLIP_S

A touch is made, not assumed: inverse kinematics puts the thumb tip on the
fingertips, then the servos drive the hand there from rest in the hand sim
(SETTLE_S) and the fingers stop on each other; the pose is kept only if
they touch - a contact between the thumb's last link and the finger's, or,
for two fingers at once, each fingertip within TOUCH_NEAR palm lengths of
the thumb's and one in contact. Gesture and wave poses are settled the same
way, so no key pose has fingers inside each other. Touch and gesture clips
draw their poses from BANK settled poses of each kind.

A clip is kept as its key poses and their timing, not its frames:
`features` makes the shapes - each frame's joints through the hand's
kinematics - where they are needed (on the GPU, from its hand.xml), so the
file stays small enough to send.

    hand_library.npz
        kind (C,): random, touch, gesture, wave or recorded; source (C,):
            the kind, or the recording's folder; recorded (C,)
        length (C,) frames; start (C, J) rad: the joints a generated clip
            starts in (NaN for a recorded one: it starts at rest)
        key_joints (K, J) rad - the hand's joints in its model's order -
            key_hold (K,), key_move (K,) frames: each key pose held, then
            moving to the next (0 for a clip's last); key_first (C + 1,):
            each clip's first key
        recorded_features (R, FEATURES), recorded_first (C + 1,): the
            recorded clips' shapes
        rate, made
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ..paths import ROOT
from ..sim.task import hand_spec
from .handmotion import FEATURES, RATE, Kinematics, human_features, robot_chain

LIBRARY = ROOT / "data" / "learning" / "hand_library.npz"
SIZES = {"random": 1500, "touch": 2500, "gesture": 1500, "wave": 500}   # clips of each kind
BANK = 150                    # settled poses per touch group and per gesture
POSES = (4, 9)                # key poses a random clip moves through
MOVE_S = (0.4, 1.4)
HOLD_S = (0.0, 0.5)
FAST_MOVE_S = (0.25, 0.8)     # touch and wave clips
SIGN_MOVE_S = (0.3, 1.0)      # gesture clips
SIGN_HOLD_S = (0.2, 0.8)
TOUCHES = (3, 8)              # touches in a touch clip
SIGNS = (3, 7)                # gestures in a gesture clip
OPEN_BETWEEN = 0.35           # chance a touch clip opens the hand before a touch
UNIFORM_SHARE = 0.3
SETTLE_S = 0.6
TOUCH_GROUPS = (("index",), ("middle",), ("ring",), ("index", "middle"), ("middle", "ring"))
TWO_SHARE = 0.2               # touches on two fingertips at once
TOUCH_NEAR = 0.2              # palm lengths
FINGERS = ("index", "middle", "ring")
CURL = np.array([1.55, 1.6, 1.1])          # rad: a finger's three flexing joints, curled
SPREAD = {"index": 0.3, "middle": 0.0, "ring": -0.3}   # rad on each first joint: apart
THUMB = {"out": (0.0, 0.0, 0.0, 0.0), "tucked": (1.1, 1.3, 0.8, 0.6),
         "opposed": (1.0, 0.9, 0.4, 0.3)}   # rad: proximal, adduction, middle, distal
CLAW = (0.15, 1.3, 1.0)
# A sign: each finger's curl (index, middle, ring - a share of CURL, None for
# a finger the thumb touches, or the joints themselves), how far apart the
# fingers are (a share of SPREAD), and the thumb: a THUMB pose, or the
# fingertips it touches.
GESTURES = {
    "open": ((0.0, 0.0, 0.0), 0.3, "out"),
    "spread": ((0.0, 0.0, 0.0), 1.0, "out"),
    "fist": ((1.0, 1.0, 1.0), 0.0, "tucked"),
    "point": ((0.0, 1.0, 1.0), 0.0, "tucked"),
    "peace": ((0.0, 0.0, 1.0), 1.0, "tucked"),
    "ok": ((None, 0.1, 0.1), 0.6, ("index",)),
    "thumbs up": ((1.0, 1.0, 1.0), 0.0, "out"),
    "three": ((0.0, 0.0, 0.0), 0.5, "tucked"),
    "claw": ((CLAW, CLAW, CLAW), 0.3, "opposed"),
    "rock": ((0.0, 1.0, 0.0), 0.8, "tucked"),
}
NOISE = 0.06                  # rad on every joint of a sign
CLIP_S = 6.0
MIN_CLIP_S = 1.0
SEED = 0


# ------------------------------------------------------------------ building
def build(progress=lambda done, total, text="": None, seed=SEED, sizes=None, path=LIBRARY,
          recordings=True):
    """Generate the library (`sizes`: clips of each kind, SIZES by default)
    and add every recorded hand; saved to `path`. Returns a summary."""
    sizes = {**SIZES, **(sizes or {})}
    rng = np.random.default_rng(seed)
    hand = _Hand()
    clips = []                              # (kind, source, keys or features)
    bank = _bank(hand, rng, min(BANK, max(sizes.values())), progress)
    makers = {"random": _random_clip, "touch": _touch_clip, "gesture": _gesture_clip,
              "wave": _wave_clip}
    total = sum(sizes.values())
    for kind, maker in makers.items():
        for number in range(sizes[kind]):
            clips.append((kind, kind, maker(hand, bank, rng)))
            done = len(clips)
            if done % 100 == 0:
                progress(done, total, f"{kind} clips: {number + 1} of {sizes[kind]}")
    if recordings:
        for folder, shapes in _recorded():
            for piece in _pieces(shapes):
                clips.append(("recorded", folder, piece))
    return _save(clips, hand.model.njnt, path)


def load(path=LIBRARY):
    if not path.exists():
        return {}
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def features(library, kinematics):
    """Every clip's shapes, (C, L, FEATURES) float32, each padded with its
    last: the generated ones through `kinematics` (handmotion.Kinematics,
    the hand's joints in the library's order)."""
    lengths = library["length"]
    out = np.empty((len(lengths), int(lengths.max()), FEATURES), np.float32)
    key_first, recorded_first = library["key_first"], library["recorded_first"]
    for clip in range(len(lengths)):
        if library["recorded"][clip]:
            shapes = library["recorded_features"][recorded_first[clip]:recorded_first[clip + 1]]
        else:
            keys = slice(key_first[clip], key_first[clip + 1])
            shapes = kinematics.shapes(frames(library["key_joints"][keys],
                                              library["key_hold"][keys],
                                              library["key_move"][keys]))
        out[clip, :len(shapes)] = shapes
        out[clip, len(shapes):] = shapes[-1]
    return out


def frames(joints, hold, move):
    """A clip's joints at RATE (T, J) from its key poses: the first, then
    each held `hold` more frames and moving to the next in `move`,
    minimum-jerk."""
    out = [joints[0]]
    for i, q in enumerate(joints):
        out += [q] * int(hold[i])
        if i + 1 < len(joints):
            steps = int(move[i])
            for step in range(1, steps + 1):
                s = step / steps
                out.append(q + (joints[i + 1] - q) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))
    return np.array(out)


def export_bundle(folder):
    """What the GPU trains on (gpu_hand.py), into `folder`: the library,
    the hand sim as MJCF (hand.xml) and bundle.json - its joints in the
    library's order, where the 16 points sit (body names, points in the
    bodies' frames) and the palm frame."""
    library = load()
    if not library:
        raise RuntimeError("no hand library - build it first")
    if "key_first" not in library:
        raise RuntimeError("the hand library is from before touches and signs - build it again")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(LIBRARY, folder / LIBRARY.name)
    spec, _, mount, _ = hand_spec()
    model = spec.compile()
    chain, (wrist, axes, length) = robot_chain(model)
    # Attaching the robot leaves an empty default class in the export, which
    # newer MuJoCo (3.17, the agent's) refuses; it holds nothing.
    xml = spec.to_xml().replace("  <default>\n    <default/>\n  </default>\n", "", 1)
    (folder / "hand.xml").write_text(xml)
    (folder / "bundle.json").write_text(json.dumps({
        "joints": [model.joint(j).name for j in range(model.njnt)],
        "bodies": [model.body(body).name for body, _ in chain],
        "points": [point.tolist() for _, point in chain],
        "wrist": wrist.tolist(), "axes": axes.tolist(), "palm_length": length,
        "mount_position": mount[0].tolist(), "mount_quat": mount[1].tolist(),
        "mujoco": mujoco.__version__, "library_made": str(library["made"]),
        "made": datetime.now().isoformat(timespec="seconds")}, indent=2))
    return folder


# ------------------------------------------------------------------ the hand
class _Hand:
    """The hand sim, for making key poses: its joints by finger, settling
    into a pose with the servos, touching, reaching with the thumb."""

    TIPS = {"thumb": 3, "index": 7, "middle": 11, "ring": 15}     # in robot_chain's order
    ORDER = {"thumb": ("proximal", "adduction", "middle", "distal"),
             **{f: ("adduction", "proximal", "middle", "distal") for f in FINGERS}}

    def __init__(self):
        self.model = hand_spec()[0].compile()
        self.data = mujoco.MjData(self.model)
        model = self.model
        if model.nq != model.njnt or (model.jnt_type != mujoco.mjtJoint.mjJNT_HINGE).any():
            raise RuntimeError("the hand sim is expected to hold only the hand's hinges")
        self.chain, self.palm = robot_chain(model)
        self.kinematics = Kinematics(model, self.chain, self.palm)
        self.low, self.high = model.jnt_range.T
        self.joints = {f: [int(model.jnt_qposadr[model.joint(f"{f}_{part}_joint").id])
                           for part in parts] for f, parts in self.ORDER.items()}
        self.tip_body = {f: self.chain[i][0] for f, i in self.TIPS.items()}

    def clip(self, q):
        return np.clip(q, self.low, self.high)

    def settle(self, target):
        """Where the servos, driving from rest to `target`, leave the hand."""
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = np.clip(target, *model.actuator_ctrlrange.T)
        mujoco.mj_step(model, data, nstep=round(SETTLE_S / model.opt.timestep))
        return data.qpos.copy()

    def touching(self, finger):
        """Whether the thumb's last link touches `finger`'s, as the hand is."""
        pair = {self.tip_body["thumb"], self.tip_body[finger]}
        model = self.model
        return any({int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])} == pair
                   for c in self.data.contact[:self.data.ncon])

    def pinch(self, finger):
        """The thumb tip's distance to `finger`'s, palm lengths, as the hand is."""
        return float(np.linalg.norm(self._tip("thumb") - self._tip(finger))) / self.palm[2]

    def clear(self, q):
        """Whether pose `q` has no fingers inside each other."""
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        return not any(c.efc_address >= 0 for c in self.data.contact[:self.data.ncon])

    def reach(self, q, fingers, rng):
        """`q` with the thumb and `fingers` moved so the thumb tip is on each
        of their tips (best of three starts)."""
        moved = self.joints["thumb"] + [j for f in fingers for j in self.joints[f]]
        low, high = self.low[moved], self.high[moved]

        def apart(x):
            self.data.qpos[:] = q
            self.data.qpos[moved] = x
            mujoco.mj_kinematics(self.model, self.data)
            gaps = [self._tip("thumb") - self._tip(f) for f in fingers]
            return np.concatenate(gaps + [0.002 * (x - start)])

        best = None
        for _ in range(3):
            start = np.clip(q[moved] + rng.normal(0, 0.2, len(moved)), low, high)
            found = least_squares(apart, start, bounds=(low, high))
            if best is None or found.cost < best.cost:
                best = found
        out = q.copy()
        out[moved] = best.x
        return out

    def _tip(self, finger):
        body, point = self.chain[self.TIPS[finger]]
        return self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ point


# ------------------------------------------------------------------ poses
def _bank(hand, rng, count, progress):
    """{("touch", group) or ("gesture", name) or ("curled", fingers): [settled
    poses]} for the clips to draw from."""
    bank = {}
    kinds = ([("touch", group) for group in TOUCH_GROUPS]
             + [("gesture", name) for name in GESTURES]
             + [("curled", tuple(f for f, bit in zip((*FINGERS, "thumb"), bits) if bit))
                for bits in np.ndindex(2, 2, 2, 2)])
    for number, (kind, what) in enumerate(kinds):
        progress(number, len(kinds), f"settling {kind} poses: {', '.join(what) or 'open'}")
        poses = []
        for _ in range(count * 4):
            if len(poses) == count:
                break
            pose = (_touch_pose(hand, what, rng) if kind == "touch" else
                    _sign_pose(hand, GESTURES[what], rng) if kind == "gesture" else
                    _curled_pose(hand, what, rng))
            if pose is not None:
                poses.append(pose)
        if not poses:
            raise RuntimeError(f"no {kind} pose {what} could be made")
        bank[(kind, what)] = poses
    return bank


def _fingers(hand, q, curls, spread):
    """Set each finger's joints from its curl (a share of CURL, or the
    joints) and the spread (a share of SPREAD)."""
    for finger, curl in zip(FINGERS, curls):
        joints = hand.joints[finger]
        q[joints[0]] = SPREAD[finger] * spread
        q[joints[1:]] = (np.array(curl) if isinstance(curl, tuple) else curl * CURL)
    return q


def _sign_pose(hand, sign, rng):
    curls, spread, thumb = sign
    q = _fingers(hand, np.zeros(hand.model.nq),
                 [0.5 if curl is None else curl for curl in curls], spread)
    if isinstance(thumb, str):
        q[hand.joints["thumb"]] = THUMB[thumb]
    q = hand.clip(q + rng.normal(0, NOISE, len(q)))
    if isinstance(thumb, str):
        return hand.settle(q)
    return _touched(hand, hand.reach(q, thumb, rng), thumb)


def _touch_pose(hand, group, rng):
    """The thumb tip on the fingertips of `group`, the other fingers anyhow."""
    curls = [0.45 if f in group else rng.uniform(0.0, 1.0) for f in FINGERS]
    q = _fingers(hand, np.zeros(hand.model.nq), curls, rng.uniform(0.0, 0.6))
    q[hand.joints["thumb"]] = THUMB["opposed"]
    q = hand.clip(q + rng.normal(0, 0.1, len(q)))
    return _touched(hand, hand.reach(q, group, rng), group)


def _touched(hand, target, group):
    """The settled pose if the thumb touches every finger of `group`, else None."""
    settled = hand.settle(target)
    touching = [hand.touching(f) for f in group]
    near = [hand.pinch(f) < TOUCH_NEAR for f in group]
    if all(touching) or (len(group) > 1 and any(touching) and all(near)):
        return settled
    return None


def _curled_pose(hand, curled, rng):
    """The fingers in `curled` curled into the palm (the thumb tucked), the
    others open."""
    q = _fingers(hand, np.zeros(hand.model.nq),
                 [rng.uniform(0.85, 1.0) if f in curled else rng.uniform(0.0, 0.1)
                  for f in FINGERS], rng.uniform(0.0, 0.5))
    q[hand.joints["thumb"]] = THUMB["tucked" if "thumb" in curled else "out"]
    return hand.settle(hand.clip(q + rng.normal(0, NOISE, len(q))))


def _random_pose(hand, rng):
    """A pose anywhere the hand reaches, its fingers clear of each other."""
    low, high = hand.low, hand.high
    for _ in range(100):
        if rng.random() < UNIFORM_SHARE:
            q = rng.uniform(low, high)
        else:                                  # each finger curls its joints together
            q = np.zeros(hand.model.nq)
            for finger, joints in hand.joints.items():
                curl = rng.uniform(0.0, 1.0)
                for k, i in enumerate(joints):
                    centre = low[i] + curl * (high[i] - low[i]) if k else \
                        rng.uniform(low[i], high[i]) * 0.5
                    q[i] = np.clip(centre + rng.normal(0, 0.1 * (high[i] - low[i])),
                                   low[i], high[i])
        if hand.clear(q):
            return q
    return np.zeros(hand.model.nq)


# ------------------------------------------------------------------ clips
def _keys(poses, rng, move_s, hold_s):
    """Key poses with their timing: [(joints, hold frames, move frames)]."""
    holds = [round(rng.uniform(*hold_s) * RATE) for _ in poses]
    moves = [max(2, round(rng.uniform(*move_s) * RATE)) for _ in poses[1:]] + [0]
    return list(zip(poses, holds, moves))


def _pick(bank, key, rng):
    poses = bank[key]
    return poses[int(rng.integers(len(poses)))]


def _random_clip(hand, bank, rng):
    poses = [_random_pose(hand, rng) for _ in range(int(rng.integers(*POSES)))]
    return _keys(poses, rng, MOVE_S, HOLD_S)


def _touch_clip(hand, bank, rng):
    """The thumb on one fingertip (or two) after another, from an open hand,
    sometimes opening in between."""
    singles = [g for g in TOUCH_GROUPS if len(g) == 1]
    doubles = [g for g in TOUCH_GROUPS if len(g) > 1]
    poses = [_pick(bank, ("gesture", "open"), rng)]
    for _ in range(int(rng.integers(*TOUCHES))):
        if rng.random() < OPEN_BETWEEN:
            poses.append(_pick(bank, ("gesture", "open"), rng))
        groups = doubles if rng.random() < TWO_SHARE else singles
        poses.append(_pick(bank, ("touch", groups[int(rng.integers(len(groups)))]), rng))
    poses.append(_pick(bank, ("gesture", "open"), rng))
    return _keys(poses, rng, FAST_MOVE_S, HOLD_S)


def _gesture_clip(hand, bank, rng):
    names = list(GESTURES)
    poses, last = [], None
    for _ in range(int(rng.integers(*SIGNS))):
        name = names[int(rng.integers(len(names)))]
        while name == last:
            name = names[int(rng.integers(len(names)))]
        poses.append(_pick(bank, ("gesture", name), rng))
        last = name
    return _keys(poses, rng, SIGN_MOVE_S, SIGN_HOLD_S)


def _wave_clip(hand, bank, rng):
    """The fingers and thumb curling in turn: one at a time back to open,
    or counting - curling up one by one, then opening one by one."""
    order = list(rng.permutation([*FINGERS, "thumb"]))
    curled = [()]
    if rng.random() < 0.5:
        for finger in order:
            curled += [(finger,), ()]
    else:
        for count in range(1, len(order) + 1):
            curled.append(tuple(order[:count]))
        for count in range(len(order) - 1, -1, -1):
            curled.append(tuple(order[:count]))
    canonical = lambda fingers: tuple(f for f in (*FINGERS, "thumb") if f in fingers)  # noqa: E731
    poses = [_pick(bank, ("curled", canonical(fingers)), rng) for fingers in curled]
    return _keys(poses, rng, FAST_MOVE_S, (0.0, 0.3))


# ------------------------------------------------------------------ recordings
def _recorded():
    """[(folder, shapes (T, FEATURES) at RATE)] for every run of frames a
    hand is seen in, in every recording."""
    from ..recording import Episode, episodes            # here: training needs no recordings
    from ..vision.tracker import SIDES

    out = []
    for folder in episodes():
        episode = Episode(folder)
        hands = episode.clean_hands or episode.hands
        if not hands or episode.frames < 2:
            continue
        for slot, side in enumerate(SIDES):
            present = hands["present"][:, slot] & np.isfinite(hands["world"][:, slot]).all(axis=(1, 2))
            edges = np.diff(np.r_[0, present.astype(np.int8), 0])
            for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                times = episode.times[begin:end]
                if times[-1] - times[0] < MIN_CLIP_S:
                    continue
                world = hands["world"][begin:end, slot].astype(float)
                at = np.arange(times[0], times[-1], 1.0 / RATE)
                resampled = np.stack([np.interp(at, times, world[:, p, k])
                                      for p in range(21) for k in range(3)], axis=-1)
                out.append((str(folder), human_features(resampled.reshape(-1, 21, 3), side)))
    return out


def _pieces(shapes):
    step = round(CLIP_S * RATE)
    return [shapes[i:i + step] for i in range(0, len(shapes), step)
            if len(shapes[i:i + step]) >= MIN_CLIP_S * RATE]


def _save(clips, joints, path):
    kinds, sources, lengths, starts = [], [], [], []
    key_joints, key_hold, key_move, key_first = [], [], [], [0]
    recorded_features, recorded_first = [], [0]
    for kind, source, content in clips:
        kinds.append(kind)
        sources.append(source)
        if kind == "recorded":
            recorded_features.append(content)
            lengths.append(len(content))
            starts.append(np.full(joints, np.nan))
        else:
            poses, holds, moves = zip(*content)
            key_joints += poses
            key_hold += holds
            key_move += moves
            lengths.append(1 + sum(holds) + sum(moves))
            starts.append(poses[0])
        key_first.append(len(key_joints))
        recorded_first.append(recorded_first[-1] + (len(content) if kind == "recorded" else 0))
    arrays = {
        "kind": np.array(kinds), "source": np.array(sources),
        "recorded": np.array(kinds) == "recorded", "length": np.array(lengths),
        "start": np.array(starts, np.float32),
        "key_joints": np.array(key_joints, np.float32).reshape(-1, joints),
        "key_hold": np.array(key_hold, np.int16), "key_move": np.array(key_move, np.int16),
        "key_first": np.array(key_first), "recorded_first": np.array(recorded_first),
        "recorded_features": (np.concatenate(recorded_features).astype(np.float32)
                              if recorded_features else np.zeros((0, FEATURES), np.float32)),
        "rate": np.array(RATE), "made": np.array(datetime.now().isoformat(timespec="seconds"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    counts = {kind: int((arrays["kind"] == kind).sum()) for kind in
              ("random", "touch", "gesture", "wave", "recorded")}
    return {**counts, "recordings": len({s for k, s in zip(kinds, sources) if k == "recorded"}),
            "minutes": round(float(arrays["length"].sum()) / RATE / 60, 1),
            "megabytes": round(path.stat().st_size / 1e6, 1)}
