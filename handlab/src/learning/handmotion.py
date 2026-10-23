"""Hand motion: what the robot hand learns before any object - to take the
shape of a person's hand, as it moves.

    library = build(progress)                   # -> data/learning/hand_library.npz
    export_bundle(folder)                       # what the GPU trains on (gpu_hand.py)
    features = human_features(world, side)      # MediaPipe hands -> shapes
    chain = robot_chain()                       # where the robot's matching points are
    features = shape(points)                    # either hand's points -> shape
    policy_input = observation(...)             # what the policy sees (gpu_hand, hand_follow)

A hand's shape, the same for a person's hand and the robot's though they
differ in size (the robot's palm is ~1.7 times a person's) and fingers (it
has three and a thumb): FEATURES numbers, from 16 points - for the thumb and
the index, middle and ring fingers (a person's little finger has no match),
the joint at the finger's root, the next two joints, and the tip - in the
palm's frame (x from the wrist to the middle knuckle, z out of the palm, y =
z x x: towards the thumb on a left hand), measured in palm lengths (wrist to
middle knuckle):

    directions  each finger's three segments as unit vectors (12 x 3): how
                each joint bends, whatever the finger's length
    tips        the four fingertips' positions (4 x 3)
    pinch       the thumb tip's distance to each fingertip (3): pinches and
                the OK sign

A person's points are MediaPipe's world landmarks (thumb 1-4, index 5-8,
middle 9-12, ring 13-16); a right hand is mirrored (y -> -y) onto the robot's
left. The robot's are its joints - for a finger the knuckle (its flexing
joint), the next two joints, and the tip; for the thumb its second, third and
fourth joints and the tip - each a point fixed on a link (`robot_chain`), so
they follow the links wherever the joints put them.

The library: clips of shapes at RATE Hz to follow, each to learn from -
RANDOM_CLIPS of the robot moving between poses it can reach (each finger's
joints curling together, as a hand's do, or - one pose in UNIFORM_SHARE -
every joint anywhere in its range; none with fingers inside each other),
with a minimum-jerk motion between them; and every recorded hand in
data/tasks (hands_clean.npz, else hands.npz), in pieces of up to CLIP_S.
Recordings of gestures - fists, each finger, peace, pointing - add to it
when it is built again.

    hand_library.npz
        features (C, L, FEATURES): the shapes, each clip padded with its last
        length (C,): frames; start (C, J) rad: the robot's joints a random
            clip starts from (NaN for a recorded one: from rest)
        source (C,): "random" or the recording's folder
        rate, made
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime

import mujoco
import numpy as np
import torch

from ..paths import ROOT
from ..sim.task import hand_spec
from . import env as tracker

LIBRARY = ROOT / "data" / "learning" / "hand_library.npz"
RATE = 20
FEATURES = 51
SEGMENTS = 12
FUTURE_STEPS = (1, 3, 6)         # the shapes ahead the policy sees, in steps
ACTION_SCALE = tracker.ACTION_SCALE   # rad a servo target moves per step at most
CHAINS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
          "ring": [13, 14, 15, 16]}
WRIST, MIDDLE_KNUCKLE, INDEX_KNUCKLE, PINKY_KNUCKLE = 0, 9, 5, 17
RANDOM_CLIPS = 1500
POSES = (4, 9)                # poses a random clip moves through
MOVE_S = (0.4, 1.4)
HOLD_S = (0.0, 0.5)
UNIFORM_SHARE = 0.3
CLIP_S = 6.0
MIN_CLIP_S = 1.0
SEED = 0


# ------------------------------------------------------------------ the shape
def shape(points):
    """Shapes (..., FEATURES) from points (..., 16, 3) in palm frames (palm
    lengths), the chains in CHAINS' order: thumb, index, middle, ring, each
    root, two joints, tip. Torch or NumPy in, the same out."""
    lib = torch if isinstance(points, torch.Tensor) else np
    chains = points.reshape(*points.shape[:-2], 4, 4, 3)
    segments = chains[..., 1:, :] - chains[..., :-1, :]
    norm = lib.linalg.norm(segments, axis=-1, keepdims=True) if lib is np else \
        torch.linalg.norm(segments, dim=-1, keepdim=True)
    directions = segments / lib.clip(norm, 1e-9, None) if lib is np else \
        segments / torch.clamp(norm, min=1e-9)
    tips = chains[..., :, 3, :]
    gap = tips[..., 1:, :] - tips[..., :1, :]
    pinch = lib.linalg.norm(gap, axis=-1) if lib is np else torch.linalg.norm(gap, dim=-1)
    flat = (*points.shape[:-2], -1)
    parts = [directions.reshape(flat), tips.reshape(flat), pinch]
    return lib.concatenate(parts, axis=-1) if lib is np else torch.cat(parts, dim=-1)


def observation(joints, previous, targets, now, goals):
    """The policy's input (n, 3 J + FEATURES (1 + len(FUTURE_STEPS))): what
    the robot reads, its shape now, and the shapes ahead (n, steps, FEATURES)
    less the shape now - the same wherever the hand runs."""
    return torch.cat([joints, previous, targets, now,
                      (goals - now[:, None]).reshape(len(now), -1)], dim=1)


def errors(robot, goal):
    """(segment angle rad, tip error, pinch error - palm lengths), each (n,),
    between shapes (n, FEATURES)."""
    directions = SEGMENTS * 3
    dot = (robot[:, :directions].reshape(-1, SEGMENTS, 3)
           * goal[:, :directions].reshape(-1, SEGMENTS, 3)).sum(dim=-1)
    angle = torch.arccos(torch.clamp(dot, -1.0, 1.0)).mean(dim=1)
    tips = directions + 12
    tip = torch.linalg.norm((robot[:, directions:tips] - goal[:, directions:tips]).reshape(-1, 4, 3),
                            dim=-1).mean(dim=1)
    pinch = (robot[:, tips:FEATURES] - goal[:, tips:FEATURES]).abs().mean(dim=1)
    return angle, tip, pinch


def human_points(world, side):
    """MediaPipe world landmarks (..., 21, 3) of a `side` hand -> the 16
    points (..., 16, 3) in its palm frame, palm lengths; a right hand
    mirrored onto a left."""
    return in_palm(world, side)[..., np.concatenate([CHAINS[name] for name in CHAINS]), :]


def in_palm(world, side):
    """MediaPipe world landmarks (..., 21, 3) of a `side` hand -> all 21 in
    its palm frame, palm lengths; a right hand mirrored onto a left."""
    x = world[..., MIDDLE_KNUCKLE, :] - world[..., WRIST, :]
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / length
    z = np.cross(world[..., INDEX_KNUCKLE, :] - world[..., WRIST, :],
                 world[..., PINKY_KNUCKLE, :] - world[..., WRIST, :])
    z = z * (-1.0 if side == "Left" else 1.0)                 # out of the palm
    z = z - (z * x).sum(axis=-1, keepdims=True) * x
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    axes = np.stack([x, np.cross(z, x), z], axis=-1)          # columns
    local = np.einsum("...ji,...pj->...pi", axes,
                      world - world[..., WRIST:WRIST + 1, :]) / length[..., None]
    if side != "Left":
        local[..., 1] *= -1.0
    return local


def human_features(world, side):
    return shape(human_points(world, side))


# ------------------------------------------------------------------ the robot's points
def robot_chain(model=None):
    """Where the robot's 16 points are, in CHAINS' order: [(body id, point
    (3,) m in the body's frame)], from a compiled model of the hand (the
    hand-only task sim by default) - and its palm: (wrist (3,) m, axes (3,
    3) columns, palm length m) in the model's world frame at rest."""
    if model is None:
        model = hand_spec()[0].compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [model.joint(j).name for j in range(model.njnt)
             if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    groups = {}
    for name in names:
        groups.setdefault(name.split("_")[0], []).append(name)
    chain = []
    for finger in CHAINS:
        joints = groups[finger][1:]                           # after the first: the knuckle on
        for joint in joints:
            j = model.joint(joint).id
            chain.append((int(model.jnt_bodyid[j]), model.jnt_pos[j].copy()))
        chain.append(_tip(model, data, joints))
    knuckles = [data.xanchor[model.joint(groups[f][1]).id] for f in ("index", "middle", "ring")]
    palm = np.vstack([_vertices(model, data, g) for g in range(model.ngeom)  # the fixed links
                      if model.geom_bodyid[g] and model.body_weldid[model.geom_bodyid[g]] == 0])
    middle = knuckles[1]
    finger_dir = data.xanchor[model.joint(groups["middle"][2]).id] - middle
    x = finger_dir / np.linalg.norm(finger_dir)
    across = knuckles[0] - knuckles[2]                         # ring to index: towards the thumb
    z = np.cross(x, across)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    # The wrist: the palm's far end from the knuckles, in line with the middle one.
    reach = (palm - middle) @ x
    wrist = middle + x * reach.min()
    return chain, (wrist, np.column_stack([x, y, z]), float(np.linalg.norm(middle - wrist)))


def mount_palm():
    """The robot's palm frame (wrist (3,) m, axes (3, 3) columns, palm
    length m) in its mount's frame - where the World and sim files put
    things."""
    spec, _, (position, quat), _ = hand_spec()
    _, (wrist, axes, length) = robot_chain(spec.compile())
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, float))
    rotation = rotation.reshape(3, 3)
    return (wrist - position) @ rotation, rotation.T @ axes, length


def on_robot(points, palm):
    """Points (..., 3) in a palm frame, palm lengths -> where they are on
    the robot's palm `palm` (wrist, axes, length): m, in its frame."""
    wrist, axes, length = palm
    return wrist + (points * length) @ axes.T


def robot_points(model, data, chain, palm):
    """The robot's 16 points in its palm frame, palm lengths (16, 3)."""
    world = np.array([data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
                      for body, local in chain])
    wrist, axes, length = palm
    return (world - wrist) @ axes / length


def _tip(model, data, joints):
    """The finger's tip: on its last link, as far along the finger's last
    segment as the link reaches."""
    last = model.joint(joints[-1]).id
    before = model.joint(joints[-2]).id
    body = int(model.jnt_bodyid[last])
    along = data.xanchor[last] - data.xanchor[before]
    along /= np.linalg.norm(along)
    points = np.vstack([_vertices(model, data, g) for g in np.flatnonzero(model.geom_bodyid == body)])
    tip = data.xanchor[last] + along * ((points - data.xanchor[last]) @ along).max()
    return body, data.xmat[body].reshape(3, 3).T @ (tip - data.xpos[body])


def _vertices(model, data, geom):
    mesh = model.geom_dataid[geom]
    if mesh < 0:
        return np.empty((0, 3))
    points = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]
                             + model.mesh_vertnum[mesh]]
    return points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


# ------------------------------------------------------------------ the library
def build(progress=lambda done, total, text="": None, seed=SEED):
    """The hand-motion library from random poses and every recording."""
    rng = np.random.default_rng(seed)
    model = hand_spec()[0].compile()
    data = mujoco.MjData(model)
    chain, palm = robot_chain(model)
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = model.jnt_qposadr[hinges]
    low, high = model.jnt_range[hinges].T
    clips, starts, sources = [], [], []
    for number in range(RANDOM_CLIPS):
        poses = [_pose(model, data, qadr, low, high, rng)
                 for _ in range(int(rng.integers(*POSES)))]
        joints = _motion(poses, rng)
        frames = []
        for q in joints:
            data.qpos[qadr] = q
            mujoco.mj_kinematics(model, data)
            frames.append(robot_points(model, data, chain, palm))
        clips.append(shape(np.array(frames)))
        starts.append(poses[0])
        sources.append("random")
        if number % 50 == 0:
            progress(number, RANDOM_CLIPS, f"random motion {number} of {RANDOM_CLIPS}")
    for folder, features in _recorded():
        for piece in _pieces(features):
            clips.append(piece)
            starts.append(np.full(len(qadr), np.nan))
            sources.append(folder)
    return _save(clips, starts, sources)


def export_bundle(folder):
    """What the GPU trains on (gpu_hand.py), into `folder`: the library,
    the hand sim as MJCF (hand.xml) and bundle.json - its joints in the
    library's order, where the 16 points sit (body names, points in the
    bodies' frames) and the palm frame."""
    library = load()
    if not library:
        raise RuntimeError("no hand library - build it first")
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
        "joints": [model.joint(j).name for j in range(model.njnt)
                   if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE],
        "bodies": [model.body(body).name for body, _ in chain],
        "points": [point.tolist() for _, point in chain],
        "wrist": wrist.tolist(), "axes": axes.tolist(), "palm_length": length,
        "mount_position": mount[0].tolist(), "mount_quat": mount[1].tolist(),
        "mujoco": mujoco.__version__, "library_made": str(library["made"]),
        "made": datetime.now().isoformat(timespec="seconds")}, indent=2))
    return folder


def load(path=LIBRARY):
    if not path.exists():
        return {}
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def _pose(model, data, qadr, low, high, rng):
    """A pose the hand can take, its fingers clear of each other."""
    for _ in range(100):
        if rng.random() < UNIFORM_SHARE:
            q = rng.uniform(low, high)
        else:                                  # each finger curls its joints together
            q = np.zeros(len(qadr))
            for start in range(0, len(qadr), 4):
                curl = rng.uniform(0.0, 1.0)
                for k in range(4):
                    i = start + k
                    centre = low[i] + curl * (high[i] - low[i]) if k else \
                        rng.uniform(low[i], high[i]) * 0.5
                    q[i] = np.clip(centre + rng.normal(0, 0.1 * (high[i] - low[i])),
                                   low[i], high[i])
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        if not any(contact.efc_address >= 0 for contact in data.contact[:data.ncon]):
            return q
    return np.zeros(len(qadr))


def _motion(poses, rng):
    """Joint angles at RATE: still for a while at each pose, minimum-jerk
    between them."""
    out = []
    for a, b in zip(poses, poses[1:]):
        out += [a] * round(rng.uniform(*HOLD_S) * RATE)
        frames = max(2, round(rng.uniform(*MOVE_S) * RATE))
        for frame in range(1, frames + 1):
            s = frame / frames
            out.append(a + (b - a) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))
    return np.array(out + [poses[-1]] * round(rng.uniform(*HOLD_S) * RATE))


def _recorded():
    """[(folder, features (T, FEATURES) at RATE)] for every run of frames a
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


def _pieces(features):
    step = round(CLIP_S * RATE)
    return [features[i:i + step] for i in range(0, len(features), step)
            if len(features[i:i + step]) >= MIN_CLIP_S * RATE]


def _save(clips, starts, sources):
    length = max(len(c) for c in clips)
    padded = np.stack([np.concatenate([c, np.repeat(c[-1:], length - len(c), axis=0)])
                       for c in clips]).astype(np.float32)
    arrays = {"features": padded, "length": np.array([len(c) for c in clips]),
              "start": np.array(starts, np.float32), "source": np.array(sources),
              "rate": np.array(RATE), "made": np.array(datetime.now().isoformat(timespec="seconds"))}
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIBRARY.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(LIBRARY)
    recorded = arrays["source"] != "random"
    return {"clips": len(clips), "random": int((~recorded).sum()),
            "recorded": int(recorded.sum()),
            "recordings": len(set(arrays["source"][recorded].tolist()))}
