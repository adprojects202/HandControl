"""Hand motion: what the robot hand learns before any object - to take the
shape of a person's hand, as it moves.

    features = human_features(world, side)      # MediaPipe hands -> shapes
    chain, palm = robot_chain()                 # where the robot's matching points are
    features = shape(points)                    # either hand's points -> shape
    features = Kinematics(model, chain, palm).shapes(joints)   # the robot's, from joints
    policy_input = observation(...)             # what the policy sees (gpu_hand, hand_follow)
    error, touching = touch_error(robot, goal)  # how far off the fingertips that touch are

The clips it learns from: hand_library.py.

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

A goal pinch under TOUCH_PL palm lengths is a touch: the robot's thumb and
fingertip pads meet with their tip points 0.15 - 0.23 palm lengths apart
(measured on settled touches, hand_library.py).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..sim.task import hand_spec
from . import env as tracker

RATE = 20
FEATURES = 51
SEGMENTS = 12
FUTURE_STEPS = (1, 3, 6)         # the shapes ahead the policy sees, in steps
ACTION_SCALE = tracker.ACTION_SCALE   # rad a servo target moves per step at most
TOUCH_PL = 0.25
CHAINS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
          "ring": [13, 14, 15, 16]}
WRIST, MIDDLE_KNUCKLE, INDEX_KNUCKLE, PINKY_KNUCKLE = 0, 9, 5, 17


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


def touch_error(robot, goal):
    """(error (n,), touching (n,)): between shapes (n, FEATURES), the mean
    pinch error over the fingertips the goal has touching the thumb (a pinch
    under TOUCH_PL), palm lengths - 0 where none touch."""
    pinch, wanted = robot[:, FEATURES - 3:], goal[:, FEATURES - 3:]
    touching = wanted < TOUCH_PL
    count = touching.sum(dim=1)
    error = ((pinch - wanted).abs() * touching).sum(dim=1) / torch.clamp(count, min=1)
    return error, count > 0


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


class Kinematics:
    """The robot's shapes from its joints: `model` (any compile of the hand
    sim), its 16 points (`chain`: [(body id, point (3,) in the body's
    frame)]) and its palm (wrist, axes, length) in the model's world;
    `joints`: the joint names a pose lists, in order (default: the model's)."""

    def __init__(self, model, chain, palm, joints=None):
        self.model, self.data = model, mujoco.MjData(model)
        self.bodies = np.array([body for body, _ in chain])
        self.local = np.array([point for _, point in chain])
        self.palm = palm
        names = joints or [model.joint(j).name for j in range(model.njnt)]
        self.qadr = np.array([model.jnt_qposadr[model.joint(name).id] for name in names])

    def shapes(self, joints):
        """Shapes (T, FEATURES) for poses (T, J)."""
        count = len(joints)
        positions = np.empty((count, len(self.bodies), 3))
        rotations = np.empty((count, len(self.bodies), 9))
        for row, q in enumerate(joints):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            positions[row] = self.data.xpos[self.bodies]
            rotations[row] = self.data.xmat[self.bodies]
        world = positions + np.einsum("tpij,pj->tpi", rotations.reshape(count, -1, 3, 3),
                                      self.local)
        wrist, axes, length = self.palm
        return shape((world - wrist) @ axes / length)
