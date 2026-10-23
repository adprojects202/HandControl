"""Transfer to sim: the recorded object's motion in the robot hand's frame
-> sim.npz. The world shows it as a ghost box that follows the Recordings
player (World.set_demo).

    transfer(episode, progress)

Needs the 3D step (pose3d.npz). The person's palm frame (x up the fingers,
z out of the palm) is laid on the robot hand's mount frame (World: x out of
the palm, y across it, z along the fingers): fingers to fingers, palm-out to
palm-out (TO_MOUNT). The rotation is the person's. Where it goes,
`object.training` says:

    fixed pose     the motion, its median at the World section's
                   `object.position_mm`: how the object moved, not where in
                   the hand it was
    from the hand  where the person held it, relative to their palm, matched
                   to the robot's palm (`robot_palm`): along the palm (up the
                   fingers, across) scaled by the palms' lengths - wrist to
                   middle knuckle; the robot's is ~1.5 times a person's - so
                   held by the fingertips stays by the fingertips; out of the
                   palm not scaled, from palm surface to palm surface, so a
                   box resting on the person's palm rests on the robot's (the
                   box keeps its size)

The robot is a left hand; a demonstration made with the right hand would
need mirroring, which is not done yet (the summary says so).

    sim.npz   position (N, 3) m and quat (N, 4) w x y z: the box in the
              robot hand's mount frame; valid (N,); time (N,) s; size_mm
              (3,); placement (object.training); anchor (3,) m (fixed pose:
              the shift applied); scale (from the hand: robot palm over the
              person's); track, hand (which object and hand of the
              recording); made
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco

from .. import robot as robot_module, settings
from ..vision.tracker import SIDES
from .files import SIM, stale
from .pose3d import load

ROBOT_HAND = "Left"
# Rows: the mount's x (out of the palm), y (across it), z (along the fingers)
# in the person's palm axes (x up the fingers, y = z x x, z out of the palm).
TO_MOUNT = np.array([[0.0, 0.0, 1.0],
                     [0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0]])


def to_mount(in_palm_position, in_palm_rotation):
    """Positions (N, 3) and rotations (N, 3, 3) in the person's palm frame,
    in the robot hand's mount frame."""
    return in_palm_position @ TO_MOUNT.T, np.einsum("ij,njk->nik", TO_MOUNT, in_palm_rotation)


def transfer(episode, progress=lambda done, total, text="": None):
    """Save the recorded object's motion for the sim; returns a summary."""
    pose = load(episode)
    if not pose:
        raise ValueError("fit 3D first")
    track = int(np.flatnonzero(pose["boxed"])[0])         # the calibration object
    slot = int(pose["hand"][track])
    if slot < 0:
        raise ValueError(f"no hand holds {pose['names'][track]}")
    progress(0, 1, "mapping onto the robot hand")
    placement = settings.get("object.training")
    in_palm = pose["in_palm_position"][track]
    position, rotation = to_mount(in_palm, pose["in_palm_rotation"][track])
    valid = np.isfinite(position).all(axis=1) & np.isfinite(rotation).all(axis=(1, 2))
    if not valid.any():
        raise ValueError("the 3D step has no frames with the object in the hand")
    scale = np.nan
    if placement == "from the hand":
        centre, length, surface = robot_palm()
        scale = length / _palm_length(episode, slot, pose["hand_scale"][track])
        anchor = np.zeros(3)
        position = from_hand(in_palm, scale, centre, surface)
    else:
        anchor = (np.array(settings.get("object.position_mm"), float) / 1000.0
                  - np.median(position[valid], axis=0))
        position = position + anchor
    quat = np.full((len(valid), 4), np.nan)
    xyzw = Rotation.from_matrix(rotation[valid]).as_quat()
    quat[valid] = xyzw[:, [3, 0, 1, 2]]                       # MuJoCo: w x y z
    episode.save_arrays(SIM, {
        "position": position.astype(np.float32), "quat": quat.astype(np.float32),
        "valid": valid, "time": episode.times.astype(np.float32),
        "size_mm": pose["box_mm"], "anchor": anchor.astype(np.float32),
        "placement": np.array(placement), "scale": np.array(scale),
        "position_mm": np.array(settings.get("object.position_mm"), float),
        "track": np.array(track), "hand": np.array(SIDES[slot]),
        "made": np.array(datetime.now().isoformat(timespec="seconds"))})
    progress(1, 1, "done")
    return {"frames": int(valid.sum()), "hand": SIDES[slot], "placement": placement,
            **({"scale": round(float(scale), 3)} if np.isfinite(scale) else {}),
            "mirrored_needed": SIDES[slot] != ROBOT_HAND}


def current_sim(episode, progress=lambda done, total, text="": None):
    """sim.npz as the 3D fit and the World section's placement have it now:
    sent again first when missing or out of date - what Plan and Run start
    from, so a moved object is planned afresh."""
    if not episode.arrays(SIM) or SIM in stale(episode):
        progress(0, 1, "sending to sim afresh")
        transfer(episode)
        episode.reload()
    return episode.arrays(SIM)


def from_hand(in_palm, scale, centre, surface):
    """Positions (N, 3) in the person's palm frame -> the robot's mount
    frame, matched to its palm (`robot_palm`): along the palm `scale` times,
    out of it from palm surface to palm surface."""
    return centre + (in_palm * [scale, scale, 1.0] + [0.0, 0.0, surface]) @ TO_MOUNT.T


def robot_palm(robot=None):
    """The robot hand's palm framed as pose3d frames a person's: (centre (3,)
    m, mount frame - the mean of the wrist and the knuckles; palm length m -
    wrist to middle knuckle; the palm surface's distance m out of the palm
    from the centre). The knuckles are each finger's flexing joint (its
    second, after the side swing), the thumb left out and a fourth beside
    the last as a person's little finger; the wrist is the palm's bottom,
    in line with the middle knuckle; the surface, the palm's front face."""
    robot = robot or robot_module.load()
    model = robot.spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    knuckles = [data.xanchor[model.joint(robot.names[indices[1]]).id].copy()
                for group, indices in robot.groups().items() if group != "thumb"]
    palm = np.vstack([_vertices(model, data, geom)
                      for geom in np.flatnonzero(model.geom_bodyid == 0)])   # fixed parts
    middle = knuckles[len(knuckles) // 2]
    wrist = np.array([middle[0], middle[1], palm[:, 2].min()])
    little = knuckles[-1] + (knuckles[-1] - knuckles[-2])
    centre = np.mean([wrist, *knuckles, little], axis=0)
    return centre, float(np.linalg.norm(middle - wrist)), float(palm[:, 0].max() - centre[0])


def _palm_length(episode, slot, hand_scale):
    """The person's palm, wrist to middle knuckle, m: MediaPipe's hand shape
    at the size the 3D step measured (`hand_scale`)."""
    hands = episode.clean_hands or episode.hands
    world = hands["world"][hands["present"][:, slot], slot]
    length = float(np.median(np.linalg.norm(world[:, 9] - world[:, 0], axis=1)))
    return length * (hand_scale if np.isfinite(hand_scale) else 1.0)


def _vertices(model, data, geom):
    mesh = model.geom_dataid[geom]
    points = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]
                             + model.mesh_vertnum[mesh]]
    return points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
