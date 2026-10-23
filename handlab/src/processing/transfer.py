"""Transfer to sim: the person's hand and the recorded object's motion in
the robot hand's frame -> hand_sim.npz and sim.npz. The world shows them -
the hand as a skeleton on the plan stage (World.set_skeleton), the object as
a ghost box (World.set_demo) - following the Recordings player.

    send_to_sim(episode, progress)      # Send to sim: both, as far as they can be
    transfer_hand(episode, progress)    # the hand -> hand_sim.npz
    transfer(episode, progress)         # the object -> sim.npz

THE HAND needs only the hands step: the hand holding the object (when the
3D step found one), else the one seen most, its 21 landmarks in its palm's
frame and in palm lengths (learning/handmotion.py) laid on the robot's palm
- wrist on wrist, knuckles towards knuckles, palm lengths scaled to the
robot's - a right hand mirrored onto the robot's left. Its shape there is
what the hand-motion policy follows (step 6).

    hand_sim.npz   skeleton (N, 21, 3) m: the landmarks in the robot hand's
                   mount frame (NaN where the hand was not seen); features
                   (N, handmotion.FEATURES): its shape; present (N,); side;
                   time (N,) s; made

THE OBJECT needs the 3D step.
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
from ..learning import handmotion
from ..vision.tracker import SIDES
from .files import HAND_SIM, SIM, stale
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


def send_to_sim(episode, progress=lambda done, total, text="": None):
    """Step 5's Send to sim: the hand, and the object when the 3D step has
    fitted it. Returns their summaries."""
    out = {"hand": transfer_hand(episode, progress)}
    if load(episode):
        out["object"] = transfer(episode, progress)
    return out


def transfer_hand(episode, progress=lambda done, total, text="": None):
    """Save the person's hand for the sim (hand_sim.npz); returns a summary."""
    hands = episode.clean_hands or episode.hands
    if not hands:
        raise ValueError("the recording has no hands")
    progress(0, 1, "laying the hand on the robot's palm")
    slot = _hand_slot(episode, hands)
    side = SIDES[slot]
    world = hands["world"][:, slot].astype(float)
    present = hands["present"][:, slot] & np.isfinite(world).all(axis=(1, 2))
    if not present.any():
        raise ValueError(f"the {side.lower()} hand is never seen")
    skeleton = np.full(world.shape, np.nan)
    features = np.full((len(world), handmotion.FEATURES), np.nan)
    skeleton[present] = handmotion.on_robot(handmotion.in_palm(world[present], side),
                                            handmotion.mount_palm())
    features[present] = handmotion.human_features(world[present], side)
    episode.save_arrays(HAND_SIM, {
        "skeleton": skeleton.astype(np.float32), "features": features.astype(np.float32),
        "present": present, "side": np.array(side), "time": episode.times.astype(np.float32),
        "made": np.array(datetime.now().isoformat(timespec="seconds"))})
    progress(1, 1, "done")
    return {"frames": int(present.sum()), "hand": side}


def _hand_slot(episode, hands):
    """The hand holding the object, when the 3D step found one; else the one
    seen in most frames."""
    pose = load(episode)
    if pose and pose["boxed"].any():
        slot = int(pose["hand"][int(np.flatnonzero(pose["boxed"])[0])])
        if slot >= 0:
            return slot
    return int(np.argmax(hands["present"].sum(axis=0)))


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
