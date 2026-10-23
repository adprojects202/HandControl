"""Transfer to sim: the recorded object's motion in the robot hand's frame
-> sim.npz. The world shows it as a ghost box that follows the Recordings
player (World.set_demo).

    transfer(episode, progress)

Needs the 3D step (pose3d.npz). The person's palm frame (x up the fingers,
z out of the palm) is laid on the robot hand's mount frame (World: x out of
the palm, y across it, z along the fingers): fingers to fingers, palm-out to
palm-out (TO_MOUNT). Positions keep their size - it is the same box - and
are anchored so the recording's median position is where the World section
places the calibration object, against the robot's palm. The robot is a left
hand; a demonstration made with the right hand would need mirroring, which is
not done yet (the summary says so).

    sim.npz   position (N, 3) m and quat (N, 4) w x y z: the box in the
              robot hand's mount frame; valid (N,); time (N,) s; size_mm
              (3,); anchor (3,) m; track, hand (which object and hand of the
              recording); made
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

from .. import settings
from ..vision.tracker import SIDES
from .files import POSE3D, SIM

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
    pose = episode.arrays(POSE3D)
    if not pose:
        raise ValueError("fit 3D first")
    track = 0
    slot = int(pose["hand"][track])
    if slot < 0:
        raise ValueError(f"no hand holds {pose['names'][track]}")
    progress(0, 1, "mapping onto the robot hand")
    position, rotation = to_mount(pose["in_palm_position"][track],
                                  pose["in_palm_rotation"][track])
    valid = np.isfinite(position).all(axis=1) & np.isfinite(rotation).all(axis=(1, 2))
    if not valid.any():
        raise ValueError("the 3D step has no frames with the object in the hand")
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
        "track": np.array(track), "hand": np.array(SIDES[slot]),
        "made": np.array(datetime.now().isoformat(timespec="seconds"))})
    progress(1, 1, "done")
    return {"frames": int(valid.sum()), "hand": SIDES[slot],
            "mirrored_needed": SIDES[slot] != ROBOT_HAND}
