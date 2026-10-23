"""Follow the hand: the trained hand-motion policy takes the shape of the
person's hand through the recording, in the hand sim - no object, no
retraining. -> follow.npz

    follow_hand(episode, progress)

Needs hand_sim.npz (Send to sim; sent afresh when missing or out of date)
and a hand-motion run: the run `learning.hand_run` names, else the newest
with a checkpoint (Learning panel, learning hand motion). The person's hand
shape is taken at the policy's rate (handmotion.RATE) from the first frame
the hand is seen to the last, gaps filled in between; the robot hand starts
at rest.

    follow.npz  time (T,) s, the recording's; qpos (T, J) rad, the robot's
                joint order; angle_error (T,) rad - each finger segment's
                mean angle off the person's - tip_error (T,) m, on the
                robot's palm; still_angle_error, still_tip_error: had it
                stayed at rest; run (the run folder), checkpoint, made
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .. import robot as robot_module
from ..learning.hand_follow import follow
from ..learning.handmotion import RATE
from ..learning.train import chosen_run, latest_checkpoint
from .files import FOLLOW, HAND_SIM, stale
from .transfer import transfer_hand

DIRECTIONS = 36                  # the shape's first numbers: 12 unit segments


def follow_hand(episode, progress=lambda done, total, text="": None):
    """Follow the recording's hand with the hand-motion policy; returns a summary."""
    hand = current_hand(episode, progress)
    run = chosen_run("hand motion")
    if run is None:
        raise ValueError("no hand-motion run yet (Learning panel, learn: hand motion)")
    times, features = at_rate(hand)
    progress(0, 1, f"following {len(times) / RATE:.1f} s of hand motion")
    checkpoint = latest_checkpoint(run)
    result = follow(features, run, checkpoint)
    names = robot_module.load().names
    if list(result["joint_names"]) != list(names):
        raise RuntimeError("the hand sim's joints are not the robot's order")
    length = result["palm_length"]
    episode.save_arrays(FOLLOW, {
        "time": times.astype(np.float32), "qpos": result["joints"].astype(np.float32),
        "angle_error": result["angle"].astype(np.float32),
        "tip_error": (result["tip"] * length).astype(np.float32),
        "still_angle_error": result["still_angle"].astype(np.float32),
        "still_tip_error": (result["still_tip"] * length).astype(np.float32),
        "run": np.array(str(run)), "checkpoint": np.array(checkpoint.name),
        "made": np.array(datetime.now().isoformat(timespec="seconds"))})
    progress(1, 1, "done")
    return summary(episode.arrays(FOLLOW))


def current_hand(episode, progress=lambda done, total, text="": None):
    """hand_sim.npz as the hands step has it now: sent again first when
    missing or out of date."""
    if not episode.arrays(HAND_SIM) or HAND_SIM in stale(episode):
        progress(0, 1, "sending the hand to sim afresh")
        transfer_hand(episode)
        episode.reload()
    return episode.arrays(HAND_SIM)


def at_rate(hand):
    """(times (T,) s, shapes (T, FEATURES)) at RATE from the first frame the
    hand is seen to the last; between, each number interpolated over the
    frames it is seen in, the segment directions made unit again."""
    present = hand["present"]
    seen, features = hand["time"][present].astype(float), hand["features"][present].astype(float)
    times = np.arange(seen[0], seen[-1] + 0.5 / RATE, 1.0 / RATE)
    out = np.stack([np.interp(times, seen, features[:, k]) for k in range(features.shape[1])],
                   axis=1)
    segments = out[:, :DIRECTIONS].reshape(len(times), -1, 3)
    out[:, :DIRECTIONS] = (segments / np.linalg.norm(segments, axis=-1, keepdims=True)
                           ).reshape(len(times), -1)
    return times, out


def summary(result):
    """Medians: {angle_deg, tip_mm} following, and {still_angle_deg,
    still_tip_mm} had it stayed at rest."""
    return {"angle_deg": float(np.degrees(np.median(result["angle_error"]))),
            "tip_mm": float(1000 * np.median(result["tip_error"])),
            "still_angle_deg": float(np.degrees(np.median(result["still_angle_error"]))),
            "still_tip_mm": float(1000 * np.median(result["still_tip_error"]))}
