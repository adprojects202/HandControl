"""The files the later steps add to a recording, and which ones are out of
date: made before what they were made from was made again - or, for
sim.npz, made with the object placed otherwise than the World section
places it now (`object.training`, and `object.position_mm` for a fixed
pose).

    stale(episode)   {file: what to do} - {} when every one is current
"""

import numpy as np

from .. import settings

POSE3D, SIM, PLAN, TRACK = "pose3d.npz", "sim.npz", "plan.npz", "track.npz"
MADE_FROM = {SIM: POSE3D, PLAN: SIM, TRACK: SIM}       # in the order they are made
REDO = {SIM: "the 3D fit changed since: Send to sim again",
        PLAN: "the motion sent to sim changed since: Plan again",
        TRACK: "the motion sent to sim changed since: Run again"}
MOVED = "the World section places the object otherwise now: Send to sim again"


def stale(episode):
    """{file: what to do} for each step's file made before its input was
    made again, or made from a file that is itself out of date."""
    made = {name: str(episode.arrays(name).get("made", "")) for name in (POSE3D, *MADE_FROM)}
    out = {}
    for name, source in MADE_FROM.items():
        if made[name] and (made[source] > made[name] or source in out):
            out[name] = REDO[name]
        elif name == SIM and made[name] and placed_otherwise(episode.arrays(SIM)):
            out[name] = MOVED
    return out


def placed_otherwise(sim):
    """Whether sim.npz put the object otherwise than the World section's
    settings would now. Files from before it was recorded: a fixed pose."""
    placement = str(sim.get("placement", "fixed pose"))
    if placement != settings.get("object.training"):
        return True
    if placement == "fixed pose" and "position_mm" in sim:
        return not np.allclose(sim["position_mm"], settings.get("object.position_mm"))
    return False
