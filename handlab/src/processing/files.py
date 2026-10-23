"""The files the later steps add to a recording, and which ones are out of
date: made before what they were made from was made again.

    stale(episode)   {file: what to do} - {} when every one is current
"""

POSE3D, SIM, PLAN, TRACK = "pose3d.npz", "sim.npz", "plan.npz", "track.npz"
MADE_FROM = {SIM: POSE3D, PLAN: SIM, TRACK: SIM}       # in the order they are made
REDO = {SIM: "the 3D fit changed since: Send to sim again",
        PLAN: "the motion sent to sim changed since: Plan again",
        TRACK: "the motion sent to sim changed since: Track again"}


def stale(episode):
    """{file: what to do} for each step's file made before its input was
    made again, or made from a file that is itself out of date."""
    made = {name: str(episode.arrays(name).get("made", "")) for name in (POSE3D, *MADE_FROM)}
    out = {}
    for name, source in MADE_FROM.items():
        if made[name] and (made[source] > made[name] or source in out):
            out[name] = REDO[name]
    return out
