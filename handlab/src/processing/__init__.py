"""Steps run on a saved recording, after it is recorded. Each one reads the
episode and adds its own file to it (layout in recording/episode.py).

    job.py      Job: runs one step at a time off the GUI thread, with progress
    segment.py  object masks through the recording, from the task context
                (SAM 2.1 or SAM 3) -> masks.npz
    hands.py    each hand followed through the recording -> hands_clean.npz
    objects.py  each object's motion in the image and the hand -> object.npz
    pose3d.py   the hands and the object in metres -> pose3d.npz
    transfer.py that motion in the robot hand's frame -> sim.npz
    plan.py     the robot hand grips and follows it in the task sim -> plan.npz

The describe step (context.json) lives with the assistant (src/assistant).
"""

from .job import Cancelled, Job

__all__ = ["Cancelled", "Job"]
