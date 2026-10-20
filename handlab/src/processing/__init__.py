"""Steps run on a saved recording, after it is recorded. Each one reads the
episode and adds its own file to it (layout in recording/episode.py).

    job.py      Job: runs one step at a time off the GUI thread, with progress
    segment.py  object masks through the recording, from the task context
                (SAM 2.1 or SAM 3) -> masks.npz

The describe step (context.json) lives with the assistant (src/assistant).
"""

from .job import Cancelled, Job

__all__ = ["Cancelled", "Job"]
