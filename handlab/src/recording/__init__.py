"""Recordings of a person doing a task: written live, read back for review
and for the processing steps that come after (describe, segment, ...).

    episode.py   the on-disk layout, and Episode to read one back
    recorder.py  Recorder: writes an episode from the camera thread
"""

from .episode import Episode, episodes
from .recorder import Recorder

__all__ = ["Episode", "Recorder", "episodes"]
