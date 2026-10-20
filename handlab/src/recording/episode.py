"""One recording ("episode") of a person doing a task, on disk.

    data/tasks/<task>/<YYYYmmdd-HHMMSS>/
        meta.json     what was recorded, with what (format, camera, tracker, ...)
        video.mp4     the frames, H.264, as the camera took them (never mirrored)
        frames.npz    time (N,) s since the first frame - one clock for everything
        hands.npz     per frame, slot 0 = the person's left hand, 1 = right:
                        image   (N, 2, 21, 3) landmarks in 0..1 of the image (NaN: not seen)
                        world   (N, 2, 21, 3) metres around the hand's centre
                        score   (N, 2) left/right confidence (NaN: not seen)
                        present (N, 2) bool
        robot.npz     command and encoders (N, J) rad (NaN: no reading); joints (J,)
        context.json  from the describe step: what the task is (see assistant/context.py)
        masks.npz     from the segment step, one track per context object:
                        masks  (O, N, H, ceil(W/8)) uint8, bits packed along the
                               width (Episode.mask unpacks one frame)
                        area   (O, N) fraction of the image covered
                        names, roles (O,); prompt_frame (O,); prompt_box (O, 4) px
                        model, made: what made it, when

Recording writes video.mp4 as it goes and the arrays when it stops; meta.json
is written at the start with "complete": false and finished at the stop, so
a recording cut short by a crash says so. Later steps (describe, segment,
...) each add their own file and never rewrite the recording.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .. import paths, settings
from ..vision.tracker import LANDMARKS, SIDES, Hand

FORMAT = 1
META, VIDEO, FRAMES, HANDS, ROBOT, CONTEXT, MASKS = (
    "meta.json", "video.mp4", "frames.npz", "hands.npz", "robot.npz", "context.json",
    "masks.npz")
STAMP = "%Y%m%d-%H%M%S"


def root():
    """The folder that holds every task's recordings."""
    return paths.resolve(settings.get("recording.folder"))


def task_folder_name(task):
    """A task name as a folder name: letters, digits, - and _ only."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", task.strip()).strip("_") or "untitled"


def new_folder(task, now=None):
    """A new, empty episode folder for `task`."""
    stamp = (now or datetime.now()).strftime(STAMP)
    folder = root() / task_folder_name(task) / stamp
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def episodes():
    """Every episode folder, newest first."""
    found = [meta.parent for meta in root().glob(f"*/*/{META}")]
    return sorted(found, key=lambda folder: folder.name, reverse=True)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def hand_slots(hands):
    """Tracker hands -> (image, world, score, present) for one frame, one slot
    per side. Two hands given the same side: the surer one is kept."""
    image = np.full((len(SIDES), LANDMARKS, 3), np.nan, np.float32)
    world = np.full((len(SIDES), LANDMARKS, 3), np.nan, np.float32)
    score = np.full(len(SIDES), np.nan, np.float32)
    for hand in hands:
        slot = SIDES.index(hand.side)
        if np.isnan(score[slot]) or hand.score > score[slot]:
            image[slot], world[slot], score[slot] = hand.image, hand.world, hand.score
    return image, world, score, ~np.isnan(score)


class Episode:
    """A recording read back: meta, arrays, frames, context."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / META).read_text())
        self.times = self._arrays(FRAMES).get("time", np.zeros(0))
        self.hands = self._arrays(HANDS)
        self.robot = self._arrays(ROBOT)
        self._video = None
        self._next = 0                   # the frame the video reads next
        self._masks = None

    @property
    def name(self):
        return f"{self.folder.parent.name} / {self.folder.name}"

    @property
    def complete(self):
        return bool(self.meta.get("complete"))

    @property
    def frames(self):
        return len(self.times)

    @property
    def context(self):
        path = self.folder / CONTEXT
        return json.loads(path.read_text()) if path.exists() else None

    def save_context(self, context):
        write_json(self.folder / CONTEXT, context)

    @property
    def masks(self):
        """masks.npz as a dict, or {} before the segment step has run."""
        if self._masks is None:
            self._masks = self._arrays(MASKS)
        return self._masks

    def reload_masks(self):
        self._masks = None

    def mask(self, track, index):
        """Track `track`'s mask in frame `index`, (H, W) bool."""
        width = int(self.masks["width"])
        return np.unpackbits(self.masks["masks"][track, index], axis=-1,
                             count=width).astype(bool)

    def save_masks(self, arrays):
        tmp = self.folder / (MASKS + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(self.folder / MASKS)
        self._masks = None

    def image(self, index):
        """Frame `index` as RGB (the true image), or None past the end."""
        if self._video is None:
            self._video = cv2.VideoCapture(str(self.folder / VIDEO), cv2.CAP_FFMPEG)
            self._next = 0
        if index != self._next:                  # seeking costs; playing does not
            self._video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self._video.read()
        self._next = index + 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def read_frames(self, indices):
        """The frames at `indices` (sorted), RGB, in one pass through the
        video. Uses its own reader, so it may run on another thread than
        `image` while that one plays."""
        wanted, found = set(indices), {}
        video = cv2.VideoCapture(str(self.folder / VIDEO), cv2.CAP_FFMPEG)
        try:
            for index in range(max(wanted, default=-1) + 1):
                ok = video.grab()
                if not ok:
                    break
                if index in wanted:
                    ok, bgr = video.retrieve()
                    if ok:
                        found[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            video.release()
        return [found[index] for index in sorted(found)]

    def hands_at(self, index):
        """The tracked hands of frame `index`, as tracker.Hand."""
        if not self.hands:
            return []
        return [Hand(side, float(self.hands["score"][index, slot]),
                     self.hands["image"][index, slot], self.hands["world"][index, slot])
                for slot, side in enumerate(SIDES) if self.hands["present"][index, slot]]

    def sample(self, count):
        """`count` frame indices spread evenly over the recording, first and
        last included."""
        if self.frames == 0:
            return []
        count = max(1, min(count, self.frames))
        return sorted(set(np.linspace(0, self.frames - 1, count).round().astype(int).tolist()))

    def close(self):
        if self._video is not None:
            self._video.release()
            self._video = None

    def _arrays(self, name):
        path = self.folder / name
        if not path.exists():
            return {}
        with np.load(path) as arrays:
            return {key: arrays[key] for key in arrays.files}
