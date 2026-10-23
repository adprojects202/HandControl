"""One recording ("episode") of a person doing a task, on disk.

    data/tasks/<task>/<YYYYmmdd-HHMMSS>/
        meta.json     what was recorded, with what (format, camera, tracker, ...);
                      camera.calibration: the camera's calibration at the
                      recorded size when it had one (vision/calibration.py)
        video.mp4     the frames, H.264, as the camera took them (never mirrored)
        frames.npz    time (N,) s since the first frame - one clock for everything
        hands.npz     per frame, slot 0 = the person's left hand, 1 = right:
                        image   (N, 2, 21, 3) landmarks in 0..1 of the image (NaN: not seen)
                        world   (N, 2, 21, 3) metres around the hand's centre
                        score   (N, 2) left/right confidence (NaN: not seen)
                        present (N, 2) bool
        robot.npz     command and encoders (N, J) rad (NaN: no reading); joints (J,)
        video_back.mp4, frames_back.npz, hands_back.npz
                      the second camera's, when one filmed the other side of
                      the hand (meta.json "back_camera"): its own frames and
                      their times on the same clock (frames.npz's: the first
                      camera's first frame is 0), its hands as hands.npz
        context.json  from the describe step: what the task is (see assistant/context.py)
        masks.npz     from the segment step, one track per context object:
                        masks  (O, N, H, ceil(W/8)) uint8, bits packed along the
                               width (Episode.mask unpacks one frame)
                        area   (O, N) fraction of the image covered
                        names, roles (O,); prompt_frame (O,); prompt_box (O, 4) px
                        model, made: what made it, when
        hands_clean.npz  from the hands step (processing/hands.py): the same
                      hands as hands.npz, but each slot is one physical hand
                      followed through the recording and named by vote:
                        image, world, score (NaN: filled), present (N, 2, ...)
                        observed, filled (N, 2) bool; track (N, 2) int (-1: none)
                        track_side, track_frames (T,); tracks, relabelled,
                        phantoms (detections dropped), filled_frames
        object.npz    from the object step (processing/objects.py), per mask
                      track, in the image: see that module
        pose3d.npz, sim.npz, plan.npz   Fit 3D, Send to sim, Plan in sim:
                      see processing/pose3d.py, transfer.py, plan.py

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
from ..vision.calibration import Calibration, for_camera
from ..vision.regions import main_region
from ..vision.tracker import LANDMARKS, SIDES, Hand

# 2: hands.npz slots are the person's own hands. Format 1 (until 2026-10-22)
# stored them swapped - left in the right slot - and is swapped back on read.
FORMAT = 2
META, VIDEO, FRAMES, HANDS, ROBOT, CONTEXT = (
    "meta.json", "video.mp4", "frames.npz", "hands.npz", "robot.npz", "context.json")
MASKS, HANDS_CLEAN, OBJECT = "masks.npz", "hands_clean.npz", "object.npz"
# The second camera's, when one filmed the other side of the hand.
VIDEO_BACK, FRAMES_BACK, HANDS_BACK = "video_back.mp4", "frames_back.npz", "hands_back.npz"
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
        if self.hands and self.meta.get("format", 1) < 2:
            self.hands = {key: value[:, ::-1] for key, value in self.hands.items()}
        self.robot = self._arrays(ROBOT)
        self.back_times = self._arrays(FRAMES_BACK).get("time", np.zeros(0))
        self.back_hands = self._arrays(HANDS_BACK)
        self._video = None
        self._back_video, self._back_next = None, 0
        self._next = 0                   # the frame the video reads next
        self._steps = {}                 # step file name -> its arrays

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

    def arrays(self, name):
        """A step's arrays (e.g. MASKS) as a dict - {} before the step has
        run. Read once; `reload()` after a step rewrites its file."""
        if name not in self._steps:
            self._steps[name] = self._arrays(name)
        return self._steps[name]

    def save_arrays(self, name, arrays):
        tmp = self.folder / (name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(self.folder / name)
        self._steps.pop(name, None)

    def reload(self):
        self._steps.clear()

    @property
    def calibration(self):
        """The camera's calibration for this recording: the one recorded with
        it, else the one saved for that camera now (scaled to the recorded
        size), else None."""
        camera = self.meta.get("camera", {})
        if camera.get("calibration"):
            return Calibration.from_entry(camera["calibration"])
        if camera.get("name") and camera.get("width"):
            return for_camera(camera["name"], camera["width"], camera["height"])
        return None

    @property
    def masks(self):
        """masks.npz (segment step), or {}."""
        return self.arrays(MASKS)

    @property
    def clean_hands(self):
        """hands_clean.npz (hands step), or {}."""
        return self.arrays(HANDS_CLEAN)

    def mask(self, track, index, stored=False):
        """Track `track`'s mask in frame `index`, (H, W) bool: the object's
        own region (vision/regions.py) - or, with `stored`, the pixels as
        saved, specks and all."""
        width = int(self.masks["width"])
        mask = np.unpackbits(self.masks["masks"][track, index], axis=-1,
                             count=width).astype(bool)
        return mask if stored else main_region(mask)


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

    @property
    def has_back(self):
        """Whether a second camera filmed the other side of the hand."""
        return len(self.back_times) > 0

    def back_index(self, index):
        """The back camera's frame nearest in time to frame `index`."""
        return int(np.clip(np.searchsorted(self.back_times, self.times[index]), 0,
                           len(self.back_times) - 1)) if self.has_back else None

    def back_image(self, index):
        """The back camera's frame nearest frame `index`, RGB, or None."""
        if not self.has_back:
            return None
        back = self.back_index(index)
        if self._back_video is None:
            self._back_video = cv2.VideoCapture(str(self.folder / VIDEO_BACK), cv2.CAP_FFMPEG)
            self._back_next = 0
        if back != self._back_next:
            self._back_video.set(cv2.CAP_PROP_POS_FRAMES, back)
        ok, bgr = self._back_video.read()
        self._back_next = back + 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def back_hands_at(self, index):
        """The back camera's hands nearest frame `index`, as recorded."""
        if not self.has_back or not self.back_hands:
            return []
        back, hands = self.back_index(index), self.back_hands
        return [Hand(side, float(hands["score"][back, slot]), hands["image"][back, slot],
                     hands["world"][back, slot])
                for slot, side in enumerate(SIDES) if hands["present"][back, slot]]

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

    def hands_at(self, index, clean=True):
        """The hands of frame `index`, as tracker.Hand: the hands step's when
        it has run (and `clean`), else as recorded. A filled-in hand has a
        NaN score."""
        hands = (self.clean_hands if clean else {}) or self.hands
        if not hands:
            return []
        return [Hand(side, float(hands["score"][index, slot]),
                     hands["image"][index, slot], hands["world"][index, slot])
                for slot, side in enumerate(SIDES) if hands["present"][index, slot]]

    def sample(self, count):
        """`count` frame indices spread evenly over the recording, first and
        last included."""
        if self.frames == 0:
            return []
        count = max(1, min(count, self.frames))
        return sorted(set(np.linspace(0, self.frames - 1, count).round().astype(int).tolist()))

    def close(self):
        for video in (self._video, self._back_video):
            if video is not None:
                video.release()
        self._video = self._back_video = None

    def _arrays(self, name):
        path = self.folder / name
        if not path.exists():
            return {}
        with np.load(path) as arrays:
            return {key: arrays[key] for key in arrays.files}
