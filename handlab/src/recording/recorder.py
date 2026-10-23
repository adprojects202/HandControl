"""Recording an episode while the camera runs.

    recorder = Recorder(joint_names, robot_state)
    camera.recorder = recorder          # the camera thread calls add(frame, stream)
    recorder.start("stack cubes", camera_name, settings.section("vision"),
                   back_name)           # back_name: the second camera's, or None
    ...
    folder = recorder.stop()

`start` and `stop` come from the GUI thread, `add` from the camera threads;
a lock keeps them apart. Frames of the "front" camera make video.mp4,
frames.npz, hands.npz and robot.npz; a second camera ("back", filming the
other side of the hand) makes video_back.mp4, frames_back.npz and
hands_back.npz, timed on the same clock - its frames before the front's
first are dropped. Each frame's video is written as it arrives (about
2 ms); hands and robot state are kept in memory - well under 1 MB a minute -
and saved at the stop. The robot is sampled at each camera frame, so every
array shares the frame's clock. See episode.py for the layout.
"""

from __future__ import annotations

import threading
from datetime import datetime
from importlib import metadata

import cv2
import numpy as np

from ..vision.calibration import for_camera
from .episode import (FORMAT, FRAMES, FRAMES_BACK, HANDS, HANDS_BACK, META, ROBOT,
                      VIDEO, VIDEO_BACK, hand_slots, new_folder, write_json)

FOURCC = "avc1"        # H.264, written by OpenCV's FFmpeg backend
# The rate written into video.mp4, for players only: real frame times are in
# frames.npz (cameras do not keep an exact rate).
NOMINAL_FPS = 30.0
SOFTWARE = ("mediapipe", "opencv-contrib-python", "mujoco", "numpy")


class Recorder:
    def __init__(self, joint_names, robot_state):
        """`robot_state()` returns (command, encoders or None), in
        `joint_names` order."""
        self.joint_names = list(joint_names)
        self.robot_state = robot_state
        self.folder = None           # the episode being recorded, or the last one
        self.error = ""
        self._lock = threading.Lock()
        self._active = False
        self._clear()

    @property
    def active(self):
        return self._active

    @property
    def frames(self):
        return len(self._times)

    @property
    def seconds(self):
        return self._times[-1] if self._times else 0.0

    def start(self, task, camera_name, vision, back_name=None):
        """Begin a new episode of `task`; `vision` is the vision settings;
        `back_name` the second camera's name when it records too."""
        with self._lock:
            if self._active:
                return self.folder
            self._clear()
            self.error = ""
            started = datetime.now()
            self.folder = new_folder(task, started)
            self._meta = {
                "format": FORMAT,
                "task": task.strip() or "untitled",
                "started": started.isoformat(timespec="seconds"),
                "complete": False,
                "camera": {"name": camera_name, "index": vision["camera_index"],
                           "mirrored": False},
                "tracker": {key: vision[key] for key in
                            ("model", "max_hands", "min_detection", "min_tracking")},
                **({"back_camera": {"name": back_name, "index": vision["back_camera_index"],
                                    "mirrored": False}} if back_name else {}),
                "joints": self.joint_names,
                "software": {name: _version(name) for name in SOFTWARE},
            }
            write_json(self.folder / META, self._meta)
            self._active = True
            return self.folder

    def add(self, frame, stream="front"):
        """One camera frame (vision.Frame), from a camera thread."""
        with self._lock:
            if not self._active or (stream == "back" and "back_camera" not in self._meta):
                return
            try:
                if stream == "back":
                    self._write_back(frame)
                else:
                    self._write(frame)
            except Exception as failure:          # a full disk, a bad codec ...
                self.error = f"recording stopped: {failure}"
                self._finish()

    def stop(self):
        """End the episode and save it; returns its folder."""
        with self._lock:
            if self._active:
                self._finish()
            return self.folder

    def _write(self, frame):
        height, width = frame.raw.shape[:2]
        if self._video is None:
            self._video = cv2.VideoWriter(str(self.folder / VIDEO), cv2.CAP_FFMPEG,
                                          cv2.VideoWriter_fourcc(*FOURCC), NOMINAL_FPS,
                                          (width, height))
            if not self._video.isOpened():
                raise RuntimeError(f"cannot write {VIDEO} ({FOURCC})")
            self._t0 = frame.time
            known = for_camera(self._meta["camera"]["name"], width, height)
            self._meta["camera"].update(width=width, height=height,
                                        calibration=known.entry() if known else None)
        self._video.write(cv2.cvtColor(frame.raw, cv2.COLOR_RGB2BGR))
        self._times.append(frame.time - self._t0)
        self._hands.append(hand_slots(frame.hands))
        command, encoders = self.robot_state()
        self._command.append(np.asarray(command, float))
        self._encoders.append(np.full(len(self.joint_names), np.nan)
                              if encoders is None else np.asarray(encoders, float))

    def _write_back(self, frame):
        if self._video is None:                   # timed from the front's first frame
            return
        height, width = frame.raw.shape[:2]
        if self._back_video is None:
            self._back_video = cv2.VideoWriter(str(self.folder / VIDEO_BACK), cv2.CAP_FFMPEG,
                                               cv2.VideoWriter_fourcc(*FOURCC), NOMINAL_FPS,
                                               (width, height))
            if not self._back_video.isOpened():
                raise RuntimeError(f"cannot write {VIDEO_BACK} ({FOURCC})")
            known = for_camera(self._meta["back_camera"]["name"], width, height)
            self._meta["back_camera"].update(width=width, height=height,
                                             calibration=known.entry() if known else None)
        self._back_video.write(cv2.cvtColor(frame.raw, cv2.COLOR_RGB2BGR))
        self._back_times.append(frame.time - self._t0)
        self._back_hands.append(hand_slots(frame.hands))

    def _finish(self):
        self._active = False
        if self._back_video is not None:
            self._back_video.release()
        if self._back_times:
            np.savez_compressed(self.folder / FRAMES_BACK, time=np.array(self._back_times))
            image, world, score, present = (np.stack(part) for part in zip(*self._back_hands))
            np.savez_compressed(self.folder / HANDS_BACK, image=image, world=world,
                                score=score, present=present)
            self._meta["back_camera"]["frames"] = len(self._back_times)
        if self._video is not None:
            self._video.release()
        frames = len(self._times)
        if frames:
            np.savez_compressed(self.folder / FRAMES, time=np.array(self._times))
            image, world, score, present = (np.stack(part) for part in zip(*self._hands))
            np.savez_compressed(self.folder / HANDS, image=image, world=world,
                                score=score, present=present)
            np.savez_compressed(self.folder / ROBOT, command=np.stack(self._command),
                                encoders=np.stack(self._encoders),
                                joints=np.array(self.joint_names))
        duration = self._times[-1] if frames > 1 else 0.0
        self._meta.update(complete=frames > 0, frames=frames,
                          duration_s=round(duration, 3),
                          fps=round((frames - 1) / duration, 2) if duration else 0.0)
        write_json(self.folder / META, self._meta)
        self._video = None

    def _clear(self):
        self._video = self._back_video = None
        self._meta = {}
        self._t0 = 0.0
        self._times, self._hands, self._command, self._encoders = [], [], [], []
        self._back_times, self._back_hands = [], []


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
