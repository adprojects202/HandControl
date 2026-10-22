"""Recording an episode while the camera runs.

    recorder = Recorder(joint_names, robot_state)
    camera.recorder = recorder          # the camera thread calls add(frame)
    recorder.start("stack cubes", camera_name, settings.section("vision"))
    ...
    folder = recorder.stop()

`start` and `stop` come from the GUI thread, `add` from the camera thread;
a lock keeps them apart. Each frame's video is written as it arrives (about
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
from .episode import (FORMAT, FRAMES, HANDS, META, ROBOT, VIDEO, hand_slots,
                      new_folder, write_json)

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

    def start(self, task, camera_name, vision):
        """Begin a new episode of `task`; `vision` is the vision settings."""
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
                "joints": self.joint_names,
                "software": {name: _version(name) for name in SOFTWARE},
            }
            write_json(self.folder / META, self._meta)
            self._active = True
            return self.folder

    def add(self, frame):
        """One camera frame (vision.Frame), from the camera thread."""
        with self._lock:
            if not self._active:
                return
            try:
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

    def _finish(self):
        self._active = False
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
        self._video = None
        self._meta = {}
        self._t0 = 0.0
        self._times, self._hands, self._command, self._encoders = [], [], [], []


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
