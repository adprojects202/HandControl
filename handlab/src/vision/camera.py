"""The thread that owns the camera: capture, track hands, draw the overlay.

    camera = Camera()
    camera.reconfigure(settings.section("vision"))  # starts it if enabled
    frame = camera.latest()                         # newest Frame, or None
    camera.close()

One thread captures each frame, runs the hand tracker on it and draws what it
found; everything else (the preview window, the assistant) only reads
`latest()`, so a slow reader never holds the camera up and the camera never
holds the simulator up. Tracking runs on every frame, in the camera's own
thread, because the tracker follows hands from one frame to the next.

Settings changed at runtime come in through `reconfigure`: `enabled` starts
and stops the thread; a new camera or capture size reopens the camera; new
tracker settings rebuild the tracker on the next frame; `mirror` applies on
the next frame.

Tracking and recording always use the true image, as the camera took it;
`mirror` only flips the preview (`Frame.image`). With `vision.aruco`, ArUco
markers are looked for too (about 1 ms a frame) and drawn on the preview.

The real camera first needs the user's permission (permission.py): the
first time, macOS asks whether the app running python (Terminal, VS Code,
...) may use the camera, and the thread waits for the answer. If it is no,
the error says where to turn it on.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import aruco, permission
from .overlay import draw
from .tracker import HandTracker

# Settings that need the camera reopened, and those that need a new tracker.
REOPEN = ("camera_index", "capture_width", "capture_height")
TRACKER = ("model", "max_hands", "min_detection", "min_tracking")
READ_FAILURES_TO_FAIL = 30     # consecutive failed reads, ~1 s of frames
FPS_SMOOTHING = 0.9            # weight of the running average
PERMISSION_POLL = 0.25         # s, while waiting for the user's answer


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    raw: np.ndarray        # H x W x 3 RGB, as the camera took it (never mirrored)
    image: np.ndarray      # the preview: tracking drawn on, mirrored if set
    hands: list            # tracker.Hand, in `raw`'s coordinates
    time: float            # time.monotonic() at capture
    number: int            # counts up from 1; tells a reader a frame is new
    markers: tuple = ()    # ArUco markers seen: (id, corners (4, 2) px), largest first


class Camera:
    def __init__(self, open_source=None):
        """`open_source(config)` returns something with OpenCV's
        `read() -> (ok, bgr)` and `release()`; the real camera by default,
        which is only opened with the user's permission."""
        self.open_source = open_source or open_camera
        self.needs_permission = open_source is None
        self.recorder = None     # given every frame, if set (recording.Recorder)
        self.status = "off"
        self.error = ""
        self.fps = 0.0
        self._config = None
        self._frame = None
        self._thread = None
        self._stop = threading.Event()

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def latest(self):
        """The newest Frame, or None while the camera is off or starting."""
        return self._frame

    def reconfigure(self, config):
        """Take a (changed) vision settings section; see the module docstring."""
        old, self._config = self._config, dict(config)
        if not config["enabled"]:
            self.stop()
            return
        if self.running and old is not None and any(old[k] != config[k] for k in REOPEN):
            self.stop()
        if not self.running:
            self._start()

    def stop(self):
        """Turn the camera off (and clear any error: off is off)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._frame = None
        self.fps = 0.0
        self.error = ""
        self.status = "off"

    close = stop

    def _start(self):
        # A fresh event per thread: one that outlives stop()'s join can
        # never be revived by the next start.
        self._stop = threading.Event()
        self.error = ""
        self.status = f"opening camera {self._config['camera_index']}"
        self._thread = threading.Thread(target=self._run, args=(self._stop,),
                                        name="handlab-camera", daemon=True)
        self._thread.start()

    def _run(self, stop):
        config = self._config
        source = tracker = None
        try:
            if self.needs_permission and not self._permitted(stop):
                return
            source = self.open_source(config)
            tracker, built_from = HandTracker(config), _pick(config, TRACKER)
            failures = number = 0
            last = None
            while not stop.is_set():
                ok, bgr = source.read()
                if not ok or bgr is None:
                    failures += 1
                    if failures >= READ_FAILURES_TO_FAIL:
                        raise CameraError(f"camera {config['camera_index']} "
                                          "stopped sending frames")
                    continue
                failures = 0
                now = time.monotonic()
                config = self._config
                if _pick(config, TRACKER) != built_from:
                    tracker.close()
                    tracker, built_from = HandTracker(config), _pick(config, TRACKER)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # Markers are found on the camera's own image, before anything
                # is drawn: the preview's skeleton would cover them.
                markers = (tuple(aruco.detect(rgb, config["aruco_dictionary"]))
                           if config["aruco"] else ())
                hands = tracker.track(rgb, now * 1000.0)
                if last is not None and now > last:
                    rate = 1.0 / (now - last)
                    self.fps = (rate if not self.fps else
                                FPS_SMOOTHING * self.fps + (1 - FPS_SMOOTHING) * rate)
                last = now
                number += 1
                image = draw(rgb, hands, self.fps, mirror=config["mirror"])   # a copy
                if markers:
                    aruco.draw(image, markers, mirror=config["mirror"])
                frame = Frame(raw=rgb, hands=hands, time=now, number=number, image=image,
                              markers=markers)
                self._frame = frame
                if self.recorder is not None:
                    self.recorder.add(frame)
                height, width = rgb.shape[:2]
                self.status = (f"camera {config['camera_index']}: {width}x{height}, "
                               f"{self.fps:.0f} fps, {len(hands)} hand(s)")
        except Exception as failure:     # the camera dies, never the app
            self.error = str(failure) or type(failure).__name__
            self.status = "stopped"
        finally:
            if tracker is not None:
                tracker.close()
            if source is not None:
                source.release()


    def _permitted(self, stop):
        """Wait for the user's answer; False if stopped while waiting."""
        while (access := permission.camera_access()) == permission.WAITING:
            self.status = "waiting for camera permission - answer the macOS prompt"
            if stop.wait(PERMISSION_POLL):
                return False
        if access == permission.REFUSED:
            raise CameraError(permission.HOW_TO_ALLOW)
        return True


def open_camera(config):
    """OpenCV's camera number `camera_index`, asked for the capture size
    (the driver picks the nearest it supports)."""
    index = config["camera_index"]
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise CameraError(f"camera {index} did not open - is it connected, "
                          "and not in use by another app?")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["capture_width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["capture_height"])
    return capture


def _pick(config, keys):
    return tuple(config[k] for k in keys)
