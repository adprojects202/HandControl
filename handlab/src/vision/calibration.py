"""Calibrating a camera from pictures of a checkerboard: its focal length,
image centre and lens distortion - the numbers that turn pixels into
directions, and so 2D tracking into 3D.

    session = CalibrationSession((9, 6))          # inner corners across, down
    session.offer(frame.raw)                      # every frame; found on a thread
    session.take()                                # keep the current view
    session.calibrate()                           # once MIN_VIEWS are taken
    save("MacBook Pro Camera", session.result)    # settings: camera_calibration
    for_camera("MacBook Pro Camera", 640, 480)    # -> Calibration, or None

Show the board - printed, or on a phone or tablet screen - at different
angles and distances, in different parts of the view, and take a picture
each time its corners are found. The squares' overall size does not change
the camera's numbers, but their shape does: a board shown on a screen can
be stretched, and calibrating a stretched board as square bends the focal
lengths. So a square's width and height are given (`square`, mm).

A calibration holds for the capture size it was made at. At another size of
the same shape it is scaled; at another shape there is none.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from .. import settings

MIN_VIEWS = 8
SAME_VIEW_PX = 15        # corners moved less than this on average: the same view
FIND_FLAGS = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
              | cv2.CALIB_CB_ACCURACY)


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple        # k1, k2, p1, p2, k3
    rms_px: float            # reprojection error of the corners
    views: int
    made: str
    square_mm: tuple = (0.0, 0.0)   # the board's square, width x height

    def matrix(self):
        """The 3 x 3 camera matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def scaled(self, width, height):
        """The same camera at another capture size of the same shape, or
        None for another shape."""
        if (width, height) == (self.width, self.height):
            return self
        if abs(width / height - self.width / self.height) > 0.01:
            return None
        k = width / self.width
        return Calibration(width, height, self.fx * k, self.fy * k, self.cx * k,
                           self.cy * k, self.distortion, self.rms_px * k,
                           self.views, self.made, self.square_mm)

    def entry(self):
        """As kept in settings (camera_calibration.<name>)."""
        return {"width": self.width, "height": self.height, "fx": self.fx,
                "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "distortion": list(self.distortion), "rms_px": self.rms_px,
                "views": self.views, "made": self.made,
                "square_mm": list(self.square_mm)}

    @classmethod
    def from_entry(cls, entry):
        return cls(entry["width"], entry["height"], entry["fx"], entry["fy"],
                   entry["cx"], entry["cy"], tuple(entry["distortion"]),
                   entry["rms_px"], entry["views"], entry["made"],
                   tuple(entry.get("square_mm", (0.0, 0.0))))


def find_corners(rgb, pattern):
    """The board's inner corners in `rgb` as (N, 1, 2) px, or None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, tuple(pattern), flags=FIND_FLAGS)
    return corners.astype(np.float32) if found else None


def calibrate(views, pattern, size, square=(1.0, 1.0)):
    """A Calibration from corner sets (each `find_corners` output) of one
    camera at `size` (width, height); `square` is one square's width and
    height (mm) on the board."""
    columns, rows = pattern
    board = np.zeros((columns * rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * np.float32(square)
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        [board] * len(views), list(views), tuple(size), None, None)
    return Calibration(int(size[0]), int(size[1]), float(matrix[0, 0]), float(matrix[1, 1]),
                       float(matrix[0, 2]), float(matrix[1, 2]),
                       tuple(float(v) for v in distortion.ravel()[:5]), float(rms),
                       len(views), datetime.now().isoformat(timespec="seconds"),
                       tuple(float(v) for v in square))


def for_camera(name, width, height):
    """The saved calibration of camera `name`, at width x height, or None."""
    entry = settings.get("camera_calibration").get(name)
    if not entry or entry["fx"] <= 0:
        return None
    return Calibration.from_entry(entry).scaled(width, height)


def save(name, calibration):
    """Keep `calibration` in settings.json under camera `name`."""
    saved = settings.get("camera_calibration")
    saved[name] = calibration.entry()
    settings.set("camera_calibration", saved)
    settings.save()


class CalibrationSession:
    """Taking pictures of the board from a live camera. Corners are looked
    for on the session's own thread, on the newest frame offered."""

    def __init__(self, pattern, square=(1.0, 1.0)):
        self.pattern = tuple(pattern)
        self.square = tuple(square)      # mm; may change before calibrate()
        self.views = []              # corner sets taken
        self.size = None             # (width, height) of the frames
        self.latest = (None, None)   # (frame, its corners or None)
        self.result = None           # a Calibration, after calibrate()
        self.busy = False
        self.error = ""
        self._offered = None
        self._wake = threading.Event()
        self._closed = False
        threading.Thread(target=self._find, name="handlab-calibration", daemon=True).start()

    def offer(self, rgb):
        """A new frame to look for the board in (the newest wins)."""
        self._offered = rgb
        self._wake.set()

    def take(self):
        """Keep the current view. Returns why not, or ""."""
        frame, corners = self.latest
        if corners is None:
            return "no board in view"
        size = (frame.shape[1], frame.shape[0])
        if self.size not in (None, size):
            return "the capture size changed - clear and start again"
        for view in self.views:
            if np.linalg.norm(view - corners, axis=-1).mean() < SAME_VIEW_PX:
                return "the same view again - move or tilt the board"
        self.views.append(corners)
        self.size = size
        self.result = None
        return ""

    def undo(self):
        if self.views:
            self.views.pop()
            self.result = None

    def clear(self):
        self.views, self.size, self.result = [], None, None

    def calibrate(self):
        """Calibrate from the views taken, on a thread; see busy/result/error."""
        if self.busy or len(self.views) < MIN_VIEWS:
            return
        self.busy, self.error = True, ""
        threading.Thread(target=self._calibrate,
                         args=(list(self.views), self.size, self.square),
                         daemon=True).start()

    def close(self):
        self._closed = True
        self._wake.set()

    def _calibrate(self, views, size, square):
        try:
            self.result = calibrate(views, self.pattern, size, square)
        except cv2.error as failure:
            self.error = f"calibration failed: {failure.err}"
        finally:
            self.busy = False

    def _find(self):
        while not self._closed:
            self._wake.wait()
            self._wake.clear()
            frame = self._offered
            if frame is not None and not self._closed:
                self.latest = (frame, find_corners(frame, self.pattern))
