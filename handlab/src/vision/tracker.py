"""Hand tracking on camera frames: MediaPipe's hand landmarker.

    tracker = HandTracker(settings.section("vision"))
    hands = tracker.track(rgb, timestamp_ms)     # list of Hand
    tracker.close()

Each Hand has 21 landmarks in MediaPipe's order (0 wrist, 1-4 thumb, 5-8
index, 9-12 middle, 13-16 ring, 17-20 pinky; the last of each finger is its
tip), twice over:

    image   (21, 3) x, y in 0..1 of the image width and height, z depth
            relative to the wrist (same scale as x, smaller = nearer)
    world   (21, 3) metres, origin at the hand's centre - shape, not where
            the hand is in the room

It runs in VIDEO mode: it follows hands from frame to frame instead of
searching every frame from scratch, so timestamps must keep increasing and
one tracker serves one stream of frames, from one thread.

Frames go in as the camera took them, never mirrored, so landmarks are in
the true image (what is recorded). `side` is MediaPipe's label as it comes,
and on a true image it is the person's own hand: measured 2026-10-22, a left
hand recorded with the MacBook camera came out "Left" (the older MediaPipe
Hands guide says labels assume a mirrored image; this landmarker does not).
"""

from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .. import paths

LANDMARKS = 21
TIPS = (4, 8, 12, 16, 20)
SIDES = ("Left", "Right")
# Pairs of landmark indices, drawn as the hand's skeleton.
CONNECTIONS = tuple((c.start, c.end)
                    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS)


@dataclass(frozen=True)
class Hand:
    side: str              # "Left" or "Right": the person's own hand
    score: float           # how sure the model is of `side`, 0..1
    image: np.ndarray      # (21, 3), see the module docstring
    world: np.ndarray      # (21, 3) metres


class HandTracker:
    def __init__(self, config):
        """`config` is the vision settings section: model, max_hands,
        min_detection, min_tracking."""
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(paths.resolve(config["model"]))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config["max_hands"],
            min_hand_detection_confidence=config["min_detection"],
            min_hand_presence_confidence=config["min_tracking"],
            min_tracking_confidence=config["min_tracking"])
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def track(self, rgb, timestamp_ms):
        """The hands in `rgb` (H x W x 3 uint8, RGB order, not mirrored)."""
        timestamp_ms = max(int(timestamp_ms), self._last_ms + 1)
        self._last_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return [Hand(side=handedness[0].category_name,
                     score=float(handedness[0].score),
                     image=_points(landmarks),
                     world=_points(world))
                for landmarks, world, handedness in zip(
                    result.hand_landmarks, result.hand_world_landmarks,
                    result.handedness)]

    def close(self):
        self._landmarker.close()


def _points(landmarks):
    return np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
