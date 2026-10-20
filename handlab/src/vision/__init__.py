"""A camera watching the human hand, with hand tracking on every frame.

    camera.py   the camera thread: capture -> track -> overlay -> latest()
    tracker.py  MediaPipe's hand landmarker: 21 landmarks per hand
    overlay.py  drawing the tracked hands onto the frame
    devices.py  camera names in OpenCV's numbering
    permission.py  asking macOS for camera access
"""

from .camera import Camera, Frame
from .tracker import Hand, HandTracker

__all__ = ["Camera", "Frame", "Hand", "HandTracker"]
