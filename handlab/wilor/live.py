"""WiLoR live on the webcam, to look at: your hand's 21 points drawn on the
video, the thumb tip's distance to the index, middle and ring tips in palm
lengths (green under PINCH - a pinch), and the time a frame takes.

    ../.venv-wilor/bin/python wilor/live.py

The camera is settings.json's vision.camera_index. The detector looks for
the hand every DETECT_EVERY frames or when it is lost; between, the crop
comes from where the hand's points were last frame. Keys: q quits, f toggles
WiLoR's fast mode (skips some backbone blocks), s toggles smoothing.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from model import detect, hand, load

ROOT = Path(__file__).resolve().parents[1]
DETECT_EVERY = 10
FOLLOW_MARGIN = 1.3            # the crop around last frame's points, as a share of their extent
SMOOTH = 0.5                   # weight of the new frame when smoothing
PINCH = 0.25                   # palm lengths
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
         (17, 18), (18, 19), (19, 20)]


def main():
    camera_index = json.loads((ROOT / "settings.json").read_text())["vision"]["camera_index"]
    model, cfg, detector = load()
    camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    smooth, before, followed, frame_number = True, None, None, 0
    while True:
        ok, bgr = camera.read()
        if not ok:
            print("no frame from camera", camera_index)
            break
        start = time.perf_counter()
        frame_number += 1
        detected = followed is None or frame_number % DETECT_EVERY == 0
        if detected:
            found = detect(detector, bgr)
            followed = (lambda b: (b[2], b[1]))(max(found, key=lambda b: b[0])) if found else None
        text = "no hand"
        if followed is not None:
            box, is_right = followed
            joints, uv = hand(model, cfg, bgr, box, is_right)
            if smooth and before is not None:
                joints = SMOOTH * joints + (1 - SMOOTH) * before
            before = joints
            height, width = bgr.shape[:2]
            low, high = uv.min(0), uv.max(0)
            middle, half = (low + high) / 2, (high - low).max() / 2 * FOLLOW_MARGIN
            inside = (high > 0).all() and (low < [width, height]).all() and half > 10
            followed = (np.r_[middle - half, middle + half], is_right) if inside else None
            for a, b in BONES:
                cv2.line(bgr, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), (255, 200, 80), 2)
            palm = np.linalg.norm(joints[9] - joints[0])
            for row, (name, tip) in enumerate((("index", 8), ("middle", 12), ("ring", 16))):
                gap = np.linalg.norm(joints[4] - joints[tip]) / palm
                cv2.putText(bgr, f"thumb-{name}: {gap:.2f}", (10, 60 + 28 * row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (60, 220, 60) if gap < PINCH else (220, 220, 220), 2)
            text = f"{'right' if is_right else 'left'} hand"
        took = time.perf_counter() - start
        cv2.putText(bgr, f"{text}  fast {'on' if model.backbone.skip_blocks else 'off'}  "
                         f"smooth {'on' if smooth else 'off'}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(bgr, f"{took * 1000:.0f} ms ({'detected' if detected else 'followed'})",
                    (10, bgr.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("WiLoR live (q quit, f fast, s smooth)", bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            model.backbone.skip_blocks = not model.backbone.skip_blocks
        if key == ord("s"):
            smooth = not smooth
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
