"""WiLoR on the recordings: every recording's video (data/tasks/<task>/<stamp>/
video.mp4) read as live.py reads the webcam - the hand found by the detector
every DETECT_EVERY frames or when lost, followed between by the box around
last frame's points - into hands_wilor.npz beside it. Recordings that have
one newer than their video are skipped.

    ../.venv-wilor/bin/python wilor/recording.py

The hand followed is the one the recording's own steps followed (Real to
Sim's hand_sim.npz names its side; hands_clean.npz, else hands.npz, has
where MediaPipe saw it): each frame the detector's hand nearest it, and
WiLoR's crop found again whenever it strays from it. A recording without
them follows the hand found most surely, keeping its side. Nothing is
smoothed: what the reader (src/learning/reader.py) is given live.

    hands_wilor.npz
        time (N,) s, the video's frames' (frames.npz); world (N, 21, 3) m in
        camera axes, NaN where no hand; uv (N, 21, 2) px; side (N,) "Left",
        "Right" or ""; found (N,); made
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from live import DETECT_EVERY, FOLLOW_MARGIN
from model import detect, hand, load

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("Left", "Right")        # MediaPipe's slots in hands.npz (src/vision/tracker.py)
TASKS = ROOT / "data" / "tasks"
OUT = "hands_wilor.npz"


def main():
    folders = [video.parent for video in sorted(TASKS.glob("*/*/video.mp4"))
               if not (video.parent / OUT).exists()
               or (video.parent / OUT).stat().st_mtime < video.stat().st_mtime]
    if not folders:
        print("every recording has its WiLoR hands")
        return
    model, cfg, detector = load()
    for folder in folders:
        _run(model, cfg, detector, folder)


def _run(model, cfg, detector, folder):
    time = np.load(folder / "frames.npz")["time"]
    video = cv2.VideoCapture(str(folder / "video.mp4"))
    world = np.full((len(time), 21, 3), np.nan, np.float32)
    uv = np.full((len(time), 21, 2), np.nan, np.float32)
    side = np.full(len(time), "", dtype="<U5")
    followed, kept = None, None
    guide = None
    for frame in range(len(time)):
        ok, bgr = video.read()
        if not ok:
            break
        if guide is None:
            guide = _guide(folder, len(time), bgr.shape[1], bgr.shape[0])
        strayed = (followed is not None and np.isfinite(guide[frame]).all()
                   and np.abs((followed[0][:2] + followed[0][2:]) / 2 - guide[frame]).max()
                   > (followed[0][2:] - followed[0][:2]).max() / 2)
        if followed is None or strayed or frame % DETECT_EVERY == 0:
            found = detect(detector, bgr)
            if np.isfinite(guide[frame]).all():
                pick = min(found, key=lambda f: np.linalg.norm((f[2][:2] + f[2][2:]) / 2 - guide[frame]),
                           default=None)
            else:
                same = [f for f in found if kept is None or bool(f[1]) == kept]
                pick = max(same or found, key=lambda f: f[0]) if found else None
            followed = (pick[2], bool(pick[1])) if pick else None
        if followed is None:
            continue
        box, is_right = followed
        world[frame], uv[frame] = hand(model, cfg, bgr, box, is_right)
        side[frame], kept = ("Right" if is_right else "Left"), is_right
        height, width = bgr.shape[:2]
        low, high = uv[frame].min(0), uv[frame].max(0)
        middle, half = (low + high) / 2, (high - low).max() / 2 * FOLLOW_MARGIN
        inside = (high > 0).all() and (low < [width, height]).all() and half > 10
        followed = (np.r_[middle - half, middle + half], is_right) if inside else None
    video.release()
    found = np.isfinite(world).all(axis=(1, 2))
    np.savez(folder / OUT, time=time, world=world, uv=uv, side=side, found=found,
             made=np.array(datetime.now().isoformat(timespec="seconds")))
    sides = ", ".join(f"{s} {np.mean(side == s):.0%}" for s in ("Left", "Right") if (side == s).any())
    print(f"{folder.relative_to(TASKS)}: {len(time)} frames, a hand in {found.mean():.0%} ({sides})")


def _guide(folder, count, width, height):
    """(N, 2) px: the middle of the hand the recording's steps followed,
    as MediaPipe saw it - NaN where it did not, or without those steps."""
    guide = np.full((count, 2), np.nan)
    if not (folder / "hand_sim.npz").exists():
        return guide
    slot = SIDES.index(str(np.load(folder / "hand_sim.npz")["side"]))
    hands = np.load(folder / ("hands_clean.npz" if (folder / "hands_clean.npz").exists() else "hands.npz"))
    seen = hands["present"][:, slot][:count]
    points = hands["image"][:count, slot, :, :2] * [width, height]
    guide[:len(seen)][seen] = (points.min(1) + points.max(1))[seen] / 2
    return guide


if __name__ == "__main__":
    main()
