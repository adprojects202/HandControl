"""WiLoR on FreiHAND's photos -> what a camera's tracker makes of hands whose
true 3D shape is known: one half of the intent pairs (src/learning/
intent_pairs.py adds the robot pose each hand means).

    ../.venv-wilor/bin/python wilor/freihand.py

FreiHAND (Zimmermann et al., ICCV 2019; research use only):

    training     data/external/freihand_full: 32,560 poses, each
                 photographed in front of a green screen and then given
                 other backgrounds - training/rgb holds the green-screen
                 photos first, then 3 versions' 32,560 in turn: 1 another
                 background, 2 and 3 it blurred and its colours washed.
                 This runs version VERSION, as a camera would see a hand
    evaluation   data/external/freihand: 3,960 photos with real
                 backgrounds, a different setting - for testing

Each photo is padded like an uncropped view (PAD of its size each side) so
the detector finds the hand as it would in a frame. Resumes where it
stopped (every SAVE_EVERY photos it saves).

    data/learning/intent/freihand_wilor_<set>.npz
        joints (N, 21, 3) m: WiLoR's hand, camera axes (NaN where none was
        found); uv (N, 21, 2) px in the padded photo; found (N,); right (N,)
        the detector's side; tried (N,); version; made - row n is the set's
        pose n (its ground truth's row)
"""

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from model import detect, hand, load

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
OUT = ROOT / "data" / "learning" / "intent"
VERSION = 1
SETS = {"evaluation": (EXTERNAL / "freihand" / "evaluation" / "rgb", 3960, 0),
        "training": (EXTERNAL / "freihand_full" / "training" / "rgb", 32560, VERSION)}
PAD = 0.5
SAVE_EVERY = 1000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, cfg, detector = load()
    for name, (photos, poses, version) in SETS.items():
        run(model, cfg, detector, photos, poses, version, OUT / f"freihand_wilor_{name}.npz")


def run(model, cfg, detector, photos, poses, version, out):
    done = dict(np.load(out)) if out.exists() else None
    joints = done["joints"] if done else np.full((poses, 21, 3), np.nan, np.float32)
    uv = done["uv"] if done else np.full((poses, 21, 2), np.nan, np.float32)
    found = done["found"] if done else np.zeros(poses, bool)
    right = done["right"] if done else np.full(poses, np.nan, np.float32)
    tried = done["tried"] if done else np.zeros(poses, bool)
    arrays = dict(joints=joints, uv=uv, found=found, right=right, tried=tried)
    started, count = time.monotonic(), 0
    for pose in np.flatnonzero(~tried):
        bgr = cv2.imread(str(photos / f"{version * poses + pose:08d}.jpg"))
        pad = round(PAD * bgr.shape[0])
        bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        hands = detect(detector, bgr)
        if hands:
            _, is_right, box = max(hands, key=lambda h: h[0])
            joints[pose], uv[pose] = hand(model, cfg, bgr, box, is_right)
            found[pose], right[pose] = True, is_right
        tried[pose] = True
        count += 1
        if count % SAVE_EVERY == 0:
            _save(out, arrays, version)
            rate = (time.monotonic() - started) / count
            print(f"{out.stem}: {tried.sum()} of {poses}, {rate * 1000:.0f} ms each, "
                  f"~{(poses - tried.sum()) * rate / 60:.0f} min left", flush=True)
    _save(out, arrays, version)
    print(f"{out.stem}: hands found in {found.mean():.1%} of {poses} photos", flush=True)


def _save(out, arrays, version):
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays, version=np.array(version),
             made=np.array(datetime.now().isoformat(timespec="seconds")))
    tmp.replace(out)


if __name__ == "__main__":
    main()
