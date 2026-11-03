"""The reader on your own recordings: WiLoR's hands from a recording's video
(wilor/recording.py -> hands_wilor.npz) read at RATE as live - each tick the
latest frame's hand, or none - by the reader (reader.Reader) and, for
comparison, by the labelling solver (human_retarget.HumanRetarget, each
frame from the last, a lost frame held). There is no true robot pose here:
what is measured is whether the motion is one a hand makes - smooth, at a
person's pace, fingers clear of each other - how far the two agree, and
whether a pinch the camera sees becomes a touch.

    results = read(folder)             # -> folder/read.npz, the numbers
    render(folder, out)                # the video: camera | reader | solver

    read.npz
        time (T,) s; found (T,); reader (T, J) rad, meant (T, INTENTS)
        likelihoods, solver (T, J) rad - joints in the hand model's order;
        pinch (T, 3): the person's thumb tip to the index, middle and ring
        tips as WiLoR gives them, palm lengths; made
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from . import motion_space, reader as reader_module
from .hand_contacts import HandContacts
from .hand_library import HandModel
from .handmotion import RATE, TOUCH_PL, in_palm
from .human_retarget import HumanRetarget

HANDS = "hands_wilor.npz"
OUT = "read.npz"
JUMP = 20                        # degrees in a tick: a jump
PINCH_PL = 0.25                  # the person's thumb tip this near a fingertip: a pinch
TIPS = (4, 8, 12, 16)            # WiLoR's thumb, index, middle, ring tips
VIEW = (320, 320)                # each robot picture, px
CAMERA = "palm view"


def read(folder):
    """Read a recording's WiLoR hands (see the docstring); writes read.npz,
    returns the numbers."""
    hands = np.load(folder / HANDS)
    time, world, side, found = hands["time"], hands["world"], hands["side"], hands["found"]
    ticks = np.arange(time[0], time[-1], 1.0 / RATE)
    latest = np.searchsorted(time, ticks, side="right") - 1
    space = motion_space.MotionSpace(motion_space.load())
    reader = reader_module.Reader(reader_module.load(), space)
    solver = HumanRetarget()
    rest = np.clip(np.zeros(solver.hand.model.nq), solver.hand.low, solver.hand.high)
    reader.reset(rest)
    made, meant, solved, pinch = [], [], [], []
    last = rest
    for frame in latest:
        hand = world[frame] if found[frame] else None
        q, likely = reader.step(hand, str(side[frame]) or "Left")
        made.append(q)
        meant.append(likely)
        if hand is not None:
            last, _ = solver(hand, str(side[frame]), start=last)
            points = in_palm(hand.astype(float), str(side[frame]))
            pinch.append([np.linalg.norm(points[4] - points[t]) for t in TIPS[1:]])
        else:
            pinch.append([np.nan] * 3)
        solved.append(last)
    out = {"time": ticks - ticks[0], "found": found[latest], "reader": np.array(made, np.float32),
           "meant": np.array(meant, np.float32), "solver": np.array(solved, np.float32),
           "pinch": np.array(pinch, np.float32)}
    np.savez(folder / OUT, **out, made=np.array(datetime.now().isoformat(timespec="seconds")))
    return measure(out, reader.thresholds, solver.hand)


def measure(out, thresholds, hand):
    """Per way of reading (reader, solver): speed deg/s, jitter (median
    change of a joint's speed in a tick, deg), jumps (ticks a joint moves
    over JUMP), fingers inside each other, and of the ticks the person
    pinches (WiLoR, under PINCH_PL) the share made a robot touch - and
    the share made one where they do not; how far the two are apart; and
    the reader's meant touches against the person's pinches."""
    contacts = HandContacts(hand)
    pinching = out["pinch"] < PINCH_PL
    seen = np.isfinite(out["pinch"]).all(1)
    results = {"seconds": float(out["time"][-1]), "found": float(out["found"].mean()),
               "person pinches": {f: float(pinching[seen, k].mean()) for k, f in enumerate(("index", "middle", "ring"))}}
    for name in ("reader", "solver"):
        q = out[name].astype(float)
        step = np.degrees(np.abs(np.diff(q, axis=0)))
        accel = np.degrees(np.abs(np.diff(q, 2, axis=0)))
        robot_touch = hand.kinematics.shapes(q)[:, -3:] < TOUCH_PL     # its pinches, palm lengths
        results[name] = {
            "speed_deg_s": float(step.mean() * RATE), "jitter_deg": float(np.median(accel)),
            "jumps": float((step.max(1) > JUMP).mean()), "inside": contacts.share_inside(q[::2]),
            "touch": {f: {"made": _share(robot_touch[seen & pinching[:, k], k]),
                          "false": _share(robot_touch[seen & ~pinching[:, k], k])}
                      for k, f in enumerate(("index", "middle", "ring"))}}
    results["apart_deg"] = float(np.degrees(np.abs(out["reader"] - out["solver"])).mean())
    said = out["meant"][:, :3] > thresholds[:3]
    results["reader meant"] = {f: {"made": _share(said[seen & pinching[:, k], k]),
                                   "false": _share(said[seen & ~pinching[:, k], k])}
                               for k, f in enumerate(("index", "middle", "ring"))}
    return results


def _share(flags):
    return float(flags.mean()) if len(flags) else None


def render(folder, out):
    """The recording at RATE as a video `out` (.mp4): the camera's frame
    with WiLoR's hand drawn, the robot hand as the reader moves it (with
    the touches it means), and as the solver does."""
    import cv2
    import mujoco

    from ..world import BONES

    read_out, hands = np.load(folder / OUT), np.load(folder / HANDS)
    model = _viewed(HandModel())
    data = mujoco.MjData(model)
    video = cv2.VideoCapture(str(folder / "video.mp4"))
    frames = []
    while True:
        ok, bgr = video.read()
        if not ok:
            break
        frames.append(bgr)
    video.release()
    latest = np.searchsorted(hands["time"], read_out["time"] + hands["time"][0], side="right") - 1
    height = VIEW[1]
    # H.264, which browsers and QuickTime play (OpenCV's "mp4v" is MPEG-4 Part 2: they do not).
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), RATE,
                             (round(frames[0].shape[1] * height / frames[0].shape[0]) + 2 * VIEW[0], height))
    thresholds = np.array(reader_module.load()["thresholds"])
    with mujoco.Renderer(model, VIEW[1], VIEW[0]) as renderer:
        for tick, frame in enumerate(latest):
            bgr = frames[min(frame, len(frames) - 1)].copy()
            if hands["found"][frame]:
                uv = hands["uv"][frame]
                for a, b in BONES:
                    cv2.line(bgr, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), (255, 200, 80), 2)
            bgr = cv2.resize(bgr, (round(bgr.shape[1] * height / bgr.shape[0]), height))
            panels = [bgr]
            for name in ("reader", "solver"):
                data.qpos[:] = read_out[name][tick]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, CAMERA)
                panel = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(panel, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if name == "reader":
                    meant = read_out["meant"][tick, :3]
                    for row, (finger, likely, cut) in enumerate(zip(("index", "middle", "ring"), meant, thresholds)):
                        cv2.putText(panel, f"touch {finger} {likely:.2f}", (10, 50 + 22 * row),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (80, 230, 80) if likely > cut else (200, 200, 200), 1)
                panels.append(panel)
            writer.write(np.concatenate(panels, axis=1))
    writer.release()


def _viewed(hand):
    """The hand model with a camera facing its palm, the fingers up - as a
    webcam sees a hand shown to it."""
    import mujoco

    from ..sim.task import hand_spec

    wrist, axes, length = hand.palm
    spec = hand_spec()[0]
    spec.worldbody.add_camera(name=CAMERA, pos=wrist + axes @ np.array([0.9, 0.0, 3.2]) * length,
                              xyaxes=np.r_[-axes[:, 1], axes[:, 0]], fovy=45)
    return spec.compile()
