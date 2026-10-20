"""Watch what the teacher decided, in the MuJoCo viewer.

A metrics table says the offline solve is smoother. It does not say whether the
hand it produced looks like the hand that was recorded, and no table can - the
whole complaint that started this work was that the motion did not look right,
which is a judgement made with eyes.

So this plays a solved session back: a row of hands, one per mapping, the
operator's own skeleton standing beside them, at the rate it was recorded.

Driven by `learn.py replay`; nothing here parses arguments.

MJPYTHON, NOT PYTHON. macOS requires the viewer to own the main thread, exactly
as `main.py` does - the same message and the same fix.

ONE LANE PER MAPPING, every colour meaning one thing - see LANE_STYLE. The
teacher is green, the student violet, the live solver grey, a forward head
coral. Amber is not used here at all: on the bench it means the real robot's
encoders, and a colour that means two things means neither. Where the lanes
separate is the headroom, visible rather than tabulated.
"""

from __future__ import annotations

import time
from pathlib import Path

import json
import numpy as np

from ..robot import glyphs
from .sessions import DATA, sessions_under

from ..robot.mjcf import (FUTURE, FUTURE_RGBA, LIVE, LIVE_RGBA, STUDENT,
                     STUDENT_RGBA, TEACHER, TEACHER_RGBA)
from ..robot.layout import Row
from ..robot.options import Scene
from ..robot.sim import HandSim

# The operator's hand now has a lane of its own rather than floating above a
# robot, so it needs no lift by default. Pass one to raise it if a particular
# comparison reads better stacked.
SKELETON_OFFSET = (0.0, 0.0, 0.0)

# How far apart the lanes stand, in metres. Flat and in a row rather than
# superimposed: three mappings on top of each other interpenetrate and none is
# readable, and the whole reason to look at this instead of a metrics table is
# to SEE which one is wrong.
#
# Wide enough for the widest nameplate. "TEACHER" is 27 glyph columns, and at
# the old 0.22 the words ran into each other.
OBJECT_RGBA = (0.85, 0.70, 0.25, 0.55)   # the held object: amber glass, see-through so the fingers behind it show

# Amber is the real robot's encoders, here and everywhere else in this project.
# It is NOT lent to whichever mapping is on screen: nothing in a replay is an
# encoder reading, so nothing in a replay is amber, and the colour keeps
# meaning one thing for when hardware is attached.
# How far a student's own idea of the average pose may sit from the teacher's
# before it is called stale. A fresh model lands within a couple of degrees;
# the one trained before the rest alignment sat 6 degrees out.
STALE_DEGREES = 4.0

# A deeper blue than the pale one this used to be: the pale version sat
# 0.33 from the live lane's grey, close enough that the check below calls
# them the same colour, and "it is drawn as sticks so you can tell" is an
# excuse rather than a fix.
SKELETON_RGBA = "0.30 0.55 1.00 0.95"

# EVERYTHING A LANE IS, IN ONE TABLE: the MJCF prefix its copy is built under,
# the colour it renders in, the word written on the floor beneath it, and what
# the terminal legend calls that colour.
#
# One table because three separate lookups is how the future lane came out
# green. Its name was matched against a dict of the OTHER lanes and fell
# through to a default of `TEACHER` - so it was built under the teacher's
# prefix, in the teacher's colour, while the legend printed "coral" from a
# different fallback and agreed with nobody.
#
# There is deliberately NO default here. A lane whose key is missing raises
# rather than quietly becoming another lane.
LANE_STYLE = {
    "you":     (None,    SKELETON_RGBA, "YOU",     "blue skeleton"),
    "teacher": (TEACHER, TEACHER_RGBA,  "TEACHER", "green"),
    "student": (STUDENT, STUDENT_RGBA,  "STUDENT", "violet"),
    "future":  (FUTURE,  FUTURE_RGBA,   "AHEAD",   "coral"),
    "live":    (LIVE,    LIVE_RGBA,     "LIVE",    "grey"),
    # Amber stays the real robot's encoders, here as everywhere else in this
    # project. Nothing in a replay is an encoder reading, so nothing in a
    # replay is amber and the colour keeps meaning one thing.
}


def style(key):
    """(prefix, rgba, floor tag, colour name) for one lane. Raises if unknown."""
    if key not in LANE_STYLE:
        raise SystemExit(f"no style for lane {key!r} - add it to LANE_STYLE "
                         f"rather than letting it default to another lane")
    return LANE_STYLE[key]


def rgba_tuple(text):
    return tuple(float(value) for value in text.split())


def student_rollout(path, keypoints, raw, bounds, device=None, horizon=0):
    """Run a trained student causally over a session.

    Returns (q, valid, joint_names, teacher_mean) - FOUR values. It returned
    two once, and the gesture probe went on unpacking two for long enough that
    `selftest.test_call_sites_match_signatures` exists because of it.

    Both views go in - the aligned keypoints the solver was sent and the raw
    ones the distances are read from - because that is what the student was
    trained on; see `student.features`.

    Causally, and frame by frame from a sliding window, because that is the
    only thing the live path can do - and watching a model replayed any other
    way tells you about a model that cannot exist.

    The first `window - 1` frames of every run have no full history. They come
    back invalid rather than padded: padding invents a hand that was not there,
    and on the bench those frames have not happened yet either.
    """
    import torch

    from .student import Student, feature_width
    from .train import run_windows, sliding

    blob = torch.load(path, map_location="cpu", weights_only=False)
    # THE CHECKPOINT SAYS WHAT IT IS. Comparing against the module's own
    # `FEATURES` asked "were you trained with the flags this process happens
    # to have set", which is a different question and reported the answer as
    # "you predate the distance features".
    trained_belief = bool(blob.get("belief"))
    if blob.get("features") != feature_width(trained_belief):
        raise SystemExit(f"{path} predates the distance features - "
                         f"rerun `learn.py train`")
    model = Student(blob["mean"], blob["basis"], channels=blob["channels"],
                    belief=trained_belief,
                    authority=bool(blob.get("authority")))
    if trained_belief:
        from ..paths import NOISE_FIT
        from .belief import Belief
        model.belief = Belief.load(NOISE_FIT)
        model.belief.gate = bool(blob.get("contact_gate"))
    model.load_state_dict(blob["state"])
    length = blob["window"]
    horizons = tuple(blob.get("horizons", (0,)))
    if horizon not in horizons:
        raise SystemExit(f"this student has no {horizon}-frame head; "
                         f"it was trained with {horizons}")
    head = horizons.index(horizon)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()

    q = np.zeros((len(keypoints), len(blob["mean"])), dtype=np.float32)
    valid = np.zeros(len(keypoints), dtype=bool)
    for start, end in bounds:
        if end - start < length:
            continue
        aligned, raw_windows = sliding(keypoints, raw, start, end, length)
        q[start + length - 1:end] = run_windows(model, aligned, raw_windows,
                                                device, head)
        valid[start + length - 1:end] = True
    reference = blob.get("teacher_mean")
    return (q, valid, list(blob["joint_names"]),
            None if reference is None else np.asarray(reference))


def solved(path):
    """A session's teacher output, or a synthetic pair file. Same shape either
    way: keypoints in, joint vectors out, and the run bounds that say where the
    discontinuities are."""
    path = Path(path)
    for name in ("teacher.npz", "pairs.npz"):
        if (path / name).exists():
            with np.load(path / name) as handle:
                return {key: handle[key] for key in handle.files}, name
    raise SystemExit(f"{path} has no teacher.npz or pairs.npz - "
                     f"run `learn.py teacher` or `learn.py synth` on it first")




def unscaled(points, digit_scale):
    """Undo the per-digit stretch, giving the OPERATOR's own hand shape back.

    WHY THE DRAWN HAND IS NOT THE SOLVED HAND. `fit_proportions` stretches each
    digit so the solver is asked for a target its own geometry can hold, and on
    this pair of hands that is not a uniform stretch. `h4nd_left` reaches
    200 mm to its thumb tip at rest and 150 to 153 to its fingers - a thumb
    LONGER than the fingers, where this operator's is shorter at 101 mm
    against 119 to 127. So the thumb goes up 1.98x while the fingers go up
    1.18 to 1.28, and the pinky, having no robot counterpart, is left alone.

    THAT RATIO IS WHY A MILLIMETRE IS NOT A MILLIMETRE in the scaled view: an
    angular error on the thumb arrives 1.6x larger than the same error on the
    middle finger. Anything comparing digits after the stretch has to divide
    it back out first - see `selftest.test_synthetic_and_real_share_a_frame`,
    which read 70 mm on the thumb against 45 on the middle for one population
    that is 35 and 38 per unit of scale.

    Drawn straight, that hand has a thumb 40 percent longer relative to its
    fingers than the operator's really is - a thumb-to-index length ratio of
    0.727 against a true 0.521 - and it reads as a broken thumb rather than as
    what it is. The lane is labelled "you", so it shows you.
    """
    points = np.array(points, dtype=float, copy=True)
    for slot, digit in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        factor = float(digit_scale.get(digit, 1.0))
        if abs(factor - 1.0) > 1e-9:
            points[:, 4 * slot + 1:4 * slot + 5] /= factor
    return points


def play(target=None, synthetic=None, blocks="", speed=1.0,
         loop=False, robot_hand="left", skeleton_offset=SKELETON_OFFSET,
         student=None, ghost="live", scaled_skeleton=False, future=0,
         video=None, fps=30):
    """Play a solved session in the viewer. Returns when the window closes.

    With `video`, renders the span once to that file instead of opening a
    window - the same offscreen renderer `gallery` uses for a still, driven
    frame by frame into ffmpeg. Everything else is identical, so what lands in
    the file is what the viewer would have shown: same lanes, same nameplates,
    same camera.
    """
    from .sessions import load

    target = synthetic or target
    if target is None:
        found = sessions_under()
        if not found:
            print(f"no sessions under {DATA}")
            return 1
        target = found[-1]
    data, kind = solved(target)
    synthetic = kind == "pairs.npz"

    q_star = data["q_star"]
    q_live = data.get("q_live")
    keypoints = data["keypoints"]
    raw = data.get("keypoints_raw")
    bounds = data["run_bounds"]

    # What the SOLVER ate stays `keypoints`; what gets DRAWN is the operator's
    # own hand unless asked otherwise. `--scaled-skeleton` shows the solver's
    # input instead, which is the view to use when asking why it did something.
    drawn = keypoints
    if not scaled_skeleton and "digit_scale" in data and "fingers" in data:
        drawn = unscaled(keypoints,
                         dict(zip((str(f) for f in data["fingers"]),
                                  data["digit_scale"])))

    # Every mapping that has something to show, in the order they stand in.
    # Lane 0 is the operator, drawn as a skeleton; lane 1 is the simulated
    # hand; everything past that is a posed copy.
    #
    # ALL OF THEM AT ONCE, rather than one solid against one ghost. Two hands
    # forced a choice about which comparison was on screen, and the amber ghost
    # had to stand in for whichever mapping was not being driven - which
    # quietly reused the colour this project means the REAL ROBOT'S ENCODERS
    # by. A colour that means two things means neither.
    valid = np.ones(len(q_star), dtype=bool)
    lanes = [("you", "you", None), ("teacher", "teacher", q_star)]
    if student:
        if raw is None:
            raise SystemExit("this solve has no raw view for the student to "
                             "read distances from - rerun `learn.py teacher`")
        q_student, valid, stored, mean = student_rollout(student, keypoints,
                                                         raw, bounds)
        if stored != [str(n) for n in data["joint_names"]]:
            raise SystemExit("the student was trained on a different joint "
                             "order than this session was solved in")
        # IS THIS STUDENT STILL ABOUT THIS TEACHER? `teacher_mean` is the
        # average pose the solve had when this student trained, so comparing it
        # against the solve on disk answers that without trusting a timestamp.
        #
        # It has to be that and not the PCA mean, which is fitted over the
        # synthetic set too and so sits several degrees from any one session
        # even when nothing is wrong - a guard that cries wolf is worse than no
        # guard, because it gets read past.
        #
        # Worth a warning because the failure is quiet and convincing: a
        # student trained before the rest-pose alignment landed drew a
        # perfectly steady hand with its fingers splayed 15 degrees, beside a
        # teacher that had been fixed hours earlier - and it looked like a
        # model that had learned the wrong thing rather than a stale file.
        #
        # ONLY AGAINST A SESSION. `teacher_mean` is written as
        # `real["q_star"].mean(axis=0)` - the REAL session's average pose - so
        # the comparison only means anything on a session replay. Measured
        # against this student: 0.00 deg on the session it trained on, 10.77 on
        # `synthetic/seed0`, which it ALSO trained on, and 10.32 on the authored
        # gestures. A synthetic set's pose distribution is a different
        # distribution, not a stale solve, so the guard fired on every
        # `--synthetic` replay and told the reader to retrain against the very
        # data the model was trained on. That is the wolf this comment warns
        # about, so the check is scoped to where its reference exists.
        drift = (None if mean is None or synthetic else
                 float(np.degrees(np.abs(mean - q_star.mean(axis=0)).mean())))
        if drift is not None and drift > STALE_DEGREES:
            print(f"[replay] WARNING: this student is {drift:.1f} deg from the "
                  f"teacher on disk, which is more than a retrain should "
                  f"leave.\n           It was probably trained against an "
                  f"older solve - rerun `learn.py train`.")
        lanes.append(("student", "student", q_student))
        if future:
            # The SAME network, read from a different head. It leads the other
            # lanes by `future` frames on purpose - that is the whole claim, and
            # a lane that did not run ahead of the others would be showing
            # nothing.
            ahead, ahead_valid, _, _ = student_rollout(student, keypoints, raw,
                                                       bounds, horizon=future)
            valid = valid & ahead_valid
            lanes.append(("future", f"student +{future}f", ahead))
    if q_live is not None and ghost != "none":
        # The row already stands the teacher beside the student, so this only
        # decides whether the live solver joins them. It used to offer
        # "teacher" as a third choice, which did exactly what "live" did.
        lanes.append(("live", "live solver", q_live))

    pick = valid.copy()
    if blocks and "block" in data:
        # Block ids are indices into the recording's own block table, so the
        # names have to come from the session that wrote them.
        _, metas = load([Path(target)])
        wanted = {metas[0]["blocks"].index(k)
                  for k in blocks.split(",") if k}
        pick &= np.isin(data["block"], list(wanted))
        if not pick.any():
            print(f"no frames in blocks {blocks}")
            return 1

    hz = 125.0                    # only for a pair file with no timestamps:
                                  # synthetic sets are generated at `--hz`
    if "t" in data and len(data["t"]) > 1:
        hz = 1.0 / max(float(np.median(np.diff(data["t"]))), 1e-6)

    # The SIMULATED hand stands in lane 1, so moving it moves the row and every
    # posed copy comes with it. `layout.Row` owns where each lane lands - see
    # `robot/layout.py`.
    prefixes = {key: style(key)[0] for key, _, _ in lanes[2:]}
    row = Row(len(lanes), robot_hand)
    camera_pos, camera_lookat = row.camera(back=0.95, lift=0.52, look=0.16)
    sim = HandSim(hand=robot_hand, scene=Scene(
        position=row.anchor, ghost=False, main_rgba=TEACHER_RGBA,
        copies=row.copies([(style(key)[0], style(key)[1])
                           for key, _, _ in lanes[2:]]),
        camera_pos=camera_pos, camera_lookat=camera_lookat,
        show_viewer=video is None))
    sim.build()

    # The stored joint order is the wire format. Reordering by name rather than
    # trusting the two to coincide keeps this correct if the URDF is ever
    # re-exported with the joints shuffled - the failure otherwise is the ring
    # finger's angle on the thumb, silently.
    stored = [str(n) for n in data["joint_names"]]
    order = [stored.index(n) for n in sim.joint_names]

    # The keypoints were solved in the robot's own axes and metres, wrist at the
    # origin, so placing them is one translation - no second convention.
    # Lane 0. The operator stands BESIDE the robots now rather than above one
    # of them: it is being compared against all of them at once.
    origin = (np.array(row.place(0))
              + np.asarray(skeleton_offset, dtype=float))

    # The objects the hand was solved holding (`objects.json` beside the
    # pairs: see `gestures.cylinder_to_robot`), drawn as capsules in every
    # robot lane on the frames they are present. The teacher held them; the
    # student is meant to; a wrap without the thing being wrapped is
    # unreadable.
    objects = []
    listing = Path(target)
    listing = listing / "objects.json" if listing.is_dir() else listing.parent / "objects.json"
    if synthetic and listing.exists():
        objects = json.loads(listing.read_text())
        print(f"[replay] {len(objects)} object{'s' if len(objects) != 1 else ''} in the scene")
    robot_lanes = [np.array(row.place(k))
                   for k in range(1, len(lanes))]

    frames = np.flatnonzero(pick)
    print(f"[replay] {target}")
    print(f"[replay] {len(frames)} frames @ {hz:.0f} Hz"
          f"{' (synthetic)' if synthetic else ''}")
    # A legend in the terminal, because MuJoCo's passive viewer offers no way
    # to draw text into the scene. The bodies themselves ARE named per lane, so
    # switching on body labels in the viewer's rendering panel puts the names
    # on the hands.
    print("[replay] laid out left to right:")
    for index, (key, title, _) in enumerate(lanes):
        print(f"           lane {index}  "
              f"{row.place(index)[1]:+5.2f} m   "
              f"{title:<14} {style(key)[3]}")
    print("[replay] close the viewer window to stop.")

    # The lane names, written on the floor once and redrawn every frame
    # because `draw_debug` rebuilds the overlay from scratch. Colour alone was
    # not enough to tell the lanes apart - two of them read as the same green -
    # and a word does not have that failure mode.
    # Nameplates stand upright in front of the row, sized so the widest one
    # fits inside a lane. Auto-scaled rather than a fixed cell, because the
    # word lengths are not fixed and overlapping labels are worse than small
    # ones.
    widest = max(glyphs.cells(style(key)[2])[1] for key, _, _ in lanes)
    cell = min(glyphs.CELL, row.spacing * 0.82 / (widest * 1.4))
    base = sim.position[2] - 0.20 + glyphs.LIFT * cell * 1.4
    nameplates = []
    for index, (key, _, _) in enumerate(lanes):
        _, rgba, tag, _ = style(key)
        nameplates.append((tag, (0.10, row.place(index)[1], base),
                           rgba_tuple(rgba), cell))

    # OFFSCREEN, when a file was asked for: one renderer, one camera built from
    # the same framing the viewer would use, and ffmpeg fed raw frames. The
    # loop below is the viewer's loop unchanged - only the drawing target and
    # the pacing differ, so a video cannot drift from what the window shows.
    writer = renderer = camera = None
    if video is not None:
        import subprocess

        import mujoco
        width, height = 1280, 720
        renderer = mujoco.Renderer(sim.mj, height=height, width=width,
                                   max_geom=20000)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = sim.camera_lookat
        offset = np.asarray(sim.camera_pos) - np.asarray(sim.camera_lookat)
        camera.distance = float(np.linalg.norm(offset))
        camera.azimuth = float(np.degrees(np.arctan2(-offset[1], -offset[0])))
        camera.elevation = -float(np.degrees(
            np.arcsin(offset[2] / camera.distance)))
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             str(video)], stdin=subprocess.PIPE)
        loop = False

    started = time.perf_counter()
    at = 0
    while sim.viewer_alive():
        index = int(frames[at])
        sim.command(lanes[1][2][index][order])
        for key, _, values in lanes[2:]:
            sim.set_copy(prefixes[key], values[index][order])
        sim.step()

        # The operator's hand: 21 keypoints as five chains from the wrist, which
        # is what `draw_debug` wants and what the recording already is.
        points = drawn[index]
        skeleton = {}
        for slot, digit in enumerate(("thumb", "index", "middle", "ring",
                                      "pinky")):
            chain = [points[0]] + [points[4 * slot + 1 + k] for k in range(4)]
            skeleton[digit] = [(np.asarray(p) + origin).tolist() for p in chain]
        capsules = []
        for item in objects:
            first, last = item.get("frames", (0, len(q_star)))
            if not first <= index < last:
                continue
            for lane_origin in robot_lanes:
                capsules.append((np.asarray(item["p0"]) + lane_origin,
                                 np.asarray(item["p1"]) + lane_origin,
                                 float(item["radius"]), OBJECT_RGBA))
        drawing = dict(skeleton=skeleton, extra=nameplates,
                       skeleton_rgba=rgba_tuple(SKELETON_RGBA),
                       capsules=capsules)
        if writer is None:
            sim.draw_debug(**drawing)
        else:
            renderer.update_scene(sim.data, camera=camera)
            sim.draw_debug(**drawing, scene=renderer.scene)
            writer.stdin.write(renderer.render().tobytes())

        at += 1
        if at >= len(frames):
            if not loop:
                break
            at, started = 0, time.perf_counter()
        if speed > 0 and writer is None:
            # Pace against the wall clock rather than sleeping a fixed amount,
            # so a slow draw does not compound into drift over a long session.
            # Rendering to a file has no clock to keep - `fps` sets the pace.
            due = started + at / (hz * speed)
            while time.perf_counter() < due and sim.viewer_alive():
                time.sleep(0.001)

    if writer is not None:
        writer.stdin.close()
        writer.wait()
        print(f"[replay] wrote {video}  ({at} frames at {fps} fps)")
    sim.close()
    return 0
