"""The invariants this pipeline keeps breaking, checked in one place.

WHY THIS FILE EXISTS. Every bug in this package so far has been one of five
things, and none of them was caught by the checks that were already here:

  A FRAME OR A BASIS, skipped or applied twice. The synthetic hands were solved
  ninety degrees out for 150,000 frames; the robot's rest bones were dropped
  into the operator's basis and collapsed the sampled reach from 125 mm to 82;
  the flexion axis was crossed the wrong way round and curled fingers out of
  the back of the hand. A rigid rotation preserves every distance, so no
  distribution or coverage test can see any of them.

  A BOUNDARY CROSSED. Solving across a block seam warm-started a fist from a
  flat hand; measuring jerk across one counted the seam as a 108 degree
  single-frame step and reported the teacher as worse than what it improves on.

  A STALE ARTIFACT compared against a fresh one. A student trained before the
  rest alignment drew a steady, confident, splayed hand beside a teacher fixed
  hours earlier.

  A SIGNATURE CHANGED AT ONE CALL SITE. `student_rollout` grew a fourth return
  value and the gesture probe kept unpacking three.

  A DISTANCE MEASURED IN THE WRONG VIEW. A real pinch is 21 mm from the sensor
  and 136 mm after per-digit scaling and the rest rotation, and the solver
  matched the 136 for a whole recording. Directions survive those transforms;
  distances do not, and nothing had checked which view a number came from.

So the tests are written against those, not against the code's structure. They
are cheap, they run on real recorded data, and `learn.py selftest` is the thing
to run after touching anything in this package.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import tempfile
from pathlib import Path

import numpy as np

from ..paths import MODELS, ROOT
from ..robot.model import FINGERS, HandModel, ensure_urdf
from ..retargeting import Retargeter
from ..retargeting.keyvector import mano_keypoints, rest_offsets
from .sessions import (finish_keypoints, fit_bone_offsets, fit_proportions,
                       from_robot_axes, hands_from, keypoints_for, load,
                       robot_basis, robot_rest_bones, runs_for,
                       sessions_under, to_robot_axes)
from .synthetic import sampler_for

# How far apart two things that should be the same are allowed to sit.
# Worst mean-fingertip gap between generated and recorded, PER UNIT OF THE
# DIGIT'S SCALE - see `test_synthetic_and_real_share_a_frame` for why the
# division is not optional. The four digits sit at 28 to 38, and a reflection
# or an axis swap puts them past 200, so the bound has a wide moat on both
# sides and does not need to be retuned when the hand's proportions change.
FRAME_MM = 60.0
LANE_COLOUR_GAP = 0.35   # rgb distance below which two lanes look alike          # mean fingertip, synthetic against recorded
FLAT_DEGREES = 6.0       # commanded joint angle on a flat hand
ALIGN_DEGREES = 0.5      # a bone after its rest correction


class Failure(Exception):
    pass


def check(name, condition, detail=""):
    if not condition:
        raise Failure(f"{name}: {detail}")
    return f"  ok   {name}" + (f"   ({detail})" if detail else "")


# ------------------------------------------------------------------ frames
def test_student_is_causal(context):
    """Nothing the network says about frame t moves when frame t+1 does.

    THE TEST THAT MAKES THE WINDOW HONEST. Symmetric padding, a centred
    difference, a normalisation that pools over time - each is one line, each
    lets an output read a frame that has not happened live, and none shows up
    as anything but a model that is better offline than on the bench. Bump
    the newest frame of both views; every earlier output, of every head, and
    every earlier feature must be bit-for-bit what it was - and the newest
    output must have changed, or the network is not reading its input.
    """
    import torch

    from .student import FEATURES, Student, features

    rng = np.random.default_rng(0)
    aligned = torch.tensor(rng.normal(size=(2, 20, 21, 3)) * 0.05,
                           dtype=torch.float32)
    raw = torch.tensor(rng.normal(size=(2, 20, 21, 3)) * 0.05,
                       dtype=torch.float32)
    model = Student(np.zeros(16, np.float32), np.eye(16, dtype=np.float32))
    # The pose head is zero-initialised, and a head that says the mean
    # whatever it is shown would pass a causality test for the wrong reason.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    model.eval()
    bumped_aligned, bumped_raw = aligned.clone(), raw.clone()
    bumped_aligned[:, -1] += 0.02
    bumped_raw[:, -1] += 0.02
    with torch.no_grad():
        before = model.predict(aligned, raw)
        after = model.predict(bumped_aligned, bumped_raw)
    earlier_same = all(torch.equal(before[k][:, :-1], after[k][:, :-1])
                       for k in before)
    newest_moved = not torch.allclose(before["q"][:, -1], after["q"][:, -1])
    f_before, f_after = features(aligned, raw), features(bumped_aligned, bumped_raw)
    features_causal = torch.equal(f_before[:, :-1], f_after[:, :-1])
    width = f_before.shape[-1] == FEATURES
    return check("student is causal",
                 earlier_same and newest_moved and features_causal and width,
                 f"earlier frames unchanged: {earlier_same}, newest moved: "
                 f"{newest_moved}, features causal: {features_causal}, "
                 f"{f_before.shape[-1]} features")


def test_basis_roundtrip(context):
    """`to_robot_axes` and its inverse compose to the identity.

    Trivial, and it is the operation two separate bugs got wrong by being
    written out by hand at three different call sites."""
    points = np.random.default_rng(0).normal(size=(50, 21, 3))
    back = from_robot_axes(to_robot_axes(points, context["retarget"]),
                           context["retarget"])
    worst = float(np.abs(points - back).max())
    return check("basis roundtrip", worst < 1e-9, f"worst {worst:.2e}")


def test_basis_is_orthonormal(context):
    basis = robot_basis(context["retarget"])
    error = float(np.abs(basis @ basis.T - np.eye(3)).max())
    return check("robot basis orthonormal", error < 1e-9, f"worst {error:.2e}")


def test_rotation_is_not_a_reflection(context):
    """Operator-to-robot must be a proper rotation. A reflection here draws the
    other hand, and it looks like a plausible hand, so nothing else notices."""
    hand = context["hand"]
    determinant = float(np.linalg.det(context["retarget"].rotation(hand)))
    return check("rotation is proper", abs(determinant - 1.0) < 1e-6,
                 f"det {determinant:+.4f}")


def test_synthetic_and_real_share_a_frame(context):
    """Mean fingertip, generated against recorded, AFTER both are finished.

    This is the one a rotation cannot hide from - see the module docstring.

    DIVIDED BY THE DIGIT'S SCALE, because the finished view is not one space.
    `fit_proportions` multiplies each digit radially by the robot's reach over
    the operator's, and those constants are not close to each other: this hand
    stretches the thumb 1.98x and the middle 1.18x. A raw millimetre threshold
    is therefore 1.7x tighter on the thumb than on the middle finger, and it
    fails on the digit whose robot bone is longest rather than on the digit
    whose frame is wrong.

    That is not hypothetical. `h4nd_left` reaches 179 mm to the thumb tip and
    126 mm to the index, where a hand does the opposite - so the thumb's scale
    went 1.78 to 1.98 with the new URDF and this test failed at 70 mm with
    nothing about the frame having changed. Per scale, all four digits sit at
    28 to 38 mm: one population, no outlier.

    The residual that remains is the sampler covering its fitted range where
    the operator spent the recording near flat, which `curl`-matching the two
    sets collapses from 45 mm to 16 mm on the fingers. A frame error survives
    the division - a reflection moves a fingertip by twice its reach, and no
    scale in the 1.2 to 2.0 range brings that under the bound.
    """
    real, synthetic, scale = context["real"], context["synthetic"], context["scale"]
    worst, where = 0.0, ""
    for name, slot in (("thumb", 4), ("index", 8), ("middle", 12), ("ring", 16)):
        gap = float(np.linalg.norm(real[:, slot].mean(0)
                                   - synthetic[:, slot].mean(0))) * 1000.0
        gap /= scale.get(name, 1.0)
        if gap > worst:
            worst, where = gap, name
    hands = chirality(real), chirality(synthetic)
    return check("synthetic shares the robot's frame",
                 worst < FRAME_MM and hands[0] == hands[1],
                 f"worst {where} {worst:.0f} mm per unit scale, "
                 f"both {'left' if hands[0] < 0 else 'right'}"
                 if hands[0] == hands[1] else
                 f"CHIRALITY: recorded {hands[0]:+d}, generated {hands[1]:+d}")


def chirality(points):
    """+1 or -1: which way round the hand is, from its own three fingertips.

    THE ONE THING A MEAN POSITION CANNOT SEE. Negating the palm-normal axis
    moves no fingertip mean, because every fingertip mean already sits within
    10 mm of zero on that axis - so a mirrored hand passes a position test
    while being the wrong hand entirely. The signed volume spanned by the
    thumb, index and ring tips flips with any reflection and with nothing
    else, which is exactly the discrimination that is wanted.

    NEGATIVE IS LEFT, in the robot's axes: measured on the left-hand recording
    carried through the left retargeter onto `h4nd_left`, which is the left
    hand by construction. Its mirror comes out +1.
    """
    wrist = points[:, 0]
    volume = np.einsum("ij,ij->i",
                       np.cross(points[:, 8] - wrist, points[:, 16] - wrist),
                       points[:, 4] - wrist)
    return 1 if np.median(volume) > 0 else -1


# ------------------------------------------------------------- rest pose
def test_rest_offsets_align_every_bone(context):
    """All three bones of every digit, not just the first.

    The chain was re-integrated while being read, so bone k measured from a
    corrected start to an uncorrected end and only the proximal came out."""
    hand, retarget = context["hand"], context["retarget"]
    robot = robot_rest_bones(context["model"], retarget.joint_names, FINGERS)
    rotation = retarget.rotation(hand)
    points = mano_keypoints(hand, rotation,
                            bone_offsets=rest_offsets(hand, rotation, robot))
    worst, where = 0.0, ""
    for slot, digit in enumerate(("thumb", "index", "middle", "ring")):
        base = 4 * slot + 1
        for k in range(3):
            vector = points[base + k + 1] - points[base + k]
            vector = vector / np.linalg.norm(vector)
            angle = np.degrees(np.arccos(np.clip(vector @ robot[digit][k],
                                                 -1, 1)))
            if angle > worst:
                worst, where = angle, f"{digit} bone {k}"
    return check("rest offsets align every bone", worst < ALIGN_DEGREES,
                 f"worst {where} {worst:.2f} deg")


def test_flat_hand_gives_a_straight_robot(context):
    """A flat operator hand must command near-zero joints.

    The whole point of the rest alignment, and the symptom that found it: the
    fingers were commanded 13 to 19 degrees of adduction, all the same sign, so
    the hand splayed off to one side."""
    teacher, blocks = context["teacher"], context["blocks"]
    if "flat_hold" not in blocks:
        return "  skip flat hand (no flat_hold block)"
    pick = teacher["block"] == blocks.index("flat_hold")
    names = [str(n) for n in teacher["joint_names"]]
    worst, where = 0.0, ""
    for finger in FINGERS:
        index = names.index(f"{finger}_adduction_joint")
        angle = abs(float(np.degrees(teacher["q_star"][pick, index].mean())))
        if finger != "thumb" and angle > worst:      # the thumb has no zero
            worst, where = angle, finger
    return check("flat hand gives a straight robot", worst < FLAT_DEGREES,
                 f"worst {where} {worst:.1f} deg")


# ---------------------------------------------------------------- contact
def test_a_pinch_survives_into_the_raw_view(context):
    """On the frames the sensor calls a pinch, the raw thumb-to-index distance
    must be a pinch, and the aligned one must NOT be - that gap is the bug
    the two views exist to separate. Measured: 21 mm against 136."""
    from .sessions import raw_keypoints_for
    data, retarget = context["data"], context["retarget"]
    tracked = np.flatnonzero(data["tracked"])
    strong = tracked[data["pinch_strength"][tracked] > 0.9][::20]
    if len(strong) < 20:
        return "  skip pinch survives (no strong pinches recorded)"
    raw = np.median([np.linalg.norm(raw_keypoints_for(retarget, data, i)[4]
                                    - raw_keypoints_for(retarget, data, i)[8])
                     for i in strong]) * 1000.0
    aligned = np.median([np.linalg.norm(keypoints_for(retarget, data, i,
                                                      context["scale"], context["offsets"])[4]
                                        - keypoints_for(retarget, data, i,
                                                        context["scale"], context["offsets"])[8])
                         for i in strong]) * 1000.0
    return check("a pinch survives into the raw view",
                 raw < 40.0 and aligned > 2 * raw,
                 f"raw {raw:.0f} mm, aligned {aligned:.0f} mm")


def test_collision_proxy_matches_mujoco(context):
    """The boxes the teacher keeps apart are the meshes MuJoCo collides.

    Random poses through both: the proxy must agree with MuJoCo's signed
    mesh distance on whether a pair is more than 3 mm into another on all
    but a few percent of pair-frames, and read within 2 mm of it around
    touching. A proxy that drifted from the meshes - a re-exported URDF, a
    stale cache - would keep a term that optimises the wrong hand."""
    import torch

    from . import collision
    from .kinematics import ChainFK

    model = context["model"]
    names = model.joint_names()
    fk = ChainFK(model, names)
    lower, upper = model.limits(names)
    proxy = collision.for_hand(fk, lower, upper)
    census = collision.Census(names)
    rng = np.random.default_rng(3)
    q = np.concatenate([rng.uniform(lower, upper, size=(400, len(lower))),
                        collision.tips_close(fk, lower, upper, count=20000, seed=4)[:400]])
    with torch.no_grad():
        _, origins, frames = fk.walk(torch.tensor(q, dtype=torch.float32))
        said = proxy.gaps(origins, frames).numpy()
    lookup = collision.geom_pairs(census.mj, fk, proxy.palm_pairs)
    truth = np.stack([collision.mujoco_gaps(census.mj, census.data, census.qadr,
                                            lookup, row) for row in q])
    agree = np.mean((said < -3.0) == (truth < -3.0))
    near = np.abs(truth) < 3.0
    error = np.median(np.abs(said[near] - truth[near])) if near.any() else 0.0
    return check("collision proxy matches mujoco", agree > 0.97 and error < 2.0,
                 f"agrees on 3 mm penetration {100 * agree:.1f}% of pair-frames, "
                 f"median error {error:.2f} mm near touching")


def test_labels_do_not_self_collide(context):
    """The teacher's labels keep the hand out of itself, by MuJoCo's own
    contacts - measured past as-built, the way the servo guard counts it.
    Before the collision terms this was 42% of the recorded frames."""
    from . import collision, objective

    teacher = context["teacher"]
    names = [str(n) for n in teacher["joint_names"]]
    survey = collision.Census(names).survey(teacher["q_star"][::25],
                                             objective.PENETRATION_MM)
    return check("labels do not self-collide", survey["penetration_pct"] < 5.0,
                 f"{survey['penetration_pct']}% of frames deeper than "
                 f"{objective.PENETRATION_MM:.0f} mm, worst "
                 f"{survey['worst_penetration_mm']} mm")


def test_gesture_probe_plays(context):
    """The probe feeds the live lanes in the shapes they read.

    `GesturePlayback` stands in for the sensor: windows for the student in
    the student's own window length, a teacher pose per frame, a skeleton in
    `Mimic.skeleton`'s shape. Checked against the student on disk, because
    that is what the STUDENT and AHEAD lanes run when the button is pressed.
    """
    from ..leap import DIGITS
    from . import playback
    from .student import WINDOW

    if not playback.available():
        return "  skip gesture probe (no data/synthetic/gestures - run `learn.py gestures`)"
    probe = playback.GesturePlayback()
    probe.index = min(7, len(probe) - 1)
    aligned, raw = probe.windows()
    skeleton = probe.skeleton((0.0, 0.0, 0.2))
    problems = []
    if aligned.shape != (WINDOW, 21, 3) or raw.shape != (WINDOW, 21, 3):
        problems.append(f"windows {aligned.shape} / {raw.shape}")
    if set(skeleton) != set(DIGITS) or any(len(v) != 4 for v in skeleton.values()):
        problems.append("skeleton is not four points per digit")
    if len(probe.teacher()) != len(probe.joint_names):
        problems.append("teacher pose is the wrong width")
    probe.index = len(probe) - 1
    if not probe.label:
        problems.append("no gesture name on the last frame")
    return check("gesture probe plays", not problems,
                 "; ".join(problems) or f"{len(probe)} frames, {len(probe.names)} gestures")


def test_objects_are_placed_from_the_hand(context):
    """The cylinder a synthetic wrap is authored around lands in the robot's
    palm as a function of the operator's hand and nothing else: the same
    hand and radius give the same object, it rests on the palm face, its
    axis runs along the knuckle line, and it sits between the fingers."""
    from . import gestures as authored, objective
    from .kinematics import ChainFK

    names = context["retarget"].joint_names
    fk = ChainFK(context["model"], names)
    palm = authored.palm_of(fk)
    reach = float(np.mean([context["scale"].get(f, 1.0) for f in objective.OTHERS]))
    problems = []
    for radius in (12.0, 20.0, 30.0):
        item = authored.cylinder_in_hand(context["geometry"], radius)
        placed = authored.cylinder_to_robot(item, context["retarget"], reach, palm)
        again = authored.cylinder_to_robot(item, context["retarget"], reach, palm)
        p0, p1 = np.array(placed["p0"]), np.array(placed["p1"])
        centre, axis = (p0 + p1) / 2, (p1 - p0) / np.linalg.norm(p1 - p0)
        if placed != again:
            problems.append(f"R={radius:.0f}: two placements differ")
        rest = (centre[0] - placed["radius"]) * 1000
        if not 1.0 <= rest <= 6.0:
            problems.append(f"R={radius:.0f}: rests {rest:.1f} mm off the palm face")
        if abs(axis[1]) < 0.95:
            problems.append(f"R={radius:.0f}: axis {np.round(axis, 2)} is not along the knuckle line")
        knuckle = np.array(palm["knuckle"])
        if not (knuckle[1] - 0.03 < centre[1] < knuckle[1] + 0.03
                and knuckle[2] < centre[2] < knuckle[2] + 0.04):
            problems.append(f"R={radius:.0f}: centre {np.round(centre * 1000)} mm is not under the fingers")
        if abs(placed["radius"] - radius * 1e-3) > 1e-9:
            problems.append(f"R={radius:.0f}: the radius was scaled")
    return check("objects are placed from the hand", not problems,
                 "; ".join(problems) or "three radii rest on the palm face, along the knuckle line")


def test_teacher_makes_contact(context):
    """The solved labels must touch when the operator touches. The old teacher
    never did - 0% on every block - and that is the whole reason it was
    replaced."""
    from . import collision, objective
    from .kinematics import ChainFK
    teacher, blocks = context["teacher"], context["blocks"]
    if "keypoints_raw" not in teacher:
        return check("teacher makes contact", False,
                     "teacher.npz has no raw view - re-run `learn.py teacher`")
    if "thumb_tap" not in blocks:
        return "  skip teacher contact (no thumb_tap block)"
    pick = np.flatnonzero(teacher["block"] == blocks.index("thumb_tap"))[::4]
    names = [str(n) for n in teacher["joint_names"]]
    fk = ChainFK(context["model"], names)
    proxy = collision.for_hand(fk, *context["model"].limits(names))
    _, _, made, gap, _, _ = objective.score(fk, proxy, teacher["q_star"][pick],
                                         teacher["keypoints"][pick], teacher["keypoints_raw"][pick])
    return check("teacher makes contact", made["all"] > 0.7,
                 f"{100 * made['all']:.0f}% of the operator's touches on thumb_tap, "
                 f"{gap:.1f} mm gap")


def test_geometry_is_one_table(context):
    """The distance cells the objective reads, the ones stored as labels and
    the ones the robot's kinematics produce must all be the same numbers.
    Three places computing 'thumb to index' is three places to disagree."""
    import torch
    from . import geometry, objective
    from . import collision
    from .kinematics import ChainFK
    teacher = context["teacher"]
    names = [str(n) for n in teacher["joint_names"]]
    fk = ChainFK(context["model"], names)
    proxy = collision.for_hand(fk, *context["model"].limits(names))
    rows = np.arange(0, len(teacher["q_star"]), 900)
    raw = torch.tensor(teacher["keypoints_raw"][rows], dtype=torch.float32)
    aligned = torch.tensor(teacher["keypoints"][rows], dtype=torch.float32)
    q = torch.tensor(teacher["q_star"][rows], dtype=torch.float32)
    with torch.no_grad():
        _, d_robot, d_human, _, _, _ = objective.evaluate(fk, proxy, q, aligned, raw)
    table = geometry.pairwise(teacher["keypoints_raw"][rows])[:, geometry.THUMB_ROW]
    human_gap = float(np.abs(table - d_human.numpy()).max())
    # robot cells against the numpy chain, one frame
    qn = teacher["q_star"][rows[1]]
    tip = lambda f: context["model"].link_pose(f"{f}_tip_link", qn, names)[0]
    robot_gap = abs(float(d_robot[1, 0]) - np.linalg.norm(tip("thumb") - tip("index")) * 1000.0)
    stored = ("distances" in teacher and
              float(np.abs(teacher["distances"][rows][:, geometry.THUMB_ROW] - table).max()) < 1e-3)
    ttc_ok = ("time_to_contact" in teacher and bool(np.all(
        (teacher["time_to_contact"][rows] == 0) == (teacher["distances"][rows] < objective.TOUCH_MM))))
    return check("geometry is one table",
                 human_gap < 1e-3 and robot_gap < 1e-3 and stored and ttc_ok,
                 f"objective vs table {human_gap:.1e} mm, torch vs numpy FK {robot_gap:.1e} mm, "
                 f"stored {'ok' if stored else 'MISSING'}, ttc zero iff touching {'ok' if ttc_ok else 'NO'}")


# ------------------------------------------------------------- boundaries
def test_runs_never_cross_a_boundary(context):
    """No run may span two blocks or two sessions."""
    data = context["data"]
    for run in runs_for(data):
        if len(np.unique(data["block"][run])) != 1:
            return check("runs stay inside one block", False,
                         "a run spans two blocks")
        if len(np.unique(data["session"][run])) != 1:
            return check("runs stay inside one session", False,
                         "a run spans two sessions")
        if np.any(np.diff(run) != 1):
            return check("runs are contiguous", False, "a run has a hole")
    return check("runs never cross a boundary", True,
                 f"{len(runs_for(data))} runs")


def test_teacher_bounds_are_within_runs(context):
    """Every stored run boundary must be a real, contiguous stretch - anything
    measuring a difference reads these to know where not to."""
    teacher = context["teacher"]
    rows, bounds = teacher["row"], teacher["run_bounds"]
    for start, end in bounds:
        if np.any(np.diff(rows[start:end]) != 1):
            return check("teacher run bounds are contiguous", False,
                         f"[{start}, {end}) has a hole")
    covered = sum(end - start for start, end in bounds)
    return check("teacher run bounds are contiguous", covered == len(rows),
                 f"{len(bounds)} runs, {covered} frames")


# ------------------------------------------------------------- signatures
def test_call_sites_match_signatures(context):
    """Every caller of these unpacks what the callee returns.

    Cheap, and it is the check that would have caught a fourth return value
    reaching a three-way unpack in a file nobody re-ran."""
    # Imported inside the test, not at module scope. `cli` exposes this file
    # as a subcommand, so a top-level import either way round would be a cycle;
    # lazy in both directions, it is just a test exercising a command line.
    from . import cli, replay

    expected = {replay.student_rollout: 4}
    problems = []
    for function, count in expected.items():
        source = inspect.getsource(cli) + inspect.getsource(replay)
        for line in source.splitlines():
            if function.__name__ + "(" in line and "def " not in line \
                    and "=" in line:
                targets = line.split("=")[0].strip()
                if targets.count(",") + 1 != count and "," in targets:
                    problems.append(line.strip()[:60])
    return check("call sites match signatures", not problems,
                 "; ".join(problems) or "student_rollout")


# ------------------------------------------------------------------ hand
def test_digit_identity(context):
    """The thumb slot holds the thumb: the SDK pads its metacarpal to zero
    length, and no other digit has that."""
    hand = context["hand"]
    lengths = {d: np.linalg.norm(np.diff(hand.bones[d], axis=0), axis=1)
               for d in ("thumb", "index", "middle", "ring", "pinky")}
    thumb_padded = lengths["thumb"][0] < 1.0
    others_not = all(lengths[d][0] > 10.0
                     for d in ("index", "middle", "ring", "pinky"))
    return check("digit identity", thumb_padded and others_not,
                 f"thumb b1 {lengths['thumb'][0]:.1f} mm")


def test_gestures_hold_their_shape(context):
    """Every authored gesture still means what it claims on this template."""
    from . import gestures as authored

    problems = authored.verify(context["sampler"], context["geometry"])
    return check("gestures hold their shape", not problems,
                 "; ".join(problems) or f"{len(authored.GESTURES)} gestures")


def test_lanes_are_distinct(context):
    """Every lane the viewer can draw has its own prefix and its own colour,
    and the colours are far enough apart to tell apart on screen.

    Both halves matter. The future lane once fell through a lookup to the
    TEACHER default and was built under the teacher's prefix in the teacher's
    colour - two identical green hands - while a different fallback printed
    "coral" in the legend. And before that, teal and green were distinct values
    that rendered as the same colour because they shared a green channel.
    """
    from . import replay

    prefixes, colours, problems = {}, {}, []
    for key, (prefix, rgba, tag, name) in replay.LANE_STYLE.items():
        if prefix is not None and prefix in prefixes:
            problems.append(f"{key} shares a prefix with {prefixes[prefix]}")
        prefixes[prefix] = key
        colours[key] = np.array(replay.rgba_tuple(rgba)[:3])
    keys = list(colours)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            gap = float(np.linalg.norm(colours[a] - colours[b]))
            if gap < LANE_COLOUR_GAP:
                problems.append(f"{a}/{b} only {gap:.2f} apart")
    return check("lanes are distinct", not problems, "; ".join(problems)
                 or f"{len(keys)} lanes")


def test_viewer_builds_the_right_colours(context):
    """The colours in LANE_STYLE are the colours the built model carries.

    Checked on the compiled model rather than the XML, because a material can
    override an rgba and nothing in the XML would say so."""
    import mujoco

    from . import replay
    from ..robot.mjcf import ensure_sim_mjcf
    from ..robot.options import Scene

    copies, expect = [], {}
    for offset, key in enumerate(("student", "future", "live"), start=1):
        prefix, rgba, _, _ = replay.style(key)
        copies.append((prefix, rgba, 0.22 * offset))
        expect[prefix] = tuple(round(v, 2) for v in replay.rgba_tuple(rgba))
    teacher = replay.style("teacher")[1]
    path = ensure_sim_mjcf("left", scene=Scene(copies=tuple(copies),
                                               main_rgba=teacher))
    model = mujoco.MjModel.from_xml_path(str(path))

    found, problems = {}, []
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        prefix = next((p for p in expect if name.startswith(p)), None)
        if prefix is None:
            continue
        found.setdefault(prefix, set()).add(
            tuple(round(float(v), 2) for v in model.geom_rgba[index]))
    for prefix, want in expect.items():
        got = found.get(prefix, set())
        if got != {want}:
            problems.append(f"{prefix} is {sorted(got)} not {want}")
    return check("viewer builds the right colours", not problems,
                 "; ".join(problems) or f"{len(expect)} copies")


def test_lanes_stand_where_they_are_announced(context):
    """Every lane's drawn position matches the legend, including the skeleton.

    The skeleton is the one that goes wrong, because it is not a posed copy -
    it is overlay geometry placed by hand, so a change to the row's layout
    moves the four robots and leaves the operator's hand behind at the old
    origin. That is what happened when the lanes were centred: everything
    shifted except the thing being compared against.
    """
    from . import replay
    from ..robot.sim import HandSim

    seen, drawn = {}, {}
    real_init, real_draw = HandSim.__init__, HandSim.draw_debug
    real_alive = HandSim.viewer_alive
    count = {"n": 0}

    def init(self, *args, **kw):
        # The scene is one object now, so the window is switched off by
        # replacing it rather than by overriding a keyword.
        from dataclasses import replace as _replace
        from ..robot.options import Scene
        seen.update(kw)
        kw["scene"] = _replace(kw.get("scene", Scene()), show_viewer=False)
        real_init(self, *args, **kw)

    def draw(self, **kw):
        drawn.update(kw)
        real_draw(self, **kw)

    def alive(self):
        count["n"] += 1
        return count["n"] < 3

    HandSim.__init__, HandSim.draw_debug = init, draw
    HandSim.viewer_alive = alive
    replay.HandSim = HandSim
    try:
        replay.play(blocks="flat_hold", student=context.get("student"),
                    future=10 if context.get("student") else 0)
    finally:
        HandSim.__init__, HandSim.draw_debug = real_init, real_draw
        HandSim.viewer_alive = real_alive

    from ..robot.layout import Row
    lanes = len(drawn.get("extra", ()))
    row = Row(lanes, "left")
    pitch = row.spacing
    problems = []

    # The nameplates carry each lane's y, and they are built from the same
    # arithmetic the copies are - so the check that matters is the SKELETON,
    # which is placed separately.
    wrist = np.array(drawn["skeleton"]["middle"][0], dtype=float)
    want = row.place(0)[1]
    if abs(float(wrist[1]) - want) > 1e-6:
        problems.append(f"skeleton at y={wrist[1]:+.3f}, lane 0 is y={want:+.3f}")

    plate = drawn["extra"][0][1][1]
    if abs(plate - want) > 1e-6:
        problems.append(f"lane 0 nameplate at y={plate:+.3f}, not {want:+.3f}")
    return check("lanes stand where announced", not problems,
                 "; ".join(problems) or f"{lanes} lanes, pitch {pitch} m")


def test_sim_attributes_other_modules_read(context):
    """Every `sim.<name>` the rest of the project reads actually exists.

    THE CHECK THAT WOULD HAVE CAUGHT THE LAST ONE. `HandSim.__init__` was cut
    from twenty-two arguments down to three settings groups, and `self.dt`
    became `self.physics.dt` - correct inside the class, and invisible to
    pyflakes outside it, because an attribute that has moved is a name that
    still parses. The mimic loop paces itself on `sim.dt`, so `main.py run`
    died on the first tracked frame: nothing before the hardware was already
    armed.

    Static on the reader side, live on the writer side: the names come from
    grepping the source for `sim.<name>`, the object is a real built one.
    """
    import re

    from ..robot.options import Scene
    from ..robot.sim import HandSim

    # `sim.py` and `sim.xml` in a path or a comment match the same pattern as
    # an attribute read; nothing on this object is named after a file type.
    suffixes = {"py", "xml", "npz", "json", "sh", "md", "log", "png"}
    wanted = set()
    for path in sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("handcontrol/**/*.py")):
        if path.name == "sim.py":
            continue                      # inside the class, self.* is its own business
        for name in re.findall(r"\bsim\.([a-z_][a-z0-9_]*)", path.read_text()):
            if name not in suffixes:
                wanted.add(name)
    # Built and commanded once: `mj`, `data` and `targets` do not exist on a
    # bare instance, and a caller that reads them has always called these first.
    sim = HandSim(hand="left", scene=Scene(show_viewer=False)).build()
    sim.command(np.zeros(len(sim.joint_names)))
    missing = sorted(n for n in wanted if not hasattr(sim, n))
    return check("sim attributes other modules read", not missing,
                 f"missing from HandSim: {', '.join(missing)}" if missing
                 else f"{len(wanted)} names read from outside, all present")


def test_no_undefined_names(context):
    """Every name every function uses is one that is in scope where it runs.

    THE CHECK THAT WOULD HAVE SAVED THREE ROUNDS. Editing this package by
    replacing a slice of text between two markers has repeatedly taken
    neighbouring names with it - `QUANTILES`, `ROLE_FOR`, `FALLBACK`,
    `LENGTH_SPREAD` and a `from . import train as training` inside
    `cli.run_train` have all been deleted by an edit aimed at something beside
    them. Python does not notice until the line runs, so a module imports
    cleanly and dies halfway through a twenty-minute job.

    PYFLAKES, NOT A HAND-ROLLED WALK. What was here collected every name bound
    anywhere in a file into one flat set, so an import inside one function
    satisfied a use inside another - and then reported only names beginning
    with a capital, because it was written to catch deleted CONSTANTS. Both
    limits were invisible from its name: deleting `run_train`'s only import of
    `training`, which it then calls six times, left it reporting "every module
    resolves". Pyflakes is scope-aware, case-blind and already a dependency.

    Only undefined names are failures here. Unused imports and shadowed
    builtins are style, and this check is about what breaks at runtime.
    """
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    files = [str(f) for f in sorted(root.glob("handcontrol/**/*.py"))]
    files += [str(root / f) for f in ("main.py", "learn.py", "leap_mimic.py",
                                      "keyvector_bridge.py", "compare_runs.py")
              if (root / f).exists()]
    out = subprocess.run([sys.executable, "-m", "pyflakes", *files],
                         capture_output=True, text=True)
    problems = [line.replace(str(root) + "/", "")
                for line in (out.stdout + out.stderr).splitlines()
                if "undefined name" in line]
    return check("no undefined names", not problems,
                 "; ".join(problems[:3]) if problems
                 else f"{len(files)} files, pyflakes clean")


def test_executor_closes_to_contact(context):
    """The grasp executor holds a finger where it stalls and lets go when
    the operator opens - driven against the mock link's virtual object.

    The mechanism the whole object question rests on: no model, no data,
    just the encoders not following a closing command. Checked here so a
    threshold edit cannot quietly turn it into a hand that pushes through
    whatever it holds, or one that never lets go.
    """
    from ..executor import GraspExecutor, SQUEEZE_RAD, STALL_MA, FREE_AIR_MA
    from ..hardware import MockHand

    names = context["model"].joint_names()
    link = MockHand(names).connect()
    index = [k for k, n in enumerate(names) if n.startswith("index_") and "adduction" not in n]
    link.obstacle = {k: 0.50 for k in index}            # the object, met at half a radian
    executor = GraspExecutor(names)
    executor.enabled = True
    problems, held_at, released_at = [], None, None
    clock = 0.0
    for tick in range(400):                              # 4 s at 100 Hz: close 2 s, open 2 s
        clock += 0.01
        want = 1.2 * min(tick, 200) / 200 if tick < 200 else 1.2 * (400 - tick) / 200
        target = np.zeros(len(names)); target[index] = want
        positions, currents = link.read_state()
        out = executor.step(target, positions, currents)
        link.write_positions(out)
        if executor.holding["index"] and held_at is None:
            held_at = (clock, float(out[index].mean()))
        if held_at is not None and not executor.holding["index"] and released_at is None:
            released_at = (clock, want)
    if held_at is None:
        problems.append("never held against the object")
    else:
        if abs(held_at[1] - (0.50 + SQUEEZE_RAD)) > 0.02:
            problems.append(f"held at {held_at[1]:.2f} rad, not the stall point plus the squeeze")
        if held_at[0] > 1.4:
            problems.append(f"took {held_at[0] - 1.0:.2f} s past the stall to hold")
    if released_at is None:
        problems.append("never let go when the operator opened")
    elif released_at[1] > 0.50:
        problems.append(f"let go at {released_at[1]:.2f} rad, before the command opened past the stall")

    # WITHOUT CURRENT IT MUST NOT GRIP AT ALL. Current is the only stall
    # signal now, so a link that does not report it has nothing to detect a
    # stall with - and the safe failure is to follow the command, never to
    # hold. A huge command lead with no current must pass straight through.
    blind = GraspExecutor(names)
    blind.enabled = True
    positions = np.zeros(len(names))
    wish = np.zeros(len(names)); wish[index] = 0.9
    for _ in range(50):
        passed = blind.step(wish, positions, None)
    if blind.holding["index"] or not np.allclose(passed, wish):
        problems.append("gripped, or altered the command, with no current reported")

    # And it must not latch on the current a FREE finger draws, however long
    # it is held there. There is no timer any more, so `STALL_MA` alone is
    # what stands between a grip and a hand that pins itself on nothing -
    # which makes its margin over `FREE_AIR_MA` a property worth asserting
    # rather than a number worth trusting.
    cool = GraspExecutor(names)
    cool.enabled = True
    for _ in range(200):
        cool.step(wish, positions, [FREE_AIR_MA] * len(names))
    if cool.holding["index"]:
        problems.append(f"held at {FREE_AIR_MA} mA, which is free air")
    if STALL_MA < 1.5 * FREE_AIR_MA:
        problems.append(f"STALL_MA {STALL_MA} is under 1.5x the {FREE_AIR_MA} mA "
                        f"a free finger draws - too little room to tell them apart")

    # A PINCH IS NOT A GRASP. Told by the contact head that the operator is
    # closing thumb onto index, the same stall must be followed rather than
    # latched - otherwise every pinch made with GRIP on becomes a grip the
    # operator has to deliberately open out of.
    pinch = GraspExecutor(names)
    pinch.enabled = True
    clock, held = 0.0, False
    link2 = MockHand(names).connect()
    link2.obstacle = {k: 0.50 for k in index}
    for tick in range(300):
        clock += 0.01
        want = 1.2 * min(tick, 200) / 200
        target = np.zeros(len(names)); target[index] = want
        positions, currents = link2.read_state()
        out = pinch.step(target, positions, currents,
                         pinching=(0.95, 0.0, 0.0))       # index pair pinching
        link2.write_positions(out)
        held = held or pinch.holding["index"]
    if held:
        problems.append("latched a pair the contact head called a pinch")

    return check("executor closes to contact", not problems,
                 "; ".join(problems) or f"held at {held_at[0]:.2f}s, released at {released_at[0]:.2f}s")


def test_appending_to_an_older_session(context):
    """A block recorded into an existing session lands where that session says.

    TWO WAYS THIS BROKE, both of them after the operator had already sat
    through the recording. The block was numbered against TODAY's table, so
    the three blocks added since the first session was made would have been
    written as 6, 7 and 8 - which that session's table calls spread_cycle,
    finger_isolate and thumb_sweep - and every consumer resolves a name
    through `meta["blocks"]`. And `Recorder.arrays` concatenated over the
    columns it records now, so a session written before `confidence` existed
    died with a KeyError at save, with nothing kept.
    """
    import json
    import shutil
    import tempfile
    import types

    from . import survey as recording
    from .sessions import load, sessions_under

    source = sorted(sessions_under())[-1]
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "session"
        out.mkdir()
        for name in ("session.npz", "meta.json"):
            shutil.copy(source / name, out / name)
        meta = json.loads((out / "meta.json").read_text())
        recorder = recording.Recorder()
        recorder.load(out)
        existing = recorder.loaded

        # A NAME NO SURVEY TABLE HAS. This used to pick a real block missing
        # from the session and skip when there was none - so once the
        # recording was complete the append path, which `survey --append
        # --missing` depends on, was never tested again. A made-up name tests
        # the same numbering and the same save, and tests the first bug above
        # more strictly: anything still indexing today's global table instead
        # of this session's cannot find it at all.
        new = "selftest_append_probe"
        assert new not in meta["blocks"] and new not in {b.key for b in recording.BLOCKS}
        meta["blocks"].append(new)
        number = meta["blocks"].index(new)

        class Hand:
            id = 1
            width = 85.0
            pinch_strength = 0.4
            grab_strength = 0.1
            pinch_distance = 30.0
            confidence = 0.9
            palm = np.zeros(3)
            normal = np.zeros(3)
            direction = np.zeros(3)
            bones = {d: [np.zeros(3)] * 5 for d in recording.DIGITS}

        for k in range(20):
            frame = types.SimpleNamespace(timestamp=k, frame_id=k,
                                          framerate=119.0, hands=[Hand()])
            recorder.add(frame, Hand(), number)
        with contextlib.redirect_stdout(io.StringIO()):
            recording.save(out, recorder, meta)
        data, metas = load([out])

    problems = []
    if metas[0]["blocks"][number] != new:
        problems.append(f"block {number} is {metas[0]['blocks'][number]!r}, not {new!r}")
    if int((data["block"] == number).sum()) != 20:
        problems.append(f"{int((data['block'] == number).sum())} rows landed at block {number}")
    if "confidence" not in data:
        problems.append("the appended column was dropped")
    else:
        if not np.isnan(data["confidence"][:existing]).all():
            problems.append("rows recorded before the column existed are not NaN")
        if not np.isfinite(data["confidence"][existing:]).all():
            problems.append("the appended rows have no confidence")
    return check("appending to an older session", not problems,
                 "; ".join(problems)
                 or f"{new!r} appended as block {number}, columns reconciled")


def test_grip_bench_finds_the_contact(context):
    """`main.py hw grip` reports contact where the object actually is.

    The procedure that will set the executor's thresholds on the bench. It is
    the only way those numbers stop being a datasheet guess, so it is checked
    here against a virtual stop whose position is known exactly - if the
    read-out drifts, the bench run would set them wrong and nothing else would
    notice.
    """
    import argparse
    import importlib.util
    import io
    import contextlib

    from ..hardware import MockHand

    spec = importlib.util.spec_from_file_location("_main", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    names = context["model"].joint_names()
    link = MockHand(names).connect()
    stop_at = 0.5
    args = argparse.Namespace(finger="index", close_to=1.2, close_speed=0.6,
                              command_hz=200.0, obstacle=stop_at, out=None)
    args.seconds = 2.0
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        code = module.bench_grip(args, link, names)
    report = captured.getvalue()
    problems = []
    if code:
        problems.append(f"returned {code}")
    found = [line for line in report.splitlines() if "contact at" in line]
    if not found:
        problems.append("reported no contact against a stop it should have met")
    else:
        degrees = float(found[0].split(",")[1].split("deg")[0])
        if abs(degrees - np.degrees(stop_at)) > 2.0:
            problems.append(f"contact read at {degrees:.1f} deg, stop is at "
                            f"{np.degrees(stop_at):.1f}")
    if "current peaked" not in report:
        problems.append("no current read-out")
    return check("grip bench finds the contact", not problems,
                 "; ".join(problems)
                 or f"stop at {np.degrees(stop_at):.0f} deg, found there")


def test_mimic_survives_lost_tracking(context):
    """`poll` and `step` hold up when the sensor delivers no hand.

    THE FRAME WITH NOTHING IN IT IS THE COMMON CASE, not the edge one - the
    operator's hand leaves the cone constantly, and the whole retargeting
    package is built around that. Code added to the poll path is easy to write
    against the frames that DO carry a hand, and it crashes on the bench a
    second after FOLLOW is pressed, which is the worst place to find out.

    Driven here off recorded hands with deliberate gaps, so it costs a second
    and needs no sensor.
    """
    from ..leap import Frame
    from ..mimic import Mimic

    data, model = context["data"], context["model"]
    tracked = np.flatnonzero(data["tracked"])
    hands = [hands_from(data, int(i)) for i in tracked[:60]]

    class Sensor:
        """Alternates real hands with frames carrying none."""

        def __init__(self, frames):
            self.frames, self.at = list(frames), 0

        def poll(self, timeout=None):
            self.at += 1
            hand = self.frames[self.at % len(self.frames)]
            return Frame(frame_id=self.at, timestamp=self.at,
                         framerate=100.0, hands=[hand] if hand else [])

    mimic = Mimic(model=model, mode="student", calibrate=False)
    # DROPOUTS FIRST, and that ordering is the test. Put them after the hands
    # and the rest alignment is already fitted by the time one arrives, so the
    # guard short-circuits before it ever reads the missing hand - which is how
    # the first version of this passed against the very crash it was written
    # for. On the bench the sensor delivers empty frames long before the
    # operator holds still enough to fit anything.
    mimic.connection = Sensor([None] * 10 + hands + [None] * 20 + hands)
    mimic.enabled = True
    try:
        for _ in range(200):
            mimic.poll()
        for _ in range(120):
            mimic.step(None, 1.0 / 100.0)
    except Exception as problem:
        return check("mimic survives lost tracking", False,
                     f"{type(problem).__name__}: {problem}")
    return check("mimic survives lost tracking", True,
                 f"rest fitted: {mimic.bone_offsets is not None}")


def test_every_label_can_be_drawn(context):
    """Every word the viewers write on the floor has glyphs for its letters.

    A missing glyph falls back to a blank, so it is INVISIBLE rather than
    loud - "ROBOT" rendered as "RO OT" for a while because the font carried
    only the letters the lane names happened to use at the time, and nine
    letters were absent with nothing to say so.
    """
    from ..robot import glyphs
    from . import replay

    words = {tag for _, _, tag, _ in replay.LANE_STYLE.values()}
    words |= {"YOU", "ROBOT", "TEACHER", "STUDENT", "ENCODER"}
    problems = [f"{word} lacks {glyphs.unknown(word)}"
                for word in sorted(words) if glyphs.unknown(word)]
    return check("every label can be drawn", not problems,
                 "; ".join(problems) or f"{len(words)} labels")


def test_posed_lanes_do_not_sag(context):
    """A comparison lane nobody has posed yet stays flat instead of falling.

    The copies carry joints and no actuators, so gravity acts on them and
    nothing opposes it. Measured before this was held: 25 degrees of sag in two
    seconds - which is what the viewer showed on a fresh start, before any
    mapping had an answer to pose the lanes with.

    The encoder ghost never had this problem because `step` rewrites its qpos
    every tick. The copies were added without that, and the asymmetry is the
    whole bug.
    """
    from ..robot.mjcf import STUDENT, STUDENT_RGBA, TEACHER, TEACHER_RGBA
    from ..robot.options import Scene
    from ..robot.sim import HandSim

    sim = HandSim(hand="left", scene=Scene(
        show_viewer=False, position=(0.0, 0.32, 0.20),
        ghost=True, ghost_offset=0.96,
        copies=((TEACHER, TEACHER_RGBA, 0.32), (STUDENT, STUDENT_RGBA, 0.64))))
    sim.build()
    try:
        for _ in range(300):
            sim.command(np.zeros(len(sim.joint_names)))
            sim.step()
        worst, where = 0.0, ""
        for prefix, (qadr, _) in sim.copy_qadr.items():
            sag = float(np.degrees(np.abs(sim.data.qpos[qadr])).max())
            if sag > worst:
                worst, where = sag, prefix
        ghost = float(np.degrees(np.abs(sim.data.qpos[sim.ghost_qadr])).max())
    finally:
        sim.close()
    return check("posed lanes do not sag", max(worst, ghost) < 1.0,
                 f"worst {where or 'ghost'} {max(worst, ghost):.2f} deg")


def test_the_thumb_split_is_the_split_it_claims(context):
    """`thumb_free`/`thumb_contact` recomputed here must equal what `score` says.

    WHY AN ORACLE AND NOT A RECONCILIATION. The first version of this check
    only asked whether the two halves average back to the pooled number, and a
    mutation walked straight through it: masking the contact half on the INDEX
    touching rather than on any digit touching still reconciles, because the
    index carries almost every touch and the mean over the two frame sets is
    the same 84.2 deg either way. Arithmetic that balances is not evidence the
    frames were partitioned.

    So the halves are computed again here, from `evaluate` and `TOUCH_MM`
    directly, and `score` has to agree with both of them AND with the pooled
    number it splits. That is the same shape as `test_geometry_is_one_table`:
    the way to hold one number honest is a second route to it.

    The split matters because the thumb is the digit `THUMB_RELEASE` releases
    COMPLETELY, so the pooled thumb figure averages a regime the objective
    targets with one it does not, and moves with how much of a block touches.
    """
    import torch
    from . import collision, objective
    from .kinematics import ChainFK
    teacher = context["teacher"]
    if "keypoints_raw" not in teacher:
        return "  skip thumb split (teacher.npz has no raw view)"
    names = [str(n) for n in teacher["joint_names"]]
    fk = ChainFK(context["model"], names)
    proxy = collision.for_hand(fk, *context["model"].limits(names))
    rows = np.arange(0, len(teacher["q_star"]), 37)
    q, aligned, raw = (teacher[k][rows] for k in ("q_star", "keypoints", "keypoints_raw"))
    with torch.no_grad():
        _, pooled, _, _, _, posture = objective.score(fk, proxy, q, aligned, raw)
        bones, _, d_human, _, _, _ = objective.evaluate(
            fk, proxy, *(torch.tensor(np.asarray(a), dtype=torch.float32,
                                      device=fk.xyz.device) for a in (q, aligned, raw)))
    # The second route. `free` is the frame touching NOTHING, which is what the
    # objective's own `any_touch` is built from.
    mine = bones[:, fk.fingers.index("thumb")].mean(dim=1)
    free = ~(d_human < objective.TOUCH_MM).any(dim=1)
    share = float((~free).float().mean())
    want = {"thumb_free": float(mine[free].mean()), "thumb_contact": float(mine[~free].mean())}

    off = {k: abs(posture[k] - v) for k, v in want.items()}
    rebuilt = share * posture["thumb_contact"] + (1.0 - share) * posture["thumb_free"]
    off["pooled"] = abs(rebuilt - pooled)
    return check("the thumb split is the split it claims", max(off.values()) < 0.05,
                 f"free {posture['thumb_free']:.1f}d vs {want['thumb_free']:.1f}d, "
                 f"contact {posture['thumb_contact']:.1f}d vs {want['thumb_contact']:.1f}d, "
                 f"{share:.2f} touching rebuilds {rebuilt:.2f}d against {pooled:.2f}d pooled "
                 f"(worst gap {max(off.values()):.3f})")


def test_the_thumb_release_reaches_both_scorers(context):
    """THUMB_RELEASE must move `loss` AND `per_frame`, on real touching frames.

    THE SHAPE OF THE BUG THIS GUARDS. The thumb's direction weight is written
    out twice - once in `loss`, which is what the teacher solves, and once in
    `per_frame`, which is what `train.py --direct` scores a student against.
    They are meant to be the same objective. Two copies of a line is two
    places to change, and a constant threaded through only one of them is
    silent: both functions still run, both return a number, and the two
    disagree about what the thumb is being asked for.

    So this scores frames where the operator's thumb IS touching - the only
    frames the constant can act on - at both ends of its range, and demands
    that each function notice. Patching the module global is the whole point:
    it is how the sweep that chose the value was run, and if a call site went
    back to a literal, the sweep was measuring one function and shipping two.
    """
    import torch
    from . import collision, objective
    from .kinematics import ChainFK
    teacher, blocks = context["teacher"], context["blocks"]
    if "keypoints_raw" not in teacher or "thumb_tap" not in blocks:
        return "  skip thumb release (needs a solved thumb_tap block)"
    names = [str(n) for n in teacher["joint_names"]]
    fk = ChainFK(context["model"], names)
    fk.reach_scale = objective.reach_scale(
        {str(k): float(v) for k, v in zip(teacher["fingers"], teacher["digit_scale"])})
    proxy = collision.for_hand(fk, *context["model"].limits(names))
    lower, upper = (torch.tensor(b, dtype=torch.float32)
                    for b in context["model"].limits(names))

    rows = np.flatnonzero(teacher["block"] == blocks.index("thumb_tap"))[::40]
    pick = lambda key: torch.tensor(teacher[key][rows], dtype=torch.float32)
    q, aligned, raw = pick("q_star"), pick("keypoints"), pick("keypoints_raw")
    intent = pick("pinch_strength")
    # Only frames the constant can reach. If the block has none, the check
    # would pass on two identical numbers for the wrong reason.
    with torch.no_grad():
        _, _, d_human, _, _, _ = objective.evaluate(fk, proxy, q, aligned, raw)
    touching = (d_human < objective.TOUCH_MM).any(dim=1)
    if int(touching.sum()) < 8:
        return "  skip thumb release (no touching frames in the stride)"
    q, aligned, raw = q[touching], aligned[touching], raw[touching]
    intent = intent[touching]

    was = objective.THUMB_RELEASE
    moved = {}
    try:
        for name, fn in (("loss", objective.loss), ("per_frame", objective.per_frame)):
            seen = []
            for value in (1.0, 0.0):
                objective.THUMB_RELEASE = value
                with torch.no_grad():
                    seen.append(float(fn(fk, proxy, q, aligned, raw, intent, lower, upper)))
            moved[name] = abs(seen[1] - seen[0])
    finally:
        objective.THUMB_RELEASE = was

    deaf = [n for n, delta in moved.items() if delta < 1e-6]
    return check("the thumb release reaches both scorers", not deaf,
                 (f"{', '.join(deaf)} ignore THUMB_RELEASE on "
                  f"{int(touching.sum())} touching frames"
                  if deaf else "  ".join(f"{n} moves {d:.4f}" for n, d in moved.items())))


def test_objective_has_what_it_reads_off_the_fk(context):
    """Every `fk.<name>` in objective.py, supplied by every chain BUILDER.

    THE BUG THIS EXISTS FOR cost two and a half hours of queue. `reach_scale`
    was set onto the FK object by `Teacher.__init__` and by nothing else, so
    `objective.per_frame` worked when it was reached through a Teacher and
    raised AttributeError when `train.py --direct` reached it with a bare
    `ChainFK`. Nothing caught it: the module imports, argparse is wired, the
    student trains - it only fails once a loss is actually computed, three
    stages into an overnight run.

    So the check is on the BUILDERS, not on a chain assembled here. Building
    one and patching it would prove only that the attribute can be set, which
    was never in doubt - what has to be true is that every file which
    constructs a chain AND hands it to the objective sets what the objective
    goes on to read. Both name sets come out of the source, so a new
    `fk.something` is a failure the moment it is written.
    """
    import re

    from .kinematics import ChainFK

    here = Path(__file__).parent
    source = (here / "objective.py").read_text()
    wanted = sorted(set(re.findall(r"\bfk\.([a-z_]+)", source)))

    # Whatever `ChainFK.__init__` already provides is not a builder's job.
    bare = ChainFK(context["model"], list(context["retarget"].joint_names))
    owed = [n for n in wanted if not hasattr(bare, n)]

    problems = []
    for path in sorted(here.parent.rglob("*.py")):
        if path.name in ("selftest.py", "objective.py"):
            continue
        text = path.read_text()
        if "ChainFK(" not in text or "objective." not in text:
            continue
        for name in owed:
            if not re.search(rf"\.{name}\s*=", text):
                problems.append(f"{path.name} builds a chain but never sets {name}")
    return check("objective has what it reads off the fk", not problems,
                 "; ".join(problems) if problems else
                 f"{len(wanted)} read, {', '.join(owed)} owed by builders, "
                 f"all supplied")


def test_a_collapsed_run_is_stopped_and_a_healthy_one_is_not(context):
    """`train.collapsed` on the four runs that set it, epoch by epoch.

    THE BUG THIS EXISTS FOR: a 192-channel run sat at 100% held-out
    self-penetration on all 21 epochs and saved a normal-looking checkpoint.

    The data below is those runs' own first epochs, measured, and it is why
    the rule is the shape it is. Two HEALTHY runs opened at 54% - so a check
    at epoch 1, or on the current value rather than the lowest, stops runs
    that were about to be fine. This fails if the rule is ever tuned into
    doing that, as well as if it stops catching the collapse.
    """
    from .train import collapsed

    runs = {"96 seed 0": ([0.1, 0.2, 0.9, 0.4], False),
            "96 seed 1": ([54.7, 0.1, 0.1, 0.5], False),
            "192 seed 0": ([53.8, 0.1, 0.3, 0.4], False),
            "192 seed 1": ([100.0, 100.0, 100.0, 100.0], True)}
    problems = []
    for name, (penetrations, should_stop) in runs.items():
        stopped = any(collapsed(penetrations[:epoch])
                      for epoch in range(1, len(penetrations) + 1))
        if stopped != should_stop:
            problems.append(f"{name} {'stopped' if stopped else 'ran on'} "
                            f"but should have {'stopped' if should_stop else 'run on'}")
    if collapsed([float("nan")] * 3):
        problems.append("an empty held-out set read as a collapse")
    return check("a collapsed run is stopped and a healthy one is not",
                 not problems, "; ".join(problems) or
                 "stops 192 seed 1; lets through both runs that opened at 54%")


def test_architecture_is_chosen_by_argument_not_by_flag(context):
    """No module-level switch decides what a student IS.

    THE BUG THIS EXISTS FOR happened twice. `student.BELIEF` and
    `student.AUTHORITY` were module globals that decided an input width and two
    heads. First a checkpoint trained with one loaded into a process without it
    and was rejected. Then, converted to arguments by halves, one line inside
    `train` still read the global - and built a 102-channel input layer that
    it fed 81. They are gone, along with `DERIVATIVES` and
    `belief.CONTACT_GATE`: flags nothing ever set, each one a switch that
    looked live and changed some paths and not others.

    So this refuses the whole class rather than two names. `student.py` and
    `belief.py` may not define a module-level boolean - what a network is must
    arrive by argument, where a checkpoint can record it and a loader can
    rebuild it.
    """
    import ast

    problems = []
    for name in ("student.py", "belief.py"):
        tree = ast.parse((Path(__file__).parent / name).read_text())
        for node in tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, bool)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        problems.append(f"{name}:{node.lineno} {target.id} = {node.value.value}")
    return check("architecture is chosen by argument, not by flag", not problems,
                 "; ".join(problems) or "no module-level switches in student.py or belief.py")


def test_the_noise_fit_is_where_its_readers_look(context):
    """`paths.NOISE_FIT` exists and `Belief` can load it.

    THE BUG THIS EXISTS FOR is a name collision that cost the other half of
    that queue. There are two constants called DATA - `paths.DATA` is data/
    and `sessions.DATA` is data/leap - and the noise fit lives under the
    second. Three call sites imported the first, so `--belief` and `--intent`
    died on FileNotFoundError one second into each of six queued runs.

    Both are now spelled `NOISE_FIT`, and this checks the file it names is
    real and parses, because a constant pointing at nothing fails exactly as
    late as the string did.
    """
    import torch

    from ..paths import NOISE_FIT
    from .belief import Belief

    if not NOISE_FIT.exists():
        return check("the noise fit is where its readers look", False,
                     f"{NOISE_FIT} does not exist - run `learn.py noise`")
    belief = Belief.load(NOISE_FIT)
    sigma = belief.sigma_for(torch.as_tensor(
        context["real"][:8], dtype=torch.float32)[None])
    ok = bool(torch.isfinite(sigma).all()) and float(sigma.min()) >= 0.0
    return check("the noise fit is where its readers look", ok,
                 f"{NOISE_FIT.relative_to(NOISE_FIT.parents[2])}, "
                 f"sigma {float(sigma.min()):.2f} to {float(sigma.max()):.2f} mm")


def test_written_gains_survive_arming(context):
    """A register set with `hw gains` is still set after the hand is armed.

    THE BUG THIS EXISTS FOR shipped as a comment instead of a fix. `configure`
    rewrites the profile registers and the PID gains from the link's OWN
    fields, and `arm` calls `configure` - so a value written live was reverted
    at exactly the moment it was wanted. It was found because
    PROFILE_ACCELERATION could not be taken off 0 through the tuning path, and
    the response was to change the module default, which masks it for that one
    register and leaves every other one broken.

    `write_gains` now folds what it wrote back into those fields. This drives
    the real `DynamixelHand` with its bus calls captured, because the mock
    keeps every register in one dict and has a no-op `configure`, so it cannot
    show the revert either way.
    """
    from ..hardware import DynamixelHand

    names = ["index_proximal_joint", "index_middle_joint"]
    hand = DynamixelHand.__new__(DynamixelHand)
    hand.joint_names = names
    hand.calibration = {"motors": {n: {"id": i} for i, n in enumerate(names, 1)}}
    hand.profile_velocity, hand.profile_acceleration = 200, 20
    hand.current_ma = 300
    hand.gains = {"position_p": 250, "position_i": 0, "position_d": 600}
    hand.torque_on = False
    written = []
    hand._write = lambda mid, field, value: written.append((field, value))
    hand._read = lambda mid, field: dict(written).get(field)

    hand.write_gains({"profile_acceleration": 60, "position_p": 900})
    written.clear()
    hand.configure()                      # what `arm` calls
    seen = dict(written)
    problems = [f"{f} reverted to {seen.get(f)}, not {want}"
                for f, want in (("profile_acceleration", 60), ("position_p", 900))
                if seen.get(f) != want]
    return check("written gains survive arming", not problems,
                 "; ".join(problems) or
                 "profile_acceleration and position_p both held through configure")


def test_every_command_runs(context):
    """Three subcommands execute their real code path, cheaply.

    `--help` proves argparse is wired and nothing else. A handler can reference
    a name it never defines and still pass a help check - `run_gestures` did,
    using `offsets` that was only ever bound in a different function - so these
    run the shortest real invocation of each instead. Only three of the nine:
    the rest solve, train or open a window, and none of that belongs in a
    check that has to finish in seconds.

    A NON-ZERO RETURN IS A FAILURE, and so is a `SystemExit`. Both used to
    pass: the return code was discarded, and the `except SystemExit` that was
    there to let `--list` exit cleanly also swallowed every `parser.error` and
    every refusal `validate.check_frame` raises - which is the one this check
    exists to catch.
    """
    from . import cli

    import argparse
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser.add_subparsers(dest="stage"))
    # `noise` is NOT among these: it rewrites data/leap/noise.json, which is
    # the fit `learn.py train` loads, and a check has no business changing
    # what the next training run reads. `--out` sends it to a scratch file.
    with tempfile.TemporaryDirectory() as scratch:
        trials = (["synth", "--compare", "--hands", "2", "--seconds", "1"],
                  ["noise", "--no-plot", "--out", str(Path(scratch) / "noise.json")],
                  ["survey", "--list"])
        problems = []
        for argv in trials:
            try:
                args = parser.parse_args(argv)
                code = args.handler(args, parser)
                if code:
                    problems.append(f"{argv[0]}: returned {code}")
            except SystemExit as stop:
                if stop.code:
                    problems.append(f"{argv[0]}: exited with {stop.code}")
            except Exception as failure:
                problems.append(f"{argv[0]}: {type(failure).__name__}: {failure}")
    return check("every command runs", not problems, "; ".join(problems)
                 or f"{len(trials)} commands, none refusing")


# Checks that read only the SOURCE, not the recordings. They run FIRST and
# without a context, because the thing they catch - a module-level name deleted
# by an edit aimed at its neighbour - breaks `build_context` itself. Run after
# it, the harness dies before reaching them and reports a stack trace instead
# of the one-line answer.
STATIC = (test_no_undefined_names, test_call_sites_match_signatures,
          test_lanes_are_distinct, test_viewer_builds_the_right_colours,
          test_every_label_can_be_drawn)

# The checks that read `context["teacher"]`, and so cannot run before
# `learn.py teacher` has. Listed, not guessed from the source text.
NEEDS_TEACHER = (test_teacher_makes_contact, test_labels_do_not_self_collide,
                 test_flat_hand_gives_a_straight_robot,
                 test_teacher_bounds_are_within_runs, test_geometry_is_one_table)

TESTS = (test_student_is_causal, test_collision_proxy_matches_mujoco,
         test_labels_do_not_self_collide, test_gesture_probe_plays,
         test_executor_closes_to_contact, test_basis_roundtrip, test_basis_is_orthonormal,
         test_rotation_is_not_a_reflection, test_synthetic_and_real_share_a_frame,
         test_rest_offsets_align_every_bone, test_flat_hand_gives_a_straight_robot,
         test_runs_never_cross_a_boundary, test_teacher_bounds_are_within_runs,
         test_a_pinch_survives_into_the_raw_view, test_teacher_makes_contact,
         test_geometry_is_one_table,
         test_digit_identity,
         test_gestures_hold_their_shape, test_objects_are_placed_from_the_hand,
         test_grip_bench_finds_the_contact, test_appending_to_an_older_session,
         test_sim_attributes_other_modules_read,
         test_lanes_stand_where_they_are_announced,
         test_mimic_survives_lost_tracking, test_posed_lanes_do_not_sag,
         test_the_thumb_split_is_the_split_it_claims,
         test_the_thumb_release_reaches_both_scorers,
         test_objective_has_what_it_reads_off_the_fk,
         test_the_noise_fit_is_where_its_readers_look,
         test_architecture_is_chosen_by_argument_not_by_flag,
         test_a_collapsed_run_is_stopped_and_a_healthy_one_is_not,
         test_written_gains_survive_arming,
         test_every_command_runs)


def build_context(hand="left", seed=0):
    """Everything the tests share, built once off the real recordings."""
    paths = sessions_under()
    if not paths:
        raise Failure("no sessions under data/leap - record one first")
    data, metas = load(paths)
    model = HandModel.load(ensure_urdf(hand, None))
    retarget = Retargeter(model=model)
    tracked = np.flatnonzero(data["tracked"])
    scale = fit_proportions(retarget,
                            [hands_from(data, i) for i in tracked[::200]],
                            FINGERS)
    offsets = fit_bone_offsets(retarget, model,
                               [hands_from(data, i) for i in tracked[::200]],
                               FINGERS)
    sampler = sampler_for(data, metas[0], retarget, seed)
    geometry = {key: sampler.template[key]
                for key in ("knuckle", "length", "direction")}
    synthetic = finish_keypoints(
        np.concatenate([sampler.sequence(3.0) for _ in range(8)]),
        scale, retarget, offsets)

    context = {
        "data": data, "meta": metas[0], "blocks": metas[0]["blocks"],
        "model": model, "retarget": retarget, "scale": scale,
        "offsets": offsets, "sampler": sampler, "geometry": geometry,
        "synthetic": synthetic,
        "hand": hands_from(data, int(tracked[len(tracked) // 2])),
        "student": (str(MODELS / "student.pt")
                    if (MODELS / "student.pt").exists() else None),
        "real": np.stack([keypoints_for(retarget, data, i, scale, offsets)
                          for i in tracked[::50]]),
    }
    path = paths[-1] / "teacher.npz"
    if path.exists():
        with np.load(path) as handle:
            context["teacher"] = {k: handle[k] for k in handle.files}
    return context


def run(hand="left"):
    """Every invariant, against the real recordings. Returns the failures.

    The source-only checks run before anything is loaded, so a module broken by
    an edit is reported as a failed check rather than as a stack trace out of
    the harness that was trying to test it.
    """
    failures = []

    def attempt(test, context=None):
        name = test.__name__.replace("test_", "").replace("_", " ")
        try:
            print(test(context))
        except Failure as problem:
            failures.append(str(problem))
            print(f"  FAIL {problem}")
        # SystemExit too: a subcommand refusing a stale model exits, and an
        # exit is a failed check here, not the end of the run.
        except (Exception, SystemExit) as problem:   # a broken test is a failure
            failures.append(f"{name}: {type(problem).__name__}: {problem}")
            print(f"  ERROR {name}: {type(problem).__name__}: {problem}")

    for test in STATIC:
        attempt(test)
    if failures:
        print("\n  source checks failed - not loading the recordings, because "
              "whatever is broken above would only fail again less clearly")
        return failures

    context = build_context(hand)
    for test in TESTS:
        name = test.__name__.replace("test_", "").replace("_", " ")
        # Skipped on what a test READS, not on what its prose mentions. A
        # substring search over the source dropped two checks that need no
        # labels at all - the proxy against MuJoCo, and the gesture probe -
        # because each says the word "teacher" in a comment.
        if "teacher" not in context and test in NEEDS_TEACHER:
            print(f"  skip {name} (no teacher.npz)")
            continue
        attempt(test, context)
    return failures
