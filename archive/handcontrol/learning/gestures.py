"""Hand-authored gestures the student was never trained on.

WHY THESE, AND WHY AUTHORED. Every number the student has been judged on so far
came from held-out BLOCKS of the same session - a different fist, but still a
fist, and still this operator's hand doing the thing the sampler was fitted to.
That measures interpolation. It does not answer the question that decides
whether any of this ships: what does it do with a shape it has never seen.

A peace sign and an OK sign are exactly that - or were. The synthetic sampler
draws one shared curl per hand with a small per-finger departure (`FLEX_OWN`),
because real fingers move together, so "two fingers straight, two folded" was
a corner it essentially never visited. The real recording still has no peace
or OK sign in it - the blocks are holds, curls, taps, spreads and rotations -
but the sampler now oversamples that corner deliberately
(`synthetic.INDEPENDENT`, a quarter of hands against the recording's 1.3%),
BECAUSE this probe found the gap. So these poses are outside the real half of
the training set and inside the synthetic one, which is the whole point: the
probe measures what the synthetic half is worth on a shape no operator gave
it.

SIGNS ARE MEASURED, NOT GUESSED. Rotating a digit about the palm normal moves
it away from the middle finger in a direction that differs per digit - swept on
the fitted template, away is NEGATIVE for the index and the thumb and POSITIVE
for the ring and the pinky. Written down from a guess, a "spread" gesture
closes half the hand.

EVERY GESTURE CARRIES A PREDICATE, and `verify` checks it. An authored pose is
a claim about a shape, and this file has already been the place where a
confident claim was wrong twice - the frame and the flexion sign, both of which
passed every distribution check before showing up on screen. A predicate is
cheap and it fails loudly.
"""

from __future__ import annotations

import numpy as np

from ..leap import DIGITS

# (flex, spread, pip, dip) in radians, per digit; anything omitted is neutral.
# Positive flex curls toward the palm. Spread is about the palm normal, and its
# helpful direction differs per digit - see the module docstring.
CURL = 1.15                 # a folded finger

# The pinky folds tighter than the rest, and has to. It is the shortest digit,
# so the same knuckle angle leaves its tip a larger FRACTION of its own reach
# from the wrist - measured, 0.59 against 0.46 for the middle - and a fist with
# the pinky sticking half out is not a fist. A real hand does the same thing.
PINKY_CURL = 1.45


def _finger(flex, spread=0.0):
    """A finger obeying the tendon coupling, so an authored pose is still a
    pose a hand could hold."""
    pip = flex * 1.15
    return (flex, spread, pip, pip * 0.66)


GESTURES = {
    "open": {},
    "spread": {"index": _finger(0.0, -0.50), "middle": _finger(0.0, 0.05),
               "ring": _finger(0.0, 0.40), "pinky": _finger(0.0, 0.62),
               "thumb": _finger(0.0, -0.70)},
    # Fingers folded, thumb left where it rests: the halfway house between
    # open and fist that a slow closing hand passes through, and the shape
    # in which a student that snaps between poses shows it.
    "curl": {digit: _finger(CURL) for digit in ("index", "middle", "ring")}
            | {"pinky": _finger(PINKY_CURL)},
    # Solved from the hand's lengths in `wrap_pose`; this entry only makes
    # the name known.
    "wrap": {},
    "fist": {digit: _finger(CURL) for digit in ("index", "middle", "ring")}
            | {"pinky": _finger(PINKY_CURL),
               "thumb": (0.45, 0.55, 0.5, 0.3)},
    "peace": {"index": _finger(0.0, -0.32), "middle": _finger(0.0, 0.16),
              "ring": _finger(CURL), "pinky": _finger(PINKY_CURL),
              "thumb": (0.45, 0.55, 0.5, 0.3)},
    # The OK sign is NOT written down as angles - see `CONTACT` and
    # `solve_contact`. Only the digits that are merely extended are stated
    # here; the two that have to meet are solved for the template in hand.
    "ok": {"middle": _finger(0.0, 0.10), "ring": _finger(0.0, 0.30),
           "pinky": _finger(0.0, 0.45)},
    "point": {"index": _finger(0.0), "middle": _finger(CURL),
              "ring": _finger(CURL), "pinky": _finger(PINKY_CURL),
              "thumb": (0.45, 0.55, 0.5, 0.3)},
    "ring_out": {"index": _finger(CURL), "middle": _finger(CURL),
                 "ring": _finger(0.0, 0.20), "pinky": _finger(PINKY_CURL),
                 "thumb": (0.45, 0.55, 0.5, 0.3)},
    "thumb_out": {"index": _finger(CURL), "middle": _finger(CURL),
                  "ring": _finger(CURL), "pinky": _finger(PINKY_CURL),
                  "thumb": _finger(0.0, -0.75)},
}

# The order the test runs in. Spread and fist first because they are the two
# the student HAS seen - if it fails those, the failure is not generalisation.
SEQUENCE = ("open", "spread", "fist", "peace", "ok", "point", "ring_out",
            "thumb_out", "open")

# Named sequences `learn.py gestures --sequence` can author. The probe is
# the generalisation test; `slow_curl` is a continuity test - fingers first,
# then the thumb over them, then open again, meant to be run with a long
# `--move` so a student that jumps between poses rather than following a
# slow hand has nowhere to hide it.
SEQUENCES = {"probe": SEQUENCE,
             "slow_curl": ("open", "curl", "fist", "curl", "open"),
             "wrap": ("open", "wrap", "open")}

# A cylinder resting on the palm, and the hand closing around it. Not
# authored as angles: the cylinder is placed in the hand's own frame - its
# axis along the knuckle line, just proximal of the knuckles, its radius
# above the palm - and each digit's joints are swept for the pose that lays
# its phalanges on the surface without passing through it. The same object,
# radius and all, is then carried into the robot's frame FROM THE HAND for
# the teacher to solve against (`cylinder_to_robot`); a physical object does
# not scale with the hand that holds it.
WRAP_RADIUS_MM = 20.0
WRAP_SKIN_MM = 8.0          # an operator's finger, joint line to pad
WRAP_PROXIMAL = 0.2         # the axis sits this many radii proximal of the knuckles
WRAP_LIFT_MM = 3.0          # knuckles sit about this far above the palm plane


def wrap_cylinder(geometry, radius_mm=None):
    """(centre, lateral unit axis) of the cylinder in the sampler's frame, mm."""
    from .synthetic import PALM_NORMAL

    radius_mm = WRAP_RADIUS_MM if radius_mm is None else radius_mm
    knuckle = {d: np.asarray(geometry["knuckle"][d], dtype=float) for d in ("index", "middle", "ring")}
    lateral = knuckle["index"] - knuckle["ring"]
    lateral = lateral / max(np.linalg.norm(lateral), 1e-9)
    along = np.mean([np.asarray(geometry["direction"][d][0], dtype=float) for d in knuckle], axis=0)
    along = along / max(np.linalg.norm(along), 1e-9)
    centre = (np.mean(list(knuckle.values()), axis=0) - WRAP_PROXIMAL * radius_mm * along
              + (radius_mm + WRAP_LIFT_MM) * PALM_NORMAL)
    return centre, lateral


def _line_distance(points, centre, axis):
    """Distance from each point (n, 3) to the infinite line through `centre`."""
    rel = points - centre
    return np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)


def wrap_pose(sampler, geometry, radius_mm=None):
    """Joints per digit that wrap the cylinder of `wrap_cylinder`: a sweep
    per digit, scored by how close each phalanx sits to the surface and
    charged tenfold for going inside it."""
    radius_mm = WRAP_RADIUS_MM if radius_mm is None else radius_mm
    centre, axis = wrap_cylinder(geometry, radius_mm)
    surface = radius_mm + WRAP_SKIN_MM
    out = {digit: (0.0, 0.0, 0.0, 0.0) for digit in DIGITS}

    def cost(digit, tuple_):
        trial = dict(out); trial[digit] = tuple_
        points = sampler.keypoints(geometry, trial)
        slot = DIGITS.index(digit)
        chain = points[4 * slot + 1:4 * slot + 5]
        total = 0.0
        for k in range(3):
            samples = chain[k] + (chain[k + 1] - chain[k]) * np.linspace(0.0, 1.0, 5)[:, None]
            d = _line_distance(samples, centre, axis).min()
            total += 10.0 * max(surface - d, 0.0) + abs(d - surface)
        return total

    for digit in ("index", "middle", "ring"):
        best = None
        for flex in np.linspace(0.1, 1.4, 14):
            for pip in np.linspace(0.0, 1.6, 12):
                for dip in np.linspace(0.0, 1.2, 9):
                    trial = (float(flex), 0.0, float(pip), float(dip))
                    score = cost(digit, trial)
                    if best is None or score < best[0]:
                        best = (score, trial)
        out[digit] = best[1]
    out["pinky"] = out["ring"]
    best = None
    for flex in np.linspace(0.0, 1.2, 12):
        for spread in np.linspace(0.0, 1.4, 12):
            trial = (float(flex), float(spread), float(flex) * 0.9, float(flex) * 0.5)
            score = cost("thumb", trial)
            if best is None or score < best[0]:
                best = (score, trial)
    out["thumb"] = best[1]
    return out


def cylinder_in_hand(geometry, radius_mm):
    """The cylinder the wrap was authored around, in the hand's own frame:
    {centre mm, axis unit, knuckle mm (the finger knuckle line's centre),
    radius mm}. What the sampler records per object, so the robot's copy can
    be placed FROM THE HAND - see `cylinder_to_robot`."""
    centre, axis = wrap_cylinder(geometry, radius_mm)
    knuckle = np.mean([np.asarray(geometry["knuckle"][d], dtype=float)
                       for d in ("index", "middle", "ring")], axis=0)
    return {"centre": [float(v) for v in centre], "axis": [float(v) for v in axis],
            "knuckle": [float(v) for v in knuckle], "radius_mm": float(radius_mm)}


def palm_of(fk):
    """The robot's own landmarks a cylinder is placed against: the centre of
    its finger knuckle line (index, middle, ring proximal axes, metres) and
    its palm face, the x = 0 plane the fingers stand on."""
    import torch

    with torch.no_grad():
        q = torch.zeros(1, len(fk.joint_names), device=fk.xyz.device, dtype=fk.xyz.dtype)
        origins = fk.walk(q)[1][0]
    rows = [fk.fingers.index(f) for f in ("index", "middle", "ring")]
    knuckle = origins[rows, 1].mean(0).cpu().numpy()
    return {"knuckle": [float(v) for v in knuckle], "face": 0.0}


# Where a cylinder resting on the ROBOT's palm sits along its fingers, as a
# function of radius: `REST_ALONG_MM + REST_PER_RADIUS * R` distal of the
# knuckle axes. MEASURED, not chosen: it is the line through where the
# compliant teacher (object free to move in the solve) put the operator's own
# 20 and 35 mm wraps - 12.4 and 17.5 mm past the axes. The operator's cylinder
# sits 0.2 R PROXIMAL of the knuckle line, but the robot's knuckle axes are
# 16 mm inside its palm and its proximal links 21 mm thick, so a proximal link
# flexed 30-60 degrees meets an object on the palm face 10-20 mm past the
# axis; placed where the operator's knuckles say, the teacher could lay one
# link on it (1.1 links, 1.1 digits) against 3.7 links and three digits here.
REST_ALONG_MM = 5.6
REST_PER_RADIUS = 0.34


def cylinder_to_robot(item, retarget, scale, palm, half_length=0.07):
    """The same physical cylinder in the robot's palm, metres in its frame.

    PLACED FROM THE HAND, not by the solve. An object left free to settle in
    the solve landed somewhere different in every training hand, so one
    operator wrap meant many labels, and a student with no object input could
    not learn where the thing was. What the operator's hand decides: the
    radius (kept - a physical object does not scale with the hand that holds
    it), the axis (the knuckle line, rotated into the robot's axes with the
    keypoints), and the lateral position (the offset from the operator's
    knuckle-line centre, scaled by the finger-length ratio the retargeting
    uses, laid off the robot's). What the ROBOT decides, because its links are
    thick where the operator's are thin: the object rests on its palm face
    `WRAP_LIFT_MM` up, at the along-finger distance its own proximal links
    can close on - see `REST_ALONG_MM`."""
    from .sessions import to_robot_axes

    offset = (np.asarray(item["centre"], dtype=float) - np.asarray(item["knuckle"], dtype=float)) * 1e-3
    offset = to_robot_axes(offset, retarget) * float(scale)
    knuckle = np.asarray(palm["knuckle"], dtype=float)
    radius = float(item["radius_mm"])
    centre = np.array([palm["face"] + (radius + WRAP_LIFT_MM) * 1e-3,
                       knuckle[1] + offset[1],
                       knuckle[2] + (REST_ALONG_MM + REST_PER_RADIUS * radius) * 1e-3])
    axis = to_robot_axes(np.asarray(item["axis"], dtype=float), retarget)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    out = {"kind": "capsule", "p0": [float(v) for v in centre - half_length * axis],
           "p1": [float(v) for v in centre + half_length * axis],
           "radius": radius * 1e-3}
    if "frames" in item:
        out["frames"] = list(item["frames"])
    return out

# Gestures whose meaning is a CONTACT rather than a set of angles: these two
# fingertips have to touch, and whatever angles achieve that are the gesture.
#
# WHY THIS IS NOT A TABLE OF NUMBERS. It was, and the numbers were a sweep run
# once against one template and pasted in - which is a hand-tweak wearing a
# fit's clothing. The angles here are relative to the sampler's REST pose, and
# that rest moved the moment it was aligned onto the robot's zero: the pasted
# numbers then left thumb and index 77 mm apart, and the only thing that
# noticed was the predicate.
#
# An OK sign means "thumb touches index", on any hand, in any rest frame. So it
# is stored as that and solved for whatever template is loaded, which is what
# makes it survive a new operator, a re-fit, or a different robot.
CONTACT = {"ok": ("thumb", "index")}

TIP = {digit: 4 * index + 4 for index, digit in enumerate(DIGITS)}

# A digit is "out" when its tip is further from the wrist than this fraction of
# its own reach at rest, and "in" below it. Measured on the template: a folded
# finger reaches about 40% of straight, a straight one 100%.
OUT, IN = 0.80, 0.55
# What "touching" means, in the sampler's own millimetres. The sampler's frame
# is the operator's, unscaled and unrotated, so its distances are the RAW ones
# the teacher's contact term reads - one definition of a touch, in one place.
from .objective import TOUCH_MM as TOUCHING


def solve_contact(sampler, geometry, first, second, base, steps=16):
    """Angles bringing two fingertips together, for THIS hand's proportions.

    The sampler owns the sweep - its own pinches are the same operation - so
    this is one call. `first` is the thumb on every gesture that has a contact;
    the OK sign is solved to touching rather than to the sensor's 20 mm, since
    an authored pose has no tracker slack to reproduce.
    """
    assert first == "thumb", "contact gestures are thumb against a finger"
    return sampler.contact_pose(geometry, second, base, target_mm=0.0, steps=steps)


def pose(name, sampler=None, geometry=None):
    """One gesture as a full per-digit pose, neutral where unspecified.

    A contact gesture is solved here rather than read from a table, so it means
    the same thing on any hand - pass the sampler and geometry to get one.
    """
    authored = GESTURES[name]
    if name in CONTACT and sampler is not None and geometry is not None:
        authored = solve_contact(sampler, geometry, *CONTACT[name], authored)
    if name == "wrap" and sampler is not None and geometry is not None:
        authored = wrap_pose(sampler, geometry)
    return {digit: authored.get(digit, (0.0, 0.0, 0.0, 0.0))
            for digit in DIGITS}


def ease(a, b, blend):
    """Cosine easing between two poses: zero velocity at each end, so the path
    has no corners for a jerk metric to object to."""
    smooth = 0.5 - 0.5 * np.cos(np.pi * blend)
    return {digit: tuple((1 - smooth) * np.asarray(a[digit])
                         + smooth * np.asarray(b[digit])) for digit in DIGITS}


def sequence(sampler, geometry, names=SEQUENCE, hold=0.8, move=0.9, hz=125.0):
    """Keypoints through a scripted gesture sequence. (n, 21, 3) millimetres.

    Held at each gesture and eased between them, rather than swept straight
    through: a held pose is where a teleoperated hand has to be steady, and it
    is the part of the motion a jerk metric can actually see.
    """
    poses, labels = [], []
    solved = {n: pose(n, sampler, geometry) for n in set(names)}
    for index, name in enumerate(names):
        here = solved[name]
        for _ in range(int(hold * hz)):
            poses.append(here)
            labels.append(index)
        if index + 1 < len(names):
            there = solved[names[index + 1]]
            steps = max(int(move * hz), 1)
            for step in range(steps):
                poses.append(ease(here, there, (step + 1) / steps))
                labels.append(index)
    points = np.stack([sampler.keypoints(geometry, p) for p in poses])
    return points, np.array(labels, dtype=np.int16)


def verify(sampler, geometry):
    """Check every gesture is the shape it claims to be. Returns the failures.

    Predicates, not eyeballs. Each one states what the gesture MEANS - a peace
    sign is two digits out and two in - so a sign error or a bad angle fails
    here rather than being explained away later as the student generalising
    badly.
    """
    rest = sampler.keypoints(geometry, pose("open", sampler, geometry))
    reach = {digit: np.linalg.norm(rest[TIP[digit]]) for digit in DIGITS}

    def shape(name):
        points = sampler.keypoints(geometry, pose(name, sampler, geometry))
        return points, {d: np.linalg.norm(points[TIP[d]]) / reach[d]
                        for d in DIGITS}

    problems = []

    def want(name, condition, description):
        if not condition:
            problems.append(f"{name}: {description}")

    points, out = shape("fist")
    for digit in ("index", "middle", "ring", "pinky"):
        want("fist", out[digit] < IN, f"{digit} not folded ({out[digit]:.2f})")

    points, out = shape("curl")
    for digit in ("index", "middle", "ring", "pinky"):
        want("curl", out[digit] < IN, f"{digit} not folded ({out[digit]:.2f})")
    want("curl", out["thumb"] > OUT, f"thumb folded too ({out['thumb']:.2f})")

    points, out = shape("wrap")
    for digit in ("index", "middle", "ring"):
        want("wrap", out[digit] < 0.85, f"{digit} not around the cylinder ({out[digit]:.2f})")

    points, out = shape("spread")
    span = np.linalg.norm(points[TIP["index"]] - points[TIP["pinky"]])
    flat = sampler.keypoints(geometry, pose("open", sampler, geometry))
    want("spread", span > 1.4 * np.linalg.norm(flat[TIP["index"]]
                                               - flat[TIP["pinky"]]),
         f"index-to-pinky only {span:.0f} mm")

    points, out = shape("peace")
    for digit in ("index", "middle"):
        want("peace", out[digit] > OUT, f"{digit} not extended ({out[digit]:.2f})")
    for digit in ("ring", "pinky"):
        want("peace", out[digit] < IN, f"{digit} not folded ({out[digit]:.2f})")
    want("peace", np.linalg.norm(points[TIP["index"]] - points[TIP["middle"]])
         > 30.0, "index and middle not parted")

    points, out = shape("ok")
    gap = np.linalg.norm(points[TIP["thumb"]] - points[TIP["index"]])
    want("ok", gap < TOUCHING, f"thumb and index {gap:.0f} mm apart")
    for digit in ("middle", "ring", "pinky"):
        want("ok", out[digit] > OUT, f"{digit} not extended ({out[digit]:.2f})")

    for name, alone in (("point", "index"), ("ring_out", "ring"),
                        ("thumb_out", "thumb")):
        points, out = shape(name)
        want(name, out[alone] > OUT, f"{alone} not extended ({out[alone]:.2f})")
        for digit in DIGITS:
            if digit in (alone, "pinky") or digit == "thumb" and alone != "thumb":
                continue
            want(name, out[digit] < IN, f"{digit} not folded ({out[digit]:.2f})")
    return problems
