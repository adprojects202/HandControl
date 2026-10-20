"""Plausible human hands from nothing at all: Phase 2's data source.

WHY SYNTHESISE AT ALL, when there are recordings. Because the labels do not
come from the recordings. They come from the teacher, and the teacher will
solve any hand it is handed - so the expensive, bench-bound, operator-dependent
part of collecting training data is not needed to produce PAIRS. It is needed
for exactly two things, and this file is what makes that split pay: the noise
model, and a distribution to check this one against.

A recorded session gives fifty thousand frames of one hand doing thirteen
scripted things. This gives as many as there is time to solve, of hands that
differ in proportion, doing motions nobody had to perform.

THE TEMPLATE IS MEASURED, NOT INVENTED. Published anthropometry would be a
second source of truth about hand geometry, free to disagree with the operator
the robot is actually driven by, and every disagreement would land in the
student as a systematic bias. So the rest shape - where each metacarpal starts,
which way each bone points, how long it is - is fitted off the open-hand blocks
of a real recording, and the sampler perturbs THAT.

WHAT IS SAMPLED, AND WHAT IS NOT:

  Geometry varies. Bone lengths and knuckle positions are drawn per hand, so
  the student meets proportions the operator does not have and cannot overfit
  to one pair of hands.

  Pose is sampled through the tendon coupling, not over the joints. A uniform
  grid over twelve finger angles is mostly hands nobody can make - the DIP
  follows the PIP at about two thirds and the PIP follows the knuckle, and
  sampling those ratios with slack is what makes the set a set of HANDS.
  Same constants the pose bank uses, for the same reason.

  Motion is a smooth path, not independent frames. The student reads a window
  of twenty, so a shuffled pile of poses would teach it nothing about time.
  Keyframes are drawn and interpolated at the sensor's own rate.

  Noise is NOT added here. This file emits clean hands; `augment.py` adds the
  sensor's error, and keeping them apart is what allows the same clean sample
  to be shown noisy in different ways.
"""

from __future__ import annotations

import numpy as np

from ..leap import DIGITS
from ..retargeting.keyvector import wrist_point
from .sessions import hands_from

# The coupling a real finger obeys, and how far it departs from it. Shared with
# `posebank.py` by value rather than by import: the pose bank samples the
# ROBOT's joints and this samples a human's, and the day one of them needs a
# different slack is not a day the other should change.
PIP_OF_KNUCKLE = 1.15
DIP_OF_PIP = 0.66
COUPLING_SLACK = 0.16

# Fallbacks for the two coupling widths - how far one finger departs from what
# the whole hand is doing. Used only when there is no recording to fit them
# from; `fit_coupling` measures both, and measured they came out 0.03 for curl
# against an assumed 0.16, and 0.23 for spread against an assumed 0.12. The
# sampler was wrong in both directions and by a factor of five in one.
SPREAD_OWN = 0.23
# How far apart, in quantile of their own ranges, neighbouring fingers'
# spreads may sit. The fallback when no recording is fitted; `fit_coupling`
# measures it as the 99th percentile of the real hand's adjacent differences.
ADJACENT_SPREAD = 0.35
FLEX_OWN = 0.03

# How often the sampler forces a clean extended/folded split, and how much of
# that is a split rather than merely loose fingers.
#
# THESE ARE A POLICY, NOT A MEASUREMENT, and the distinction matters. Measured
# on the recording, only 1.3% of real frames have fingers spread across more
# than half their curl range, and none exceed 0.7 - a peace sign simply is not
# in the survey. Fitting to that would set this near zero and reproduce exactly
# the gap the gesture probe found: novel shapes cost the student 8 to 10
# degrees where familiar ones cost 5.
#
# So the corner is oversampled on purpose. A quarter rather than the real 1.3%,
# because the student has to have seen the shape, and not a half, because the
# marginals still have to cover the recording - which they do, at 98%.
INDEPENDENT = 0.25
SPLIT = 0.6

# HOW INDEPENDENT THE FINGERS ARE ALLOWED TO BE, as named policies rather than
# values edited in place. `measured` overrides nothing, so it is the fitted
# coupling plus the constants above - the behaviour before this table existed.
#
# `wide` IS A MEASURED DEAD END, KEPT SO THE RESULT STAYS REPRODUCIBLE.
#
# The reasoning that produced it: the sampler is fitted to one recording of one
# operator who moved their fingers together, so the prior should exclude a
# peace sign, so widening it should let the gesture probe in. Every step of
# that is true except the last. Distance from the probe's gestures to the
# nearest of 20000 sampler draws, worst joint, in degrees:
#
#     measured 18.7   wide 18.9   always-split 32.5   always-loose 18.7
#
# Widening adds variance without aiming: in sixteen dimensions the volume grows
# faster than the density, so some gestures come closer and others go further
# and the mean does not move. Forcing a split every keyframe is much worse - it
# deletes the whole-hand poses a fist and a spread need.
#
# What the sampler is actually missing is not diversity. See the note on
# `schedule` in `train.py`: the student is fine-tuned on the real hand and
# scores 8.95 degrees on the synthetic set it PRETRAINED on, against 8.57 on
# the gestures and 2.5 on the real hand. The probe measures the cost of that
# fine-tune, not a hole in this distribution, and no setting here reaches it.
DIVERSITY = {
    "measured": {},
    "wide": {"flex_own": 0.15, "spread_own": 0.40,
             "coupling_slack": 0.35, "independent": 0.45, "split": 0.6},
}

# HOW OFTEN A KEYFRAME IS A PINCH - the thumb tip solved onto a fingertip.
#
# A POLICY, and the one that the contact work made necessary. The recording
# touches on 14% of its frames, and the fitted sampler reproduces that: it draws
# hands from a distribution in which the thumb meeting a fingertip is a rare
# accident. But contact is the event the teacher now labels exactly and the
# student has to learn to arrive at, so it is oversampled on purpose - about a
# third of keyframes, each held and then left, which is what a tap is. Which
# fingertip is drawn uniformly over index, middle and ring; the pinky has no
# counterpart on the robot.
PINCH_RATE = 0.3

# OBJECTS IN THE SYNTHETIC HANDS. This share of keyframes closes on a
# cylinder of a radius drawn from OBJECT_RADIUS_MM, solved for this hand by
# `gestures.wrap_pose`, and the cylinder is present from halfway out of the
# keyframe before to halfway into the one after - the hand approaches it,
# holds it, lets go. The teacher solves those frames with the object in the
# objective (`objective.held`), so the labels hold it; the student never
# sees the object, only a hand closed the way a hand closes on one, and
# learns to answer that with a pose that holds it.
OBJECT_RATE = 0.2
OBJECT_RADIUS_MM = (12.0, 30.0)
# The gap a pinch is solved to, in the sampler's own millimetres. The sensor
# reports 21.5 mm at the median of the operator's real pinches - tips are
# spheres and the fit has slack - so a synthetic one is aimed there rather than
# at zero, which no tracker ever reports.
PINCH_GAP_MM = 20.0

# How far past the measured range to sample, as a fraction of it. The range is
# a percentile band of what one operator did in one session, not the edge of
# what a hand can do, and a student asked for a pose just outside its training
# set has nothing to fall back on. Small, because the padding is extrapolation.
MARGIN = 0.15

# The shared curl and spread draws are pushed toward their ends rather than
# drawn flat. With a single shared quantile per hand, "every finger open at
# once" is a corner of the space and a uniform draw almost never lands there -
# measured, the synthetic thumb-to-index gap reached 89 mm where the real hand
# reached 115, so the open extreme was missing entirely. Beta(0.7, 0.7) is
# still smooth, and puts enough weight on both ends to cover them.
EXTREME = 0.7

# Fallback pose limits in radians, used only when there is no recording to fit
# them from. Deliberately generous - a guess that is too tight silently removes
# poses the operator makes, which is worse than a guess that is too loose.
FALLBACK = {"flex": (-0.35, 1.40), "spread": (-0.60, 0.70),
            "pip": (-0.10, 1.65), "dip": (-0.10, 0.95)}

# Which measured role feeds which of the sampler's four angles. The roles come
# from `model.joint_role` by way of `Retargeter.measure`, so the thumb's are
# named for what they MOVE rather than for the joint they sit on - which is the
# project's one invariant and the reason this is a table rather than a guess.
ROLE_FOR = {"flex": "knuckle", "spread": "spread", "pip": "middle",
            "dip": "tip"}
MEASURED_DIGITS = ("thumb", "index", "middle", "ring")
PINKY_LIKE = "ring"

# How finely a measured distribution is kept. 101 points is every percent.
QUANTILES = 101

# How far a hand's proportions may differ from the template's. Fifteen percent
# on bone length is roughly the spread between a small adult hand and a large
# one; the knuckles move less, because a palm is a plate.
LENGTH_SPREAD = 0.15
KNUCKLE_SPREAD = 0.06

# The thumb's spread band, in radians, and the one range still written down.
#
# IT RESISTS MEASUREMENT, twice over. `Retargeter.measure` reports the thumb's
# spread signed from a fixed 55 degree rest offset and running the opposite way
# to the sampler's rotation, so its percentiles describe travel from somewhere
# else - swept, +40 degrees of that angle CLOSES the thumb-to-index gap to
# 33 mm where -40 opens it to 132. And measuring the angle directly about the
# thumb's own fitted swing axis wraps: the percentiles come out at -175 and
# +173 degrees, which is a full circle and says nothing.
#
# So this stays a written-down band, calibrated against the thumb-to-index gap
# - the one quantity observable on both hands. It is the last hardcode here
# that is not a stated policy, and it is flagged as such rather than dressed up
# as a fit.
THUMB_SPREAD_BAND = (-0.79, 0.70)


def pose_ranges(data, retarget, stride=15, low=1.0, high=99.0,
                thumb_spread=None):
    """The angles this operator's hand actually produces, per digit and role.

    MEASURED, NOT ASSUMED, for the same reason the spread axis is. Written-down
    limits made synthetic hands that reached 173 mm between thumb and index
    where the real hand's widest was 135 - a fifth of the sampled poses sat
    outside anything a hand had done, and a student trained on them spends
    capacity on a region it will never be asked about.

    Percentiles rather than min and max: a single mis-tracked frame at full
    occlusion should not become the edge of the sampling range.

    The pinky is not in `model.FINGERS` - the robot has no little finger, so
    nothing measures its angles - and it takes the ring finger's. It exists in
    the layout only so DexPilot's indexing finds something human in slot 20.

    `thumb_spread` overrides `THUMB_SPREAD_BAND` for a caller recalibrating it.
    IT IS A WEAK LEVER, and that is measured: swept from (-0.79, 0.70) down to
    (-0.25, 0.20), the generated thumb-to-index gap moved only at the top -
    p99 303 mm to 283 - while p1 stayed pinned at 135 and the thumb tip stayed
    64 to 70 mm from the recorded mean throughout. So the band is not what
    separates the generated thumb from the operator's, and retuning it is not
    the fix for anything that gap is blamed for.
    """
    angles = {}
    for index in np.flatnonzero(data["tracked"])[::stride]:
        for (finger, role), value in retarget.measure(
                hands_from(data, index)).items():
            angles.setdefault((finger, role), []).append(value)

    grid = np.linspace(low, high, QUANTILES)
    out = {}
    for digit in DIGITS:
        source = PINKY_LIKE if digit not in MEASURED_DIGITS else digit
        limits = {}
        for name, role in ROLE_FOR.items():
            values = angles.get((source, role))
            if values:
                # The whole measured distribution, as a quantile grid, not just
                # its two edges. Spread is sampled THROUGH this rather than
                # uniformly between the edges - see `pose`.
                limits[name] = np.percentile(values, grid)
            else:
                lo, hi = FALLBACK[name]
                limits[name] = np.linspace(lo, hi, QUANTILES)
        if digit == "thumb":
            limits["spread"] = np.linspace(*(thumb_spread or THUMB_SPREAD_BAND),
                                           QUANTILES)
        out[digit] = limits
    return out


# The hand frame from `geometry.hand_frame` is (forward, normal, thumbward), so
# in the local coordinates this file works in, index 1 is the palm normal and
# index 2 is the thumb side. Getting those two the wrong way round is the bug
# this constant exists to prevent a repeat of: with spread taken about the
# thumbward axis and flexion about the normal, a synthetic fist splayed its
# fingers apart instead of closing them - median index-to-middle tip distance
# 67 mm against a real 25.
#
# Measured on the recording, flat hand to fist, the index tip moves
# (-78, +23, +3) mm: back and up out of the palm, which is rotation about
# THUMBWARD. Flat hand to spread it moves (+2, -10, +36): sideways across the
# palm, which is rotation about the NORMAL.
PALM_NORMAL = np.array([0.0, 1.0, 0.0])
THUMBWARD = np.array([0.0, 0.0, 1.0])

def template(data, meta, blocks=("flat_hold", "spread_hold"), samples=200):
    """The operator's rest hand, in its own frame: knuckles, directions, lengths.

    Taken from the OPEN blocks. A fist's bone lengths are the same numbers
    measured through the tracker's worst occlusion, and its bone directions are
    not rest directions at all.
    """
    from ..retargeting.geometry import hand_frame

    names = meta["blocks"]
    wanted = {names.index(b) for b in blocks if b in names}
    rows = [i for i in np.flatnonzero(data["tracked"])
            if int(data["block"][i]) in wanted]
    if not rows:
        rows = list(np.flatnonzero(data["tracked"]))
    rows = rows[::max(1, len(rows) // samples)]

    knuckles, directions, lengths = {}, {}, {}
    for index in rows:
        hand = hands_from(data, index)
        basis = np.array(hand_frame(hand))          # rows: forward, normal, thumbward
        wrist = wrist_point(hand)
        for digit in DIGITS:
            joints = hand.bones[digit]
            local = (joints - wrist) @ basis.T
            knuckles.setdefault(digit, []).append(local[1])
            bones = np.diff(local[1:], axis=0)      # three bones, knuckle out
            span = np.linalg.norm(bones, axis=1)
            lengths.setdefault(digit, []).append(span)
            directions.setdefault(digit, []).append(
                bones / np.maximum(span, 1e-9)[:, None])
    return {
        "knuckle": {d: np.median(np.stack(v), axis=0) for d, v in knuckles.items()},
        "length": {d: np.median(np.stack(v), axis=0) for d, v in lengths.items()},
        "direction": {d: np.median(np.stack(v), axis=0)
                      for d, v in directions.items()},
    }


class HandSampler:
    """Draws hands, and paths through the poses one hand can make.

    A sampler holds one TEMPLATE and draws many geometries from it; a geometry
    is one person's hand and stays fixed for the length of a trajectory, which
    is what a real recording would give.
    """

    def __init__(self, template, ranges=None, rng=None, coupling=None,
                 diversity=None):
        self.template = template
        self.ranges = ranges or {d: dict(FALLBACK) for d in DIGITS}
        # (flex, spread) coupling widths, measured by `fit_coupling`. Held on
        # the sampler rather than read from module constants so a different
        # operator's hand brings its own.
        coupling = coupling or (FLEX_OWN, SPREAD_OWN, ADJACENT_SPREAD)
        self.flex_own, self.spread_own = coupling[0], coupling[1]
        # How far, in quantile, a finger's spread may sit from its
        # neighbour's. Measured off the recording by `fit_coupling`; without
        # it the sampler drew neighbours a whole range apart and put one
        # finger through the other on 84% of its frames.
        self.adjacent = coupling[2] if len(coupling) > 2 else ADJACENT_SPREAD
        # Every knob a PRESET may override lives here, so `DIVERSITY` can set
        # any of them without editing code. `pose` still reads module scope
        # for the constants no preset varies - EXTREME, MARGIN and the tendon
        # ratios. Reading an overridable constant directly
        # would make it the one setting a preset silently could not change.
        self.coupling_slack = COUPLING_SLACK
        self.independent = INDEPENDENT
        self.split = SPLIT
        self.pinch_rate = PINCH_RATE
        self.object_rate = OBJECT_RATE
        self.objects = []
        for name, value in (diversity or {}).items():
            if not hasattr(self, name):
                raise ValueError(f"no sampler setting named {name!r}")
            setattr(self, name, value)
        self.rng = rng or np.random.default_rng()

    def settings(self):
        """What this sampler actually drew with. Written beside the pairs so a
        training set says which policy produced it."""
        return {"flex_own": self.flex_own, "spread_own": self.spread_own,
                "adjacent_spread": self.adjacent, "object_rate": self.object_rate,
                "coupling_slack": self.coupling_slack,
                "independent": self.independent, "split": self.split,
                "pinch_rate": self.pinch_rate}

    # ------------------------------------------------------------- geometry
    def geometry(self):
        """One hand's proportions. Held fixed across a trajectory."""
        out = {"knuckle": {}, "length": {}, "direction": {}}
        # One scale for the whole hand plus a small per-bone departure: hands
        # differ mostly in overall size, and a purely per-bone draw would make
        # long-fingered short-palmed hands as common as ordinary ones.
        overall = float(self.rng.normal(1.0, LENGTH_SPREAD * 0.6))
        for digit in DIGITS:
            jitter = self.rng.normal(1.0, LENGTH_SPREAD * 0.5, size=3)
            out["length"][digit] = (self.template["length"][digit]
                                    * overall * jitter)
            out["knuckle"][digit] = (self.template["knuckle"][digit] * overall
                                     * self.rng.normal(1.0, KNUCKLE_SPREAD,
                                                       size=3))
            out["direction"][digit] = self.template["direction"][digit]
        return out

    # ----------------------------------------------------------------- pose
    def pose(self):
        """One plausible configuration: per digit (flex, spread, pip, dip).

        Three constraints, and each was added because dropping it produced
        hands the operator does not make:

        COUPLING. The DIP follows the PIP at about two thirds, the PIP follows
        the knuckle. Four independent angles per finger is mostly postures
        nobody can hold.

        THE MEASURED RANGE, from `pose_ranges`. Written-down limits reached
        173 mm between thumb and index where the real hand's widest was 135.

        SHARED SPREAD, sampled through the measured distribution rather than
        uniformly across it. Fingers abduct together - a hand opens its
        fingers, it does not pick four angles - and drawing them independently
        put the median index-to-ring distance at 78 mm against a real 47.
        """
        out = {}
        # One "how open is this hand" draw, mapped through each digit's OWN
        # measured distribution. That does two things at once: the marginals
        # come out right by construction, and the fingers stay correlated,
        # because they all read the same quantile.
        together = float(self.rng.beta(EXTREME, EXTREME))
        curl = float(self.rng.beta(EXTREME, EXTREME))

        # Most hands move their fingers together; a minority do not, and a
        # subset of THOSE split cleanly into extended and folded - which is
        # what a peace sign, a point and a ring-finger-out all are. See
        # INDEPENDENT.
        loose = self.rng.random() < self.independent
        extended = None
        if loose and self.rng.random() < self.split:
            fingers = [d for d in DIGITS if d != "thumb"]
            count = int(self.rng.integers(1, len(fingers)))
            extended = set(self.rng.choice(fingers, size=count, replace=False))

        # Spread quantiles first, all digits, so neighbours can be held
        # within `adjacent` of each other before any angle is built. Walked
        # index -> middle -> ring -> pinky, each clamped to the one before.
        quantiles, previous = {}, None
        for digit in DIGITS:
            quantile = float(np.clip(
                together + self.rng.normal(0.0, self.spread_own), 0.0, 1.0))
            if digit != "thumb":
                if previous is not None:
                    quantile = float(np.clip(quantile, previous - self.adjacent,
                                             previous + self.adjacent))
                previous = quantile
            quantiles[digit] = quantile

        for digit in DIGITS:
            limits = self.ranges[digit]
            quantile = quantiles[digit]
            band = limits["spread"]
            middle = float(band[len(band) // 2])
            spread = middle + (float(_at(band, quantile)) - middle) * (1.0 + MARGIN)

            if extended is not None and digit != "thumb":
                # A clean split: extended digits near straight, folded ones
                # near shut, each with enough jitter that the set is not eight
                # fixed poses repeated.
                target = 0.12 if digit in extended else 0.88
                fraction = float(np.clip(self.rng.normal(target, 0.10), 0.0, 1.0))
            elif loose:
                fraction = float(self.rng.beta(EXTREME, EXTREME))
            else:
                own = self.flex_own * (2.0 if digit == "thumb" else 1.0)
                fraction = float(np.clip(curl + self.rng.normal(0.0, own),
                                         0.0, 1.0))
            low, high = _padded(limits["flex"])
            flex = low + fraction * (high - low)

            pip = flex * PIP_OF_KNUCKLE * float(
                self.rng.normal(1.0, self.coupling_slack))
            dip = pip * DIP_OF_PIP * float(
                self.rng.normal(1.0, self.coupling_slack))
            out[digit] = (flex, spread,
                          float(np.clip(pip, limits["pip"][0], limits["pip"][-1])),
                          float(np.clip(dip, limits["dip"][0], limits["dip"][-1])))
        return out

    def contact_pose(self, geometry, finger, base, target_mm=PINCH_GAP_MM,
                     steps=12):
        """A pose with the thumb tip `target_mm` from `finger`'s tip, for THIS
        hand's proportions, everything else as in `base`.

        A coarse sweep over the thumb's flex and spread and the finger's curl,
        rather than an optimiser: three free numbers, one distance, well under
        a second. The gestures probe's OK sign and the sampler's pinches are
        the same operation, so they are the same code.
        """
        best, found = None, dict(base)
        thumb = "thumb"
        # The finger's spread sweeps TOWARD the thumb as well: held at -0.10
        # the ring could not be reached at all - 1% of frames against the
        # index's 20% - because a ring pinch is mostly adduction.
        for flex in np.linspace(0.0, 1.0, steps):
            for spread in np.linspace(0.0, 1.4, steps):
                for curl in np.linspace(0.3, 1.2, steps):
                  for adduct in (-0.10, -0.40, -0.70):
                    trial = dict(base)
                    trial[thumb] = (flex, spread, flex * 0.9, flex * 0.5)
                    pip = curl * PIP_OF_KNUCKLE
                    trial[finger] = (curl, adduct, pip, pip * DIP_OF_PIP)
                    full = {d: trial.get(d, (0.0, 0.0, 0.0, 0.0)) for d in DIGITS}
                    points = self.keypoints(geometry, full)
                    gap = float(np.linalg.norm(points[4 * DIGITS.index(thumb) + 4]
                                               - points[4 * DIGITS.index(finger) + 4]))
                    score = abs(gap - target_mm)
                    if best is None or score < best:
                        best, found = score, dict(trial)
        return found

    def trajectory(self, seconds, hz=125.0, keyframe_hz=1.2, geometry=None):
        """A smooth path through pose space, at the sensor's own rate.

        Keyframes drawn and interpolated rather than a random walk: a walk
        wanders without ever committing to a gesture, and what the student has
        to learn is the shape of a hand going somewhere and arriving.

        With a geometry, a `pinch_rate` share of the keyframes are contact
        poses - see PINCH_RATE - so the path approaches, touches and leaves,
        which is the motion the teacher labels with a time-to-contact.
        """
        frames = max(int(seconds * hz), 2)
        count = max(int(seconds * keyframe_hz) + 2, 2)
        keys = [self.uncrossed(geometry) for _ in range(count)]
        at = np.linspace(0, frames - 1, count)
        self.objects = []
        if geometry is not None:
            from . import gestures
            for k in range(count):
                draw = self.rng.random()
                if draw < self.object_rate:
                    radius = float(self.rng.uniform(*OBJECT_RADIUS_MM))
                    keys[k] = gestures.wrap_pose(self, geometry, radius)
                    first = 0 if k == 0 else int((at[k - 1] + at[k]) / 2)
                    last = frames if k + 1 >= count else int((at[k] + at[k + 1]) / 2)
                    item = gestures.cylinder_in_hand(geometry, radius)
                    item["frames"] = [first, last]
                    self.objects.append(item)
                elif draw < self.object_rate + self.pinch_rate:
                    finger = str(self.rng.choice(["index", "middle", "ring"]))
                    contact = self.contact_pose(geometry, finger, keys[k])
                    if not self.crossed(geometry, contact):
                        keys[k] = contact
        # Cosine easing between keyframes: velocity is zero at each one, so the
        # path has no corners for a jerk term to object to.
        path = []
        for frame in range(frames):
            upper = int(np.searchsorted(at, frame, side="right"))
            upper = min(max(upper, 1), count - 1)
            span = max(at[upper] - at[upper - 1], 1e-9)
            blend = 0.5 - 0.5 * np.cos(np.pi * (frame - at[upper - 1]) / span)
            a, b = keys[upper - 1], keys[upper]
            path.append({d: tuple((1 - blend) * np.asarray(a[d])
                                  + blend * np.asarray(b[d]))
                         for d in DIGITS})
        return path

    def crossed(self, geometry, pose, allowance_mm=3.0):
        """Does this pose put a finger through its neighbour? Adjacent
        fingertips and distal joints must keep their order along the line
        from the index knuckle to the ring knuckle; `allowance_mm` lets
        neighbours touch."""
        if geometry is None:
            return False
        points = self.keypoints(geometry, pose)
        knuckle = geometry["knuckle"]
        lateral = np.asarray(knuckle["ring"], float) - np.asarray(knuckle["index"], float)
        lateral = lateral / max(np.linalg.norm(lateral), 1e-9)
        for near, far in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
            a, b = DIGITS.index(near), DIGITS.index(far)
            for slot in (3, 4):                          # distal joint, tip
                if (points[4 * a + slot] - points[4 * b + slot]) @ lateral > allowance_mm:
                    return True
        return False

    def uncrossed(self, geometry, tries=12):
        """A pose whose fingers keep their order, redrawn until they do.

        The census that motivated this: neighbours drawn independently sat
        through each other on 84% of the synthetic frames, and the quantile
        clamp above turned out too loose to stop it - the real hand's 99th
        percentile of adjacent difference is 0.84 of a range. Geometry is
        the honest test, so the draw is checked in millimetres."""
        pose = self.pose()
        for _ in range(tries):
            if not self.crossed(geometry, pose):
                break
            pose = self.pose()
        return pose

    # ------------------------------------------------------------ kinematics
    def keypoints(self, geometry, pose):
        """(21, 3) MANO keypoints in millimetres, wrist at the origin.

        Forward kinematics in the hand's own frame - the same frame the
        template was measured in, which is the frame `Retargeter.rotation`
        maps out of. So these keypoints are drop-in where a recorded hand's
        would be.
        """
        points = np.zeros((21, 3))
        for digit in DIGITS:
            base = np.asarray(geometry["knuckle"][digit], dtype=float)
            rest = np.asarray(geometry["direction"][digit], dtype=float)
            lengths = np.asarray(geometry["length"][digit], dtype=float)
            flex, spread, pip, dip = pose[digit]

            # Spread swings the whole digit about the palm normal; flexion
            # then bends each joint about the digit's own lateral axis, which
            # is perpendicular to both the normal and the bone - so a spread
            # finger still curls into the palm rather than sideways.
            #
            # THE THUMB IS APPROXIMATED HERE and the approximation is measured.
            # A finger abducts about the palm normal, near enough; a thumb
            # OPPOSES, swinging out of the palm plane, about an axis fitted off
            # the `thumb_sweep` block at (0.82, 0.30, 0.50) in (forward,
            # normal, thumbward) - nothing like the normal. Turned about the
            # normal, as it is here, a synthetic thumb cannot quite reach the
            # fingers: its mean fingertip sat 83 mm from the recorded one
            # however the spread band was searched, which is what
            # `THUMB_SPREAD_BAND` above is compensating for. A fitter for the
            # real axis was written and then removed - it was never wired into
            # this rotation, so it was measurement nobody was using.
            at = _rotate(base, PALM_NORMAL, spread)
            direction = _rotate(rest[0], PALM_NORMAL, spread)
            # cross(direction, normal), not cross(normal, direction). The
            # sign is what decides which way a positive angle curls, and the
            # wrong one curls the finger out of the back of the hand: swept
            # against the recording, +flex drove the tip's normal component to
            # -40 mm where a real fist puts it at +35.
            axis = np.cross(direction, PALM_NORMAL)
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 1e-9 else THUMBWARD.copy()

            slot = DIGITS.index(digit)
            points[4 * slot + 1] = at
            for bone, (length, angle) in enumerate(
                    zip(lengths, (flex, pip, dip))):
                direction = _rotate(direction, axis, angle)
                at = at + direction * length
                points[4 * slot + 2 + bone] = at
        return points

    def sequence(self, seconds, hz=125.0):
        """One hand, one motion. (n, 21, 3) millimetres."""
        geometry = self.geometry()
        return np.stack([self.keypoints(geometry, pose)
                         for pose in self.trajectory(seconds, hz, geometry=geometry)])


def _rotate(vector, axis, angle):
    """Rodrigues' rotation of `vector` about the unit `axis` by `angle`.

    Written out rather than through `np.cross`: this runs five thousand
    times per contact keyframe inside `contact_pose`, and numpy's cross
    product spends its time on axis bookkeeping for a 3-vector - measured,
    it was 80% of the sampler's time and forty minutes of a 200-hand draw.
    """
    ax, ay, az = axis
    length = max((ax * ax + ay * ay + az * az) ** 0.5, 1e-12)
    ax, ay, az = ax / length, ay / length, az / length
    vx, vy, vz = vector
    cosine, sine = np.cos(angle), np.sin(angle)
    cx, cy, cz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
    dot = (ax * vx + ay * vy + az * vz) * (1.0 - cosine)
    return np.array([vx * cosine + cx * sine + ax * dot,
                     vy * cosine + cy * sine + ay * dot,
                     vz * cosine + cz * sine + az * dot])


def _at(grid, quantile):
    """Read a value out of a measured quantile grid. Linear between points."""
    grid = np.asarray(grid, dtype=float)
    return np.interp(quantile * (len(grid) - 1), np.arange(len(grid)), grid)


def _padded(grid, margin=MARGIN):
    """A measured range, widened at both ends. See MARGIN."""
    low, high = float(grid[0]), float(grid[-1])
    span = (high - low) * margin
    return low - span, high + span




def fit_coupling(data, retarget, ranges=None, stride=10):
    """How far each finger departs from what the whole hand is doing.

    Two numbers, both measured, and the sampler was wrong about both in
    opposite directions:

      FLEX a hand curls as a unit. Across the recording the per-frame spread of
      the fingers' curl fractions has a median standard deviation of 0.03,
      where the sampler assumed 0.16 - so it was splaying fingers five times
      more than a real hand does, in the case that is meant to be the common
      one.

      SPREAD is far looser: 0.23 against an assumed 0.12. Fingers abduct
      together much less than they curl together.

    A third number, ADJACENT: the 99th percentile of how far a finger's
    spread fraction sits from its neighbour's, frame by frame. It is the
    bound the sampler holds neighbours to, and the reason is the collision
    census - neighbours drawn independently crossed each other on 84% of
    the synthetic frames, and a real hand's never do.

    Fitted on the fractions rather than the angles, because the sampler works
    in fractions of each joint's own measured range and a radian means a
    different thing on each digit.
    """
    from .sessions import hands_from

    ranges = ranges or pose_ranges(data, retarget)
    fingers = [d for d in DIGITS if d not in ("thumb", "pinky")]
    curl, spread, adjacent = [], [], []
    for index in np.flatnonzero(data["tracked"])[::stride]:
        angles = retarget.measure(hands_from(data, index))
        row, side = [], []
        for finger in fingers:
            for name, store in (("flex", row), ("spread", side)):
                band = ranges[finger][name]
                low, high = float(band[0]), float(band[-1])
                value = angles[(finger, ROLE_FOR[name])]
                store.append(np.clip((value - low) / max(high - low, 1e-9), 0, 1))
        curl.append(np.std(row))
        spread.append(np.std(side))
        adjacent.extend(np.abs(np.diff(side)))
    if not curl:
        return FLEX_OWN, SPREAD_OWN, ADJACENT_SPREAD
    return (float(np.median(curl)), float(np.median(spread)),
            float(np.percentile(adjacent, 99)))


def sampler_for(data, meta, retarget, seed=0, diversity=None):
    """The hand sampler, with every range and axis fitted from the recording.

    It stays in the OPERATOR's frame, deliberately. `finish_keypoints` carries
    a generated hand into the robot's the same way `mano_keypoints` carries a
    recorded one, so the two halves of the training set get one transform
    rather than two descriptions of the same intent.

    ONE BUILDER, used by the generator and by the gesture probe. Two of them is
    two rest frames, and that is not hypothetical: the probe had its own, so it
    authored peace and OK signs in one frame and scored them against a student
    trained in another.
    """
    ranges = pose_ranges(data, retarget)
    return HandSampler(template(data, meta), ranges,
                       np.random.default_rng(seed),
                       coupling=fit_coupling(data, retarget, ranges),
                       diversity=diversity)
