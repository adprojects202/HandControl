"""Human hand (Leap) -> h4nd_left joint angles.

The two hands do not correspond. The Leap reports five digits of four bones
each, in a camera frame measured in millimetres; the robot has four digits of
four revolute joints, an unhuman thumb, and joint limits of [-20, 110] degrees
that no knuckle obeys. So nothing here is a coordinate transform of anything -
it is a set of angles measured off the operator and a decision about which
robot joint each one drives.

This is the one file in the package that binds to THIS robot. The geometry, the
filtering and the calibration file format beside it do not; if the mapping is
ever replaced - see the keyvector approach in the retargeting notes - this is
the file that gets a sibling, and the rest is reused.

Everything is addressed by ROLE (spread, knuckle, middle, tip) rather than by
joint name, because the thumb's names lie: `thumb_proximal_joint` swings the
thumb across the palm and `thumb_adduction_joint` lifts it along the fingers,
which is the opposite of what a finger's identically-named joints do.
`model.joint_role` already holds that table, measured off the hardware, so
this reads the mapping from there instead of matching on strings.

Two ways to drive it, and both are here because they fail differently:

  ANGLES (`joints`) reads each human joint as a signed angle - flexion about
  the hand's own hinge, spread in the palm plane - and drives the matching
  robot joint with it. Cheap, stable, and it always produces an answer; what it
  will not do is put a fingertip anywhere in particular, because the robot's
  segments are not the operator's.

  TIP (`tips`) maps each fingertip into the robot's palm frame, scaled by hand
  length, and leaves the joints to `HandSim.solve_tip`. That gets pinches and
  contact right - the thing angle mapping cannot do - and in exchange it is
  slower, it can be handed points outside the robot's reach, and a solution
  near a singularity moves joints the operator did not move.

Which way the robot's palm faces, which side its thumb is on, and which sign
of a spread joint means "outward" are all measured from the model, so a
re-export or a change of chirality carries through instead of silently
inverting a finger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..leap import DIGITS
from ..robot.model import FINGERS, ROLES, TIP_LINKS, roles_in_order, spread_signs
from .calibration import CALIBRATION, fit_ranges, load_fit, save_fit
from .geometry import (flex_angle, flexion_axis, hand_frame, hand_length,
                       robot_axes, spread_angle, thumb_flexion_axis, unit)

# the human ring finger is the one that lines up with the robot's - mapping
# pinky -> ring instead would move the wrong finger to close a grip.
DIGIT_FOR = {"index": "index", "middle": "middle", "ring": "ring",
             "thumb": "thumb"}
CURLED = ("index", "middle", "ring")

# Robot radians per human radian, by role. One-to-one on the three flexion
# roles - those ranges genuinely are comparable - and deliberately less on
# spread, where the operator's ~15 degrees against the robot's 110 would turn
# a small wobble of the hand into the fingers flying apart.
#
# The thumb gets its own row for every role. Its robot joints are not the
# human's joints, so a gain that is right for a knuckle means nothing there.
def _neutralised(gains, ranges):
    """The gains, with the FITTED roles set to 1 and the rest left alone.

    A fitted range already carries the scaling its gain was standing in for,
    so applying both would shrink a gesture that already fits. But a fit is
    routinely partial - `calibration.fit_ranges` skips every spread key when
    the SPREAD capture is missing, and drops any joint that moved less than
    `MIN_SPAN` - and neutralising the whole table regardless put finger spread
    on gain 1.0 with no fitted range to take over from the 0.6 it lost, which
    is the fingers flying apart that `GAINS` exists to prevent.
    """
    fitted = {f"thumb_{role}" if finger == "thumb" else role
              for finger, role in ranges}
    return {role: (1.0 if role in fitted else value)
            for role, value in gains.items()}


GAINS = {"spread": 0.6, "knuckle": 1.0, "middle": 1.0, "tip": 1.0,
         "thumb_spread": 1.15, "thumb_knuckle": 1.0, "thumb_middle": 1.1,
         "thumb_tip": 1.0}

# Where a relaxed thumb sits: the in-plane angle from the palm's forward
# direction out to the thumb metacarpal. The robot's thumb knuckle is driven
# by the difference from this, so a hand doing nothing commands nothing.
THUMB_REST_SPREAD = math.radians(55.0)

# A human DIP does not move independently: it is tendon-coupled to the PIP in
# front of it, at roughly two thirds of its angle. The Leap has to infer that
# joint anyway - it is the one hidden behind the finger in front, and the SDK
# offers no way to tell an inferred joint from a measured one - so blending its
# reported value toward the anatomical estimate trades a number the tracker
# guessed for one the hand's own tendons impose.
#
# DISTAL_TRUST is how much of the reported value survives. 0 ignores the sensor
# on that joint entirely; 1 is the old behaviour.
DISTAL_COUPLING = 2.0 / 3.0
DISTAL_TRUST = 0.4


@dataclass
class Retargeter:
    """Turns Leap hands into robot joint vectors, and remembers the last one.

    Stateful on purpose. Tracking drops out constantly - a hand leaves the
    cone, a finger occludes another - and a robot that snapped to zero on
    every dropped frame would be unusable, and with the servos live, unsafe.
    Every path out of here goes through the same smoothing and rate limit, so
    angle mode and tip mode move the hand at the same speed.
    """

    model: object
    joint_names: list = None
    gains: dict = field(default_factory=lambda: dict(GAINS))
    smoothing: float = 0.35            # per-frame blend toward the new pose
    max_rate: float = 6.0              # rad/s ceiling on the commanded angles
    thumb_mode: str = "angles"         # or "pinch"
    scale: float = 1.0                 # operator multiplier on the tip mapping
    mirror: bool = False               # drive with the opposite hand; see `frame`
    distal_trust: float = DISTAL_TRUST # how far to believe a joint it cannot see

    def __post_init__(self):
        self.joint_names = list(self.joint_names or self.model.joint_names())
        self.lower, self.upper = self.model.limits(self.joint_names)
        self.index = {n: i for i, n in enumerate(self.joint_names)}
        self.roles = {f: dict(roles_in_order(self.model, f)) for f in FINGERS}
        self.spread_sign = spread_signs(self.model, self.joint_names)
        self.forward, self.normal, self.thumbward = robot_axes(self.model,
                                                               self.joint_names)
        zero = np.zeros(len(self.joint_names))
        self.rest = {f: self.model.link_pose(TIP_LINKS[f], zero,
                                             self.joint_names)[0]
                     for f in FINGERS}
        self.robot_length = float(np.linalg.norm(self.rest["middle"]))
        self.hand_scale = None         # metres of robot per mm of operator
        self.offset = np.zeros(3)      # robot-frame shift, set by calibration
        self.calibrated = False
        # (finger, role) -> the human angles that span this joint's travel.
        # Empty means every joint is on its fixed gain; see `fit_ranges`.
        self.ranges = {}
        self.command = np.zeros(len(self.joint_names))
        self.raw = np.zeros(len(self.joint_names))

    # -------------------------------------------------------- which hand
    def chirality(self):
        """Which of the operator's hands this robot IS, measured off the model.

        (forward, normal, thumbward) is a right-handed triple for a right hand
        and a left-handed one for a left hand - that is what chirality means -
        so the sign of the determinant answers it, for the operator and the
        robot alike, with no table to keep in step.

        Measured rather than read off `--hand`, deliberately: that label names
        the hand on the bench and `model.MIRRORED_FOR` records that the label
        and the geometry disagree on purpose. Only the geometry can settle
        which hand to hold over the sensor.
        """
        determinant = np.linalg.det(np.array([self.forward, self.normal,
                                              self.thumbward]))
        return "right" if determinant > 0 else "left"

    def frame(self, hand):
        """The operator's frame, mirrored if asked.

        Two opposite-chirality hands cannot correspond both anatomically and
        in world motion, and this chooses which. Normally the mapping is
        anatomical: your index is its index, and spreading outward spreads
        outward. Hold the WRONG hand over the sensor and that still holds -
        but "toward my thumb" is now the opposite direction in the room, so
        the robot appears to move the wrong way.

        `mirror` trades the one for the other: world motion follows your hand,
        and in exchange your index drives the robot's index the anatomically
        opposite way. It is for when the hand you have free is not the hand
        the robot is. Matching hands is better; this beats nothing.
        """
        forward, normal, thumbward = hand_frame(hand)
        return (forward, normal, -thumbward) if self.mirror else (forward,
                                                                  normal,
                                                                  thumbward)

    def joint(self, finger, role):
        return self.roles[finger][role]

    def gain(self, finger, role):
        return self.gains[f"thumb_{role}" if finger == "thumb" else role]

    def limit(self, finger, role):
        i = self.index[self.joint(finger, role)]
        return self.lower[i], self.upper[i]

    def travel(self, finger, role):
        """The slice of a joint's range that a fitted human range maps onto.

        Not the whole declared range. A flexion joint's lower limit is -20
        degrees of hyperextension, and mapping a relaxed open hand onto that
        would leave the robot's neutral pose bent backwards; the operator's
        open hand is straight, so it maps to zero. Finger spread gets a
        symmetric band instead - the URDF gives index +110/-20 and ring
        +20/-110, and using either asymmetry whole would make two mechanically
        identical fingers spread by different amounts for the same gesture.
        """
        lower, upper = self.limit(finger, role)
        if role == "spread" and finger != "thumb":
            span = min(abs(lower), abs(upper))
            return -span, span
        return max(0.0, lower), upper

    def fit_ranges(self, poses):
        """Map the operator's own range of motion onto the robot's.

        `poses` is {name: measured angles} from `measure`, for the three
        captures: open, fist, spread. Fitted, a full fist is a full fist and a
        flat hand is flat, whoever is doing it.

        The fit itself lives in `calibration.py` and knows nothing about this
        robot; what this adds is which keys exist and which of them are
        spreads, both of which come from the model.
        """
        keys = [(finger, role) for finger in FINGERS for role in ROLES]
        spread_keys = {(finger, "spread") for finger in FINGERS
                       if finger != "thumb"}
        ranges = fit_ranges(poses, keys, spread_keys)
        if not ranges:
            return False
        self.ranges = ranges
        self.gains = _neutralised(self.gains, ranges)
        return True

    def clear_ranges(self):
        self.ranges = {}
        self.gains = dict(GAINS)

    # ------------------------------------------------------------ on disk
    def _key_for_name(self):
        return {self.joint(f, r): (f, r) for f in FINGERS for r in ROLES}

    def save(self, path=CALIBRATION):
        return save_fit(path,
                        {self.joint(*key): span
                         for key, span in self.ranges.items()},
                        self.hand_scale, self.offset, self.calibrated)

    def load(self, path=CALIBRATION):
        """Restore a previous fit. Missing file is the normal first run."""
        ranges, scale, offset, calibrated = load_fit(path, self._key_for_name())
        if ranges:
            self.ranges = ranges
            self.gains = _neutralised(self.gains, ranges)
        if calibrated:
            self.hand_scale, self.offset, self.calibrated = scale, offset, True
        return bool(ranges) or self.calibrated

    # ------------------------------------------------------- angle mapping
    def measure(self, hand):
        """Every human angle this cares about, in radians, keyed (finger, role).

        Split out from `joints` so it can be printed. Reading this table with
        a hand over the sensor is how you find out whether a gain is wrong
        rather than a sign.
        """
        forward, normal, thumbward = self.frame(hand)
        hinge = flexion_axis(forward, normal)
        angles = {}

        middle_base = hand.bones["middle"][0]
        for finger in CURLED:
            digit = DIGIT_FOR[finger]
            metacarpal = hand.bone_direction(digit, "metacarpal")
            proximal = hand.bone_direction(digit, "proximal")
            intermediate = hand.bone_direction(digit, "intermediate")
            distal = hand.bone_direction(digit, "distal")

            angles[finger, "knuckle"] = flex_angle(metacarpal, proximal, hinge)
            angles[finger, "middle"] = flex_angle(proximal, intermediate, hinge)

            # The distal joint, blended toward what the PIP in front of it
            # implies. Fingers only: a thumb IP is usually in plain view of the
            # camera and is not coupled the same way, so it is left alone.
            measured = flex_angle(intermediate, distal, hinge)
            implied = DISTAL_COUPLING * angles[finger, "middle"]
            trust = min(max(self.distal_trust, 0.0), 1.0)
            angles[finger, "tip"] = trust * measured + (1.0 - trust) * implied

            # Which way "away from the middle finger" points for this finger,
            # in the operator's own frame. The middle finger is the reference
            # and has no outward, so it is measured against the palm instead.
            if finger == "middle":
                reference, outward = forward, thumbward
            else:
                offset = hand.bones[digit][0] - middle_base
                reference = metacarpal
                outward = thumbward * np.sign(np.dot(offset, thumbward) or 1.0)
            angles[finger, "spread"] = spread_angle(reference, proximal,
                                                    normal, outward)

        angles.update(self._thumb(hand, forward, normal, thumbward, hinge))
        return angles

    def _thumb(self, hand, forward, normal, thumbward, hinge):
        """The thumb, where the two hands agree least.

        The Leap gives every digit four bones, so a thumb gets a zero-length
        metacarpal to pad it out: its `proximal` bone is anatomically the
        metacarpal, and everything shifts down one.

        Against that, the robot thumb is a swing and three hinges. Its spread
        role carries the tip toward the palm - opposition - and its knuckle
        role carries it up toward the fingers. So the operator's out-of-plane
        lift drives the spread, the closing of the thumb-index angle drives
        the knuckle, and the two phalanx angles drive the rest.
        """
        metacarpal = hand.bone_direction("thumb", "proximal")
        proximal = hand.bone_direction("thumb", "intermediate")
        distal = hand.bone_direction("thumb", "distal")
        # Not the palm normal: the tip curls within the palm plane, so its
        # sense IS the chirality - see `thumb_flexion_axis`.
        thumb_hinge = thumb_flexion_axis(forward, thumbward)

        # Out of the palm plane toward the palm side: opposition.
        oppose = math.asin(float(np.clip(np.dot(unit(metacarpal), normal),
                                         -1.0, 1.0)))
        # Within the palm plane: how far the thumb has closed from rest.
        close = THUMB_REST_SPREAD - spread_angle(forward, metacarpal, normal,
                                                 thumbward)

        if self.thumb_mode == "pinch":
            # The Leap's own pinch estimate fuses thumb and finger, so it
            # survives the thumb being half-hidden behind the fingers - which
            # is exactly the pose where its joint angles get least
            # trustworthy, and the one that decides whether the hand grips.
            # Fed in as a fraction of travel, so it needs no gain.
            strength = float(np.clip(hand.pinch_strength, 0.0, 1.0))
            return {("thumb", "spread"): 0.85 * strength
                                         * self.limit("thumb", "spread")[1]
                                         / max(self.gain("thumb", "spread"), 1e-6),
                    ("thumb", "knuckle"): 0.75 * strength
                                          * self.limit("thumb", "knuckle")[1]
                                          / max(self.gain("thumb", "knuckle"), 1e-6),
                    ("thumb", "middle"): flex_angle(metacarpal, proximal, hinge),
                    ("thumb", "tip"): flex_angle(proximal, distal, thumb_hinge)}

        return {("thumb", "spread"): max(oppose, 0.0),
                ("thumb", "knuckle"): close,
                ("thumb", "middle"): flex_angle(metacarpal, proximal, hinge),
                ("thumb", "tip"): flex_angle(proximal, distal, thumb_hinge)}

    def to_vector(self, angles):
        """Measured human angles -> a robot joint vector, gains and signs applied.

        A spread joint additionally picks up `spread_signs`: index and ring
        share one axis, so the same positive angle spreads one finger and
        closes the other, and only the sign table knows which is which.
        """
        q = self.raw.copy()
        for (finger, role), value in angles.items():
            name = self.joint(finger, role)
            fitted = self.ranges.get((finger, role))
            if fitted:
                low, high = fitted
                start, end = self.travel(finger, role)
                value = start + (value - low) / (high - low) * (end - start)
            value = value * self.gain(finger, role)
            if role == "spread" and finger != "thumb":
                value *= self.spread_sign[name]
            q[self.index[name]] = value
        return q

    def to_vector_raw(self, angles):
        """Measured human angle straight onto the matching robot joint.

        No gain, no fitted range, no calibration - one radian of you is one
        radian of it, clamped where the joint stops. This is the baseline
        every other mapping should have to beat, and it is worth having on a
        button precisely because it is the one thing that cannot be subtly
        misconfigured.

        The one correction kept is `spread_signs`, and it is not a tuning
        choice: the three finger adduction joints share a single axis, so the
        same positive angle spreads the index and closes the ring. Dropping
        that would not be rawer, it would be wrong - the ring finger would
        visibly move the opposite way to yours.
        """
        q = self.raw.copy()
        for (finger, role), value in angles.items():
            name = self.joint(finger, role)
            if role == "spread" and finger != "thumb":
                value = value * self.spread_sign[name]
            q[self.index[name]] = value
        return q

    def direct(self, q, dt=1.0 / 60.0):
        """Clamp and rate-limit, without the smoothing blend.

        The rate limit stays because it is the safety, not a preference: it is
        what stops a tracking glitch becoming a fast movement on sixteen live
        servos. The exponential blend is what adds lag, and raw mode is here to
        be seen without it.
        """
        target = np.clip(np.asarray(q, dtype=float), self.lower, self.upper)
        step = self.max_rate * max(dt, 1e-4)
        self.raw = target
        self.command = self.command + np.clip(target - self.command,
                                              -step, step)
        return self.command

    def joints(self, hand, dt=1.0 / 60.0):
        """A full clamped joint vector for `hand`, smoothed and rate-limited."""
        self.raw = self.to_vector(self.measure(hand))
        return self._settle(self.raw, dt)

    # --------------------------------------------------------- tip mapping
    def rotation(self, hand):
        """The 3x3 that says a point of the operator's hand in the robot's axes.

        Their forward becomes our forward, their palm normal our palm normal,
        their thumb side our thumb side - so it is the change of basis between
        two frames that were each measured from the thing they describe, and
        it stays right for a left hand over the sensor or a mirrored robot.

        Millimetres in, millimetres out: scale and placement are `place`'s job.
        """
        return (np.array([self.forward, self.normal, self.thumbward]).T
                @ np.array(self.frame(hand)))

    def _in_robot_axes(self, hand):
        """Each fingertip relative to the palm, rotated into the robot's frame
        and nothing else. What the calibration fit is fitted to."""
        rotation = self.rotation(hand)
        return {finger: rotation @ (hand.tip(DIGIT_FOR[finger]) - hand.palm)
                for finger in FINGERS}

    def place(self, hand, origin=(0.0, 0.0, 0.0)):
        """A function mapping device-frame points to robot world metres.

        One transform, built once per frame, used for the fingertips the IK
        chases and for the skeleton drawn beside them - so what you see on
        screen is literally what the solver was handed, and a mapping error
        looks like a hand in the wrong place rather than a finger that is
        subtly off.
        """
        if self.calibrated:
            scale, shift = self.hand_scale, self.offset
        else:
            scale = self.robot_length / max(hand_length(hand), 1e-6)
            shift = np.zeros(3)
        matrix = self.rotation(hand) * (self.scale * scale)
        base = np.asarray(origin, dtype=float) + shift

        def mapped(points):
            return base + np.atleast_2d(np.asarray(points, dtype=float)
                                        - hand.palm) @ matrix.T
        return mapped

    def skeleton(self, hand, origin=(0.0, 0.0, 0.0)):
        """The operator's hand, all five digits, in robot world metres.

        The pinky is in it even though the robot has no little finger. It
        costs nothing to draw and it is the clearest possible statement of
        what the mapping is throwing away.
        """
        mapped = self.place(hand, origin)
        return {digit: mapped(hand.bones[digit]) for digit in DIGITS}

    def calibrate(self, hand):
        """Fit scale and offset so a flat operator hand IS the robot's zero pose.

        Without this the tip mapping is off by a constant nobody can see the
        cause of: the operator's tips are measured from the middle of the palm
        and the robot's from the base of its palm link, so an honestly scaled
        flat hand still lands short and a finger-width inside the palm. The
        finger IK then spends its range correcting a fixed error.

        One scale and one translation, least-squares over the three fingers -
        four unknowns, nine equations. The thumb is left out of the fit and
        rides on the result: the robot's zero-pose thumb points out sideways
        and backwards, nothing a human thumb does, so fitting to it would drag
        the fingers off to satisfy a pose that is not a neutral hand.

        Call it with an open, flat hand. `--no-calibrate` keeps the raw
        length-ratio scaling instead.
        """
        vectors = self._in_robot_axes(hand)
        source = np.array([vectors[f] for f in CURLED])
        target = np.array([self.rest[f] for f in CURLED])
        centred_source = source - source.mean(axis=0)
        centred_target = target - target.mean(axis=0)
        spread = float((centred_source * centred_source).sum())
        if spread < 1e-9:
            return False
        self.hand_scale = float((centred_source * centred_target).sum()) / spread
        self.offset = target.mean(axis=0) - self.hand_scale * source.mean(axis=0)
        self.calibrated = True
        return True

    def tips(self, hand, origin=(0.0, 0.0, 0.0)):
        """Fingertip targets in world metres, for `HandSim.solve_tip`.

        `origin` is where the robot's palm link sits in the world - the `pos`
        the entity was added at - because the solver works in world
        coordinates and the palm is not at the origin.

        Uncalibrated, the scale is the ratio of the two hands' lengths, which
        is right in size and wrong in placement; see `calibrate`.
        """
        mapped = self.place(hand, origin)
        return {finger: mapped(hand.tip(DIGIT_FOR[finger]))[0]
                for finger in FINGERS}

    # ----------------------------------------------------------- smoothing
    def _settle(self, target, dt=1.0 / 60.0):
        target = np.clip(target, self.lower, self.upper)
        blended = self.command + self.smoothing * (target - self.command)
        step = self.max_rate * max(dt, 1e-4)
        self.command = self.command + np.clip(blended - self.command, -step, step)
        return self.command

    def accept(self, q, dt=1.0 / 60.0):
        """Push an externally solved vector - IK output - through the same
        smoothing, so both modes move the hand at the same speed."""
        self.raw = np.clip(np.asarray(q, dtype=float), self.lower, self.upper)
        return self._settle(self.raw, dt)

    def relax(self, rate=0.08):
        """Where the hand drifts when tracking is lost, if it is not to hold."""
        self.raw = self.raw * (1.0 - rate)
        self.command = self.command * (1.0 - rate)
        return self.command
