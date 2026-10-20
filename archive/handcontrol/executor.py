"""Closing to contact: the piece that makes a grasp work without an object model.

THE IDEA IS THE OPERATOR'S. If you are trying to grab something, the goal is
the fingers closing, and how well they wrap is the object's business, not
the mapping's. So the mapping keeps sending "close" and this watches the
motors: when a finger pushing to close starts drawing current, it has met
something - the object, or its own palm - and from then on it is HELD at
that point plus a small squeeze, which the servo's current limit turns into
a grip force. When the operator opens past where it stalled, it FOLLOWS
again. Any shape, no model, nothing learned.

WHAT COUNTS AS A STALL: present current past `STALL_MA`, per finger, over
its flexion joints. Adduction is left alone - a finger pressed sideways is a
different event. Nothing else: the position test that used to sit beside it
was removed after the bench showed it could not work, and `STALL_MA` records
why. That makes current the ONLY signal, so a link that does not report it
cannot grip at all - `step` refuses to pretend otherwise.

This sits in the bus thread between the mapping's target and the rate
limiter, so a held finger is held whatever the mapping does next, and the
simulator's picture is the command the executor let through.
"""

from __future__ import annotations

import numpy as np

from .robot.model import FINGERS, joint_role

# WHAT COUNTS AS A STALL: PRESENT CURRENT, AND NOTHING ELSE.
#
# This used to be "the command is STALL_RAD ahead of the encoder, for
# STALL_SECONDS, while the encoder is not moving", with current as a shortcut
# beside it. The bench says the position half cannot work on this hand, and
# that it was the only thing that could produce a false hold:
#
#                        free air     on a cylinder
#     command lead        28.5 deg      17.9 deg      ANTI-CORRELATED
#     peak current          77 mA        280 mA      3.6x separation
#
# THE LEAD IS BIGGER IN FREE AIR THAN ON CONTACT. In free air the servo is
# still climbing its acceleration ramp so the command outruns it badly; on
# contact it is already at speed and the object stops it gradually. No
# threshold separates 17.9 from 28.5 in the direction wanted - anything that
# fires on the touch fires harder on nothing at all.
#
# The "not moving" half does not rescue it. The finger keeps creeping into the
# object at 0.18-0.51 rad/s, so the encoder reads as stopped on 10% of
# post-contact frames. Between them the two halves detected nothing and
# carried all of the false-trigger risk, which is what `STALL_SECONDS` needed
# its 2.14x margin to cover.
#
# Current is left, and it is the honest signal anyway: a position-controlled
# servo that is blocked commands more torque, so a stall RAISES current by
# construction. Measured on an index finger against a rigid cylinder, one run.
STALL_MA = 180          # present current past which a finger is on something
FREE_AIR_MA = 77        # what a finger running free actually draws, measured
#
# 180 is 2.3x over the 77 mA a free-running finger draws, and fires 0.59 s
# after contact with the finger 9.9 deg into the object. Lowering it buys
# latency and almost no depth - 120 mA fires 0.14 s sooner and still 8.8 deg
# in - because what lags is the current ramp, not the travel. It is not a
# release threshold: see `RELEASE_RAD`.
SQUEEZE_RAD = 0.06      # held past the stall point, for grip through the current limit
RELEASE_RAD = 0.05      # the command must open past the stall point by this to let go

# What a digit closes with: everything except the joint that moves it
# sideways. Keyed on what the joint MOVES, via `model.joint_role`, and not on
# its name - the same rule, and for the same reason, as `hardware.default_sign`.
#
# MATCHING NAMES GETS THIS WRONG TWICE. Selecting the segments "proximal",
# "middle" and "distal" out of the joint names picked up `middle_adduction_joint`,
# because the finger is called middle and the test was a substring; and on the
# thumb it took `thumb_proximal` - which `THUMB_ROLE` says is the SPREAD - while
# leaving out `thumb_adduction`, which is the flexion. So a held thumb was
# squeezed sideways across the palm while the joint that actually closes it was
# never watched for contact at all. `default_sign` carries the story of the
# first time this exact substring split was made.
CLOSING_ROLES = ("knuckle", "middle", "tip")

# The thumb pairs the student's contact head scores, in its own order.
PINCH_PAIRS = ("index", "middle", "ring")
PINCH_SURE = 0.5        # probability past which a stall is read as a pinch


class GraspExecutor:
    """Per-finger FOLLOW / HOLD, on the closing joints."""

    def __init__(self, joint_names):
        self.names = list(joint_names)
        self.fingers = {}
        for finger in FINGERS:
            self.fingers[finger] = [k for k, name in enumerate(self.names)
                                    if name.startswith(finger + "_")
                                    and joint_role(name) in CLOSING_ROLES]
        self.enabled = False
        self.holding = {finger: False for finger in self.fingers}
        self.hold_at = {finger: None for finger in self.fingers}
        self.currents = None

    @staticmethod
    def pinched(pinching, sure=PINCH_SURE):
        """Which digits the contact head says are closing on each other.

        The thumb is in the set whenever any pair is, because a pinch stalls
        both sides of it.
        """
        if pinching is None:
            return frozenset()
        out = {finger for finger, p in zip(PINCH_PAIRS, pinching) if p > sure}
        return frozenset(out | {"thumb"} if out else out)

    def reset(self):
        for finger in self.fingers:
            self.holding[finger] = False
            self.hold_at[finger] = None

    def step(self, target, positions, currents=None, pinching=None):
        """The target the bus should carry: `target` with held fingers pinned.

        `positions` are the latest encoder angles (None until a read lands) -
        needed to know WHERE to hold - and `currents` the present current in
        mA per joint, which is how a stall is detected at all. Off, blind, or
        without currents, it passes the target straight through.

        `pinching` is the student's contact head: one probability per thumb
        pair, in `PINCH_PAIRS` order. A FINGER THAT IS PINCHING IS NOT
        GRASPING. Both look identical from here - a finger pressing on
        something raises its current either way - but a stall against the
        operator's own thumb is a gesture, not an object, and latching it
        would make every pinch a grip the operator has to deliberately open
        out of. Told that a pair is closing on itself, this follows it instead
        of holding it.
        """
        target = np.asarray(target, dtype=float).copy()
        if not self.enabled or positions is None or currents is None:
            return target
        positions = np.asarray(positions, dtype=float)
        pinched = self.pinched(pinching)
        for finger, joints in self.fingers.items():
            if not joints:
                continue
            if self.holding[finger]:
                held = self.hold_at[finger]
                if float(np.mean(target[joints] - held)) < -RELEASE_RAD:
                    self.holding[finger] = False
                    self.hold_at[finger] = None
                else:
                    target[joints] = held + SQUEEZE_RAD
                continue
            # CURRENT ALONE - see `STALL_MA` for the measurement that removed
            # the position test that used to sit beside it. No timer either:
            # current on this hand rises over half a second of pushing, so it
            # is already its own persistence check.
            hot = float(np.max(np.abs(np.asarray(currents)[joints]))) > STALL_MA
            if hot and finger not in pinched:
                self.holding[finger] = True
                self.hold_at[finger] = positions[joints].copy()
                target[joints] = self.hold_at[finger] + SQUEEZE_RAD
        return target

    def state(self):
        return {"enabled": self.enabled,
                "holding": [f for f, on in self.holding.items() if on]}
