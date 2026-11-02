"""A person's true 3D hand -> the robot hand's pose that does the same thing:
the label of what a hand means, for learning to read intent from a camera
(step 1 of the plan: datasets whose hands are measured in 3D, retargeted
here; their photos run through the tracker give what a camera would see).

    retarget = HumanRetarget()
    q, touching = retarget(world, side)   # (21, 3) m, MediaPipe/FreiHAND order -> (J,), (3,)

What carries over is what the hand does, not its shape - the robot's thumb
sits much lower and its fingers are longer for its palm, so no robot pose
has a person's shape. In the palm's frame and palm lengths
(handmotion.in_palm; a right hand mirrored onto the robot's left), a least
squares fit over the robot's joints (within their ranges) asks for:

    directions   each finger segment pointing as the person's (DIRECTION)
    opposition   the thumb tip facing each fingertip as the person's does
                 (OPPOSITION): which fingers the thumb works with
    touches      where the person's thumb tip is within TOUCH_HUMAN palm
                 lengths of a fingertip - touching - the robot's tips
                 together (TOUCH); then, from that pose, the thumb and those
                 fingers alone moved to close them exactly (HandModel.reach,
                 as the touch library makes its touches)
    apart        elsewhere the thumb's distance to each fingertip as the
                 person's, scaled from a person's span to the robot's
                 (APART, soft): open stays open
    rest         a little towards rest (REST), for joints nothing decides

then the hand sim settles into it with its servos (hand_library.HandModel.
settle): fingers stop on each other, so a touch is a contact and nothing
passes through itself. `touching`: which fingers the thumb then touches - a
contact, or for two fingertips pinched at once one contact and the other
within TOUCH_NEAR (the touch library's rule). On 400 FreiHAND hands (true
3D): 81% of the person's pinches became touches before that rule (index 95%,
middle 79%, ring 58% - the robot's low thumb reaches its ring finger worst;
gaps under 0.1 palm lengths 98%), none of their open hands did; finger
segments 10 degrees from the person's, the thumb 21.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .hand_library import TOUCH_NEAR, HandModel
from .handmotion import CHAINS, in_palm, robot_points

DIRECTION, OPPOSITION, TOUCH, APART, REST = 1.0, 0.5, 4.0, 0.3, 0.05
START_MARGIN = 0.05           # rad
TOUCH_HUMAN = 0.2             # palm lengths between a person's thumb tip and a fingertip: touching
HUMAN_OPEN, ROBOT_OPEN = 1.3, 2.6     # thumb tip to fingertips, open hand, palm lengths (FreiHAND 90th pct; robot at rest)
ROBOT_TOUCH = 0.14
FINGERS = ("index", "middle", "ring")
CHOSEN = np.concatenate([CHAINS[name] for name in CHAINS])      # the 16 points' landmarks
TIP = {"thumb": 3, "index": 7, "middle": 11, "ring": 15}          # in the 16


class HumanRetarget:
    def __init__(self, hand=None):
        self.hand = hand or HandModel()
        self.rest = np.zeros(self.hand.model.nq)

    def __call__(self, world, side):
        """A person's hand (21, 3) -> (robot joints (J,), touching (3,) bool)."""
        person = in_palm(np.asarray(world, float), side)[CHOSEN]
        q = self.solve(person)
        pinching = tuple(f for f, d in zip(FINGERS, _pinches(person)) if d < TOUCH_HUMAN)
        if pinching:                     # the touch made exact, as the touch library makes them
            q = self.hand.reach(q, pinching)
        settled = self.hand.settle(q)
        contact = np.array([self.hand.touching(f) for f in FINGERS])
        near = np.array([self.hand.pinch(f) < TOUCH_NEAR for f in FINGERS])
        # Two fingertips at once: one in contact and the other near counts, as
        # in the touch library - the thumb's last link seldom meets both.
        together = contact.any() and np.isin(FINGERS, pinching).sum() > 1
        return settled, contact | (together & near & np.isin(FINGERS, pinching))

    def solve(self, person):
        """The least-squares pose for the person's 16 points (palm frame)."""
        hand = self.hand
        wanted = _segments(person)
        pinch = _pinches(person)
        facing = np.array([_unit(person[TIP[f]] - person[TIP["thumb"]]) for f in FINGERS])
        touch = pinch < TOUCH_HUMAN
        apart = ROBOT_TOUCH + (pinch - TOUCH_HUMAN) * (ROBOT_OPEN - ROBOT_TOUCH) / (HUMAN_OPEN - TOUCH_HUMAN)

        def residuals(q):
            hand.data.qpos[:] = q
            mujoco.mj_kinematics(hand.model, hand.data)
            robot = robot_points(hand.model, hand.data, hand.chain, hand.palm)
            out = [DIRECTION * (_segments(robot) - wanted).ravel(), REST * (q - self.rest)]
            for k, finger in enumerate(FINGERS):
                gap = robot[TIP[finger]] - robot[TIP["thumb"]]
                out.append(OPPOSITION * (_unit(gap) - facing[k]))
                if touch[k]:
                    out.append(TOUCH * gap)
                else:
                    out.append([APART * (np.linalg.norm(gap) - apart[k])])
            return np.concatenate(out)

        # Started a little inside the joints' limits: at rest the thumb sits on
        # one, and from there the solver's first step fails and it stops.
        start = np.clip(self.rest, hand.low + START_MARGIN, hand.high - START_MARGIN)
        found = least_squares(residuals, start, bounds=(hand.low, hand.high))
        return found.x


def _pinches(points):
    """The thumb tip's distance to each fingertip, of 16 points (3,)."""
    return np.array([np.linalg.norm(points[TIP["thumb"]] - points[TIP[f]]) for f in FINGERS])


def _segments(points):
    chains = points.reshape(4, 4, 3)
    segments = chains[:, 1:] - chains[:, :-1]
    return segments / np.linalg.norm(segments, axis=-1, keepdims=True)


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)
