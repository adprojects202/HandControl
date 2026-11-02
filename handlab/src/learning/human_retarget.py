"""A person's true 3D hand -> the robot hand's pose that does the same thing:
the label of what a hand means, for learning to read intent from a camera
(step 1 of the plan: datasets whose hands are measured in 3D, retargeted
here; their photos run through the tracker give what a camera would see).

    retarget = HumanRetarget()
    q, touching = retarget(world, side)   # (21, 3) m, MediaPipe/FreiHAND order -> (J,), (3,)
    q, touching = retarget(world, side, holding, start)   # a hand on an object, in motion

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

A hand on an object (`holding`: the thumb and a fingertip both touching it,
as datasets of people handling objects label): no touch is made there,
however close the tips - the object is between them - and the gap asked
for is the person's in metres, the object's size, not scaled to the robot's
span; asked softly, as an open hand's (APART). In GRAB 65% of the right
hands' thumb-index gaps under TOUCH_HUMAN are so (left 45%). Asked as firmly
as a touch, finger segments went 1.6 degrees further from the person's and
steps over 20 degrees doubled - the robot's proportions are not a person's,
and the labels flicker; an object's real shape comes with its mesh.

In motion, `start` - the last frame's pose - is where the solve begins and
what it is held towards (STILL): the robot's thumb can do much the same in
more than one way, and solved afresh each frame it hopped between them. On
GRAB takes (20 Hz) steps over 20 degrees went from 3.6% to 1.3%, the
largest 73 to 43 (the people's own segments turn up to 25 a step, 99th
percentile), finger segments 7.8 degrees from the person's against 7.6.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .hand_library import TOUCH_NEAR, HandModel
from .handmotion import CHAINS, MIDDLE_KNUCKLE, WRIST, in_palm, robot_points

DIRECTION, OPPOSITION, TOUCH, APART, REST = 1.0, 0.5, 4.0, 0.3, 0.05
STILL = 0.5                   # per rad from `start` (the last frame's pose), in motion
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

    def __call__(self, world, side, holding=None, start=None):
        """A person's hand (21, 3) -> (robot joints (J,), touching (3,) bool).
        holding (3,) bool: the thumb and the index, middle, ring tip both on an
        object; start (J,): the pose to solve from (rest)."""
        world = np.asarray(world, float)
        person = in_palm(world, side)[CHOSEN]
        holding = np.zeros(len(FINGERS), bool) if holding is None else np.asarray(holding, bool)
        # The person's gap in the robot's palm lengths, unscaled: the object's size.
        held = np.where(holding, _pinches(person) * np.linalg.norm(world[MIDDLE_KNUCKLE] - world[WRIST])
                        / self.hand.palm[2], np.nan)
        q = self.solve(person, held, start)
        pinching = tuple(f for f, d, h in zip(FINGERS, _pinches(person), holding)
                         if d < TOUCH_HUMAN and not h)
        if pinching:                     # the touch made exact, as the touch library makes them
            q = self.hand.reach(q, pinching)
        settled = self.hand.settle(q)
        contact = np.array([self.hand.touching(f) for f in FINGERS])
        near = np.array([self.hand.pinch(f) < TOUCH_NEAR for f in FINGERS])
        # Two fingertips at once: one in contact and the other near counts, as
        # in the touch library - the thumb's last link seldom meets both.
        together = contact.any() and np.isin(FINGERS, pinching).sum() > 1
        return settled, contact | (together & near & np.isin(FINGERS, pinching))

    def solve(self, person, held=None, start=None):
        """The least-squares pose for the person's 16 points (palm frame);
        held (3,): the gaps an object keeps (robot palm lengths, NaN where none);
        start: the last frame's pose, solved from and held towards (STILL)."""
        hand = self.hand
        wanted = _segments(person)
        pinch = _pinches(person)
        facing = np.array([_unit(person[TIP[f]] - person[TIP["thumb"]]) for f in FINGERS])
        held = np.full(len(FINGERS), np.nan) if held is None else np.asarray(held, float)
        touch = (pinch < TOUCH_HUMAN) & np.isnan(held)
        apart = ROBOT_TOUCH + (pinch - TOUCH_HUMAN) * (ROBOT_OPEN - ROBOT_TOUCH) / (HUMAN_OPEN - TOUCH_HUMAN)
        apart = np.where(np.isnan(held), apart, held)

        def residuals(q):
            hand.data.qpos[:] = q
            mujoco.mj_kinematics(hand.model, hand.data)
            robot = robot_points(hand.model, hand.data, hand.chain, hand.palm)
            out = [DIRECTION * (_segments(robot) - wanted).ravel(), REST * (q - self.rest)]
            if previous is not None:
                out.append(STILL * (q - previous))
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
        previous = start
        start = np.clip(self.rest if start is None else start, hand.low + START_MARGIN,
                        hand.high - START_MARGIN)
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
