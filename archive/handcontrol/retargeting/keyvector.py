"""The operator's hand as 21 keypoints, in the layout every retargeter expects.

This file does NOT solve anything. It is the adapter: Leap bones in, MANO-21
keypoints out, expressed in the robot's own axes and in metres. The solving
happens in `keyvector_bridge.py`, in a different interpreter, for dependency
reasons written up there.

WHY 21 AND IN THIS ORDER. MANO - and MediaPipe after it - fixed a keypoint
layout that the whole retargeting literature now assumes: index 0 is the wrist,
then four points per digit running thumb, index, middle, ring, pinky. Fingertips
therefore land on 4, 8, 12, 16, 20, which is exactly the rule DexPilot uses to
pair a robot's fingertip links with human keypoints. Emit any other order and
the solver will retarget your ring finger onto the robot's thumb without
complaining once.

WHY THE ROBOT'S AXES. Keyvectors are compared against the robot's own forward
kinematics, so both sides have to be in the same frame. `Retargeter.rotation`
already produces that change of basis - measured off the model, correct for
either chirality, and already carrying the mirror option - so this reuses it
rather than inventing a second convention that can drift from the first.

WHY THE PINKY IS SYNTHESISED. The robot has no little finger, but the layout
has a slot for one and DexPilot indexes into it by position. Leaving it at zero
would put a keypoint at the wrist and quietly distort any vector touching it.
The operator has a real pinky; it is reported; it goes in its slot, and the
solver ignores it because no robot link is paired with it.
"""

from __future__ import annotations

import numpy as np

from ..leap import DIGITS

# MANO slot -> (Leap digit, which of that digit's five joint positions).
#
# The Leap gives every digit five points: the base of the metacarpal, then the
# far end of each of its four bones. A finger's MCP is therefore point 1, not
# point 0. The thumb is the exception it always is - its metacarpal is
# zero-length padding, so points 0 and 1 coincide and the anatomical CMC is
# point 1, MCP is 2, IP is 3.
MANO_LAYOUT = (
    ("thumb", 1), ("thumb", 2), ("thumb", 3), ("thumb", 4),
    ("index", 1), ("index", 2), ("index", 3), ("index", 4),
    ("middle", 1), ("middle", 2), ("middle", 3), ("middle", 4),
    ("ring", 1), ("ring", 2), ("ring", 3), ("ring", 4),
    ("pinky", 1), ("pinky", 2), ("pinky", 3), ("pinky", 4),
)
WRIST_DIGITS = ("index", "middle", "ring")
MM_TO_M = 0.001


def shortest_arc(source, target):
    """The rotation taking unit vector `source` onto unit vector `target`."""
    source = source / max(np.linalg.norm(source), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    axis = np.cross(source, target)
    sine = np.linalg.norm(axis)
    cosine = float(np.dot(source, target))
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else -np.eye(3)
    axis = axis / sine
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sine * cross + (1.0 - cosine) * (cross @ cross)


def rest_offsets(hand, rotation, robot_bones):
    """Per bone, the rotation that makes your flat hand the robot's zero pose.

    The two rest poses are not the same shape. Measured against a real
    operator's straightest tracked frame - index knuckle at -2.8 degrees, so
    genuinely flat - individual bones sat 5 to 27 degrees away from where the
    robot's are at zero. That is a constant on every bone, and every comparison
    downstream inherits it: it is what was left in the retrieval's knuckle
    after the mounting bracket was removed, and it is why a straight hand did
    not produce a straight robot.

    Fitted once from an open hand, which the CALIBRATE capture already holds.
    `robot_bones` is {digit: [unit directions at the robot's zero pose]}.
    """
    offsets = {}
    for digit, targets in robot_bones.items():
        joints = hand.bones[digit]
        found = []
        for index, target in enumerate(targets, start=1):
            local = rotation @ (joints[index + 1] - joints[index])
            found.append(shortest_arc(local, target))
        offsets[digit] = found
    return offsets


def wrist_point(hand):
    """Where slot 0 goes.

    The mean of the finger metacarpal bases rather than `hand.palm`: the palm
    centre sits forward of the wrist by a couple of centimetres, and every
    wrist-to-fingertip keyvector would inherit that offset as a constant
    error the solver then spends joint range correcting.

    The thumb's base is left out of the average because it sits far to the
    side and would drag slot 0 off the hand's midline.
    """
    return np.mean([hand.bones[digit][0] for digit in WRIST_DIGITS], axis=0)


def digit_lengths(hand):
    """Wrist to fingertip along each digit, summed bone by bone, in metres."""
    wrist = wrist_point(hand)
    out = {}
    for digit in DIGITS:
        joints = hand.bones[digit]
        span = np.linalg.norm(joints[1] - wrist)
        span += np.linalg.norm(np.diff(joints[1:], axis=0), axis=1).sum()
        out[digit] = float(span) * MM_TO_M
    return out


def apply_bone_offsets(points, bone_offsets):
    """Turn every bone of a 21-keypoint hand by its rest-pose correction.

    Re-integrates each digit from its knuckle outward rather than rotating the
    points where they lie, which keeps the chain self-consistent: the
    fingertips, and so the pinch distances taken from them, stay the tips of
    the bones that were corrected.

    THE ORIGINAL BONES ARE READ FIRST, before anything is written back. The
    loop overwrites `points[base + k + 1]`, which is the START of bone k + 1 -
    so reading them as it goes measured each bone from a corrected start to an
    uncorrected end, and only the first bone of every digit came out aligned.
    Measured against the robot's zero pose, the proximal bone landed at 0.00
    degrees and the two past it at 13 to 33.

    Mutates `points` and returns it.
    """
    for slot, digit in enumerate(DIGITS):
        turns = bone_offsets.get(digit)
        if not turns:
            continue
        base = 4 * slot + 1
        bones = [points[base + k + 1] - points[base + k]
                 for k in range(len(turns))]
        at = points[base].copy()
        for k, (turn, bone) in enumerate(zip(turns, bones)):
            at = at + turn @ bone
            points[base + k + 1] = at
    return points


def mano_keypoints(hand, rotation, scale=MM_TO_M, digit_scale=None,
                   bone_offsets=None):
    """(21, 3) keypoints in the robot's axes, metres, wrist at the origin.

    `rotation` is `Retargeter.rotation(hand)` - the change of basis from the
    operator's frame into the robot's.

    Wrist-relative, deliberately: DexPilot only ever looks at differences
    between these points, so the absolute placement is unused, and anchoring at
    the wrist keeps the numbers small and the frame unambiguous.

    `digit_scale` stretches each digit radially from the wrist by its own
    factor. A single global scale is wrong for this pair of hands: the robot's
    fingers are 1.04 times the operator's, but its THUMB is 1.27 times theirs.
    Scaled globally, the thumb is asked for a point 18 percent short of where
    its own thumb can comfortably sit, and it curls hard reaching for somewhere
    that is not there. Per digit, each one is asked for a target its own
    proportions can actually hold.
    """
    wrist = wrist_point(hand)
    points = np.empty((21, 3), dtype=float)
    points[0] = 0.0
    for slot, (digit, index) in enumerate(MANO_LAYOUT, start=1):
        local = rotation @ (hand.bones[digit][index] - wrist)
        points[slot] = local * (digit_scale or {}).get(digit, 1.0)

    if bone_offsets:
        apply_bone_offsets(points, bone_offsets)
    return points * scale
