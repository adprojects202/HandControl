"""The measurement half of retargeting: frames, hinges and angles.

Pure geometry on arrays. Nothing here opens a file, keeps state, or knows what
a robot is beyond being handed one to probe - which is what makes it testable
against a hand you construct by hand, and portable to a different robot or a
different sensor.
"""

from __future__ import annotations

import math

import numpy as np

from ..robot.model import FINGERS, TIP_LINKS, roles_in_order


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros(3)


def flexion_axis(forward, normal):
    """The hinge every finger joint turns about: `forward` x `normal`.

    Chosen so that rotating `forward` about it by a positive angle carries it
    toward `normal` - into the palm, the way a finger curls - which holds for
    either chirality, because `thumbward` is the axis whose sign depends on
    which hand it is and this one is not it.
    """
    return unit(np.cross(forward, normal))


def thumb_flexion_axis(forward, thumbward):
    """The axis the thumb's interphalangeal joint turns about: `thumbward` x
    `forward`.

    The finger hinge (`flexion_axis`) is deliberately chirality-free, because
    a finger curls out of the palm plane and both hands do it the same way
    about `forward` x `normal`. The thumb's tip does not: it curls WITHIN the
    palm plane, from `thumbward` toward `forward`, and that is a rotation
    whose sense is the hand's chirality itself. Measured with the palm normal
    instead - which is the same vector for either hand - the same anatomical
    curl reads positive on a right hand and negative on a left, and the sign
    was never applied anywhere downstream: on a left hand the robot's thumb
    tip was driven backwards into its hyperextension stop. Verified by
    measuring a recorded hand and its mirror image: every other joint agreed
    to zero, the thumb tip flipped.

    Taking the cross product carries the sign for free - it is +normal for a
    right hand and -normal for a left - which is the same reason
    `flexion_axis` is a cross product rather than a table.
    """
    return unit(np.cross(thumbward, forward))


def flex_angle(reference, bone, axis):
    """Signed angle from `reference` to `bone` about `axis`, positive curling.

    About a FIXED axis, not in the plane spanned by the reference bone and the
    palm normal. That plane collapses the moment the reference bone lines up
    with the normal - `normal - (normal . ref) ref` is then the zero vector -
    and a finger's intermediate phalanx points straight into the palm at 90
    degrees of cumulative curl, which is an ordinary fist, not an edge case.
    The old form read that joint as 0 there and as NEGATIVE past it, so the
    tip joint quietly stopped tracking exactly when the hand closed.

    Projecting onto the plane perpendicular to a fixed hinge has no such
    singularity anywhere a finger can go, and still keeps sideways splay out
    of the flexion number.
    """
    reference = reference - np.dot(reference, axis) * axis
    bone = bone - np.dot(bone, axis) * axis
    if np.linalg.norm(reference) < 1e-9 or np.linalg.norm(bone) < 1e-9:
        return 0.0
    return math.atan2(float(np.dot(np.cross(reference, bone), axis)),
                      float(np.dot(reference, bone)))


def spread_angle(reference, bone, normal, outward):
    """Signed angle from `reference` to `bone` within the palm plane, positive
    toward `outward` - which is away from the middle finger."""
    reference = unit(reference - np.dot(reference, normal) * normal)
    if not reference.any():
        return 0.0
    outward = outward - np.dot(outward, normal) * normal
    outward = unit(outward - np.dot(outward, reference) * reference)
    flat = bone - np.dot(bone, normal) * normal
    # A curled finger points into the palm, so what is left of it in the palm
    # plane is a stub, and the angle of a stub is noise. Reported as no spread
    # rather than as whatever the noise says - otherwise the fingers scissor
    # about every time the hand closes.
    if np.linalg.norm(flat) < 0.2 * max(np.linalg.norm(bone), 1e-9):
        return 0.0
    return math.atan2(float(np.dot(flat, outward)),
                      float(np.dot(flat, reference)))


def hand_frame(hand):
    """An orthonormal frame for the operator's hand: (forward, normal, thumbward).

    `forward` runs palm to fingers and `normal` out of the front of the palm,
    both straight off the Leap. Which way the third axis points is settled by
    looking at where that hand's own thumb is, so a left hand over the sensor
    is described in its own terms rather than mirrored into a right one.
    """
    forward = unit(hand.direction)
    normal = unit(hand.normal - np.dot(hand.normal, forward) * forward)
    lateral = np.cross(forward, normal)
    across = hand.bones["thumb"][0] - hand.bones["pinky"][0]
    return forward, normal, lateral * (1.0 if np.dot(lateral, across) >= 0 else -1.0)


def robot_axes(model, names=None, probe=0.5):
    """The same three axes for the robot, measured off the URDF.

    `forward` is the axis the fingers extend along, `normal` is where a
    fingertip goes when its knuckle flexes - which is what makes it the palm
    side - and `thumbward` is the remaining axis, pointed at the thumb.
    Measured rather than asserted, so mirroring the model does not quietly
    invert the mapping.
    """
    names = names or model.joint_names()
    zero = np.zeros(len(names))
    tips = {f: model.link_pose(TIP_LINKS[f], zero, names)[0] for f in FINGERS}

    forward = np.zeros(3)
    axis = int(np.argmax(np.abs(tips["middle"])))
    forward[axis] = np.sign(tips["middle"][axis])

    knuckle = dict(roles_in_order(model, "index"))["knuckle"]
    flexed = zero.copy()
    flexed[names.index(knuckle)] = probe
    delta = model.link_pose(TIP_LINKS["index"], flexed, names)[0] - tips["index"]
    delta = delta - np.dot(delta, forward) * forward
    normal = np.zeros(3)
    axis = int(np.argmax(np.abs(delta)))
    normal[axis] = np.sign(delta[axis])

    thumbward = np.cross(forward, normal)
    if np.dot(thumbward, tips["thumb"] - tips["middle"]) < 0:
        thumbward = -thumbward
    return forward, normal, thumbward


def hand_length(hand, digit="middle"):
    """Wrist to fingertip along one digit, summed bone by bone.

    Summed rather than measured straight through: the straight-line distance
    shrinks as the finger curls, and the tip-mapping scale factor would then
    breathe with the operator's grip.
    """
    return float(np.linalg.norm(np.diff(hand.bones[digit], axis=0),
                                axis=1).sum())
