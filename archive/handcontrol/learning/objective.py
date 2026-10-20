"""What the teacher minimises, and why every term is shaped the way it is.

The old teacher matched bone directions and three thumb-to-fingertip vectors,
all computed on ALIGNED keypoints - per-digit scaled and rest-rotated. Two
measurements ended that:

  A PINCH DID NOT SURVIVE THE ALIGNMENT. Raw from the sensor, an operator's
  thumb tip sits about 21 mm from the index tip on a pinch. Scaling the thumb
  and the index radially from the wrist by different factors sends two
  coincident tips to two different places, and the rest rotation does the same
  by another route; both together are what the solver saw. It matched that
  faithfully and reported a pinch.

  THE SPLIT ABOVE (64 mm from scale, 80 from rotation, 136 together) was
  measured on the first hand, whose thumb scale was 1.78. `h4nd_left` stretches
  the thumb 1.98x, so the effect is LARGER, not smaller: the same pinch now
  reads 22 mm raw and 189 mm aligned. Only the total is re-measured per run,
  by `selftest.test_a_pinch_survives_into_the_raw_view`. Directions survive both transforms;
  distances survive neither. So this file reads two views of the hand: the
  aligned one for directions, the rotation-only one for anything that is a
  distance.

  MATCHING THE THUMB'S DIRECTIONS IS INCOMPATIBLE WITH CONTACT ON THIS HAND.
  Solved per frame on 60 held-out pinches: matching the aligned bone directions
  to 3 degrees puts the tips 163 mm apart; carrying the rotation-from-rest
  instead is no better at 184. Touching needs the thumb 79 to 85 degrees off
  either. The robot's thumb is 1.4x longer relative to its index than the
  operator's, on a different base, and the rotation that closes their pinch
  does not close its own. So the thumb gets a task-space objective - where its
  tip is relative to the fingers - and only a weak direction prior, only when
  nothing is being touched. The fingers keep their directions; their axes do
  match.

WHAT THE TERMS ARE, AND THE NUMBER BEHIND EACH CONSTANT.

  Finger bone directions, on aligned keypoints. Full weight, except the finger
  being touched, released to 0.1 - the per-frame floor on this hand is that
  seven degrees of index buys 82% contact and the last 18% costs 28 more.

  Thumb bone directions, weight 0.3, faded out by contact. Tried at 1.0: the
  fist fell apart (fingers 7.5 -> 14.4 degrees) because the aligned thumb
  direction is wrong on any curled thumb, not just a pinching one, and forcing
  it dragged the fingers with it.

  Contact: relu(robot distance - touch floor), per finger, weighted by how
  close the operator's own tips are, NORMALISED OVER THE TOUCHING FRAMES AND
  FLOORED. Touching frames are 14% of a recording; averaged into the rest the
  term vanished. Normalised without a floor, the sigmoid's tails on a flat hand
  summed to 4 over 3575 frames and the term became the full mean gap - 233 -
  at full weight, pulling the thumb in on an open palm. Floored at 5% of the
  run: 0.05.

  Tracking: the fingers NOT being touched hold the operator's raw distance to
  the thumb, scaled by the finger-length ratio. Without it, 95% contact came
  as a fist - every finger curled onto the thumb. Gated by the SDK's own pinch
  intent, saturating at 0.25, because distance cannot tell a pinch approach
  from this operator's flat hold, where the thumb rests on the index at 77 mm;
  intent reads 0.00 there and ramps past 0.25 on the way into a tap.
  Ungated, it also positions the thumb on the approach and contact went
  67% -> 97%; gated on distance alone it went back to 67.

  A fist detector: when every finger is within reach of the thumb, tip contact
  is suppressed. A fist is not a pinch - the thumb wraps the side of the index
  and does not meet its tip - and the tip term half-firing there fought the
  thumb prior.

  Acceleration and joint limits. The learning-rate decay matters more than the
  weight: at a fixed rate the holds jittered 32x; decayed, they sit at parity
  with the solver they replace.

  SELF-COLLISION, and contact measured between SURFACES. Both came from one
  census, run after everything above had shipped: MuJoCo's own contacts said
  the labels put the hand through itself on 42% of the recorded frames and
  95% of the synthetic ones. Two causes. The contact term drove tip ORIGINS
  to the closest the kinematics allow, and the tip meshes then overlapped by
  ten millimetres on every pinch; and the sampler crossed neighbouring
  fingers freely, which the teacher copied faithfully. So the contact term
  now reads the gap between the two tips' SURFACES (`collision.Proxy`: every
  body an oriented box, one offset per pair measured off the meshes)
  and drives it to zero, and a penalty on every cross-digit pair charges
  penetration beyond the proxy's own honesty. Contact is not the enemy - a
  hand that cannot touch itself cannot pinch. Passing through is.

WHAT IT MEASURED when it first replaced the old objective - whole recording,
every block, against the live solver: fingers 6.90 -> 3.16 degrees and better
on every block (a flat hand from 3.21 to 0.08), contact made 0% -> 82% (index
93, middle 58, ring 64), gap at touch 135 -> 3.9 mm, jerk 8.1e3 -> 3.5e3.
Those numbers are the argument for the change, not today's: the terms added
since put it at 2.69 degrees and 95%, and `learning/README.md` has the
current table. The
development passes that set each constant scored per block: fist 100, rot45
98, curl_slow 88, thumb_tap 87. Motion blocks stay 1.4 to 3.3x rougher than
the old teacher - one smoothing pass against its two plus a re-solve - and
that roughness is mostly real: the last 27 mm of a tap takes 56 ms, and a
thumb that has to arrive when the operator's does moves 23 degrees in that
time.
"""

from __future__ import annotations

import numpy as np
import torch

from ..robot.model import FINGERS
from . import geometry

# MANO slot of each digit's first joint; the three bones are the differences
# of the four slots from it. Tip slot is base + 3.
HUMAN_BASE = {"thumb": 1, "index": 5, "middle": 9, "ring": 13}
TIP = {digit: base + 3 for digit, base in HUMAN_BASE.items()}
OTHERS = tuple(f for f in FINGERS if f != "thumb")


def reach_scale(digit_scale):
    """Robot over operator finger length, for the tracking term.

    THE FINGERS ONLY, never the thumb: `fit_proportions` stretches this
    operator's thumb 1.98x against the fingers' 1.18 to 1.28, so including it
    drags the mean by a fifth and the tracking term then asks every digit to
    reach where only the thumb can. A rest-pose ratio was tried instead and
    gave 3.77, because the robot's zero pose points thumb and fingers apart.

    ONE DEFINITION. `Teacher.__init__` used to set this straight onto the FK
    object, so `per_frame` worked through a Teacher and raised
    AttributeError through `train.py --direct`, which builds a bare ChainFK.
    """
    return float(np.mean([digit_scale.get(f, 1.0) for f in OTHERS]))
# The three distance cells this objective reads - thumb against index, middle,
# ring - in that order, out of the one table every distance goes through.
THUMB_ROW = list(geometry.THUMB_ROW)

# ----------------------------------------------------------------- contact
TOUCH_MM = 33.0        # raw thumb-to-tip distance that counts as touching. The
                       # sensor's pinch median is 21.5; 30-40 is still a pinch.
TOUCH_RAMP_MM = 3.5    # sigmoid width around it. Tails are ~0 by 60 mm.
FIST_MM = 60.0         # every finger inside this = a fist, not a pinch
FIST_RAMP_MM = 10.0
INTENT_GAIN = 4.0      # SDK pinch_strength x this, clamped: saturates at 0.25
MADE_MM = 8.0          # surface gap under which a touch scores as "made"

# --------------------------------------------------------------- collision
# How far into each other two bodies may read before it costs anything, mm.
# The proxy's own error: validated against MuJoCo on the operator's pinches
# it reads within a millimetre around touching. Swept on three runs against
# 2 mm and 0: at 2 the thumb-tap run still put 8% of its frames more than
# 3 mm into a neighbour, at 1 it was 0.1% with contact and posture unchanged,
# and 0 bought nothing more for a tenth of a degree of posture.
COLLIDE_TOL_MM = 1.0
W_COLLIDE = 0.3        # per mm^2 past the tolerance, summed over pairs
PENETRATION_MM = 3.0   # what `score` counts as a penetrating frame

# CLEARANCE, for the student's sake. Neighbouring fingers sit a millimetre
# apart on this hand, and a student four degrees off its label passes
# through a neighbour between two clear poses however clear the labels are
# - the clearance is smaller than the precision. So wherever the operator's
# own neighbouring fingers are apart, the labels keep this much daylight
# between the robot's, and the student inherits an error budget. Where the
# operator holds fingers together, or makes a fist, only the tolerance
# applies: a hand that cannot close its fingers is not a hand.
CLEAR_MM = 4.0
TOGETHER_MM = 28.0     # raw adjacent fingertip distance under which they are together
TOGETHER_RAMP_MM = 4.0

# FINGERS KEEP THEIR ORDER. A penalty on penetration pushes a body out along
# the shortest way, and when a finger starts a solve more than halfway
# through its neighbour the shortest way is onward - out the far side, into
# a hand with its fingers crossed and every pair clear. Seeded from the old
# labels that was 15% of the synthetic frames. So the distal joint and tip
# of each finger are held on their own side of the next finger's, along the
# palm's lateral axis, with the same squared hinge. Not the thumb: it
# crosses under the fingers to oppose them, and that is not a fault.
ORDER = (("index", "middle"), ("middle", "ring"))
W_ORDER = 0.3

# ----------------------------------------------------------------- objects
# THE OBJECT IN THE OBJECTIVE. Measured on a wrap probe: direction matching
# keeps the operator's finger curvature and closes this hand on nothing,
# because its joint axes sit at the back of 21 mm links where the operator's
# fingers are 8 - a wrap that encloses 27 mm in the operator's hand encloses
# 2 on the robot, and every geometric stand-in for "aperture" tried against
# MuJoCo's own answer failed to track it. So when there is something in the
# hand (`collision.Objects`, from objects.json beside a pair file), two terms
# read it directly: penetration into it costs like penetration into a
# neighbour, and every digit the operator has BENT around it - two bone
# bends summing past OBJECT_BEND - is asked to bring its nearest link to
# its surface. Nothing is asked of a straight finger, and nothing at all on
# frames with no object present.
W_OBJECT = 3.0          # per mm^2 of penetration past the tolerance, summed - ten times the self-collision weight: measured at 0.3 the hold term parked the phalanges 2 to 7 mm inside the object
W_HOLD = 4.0            # per mm a bent digit's phalanges stay off it, averaged
# Four, not one: the posture term is degrees of bone direction and this is
# millimetres of gap, averaged the same way, so at one-to-one a finger that
# closes 4 mm onto a fixed object pays 6 degrees on one bone to save 4 mm on
# one phalanx, and it does not close - the teacher hovered 5-15 mm off (1.7
# links). At four it lies on the object (3.7 links, three digits, the thumb)
# for about 5 degrees more posture than the object coming to the hand cost.
# THE OBJECT IS COMPLIANT. Pinned where a rule put it, holding a 35 mm
# cylinder cost the teacher 18 degrees of posture; a real object moves to
# where the hand can hold it. So its position is a variable of the solve,
# with a weak pull toward where it started and a floor at the palm face,
# and where it settles is written back for the census.
W_OBJECT_PRIOR = 2e-4   # while the object rides with the solve: per mm^2 it has moved
# The object ends FIXED where the hand placed it (`gestures.cylinder_to_robot`)
# and the fingers go to it. It used to stay wherever the solve carried it
# (compliant), which was somewhere different in every hand, so one operator
# wrap meant many labels. But solving against a fixed object from the start
# stalls: closing a finger onto a cylinder drives the fingertip through the
# surface before the middle link lies on it, and the penetration wall stops
# the finger there, hovering 5-15 mm off (1.7 links against 3.7). So the
# object first rides with the solve for `OBJECT_FREE_STEPS` - the fingers
# close and it comes to meet them, no wall to cross - then it goes back where
# the hand placed it and the hand finishes there, a few millimetres away from
# a pose that already encloses it. A stiff prior instead of the two phases
# was tried and was worse than either: Adam steps the shift by its learning
# rate whatever the gradient, so a stiff spring made the object jitter by
# centimetres through the solve.
OBJECT_FREE_STEPS = 300
OBJECT_LR = 1e-3            # metres per step the object may ride: a millimetre, not `LR` = 20 mm
OBJECT_RETURN_STEPS = 100   # the object slides back over this many steps, not in one jump
POLISH_STEPS = 100          # after the smoothing pass, with the object fixed: see `Teacher.teach`
POLISH_LR = 5e-3
OBJECT_FLOOR_MM = 2.0   # its surface may not go below the palm face by more
OBJECT_BEND = 0.5       # radians of bend across a digit that means "closed on it"
OBJECT_BEND_RAMP = 0.15

# ------------------------------------------------------------------ weights
THUMB_PRIOR = 0.3      # thumb direction term, weighted against the fingers
RELEASE = 0.9          # the touched finger keeps 1 - this of its direction term

# HOW MUCH OF THE THUMB'S DIRECTION TERM A TOUCH REMOVES. 1.0 is all of it,
# which is what this was before it became a constant - written
# `(1 - any_touch)`, beside fingers that keep a tenth of theirs through
# `RELEASE`. The thumb is the only digit released completely, and on ANY touch
# rather than its own.
#
# SWEPT, AND 1.0 IS KEPT. On thumb_tap + ring_pinch, calibrated on the open
# blocks, the thumb's error while touching against the contact it is there to
# protect (one row per setting: this solve is Adam from a fixed warm start with
# no random init, so it repeats - the 1.0 row reproduces the shipped labels
# field for field):
#
#     release   touching   contact    gap   pen
#         1.0      90.0d     92.8%  1.1mm  1.5%
#         0.9      89.4d     92.9%  1.0mm  1.7%
#         0.7      88.3d     92.6%  1.0mm  1.9%
#         0.5      87.0d     92.5%  1.1mm  2.2%
#         0.0      84.1d     91.9%  1.3mm  2.4%
#
# THE LAST ROW IS THE RESULT. With the release gone entirely - the full prior
# applied through the touch - the thumb still sits 84.1 deg off the operator's
# direction. So the release accounts for 5.9 of those 90 degrees and the hand's
# geometry accounts for the rest, which is the 163 mm measured on 60 held-out
# pinches at the top of this file, reached from the other direction. Buying the
# 5.9 costs self-penetration (1.5% of frames to 2.4%) and ring contact (89% to
# 87%). There is no setting where that trade is worth re-solving every label.
#
# AND THE LABEL IS NOT ARBITRARY, which is the other reason not to move it.
# Pairs of frames whose keypoints match within 13 mm and which are at least 300
# frames apart - a different tap, not the next frame - disagree about the
# released thumb by 2.9 deg, against 2.8 for the fingers on those same frames.
# A released target is not a random one: contact plus the limits still pin the
# pose. So the thumb's 38 deg is a pose a student can learn, not noise it can
# only average, and the number to watch is `thumb_contact_deg` beside
# `thumb_free_deg` rather than the two pooled.
THUMB_RELEASE = 1.0
W_CONTACT = 1.0
W_TRACK = 0.2
W_ACCEL = 1.5
W_LIMIT = 1e3
FLOOR = 0.05           # normaliser floor, as a fraction of the run's frames

# ------------------------------------------------------------------- solver
STEPS = 600
LR = 0.02
LR_MIN = 1e-4


def human_bones(points):
    """(b, 21, 3) -> (b, fingers, 3 bones, 3) unit directions, FINGERS order."""
    out = []
    for finger in FINGERS:
        base = HUMAN_BASE[finger]
        segment = points[:, base:base + 4]
        out.append(segment[:, 1:] - segment[:, :-1])
    bones = torch.stack(out, dim=1)
    return bones / bones.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def evaluate(fk, proxy, q, aligned, raw):
    """The quantities every term is built from, from one FK pass.

    Returns bone-direction error per bone in degrees (b, fingers, 3), the
    robot's thumb-to-fingertip ORIGIN distance in mm (b, others) for the
    tracking term, the operator's raw one (b, others), the SURFACE gap
    between thumb and each fingertip in mm (b, others) for the contact
    term, every cross-digit pair's surface gap (b, pairs), and the joint
    origins (b, F, 5, 3) the ordering and object terms read. Six values, in
    that order. Directions from the aligned view, distances from the raw.
    """
    links, origins, frames = fk.walk(q)
    robot = links[:, :, 1:] - links[:, :, :-1]
    robot = robot / robot.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cosine = (robot * human_bones(aligned)).sum(-1).clamp(-1.0, 1.0)
    bones = torch.rad2deg(torch.acos(cosine))
    tips = links[:, :, -1]
    thumb = fk.fingers.index("thumb")
    others = [fk.fingers.index(f) for f in OTHERS]
    d_robot = (tips[:, thumb, None] - tips[:, others]).norm(dim=-1) * 1000.0
    d_human = geometry.pairwise(raw)[:, THUMB_ROW]
    gaps = proxy.gaps(origins, frames)
    return bones, d_robot, d_human, gaps[:, proxy.tip_rows], gaps, origins


def lateral_axis(fk):
    """Unit vector from the index knuckle to the ring knuckle at rest - the
    direction the fingers are laid out along - as a tensor on the FK's device."""
    with torch.no_grad():
        _, origins, _ = fk.walk(torch.zeros(1, len(fk.joint_names), device=fk.xyz.device))
    index, ring = fk.fingers.index("index"), fk.fingers.index("ring")
    axis = origins[0, ring, 1] - origins[0, index, 1]
    return axis / axis.norm().clamp_min(1e-9)


def allowance(proxy, raw, fist):
    """How much gap each pair must keep, (b, pairs) mm: minus the tolerance
    everywhere, rising to CLEAR_MM on neighbouring-finger pairs while the
    operator's own neighbouring fingertips are apart and the hand is no fist.
    """
    cells = [geometry.INDEX["index-middle"], geometry.INDEX["middle-ring"]]
    apart = 1.0 - torch.sigmoid((TOGETHER_MM - geometry.pairwise(raw)[:, cells])
                                / TOGETHER_RAMP_MM)                 # (b, 2)
    apart = apart * (1.0 - fist)[:, None]
    base = torch.full((raw.shape[0], len(proxy.adjacent)), -COLLIDE_TOL_MM,
                      device=raw.device, dtype=raw.dtype)
    group = proxy.adjacent
    lift = torch.where(group[None, :] >= 0,
                       apart[:, group.clamp_min(0)], torch.zeros_like(base))
    return base + (CLEAR_MM + COLLIDE_TOL_MM) * lift


def crossing(fk, origins):
    """Squared millimetres by which any finger's distal joint or tip sits on
    the wrong side of its neighbour's, summed over the ORDER pairs, per frame."""
    lateral = getattr(fk, "lateral", None)
    if lateral is None:
        return origins.new_zeros(origins.shape[0])
    along = (origins[:, :, 3:5] @ lateral) * 1000.0            # (b, F, 2) mm
    total = origins.new_zeros(origins.shape[0])
    for near, far in ORDER:
        a, b = fk.fingers.index(near), fk.fingers.index(far)
        total = total + (torch.relu(along[:, a] - along[:, b]) ** 2).sum(-1)
    return total


def intent_from_distance(d_human_mm, near=TOUCH_MM, far=120.0):
    """A stand-in for the SDK's pinch_strength when there is no SDK.

    Synthetic and authored hands are geometry with no sensor behind them, so
    intent is read off the closest thumb-to-fingertip distance: 1 at touching,
    0 beyond `far`. Recorded hands use the real field; this only exists so the
    generated ones go through the same objective.
    """
    if torch.is_tensor(d_human_mm):
        # The training loop reads intent off the same window it feeds the
        # network, so it arrives as a tensor. It is a function of the INPUT
        # and carries no gradient toward the pose either way.
        closest = d_human_mm.min(dim=-1).values
        return ((far - closest) / (far - near)).clamp(0.0, 1.0)
    closest = np.min(np.asarray(d_human_mm), axis=-1)
    return np.clip((far - closest) / (far - near), 0.0, 1.0).astype(np.float32)


def bends(raw):
    """How far each digit is bent, (b, fingers) radians: the two angles
    between its three bones, from the raw view. FINGERS order."""
    unit = human_bones(raw)                                       # (b, F, 3, 3)
    first = torch.acos((unit[:, :, 0] * unit[:, :, 1]).sum(-1).clamp(-1.0, 1.0))
    second = torch.acos((unit[:, :, 1] * unit[:, :, 2]).sum(-1).clamp(-1.0, 1.0))
    return first + second


def held(fk, proxy, origins, frames, raw, objects, present, shift=None):
    """The two object terms for one run: (penetration, hold), scalars."""
    gaps = objects.gaps(origins, frames, proxy, present, shift)   # (b, bodies, objects) mm
    penetration = (torch.relu(-gaps - COLLIDE_TOL_MM) ** 2).sum(dim=(1, 2)).mean()
    closed = torch.sigmoid((bends(raw) - OBJECT_BEND) / OBJECT_BEND_RAMP)   # (b, F)
    per_digit = gaps.reshape(gaps.shape[0], len(fk.fingers), -1, gaps.shape[2])
    # The phalanges, not the mount: body 0 is the knuckle block, which no
    # finger joint moves, and a term that read the nearest body sat on it
    # while the fingers hovered twenty millimetres off the object. A wrap
    # lays all three phalanges on the surface, so all three are asked.
    off = torch.relu(per_digit[:, :, 1:]).mean(dim=2)             # (b, F, objects)
    weight = closed[:, :, None] * present[:, None, :]
    hold = (weight * off).sum() / weight.sum().clamp_min(1.0)
    return penetration, hold


def per_frame(fk, proxy, q, aligned, raw, intent, lower, upper):
    """Every term that scores ONE frame on its own, as a single scalar.

    Split out of `loss` so the student can be trained against this objective
    directly rather than only against the teacher's answer to it. The teacher
    solves a whole trajectory and then smooths it acausally, so there are
    poses in `q_star` that no causal network can produce; a student that is
    only ever scored against those poses inherits a floor it cannot cross.
    Scored against the objective itself, it is free to find the causal pose
    that satisfies the same terms.

    MEASURED, AND IT DOES NOT PAY. Three weights at one seed and then 0.01 at
    three: the posture score improves a long way (13.02 to 8.11 deg at w=0.20)
    but that number IS this function's first term, so training on it and
    reporting it is circular. What is independent says otherwise - distance
    from the teacher rises monotonically with the weight (7.87, 8.19, 9.93,
    11.36) and self-penetration reaches 2.3% of frames at w=0.20 against a
    baseline band of 0.3-0.7. At w=0.01, where nothing looked worse on one
    seed, the third seed penetrates on 23.2% of frames. The failure is not a
    scale that needs tuning, it is that this objective has no smoothness term
    (see below) and no label to hold it to a reachable pose.

    Left in and defaulted OFF: it is the only route to a student that is not
    bounded by an acausal label, and the acceleration gap below is a concrete
    thing to fix before retrying it.

    The acceleration term is NOT here, because it is the one term that reads
    across frames: on a shuffled batch of windows it would difference across
    the seam between two unrelated runs. `train` supplies smoothness with
    `W_VELOCITY` against the label instead. See `loss`, which adds it back.
    """
    bones, d_robot, d_human, gap, gaps, origins = evaluate(fk, proxy, q, aligned, raw)
    touching = torch.sigmoid((TOUCH_MM - d_human) / TOUCH_RAMP_MM)
    fist = torch.sigmoid((FIST_MM - d_human) / FIST_RAMP_MM).min(dim=1).values
    touching = touching * (1.0 - fist)[:, None]
    any_touch = touching.max(dim=1).values

    thumb = fk.fingers.index("thumb")
    fingers = [fk.fingers.index(f) for f in OTHERS]
    finger_weight = 1.0 - RELEASE * touching
    thumb_weight = 1.0 - THUMB_RELEASE * any_touch
    posture = ((bones[:, fingers].mean(dim=2) * finger_weight).mean()
               + THUMB_PRIOR * (bones[:, thumb].mean(dim=1) * thumb_weight).mean())

    floor = FLOOR * len(q)
    contact = ((touching * torch.relu(gap)).sum()
               / touching.sum().clamp_min(floor))
    gate = (1.0 - touching) * (intent * INTENT_GAIN).clamp(0.0, 1.0)[:, None]
    track = ((gate * (d_robot - fk.reach_scale * d_human).abs()).sum()
             / gate.sum().clamp_min(floor))
    limits = (torch.relu(lower - q) + torch.relu(q - upper)).mean() * W_LIMIT
    collide = (torch.relu(allowance(proxy, raw, fist) - gaps) ** 2).sum(dim=1).mean()
    order = crossing(fk, origins).mean()
    return (posture + W_CONTACT * contact + W_TRACK * track
            + limits + W_COLLIDE * collide + W_ORDER * order)


def loss(fk, proxy, q, aligned, raw, intent, lower, upper, objects=None, present=None,
         shift=None):
    """One scalar over a whole run. See the module docstring for each term."""
    bones, d_robot, d_human, gap, gaps, origins = evaluate(fk, proxy, q, aligned, raw)
    touching = torch.sigmoid((TOUCH_MM - d_human) / TOUCH_RAMP_MM)
    fist = torch.sigmoid((FIST_MM - d_human) / FIST_RAMP_MM).min(dim=1).values
    touching = touching * (1.0 - fist)[:, None]
    any_touch = touching.max(dim=1).values

    thumb = fk.fingers.index("thumb")
    fingers = [fk.fingers.index(f) for f in OTHERS]
    finger_weight = 1.0 - RELEASE * touching
    thumb_weight = 1.0 - THUMB_RELEASE * any_touch
    posture = ((bones[:, fingers].mean(dim=2) * finger_weight).mean()
               + THUMB_PRIOR * (bones[:, thumb].mean(dim=1) * thumb_weight).mean())

    floor = FLOOR * len(q)
    # Surfaces meet at zero. Below zero the collision term takes over, so the
    # two together hold a pinch at the skin rather than at the bone.
    contact = ((touching * torch.relu(gap)).sum()
               / touching.sum().clamp_min(floor))
    gate = (1.0 - touching) * (intent * INTENT_GAIN).clamp(0.0, 1.0)[:, None]
    track = ((gate * (d_robot - fk.reach_scale * d_human).abs()).sum()
             / gate.sum().clamp_min(floor))

    degrees = torch.rad2deg(q)
    accel = ((degrees[2:] - 2 * degrees[1:-1] + degrees[:-2]).abs().mean()
             if len(q) > 2 else q.new_zeros(()))
    limits = (torch.relu(lower - q) + torch.relu(q - upper)).mean() * W_LIMIT
    # Summed over pairs, not averaged: one finger through another is the
    # whole fault, and dividing it by ninety-five clear pairs would hide it.
    collide = (torch.relu(allowance(proxy, raw, fist) - gaps) ** 2).sum(dim=1).mean()
    order = crossing(fk, origins).mean()
    total = (posture + W_CONTACT * contact + W_TRACK * track + W_ACCEL * accel
             + limits + W_COLLIDE * collide + W_ORDER * order)
    if objects is not None and present is not None:
        _, _, frames = fk.walk(q)
        penetration, hold = held(fk, proxy, origins, frames, raw, objects, present, shift)
        total = total + W_OBJECT * penetration + W_HOLD * hold
        if shift is not None:
            moved = (shift * 1000.0) ** 2
            below = torch.relu((objects.radius + OBJECT_FLOOR_MM * 1e-3
                                - (objects.p0[:, 0] + shift[:, 0])) * 1000.0) ** 2
            total = total + W_OBJECT_PRIOR * moved.sum() + W_OBJECT * below.sum()
    return total


def solve(fk, proxy, seed, aligned, raw, intent, lower, upper, steps=STEPS,
          objects=None, present=None):
    """One run, solved as a batch, warm-started from `seed`. Returns q (n, dof).

    Adam with cosine decay to `LR_MIN`. The decay is not cosmetic: at a fixed
    rate the solve never settles and a still hand jitters at 32x the solver it
    replaces; decayed, it sits at parity.
    """
    device = fk.xyz.device
    q = torch.tensor(np.asarray(seed), dtype=torch.float32, device=device).requires_grad_(True)
    aligned = torch.tensor(np.asarray(aligned), dtype=torch.float32, device=device)
    raw = torch.tensor(np.asarray(raw), dtype=torch.float32, device=device)
    intent = torch.tensor(np.asarray(intent), dtype=torch.float32, device=device)
    lower = torch.tensor(np.asarray(lower), dtype=torch.float32, device=device)
    upper = torch.tensor(np.asarray(upper), dtype=torch.float32, device=device)

    optimiser = torch.optim.Adam([q], lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps, eta_min=LR_MIN)
    shift, returning = None, []
    if present is not None:
        present = torch.as_tensor(np.asarray(present), dtype=torch.float32, device=device)
        if OBJECT_FREE_STEPS:
            # Phase one: the object rides with the fingers. See `OBJECT_FREE_STEPS`.
            shift = torch.zeros(len(objects), 3, device=device).requires_grad_(True)
            # Adam steps every parameter by its learning rate whatever the
            # gradient, and `LR` is in radians: on the object, in metres, it
            # was 20 mm a step - the cylinder leapt 70 mm up in the first
            # fifty steps and wandered back, and where it had been when the
            # fingers closed decided whether they came out clean. A
            # millimetre a step and it drifts to the hand and stays.
            riding = torch.optim.Adam([{"params": [q]}, {"params": [shift], "lr": OBJECT_LR}], lr=LR)
            pace = torch.optim.lr_scheduler.CosineAnnealingLR(riding, OBJECT_FREE_STEPS, eta_min=LR_MIN)
            for _ in range(OBJECT_FREE_STEPS):
                value = loss(fk, proxy, q, aligned, raw, intent, lower, upper, objects, present, shift)
                riding.zero_grad()
                value.backward()
                riding.step()
                pace.step()
            # Then it slides back to where the hand placed it, the fingers
            # following, rather than jumping: a jump of a centimetre left a
            # phalanx inside the cylinder, on its axis, where the way out has
            # no gradient - and the middle finger stayed there, 30 mm deep.
            rode = shift.detach()
            returning = [rode * (1.0 - k / OBJECT_RETURN_STEPS) for k in range(1, OBJECT_RETURN_STEPS + 1)]
            shift = None        # back where the hand placed it
    for k in range(steps):
        offset = returning[k] if k < len(returning) else shift
        value = loss(fk, proxy, q, aligned, raw, intent, lower, upper, objects, present, offset)
        optimiser.zero_grad()
        value.backward()
        optimiser.step()
        schedule.step()
    with torch.no_grad():
        q.clamp_(lower, upper)
    return q.detach().cpu().numpy().astype(np.float64)


def polish(fk, proxy, q0, aligned, raw, intent, lower, upper, objects, present,
           steps=POLISH_STEPS, lr=POLISH_LR):
    """A short descent from a smoothed trajectory with the object fixed.

    The solve leaves the object clean; the smoothing pass after it does not.
    Where the thumb flips between two answers on consecutive frames - 66
    degrees in one step on a jittering wrap - a 21-frame filter draws the
    line between them straight through the cylinder, 13 mm deep for sixty
    frames. This takes the smoothed trajectory and lets the objective pull
    those frames back out, at a quarter of the solve's rate so the rest stays
    where the filter put it."""
    device = fk.xyz.device
    q = torch.tensor(np.asarray(q0), dtype=torch.float32, device=device).requires_grad_(True)
    aligned = torch.tensor(np.asarray(aligned), dtype=torch.float32, device=device)
    raw = torch.tensor(np.asarray(raw), dtype=torch.float32, device=device)
    intent = torch.tensor(np.asarray(intent), dtype=torch.float32, device=device)
    lower = torch.tensor(np.asarray(lower), dtype=torch.float32, device=device)
    upper = torch.tensor(np.asarray(upper), dtype=torch.float32, device=device)
    present = (torch.ones(len(q), len(objects), device=device) if present is None
               else torch.as_tensor(np.asarray(present), dtype=torch.float32, device=device))
    optimiser = torch.optim.Adam([q], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps, eta_min=LR_MIN)
    for _ in range(steps):
        value = loss(fk, proxy, q, aligned, raw, intent, lower, upper, objects, present, None)
        optimiser.zero_grad()
        value.backward()
        optimiser.step()
        schedule.step()
    with torch.no_grad():
        q.clamp_(lower, upper)
    return q.detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def score(fk, proxy, q, aligned, raw):
    """Posture and contact on a solved run, as plain numbers.

    Returns finger posture (deg), thumb posture (deg), the fraction of the
    operator's touches the robot made - per finger and pooled under "all" -
    the mean surface gap on those frames (mm), the fraction of frames with any
    pair read more than PENETRATION_MM into another by the proxy, and a
    `posture` dict splitting the first number by contact state.

    WHY THE SPLIT. `posture_deg` pooled over every frame is not a like-for-like
    comparison between a mapping that closes contacts and one that does not.
    This objective deliberately ABANDONS the direction target during a touch:
    the touched finger keeps only `1 - RELEASE` of its direction weight and the
    thumb keeps `1 - THUMB_RELEASE` of its, which at the shipped 1.0 is none.
    Measured, a made contact needs the thumb 79 to 85 degrees off any direction
    match, and the sweep beside THUMB_RELEASE shows 84 of those degrees survive
    even with the prior fully restored. So a hand that pinches must score worse
    on the pooled number, and the live solver's flattering 7.95 deg is bought
    by never closing at all (0% contact, 106 mm of gap).

    The dict carries `free` (frames where the operator is touching nothing -
    the frames where the direction target is actually being asked for),
    `contact`, and `untouched`, which scores only the digits the objective did
    not release, on every frame. `untouched` is the sharpest of the three: it
    is the posture number with the released target removed rather than the
    released frames removed.

    `thumb_free` and `thumb_contact` split the thumb the same way, and they
    matter more than the fingers' split does: the thumb is released COMPLETELY,
    so its pooled number is an average of a regime the objective targets and
    one it does not, and it moves when the share of touching frames moves
    rather than when the mapping gets better or worse.
    """
    device = fk.xyz.device
    q = torch.tensor(np.asarray(q), dtype=torch.float32, device=device)
    aligned = torch.tensor(np.asarray(aligned), dtype=torch.float32, device=device)
    raw = torch.tensor(np.asarray(raw), dtype=torch.float32, device=device)
    bones, d_robot, d_human, gap, gaps, _ = evaluate(fk, proxy, q, aligned, raw)
    thumb = fk.fingers.index("thumb")
    fingers = [fk.fingers.index(f) for f in OTHERS]
    on = d_human < TOUCH_MM                                   # (b, others)
    hit = on & (gap < MADE_MM)
    made = {finger: (float(hit[:, k].sum() / on[:, k].sum()) if int(on[:, k].sum())
                     else float("nan")) for k, finger in enumerate(OTHERS)}
    # Pooled over every (frame, finger) touch, not averaged per finger: the
    # index carries five times the ring's touches and should count that way.
    made["all"] = float(hit.sum() / on.sum()) if int(on.sum()) else float("nan")
    # The gap the robot leaves when the operator touches - what "pinch
    # accuracy" means now that the aperture is measured on the raw hand.
    left = float(gap[on].mean()) if int(on.sum()) else float("nan")
    penetrating = float((gaps.min(dim=1).values < -PENETRATION_MM).float().mean())

    # The same bone errors, split by whether the direction target was being
    # asked for on that frame and that digit - see the docstring.
    finger_bones = bones[:, fingers].mean(dim=2)              # (b, others)
    thumb_bones = bones[:, thumb].mean(dim=1)                 # (b,)
    free = ~on.any(dim=1)
    released = bones.new_zeros(finger_bones.shape, dtype=torch.bool) | on
    split = lambda x, m: float(x[m].mean()) if int(m.sum()) else float("nan")
    posture = {
        "all": float(finger_bones.mean()),
        "free": split(finger_bones, free),
        "contact": split(finger_bones, ~free),
        "untouched": (float(finger_bones[~released].mean())
                      if int((~released).sum()) else float("nan")),
        # The thumb's own split. It is the digit released COMPLETELY, so the
        # pooled thumb number is the one most worth not reading on its own.
        "thumb_free": split(thumb_bones, free),
        "thumb_contact": split(thumb_bones, ~free),
    }
    return (float(bones[:, fingers].mean()), float(bones[:, thumb].mean()), made,
            left, penetrating, posture)
