#!/usr/bin/env python
"""Keyvector retargeting for the h4nd_left, in its own interpreter.

    .venv-dex/bin/python keyvector_bridge.py --urdf build/h4nd_left_ik.urdf \
        --joints index_adduction_joint,... --tips thumb_tip_link,...

`main.py` spawns this; you rarely run it by hand. It listens for 21 MANO
keypoints on one UDP port and answers with 16 joint angles on another.

WHY A SEPARATE PROCESS, and not a nicer-sounding import. `dex_retargeting`
pulls in Pinocchio, NLopt and torch, and wants numpy >= 2.0; this project's
venv is on numpy 1.26. Nothing in the tree hard-caps numpy below 2, so the two
might well cohabit - but "might well" is not a thing to find out halfway
through a teleoperation session with sixteen servos live. Two interpreters, one
datagram, no version negotiation. It is the same argument that already put the
pygame panel in its own process, and the same transport.

WHAT IT ACTUALLY DOES is DexPilot's objective: match the 10 vectors between
fingertips and from the wrist to each fingertip, rather than matching joint
angles or absolute positions. Angles transfer badly because equal angles on
unequal segments put fingertips in different places; absolute positions
transfer badly because the robot's thumb cannot reach where a human thumb goes.
Differences of positions survive both problems, and a pinch is a difference.

    Handa et al., DexPilot (2020); Qin et al., AnyTeleop (RSS 2023).
    dexsuite/dex-retargeting, MIT.

SETUP, once:
    python3.11 -m venv .venv-dex
    .venv-dex/bin/pip install dex_retargeting torch
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

import numpy as np

HOST = "127.0.0.1"


# The bone set: three directions per digit - see `bone_vectors` for why not
# four - plus three thumb-to-finger distances. A digit's four MANO points are
# 4i-3 (MCP) .. 4i (TIP), 0 is the wrist, and the robot's link chain runs
# base -> proximal -> middle -> distal -> tip, so the correspondence is
# positional and needs no table.
DIGIT_SLOT = {"thumb": 1, "index": 2, "middle": 3, "ring": 4}


def bone_vectors(fingers=("thumb", "index", "middle", "ring"), posture=2):
    """(robot link pairs, human keypoint pairs, how many are directions).

    WHY DIRECTIONS AND NOT POSITIONS. Matching where a fingertip IS leaves the
    posture that reaches it unconstrained - four joints for a three-coordinate
    target - and the solver spends the slack on whatever the objective does not
    see. Measured on a real recording it spent it on a hook: the operator's DIP
    at 10 degrees, the robot's at 106.

    A bone direction constrains the thing that was free. It is also scale-free,
    which quietly removes the other problem: this robot's thumb is 1.27 times
    the operator's while its fingers are 1.04, and no single scaling factor
    serves both. Directions do not care.

    What directions cannot carry is contact - they say which way a finger
    points, never where its tip ended up - so three thumb-to-fingertip vectors
    go in at their true length. Those three are the grasp; the sixteen are the
    hand it is made with.
    """
    robot, human = [], []
    for finger in fingers:
        i = DIGIT_SLOT[finger]
        # THREE bones per digit, not four. The obvious fourth - the operator's
        # wrist-to-knuckle metacarpal - has no counterpart on this robot,
        # because this robot has no wrist: it is bolted to the bench, and
        # `palm_base_link -> proximal_link` is a mounting bracket plus the
        # adduction mechanism. Measured at rest, that bracket points 27 degrees
        # away from the metacarpal it was standing in for, on all three
        # fingers, and the solver was reconciling that fixed error every frame.
        #
        # Nothing is lost by dropping it. The hand is fixed, so the palm frame
        # is the reference, and the proximal phalanx's direction IN that frame
        # already determines the knuckle angle. The metacarpal was never
        # carrying information the other three did not.
        robot += [(f"{finger}_proximal_link", f"{finger}_middle_link"),
                  (f"{finger}_middle_link", f"{finger}_distal_link"),
                  (f"{finger}_distal_link", f"{finger}_tip_link")]
        human += [(4 * i - 3, 4 * i - 2),
                  (4 * i - 2, 4 * i - 1), (4 * i - 1, 4 * i)]
    # VectorOptimizer weights every vector equally, so the only way to say
    # "posture matters more than contact" is to say posture more times. Crude,
    # and it is exactly a weight. Measured on a real recording:
    #
    #     x1   posture 10.5 deg   contact 12.2 mm
    #     x2   posture  7.4 deg   contact 18.8 mm
    #     x4   posture  5.0 deg   contact 25.3 mm
    #
    # x2 by default: this hand's complaint is that the motion does not look
    # like a hand, and six millimetres of grip is a cheap price for a third of
    # the posture error.
    robot, human = robot * max(1, int(posture)), human * max(1, int(posture))
    directions = len(robot)
    for finger in fingers:
        if finger == "thumb":
            continue
        robot.append(("thumb_tip_link", f"{finger}_tip_link"))
        human.append((4, 4 * DIGIT_SLOT[finger]))
    return robot, human, directions


def build_bones(args):
    """A bone-direction retargeter. Measured against DexPilot on a real trace:
    posture error 10.7 degrees against 55, contact error 4-13 mm against 11-14,
    worst frame-to-frame jump 10 degrees against 40."""
    from dex_retargeting.retargeting_config import RetargetingConfig

    robot, human, directions = bone_vectors(posture=args.posture)
    retargeting = RetargetingConfig.from_dict({
        "type": "vector",
        "urdf_path": args.urdf,
        "target_joint_names": args.joints.split(","),
        "target_origin_link_names": [a for a, _ in robot],
        "target_task_link_names": [b for _, b in robot],
        "target_link_human_indices": [[a for a, _ in human],
                                      [b for _, b in human]],
        "scaling_factor": args.scaling,
        "low_pass_alpha": args.alpha,
    }).build()

    optimizer = retargeting.optimizer
    # The robot's own bone lengths, so a human direction can be compared
    # against the robot's without its length arguing about it.
    zero = np.zeros(optimizer.robot.dof)
    optimizer.robot.compute_forward_kinematics(zero)
    indices = optimizer.get_link_indices([a for a, _ in robot] + [b for _, b in robot])
    poses = [optimizer.robot.get_link_pose(i)[:3, 3] for i in indices]
    optimizer.bone_lengths = np.array(
        [float(np.linalg.norm(poses[len(robot) + k] - poses[k]))
         for k in range(directions)])
    optimizer.direction_count = directions
    return retargeting


def oriented(retargeting, distal_links, tip_links):
    """Add a fingertip-orientation term to a DexPilot optimizer, in place.

    THE PROBLEM. A finger has four joints and a fingertip has three
    coordinates, so matching tip POSITION leaves one degree of freedom
    unconstrained per finger. The solver is free to spend that however it
    likes, and it does: measured on a real recording, the operator's DIP sat
    at 3-9 degrees while the robot's sat at 94-106. Same fingertip, opposite
    posture - the robot folds its tip and straightens its knuckle to
    compensate. A hook where the hand made a curl.

    Two things make it stick. The distal joint has six times less authority
    over the tip than the spread joint has (4.7 mm against 29.2 for the same
    0.2 rad), so it barely registers in a least-squares objective; and
    `dex_retargeting` starts every joint at its limit MIDPOINT, which for a
    [-20, 110] joint is +45 degrees - already curled, with nothing pulling it
    back.

    THE FIX, from Analyzing Key Objectives in Human-to-Robot Retargeting
    (2025): add the DIP-to-fingertip vector as an objective term. That vector
    is the last bone's direction, so constraining it constrains the posture
    the position term leaves free. The paper weights it at 10 and reports that
    removing it makes "fingers bend in the wrong direction", which is this
    exactly.

    HOW, without vendoring their objective. Their loss, gradients and
    Jacobians already generalise to any number of keyvectors - the one thing
    sized to the finger count is the trailing weight block, built as
    `ones(num_fingers) * len_proj + num_fingers`. So appending four DIP-to-tip
    pairs and telling the optimizer it has twice as many fingers gives every
    trailing vector a weight of 14 and leaves the pinch projection, which
    operates only on the leading block, untouched.
    """
    import torch

    optimizer = retargeting.optimizer
    fingers = optimizer.num_fingers
    if len(distal_links) != fingers:
        raise ValueError("one distal link per fingertip is required")

    # MANO puts a digit's four points at 4i-3 .. 4i, so its DIP is 4i-1 and
    # its tip is 4i - the same pairs the tip links already use, one back.
    origin, task = optimizer.target_link_human_indices
    dip = np.array([4 * i - 1 for i in range(1, fingers + 1)])
    tip = np.array([4 * i for i in range(1, fingers + 1)])
    optimizer.target_link_human_indices = np.stack(
        [np.concatenate([origin, dip]), np.concatenate([task, tip])])

    optimizer.origin_link_names = list(optimizer.origin_link_names) + list(distal_links)
    optimizer.task_link_names = list(optimizer.task_link_names) + list(tip_links)

    # The cached index tensors are derived from those name lists, so they have
    # to be rebuilt rather than appended to - one link can appear in several
    # vectors and is only computed once.
    names = list(set(optimizer.origin_link_names) | set(optimizer.task_link_names))
    optimizer.computed_link_names = names
    optimizer.origin_link_indices = torch.tensor(
        [names.index(n) for n in optimizer.origin_link_names])
    optimizer.task_link_indices = torch.tensor(
        [names.index(n) for n in optimizer.task_link_names])
    optimizer.computed_link_indices = optimizer.get_link_indices(names)

    # Sizes the trailing weight block to cover the new vectors. Nothing else in
    # the objective reads this; the projection block is sized from a cache
    # built during __init__ and is already fixed.
    optimizer.num_fingers = 2 * fingers

    # The robot's own last-segment lengths, measured off its kinematics. The
    # operator's DIP-to-tip is 13-16 mm and the robot's is 24; compared raw,
    # that 9 mm is a residual the solver can only shed by ROTATING the segment,
    # and it rotates it into the end stop. Measured on a real recording the
    # first version of this term drove every DIP to -20 degrees, its lower
    # limit - the same locking as before, mirrored.
    #
    # Rescaling the human vector to the robot's length leaves only direction to
    # disagree about, which is the whole point of an orientation term.
    zero = np.zeros(optimizer.robot.dof)
    optimizer.robot.compute_forward_kinematics(zero)
    indices = optimizer.get_link_indices(list(distal_links) + list(tip_links))
    poses = [optimizer.robot.get_link_pose(i)[:3, 3] for i in indices]
    optimizer.orientation_lengths = np.array(
        [float(np.linalg.norm(poses[fingers + i] - poses[i]))
         for i in range(fingers)])
    return fingers


def reference_vectors(optimizer, points):
    """The keyvectors the optimizer wants, from 21 MANO keypoints.

    Split out of the serving loop because the offline teacher runs the same
    solver over recorded sessions - as its warm start, and as the
    live-equivalent row it is compared against - and has to build the
    reference EXACTLY the same way. Two copies of this rescaling would be two
    mappings, and the one that drifted would be the one seeding the labels.
    """
    origin, task = optimizer.target_link_human_indices
    reference = points[task] - points[origin]

    bones = getattr(optimizer, "bone_lengths", None)
    if bones is not None:
        # The direction block is rescaled to the robot's own bones so only its
        # direction can disagree; the pinch block keeps its true length,
        # because a grasp is a distance.
        n = optimizer.direction_count
        head = reference[:n]
        norms = np.linalg.norm(head, axis=1, keepdims=True)
        reference[:n] = head / np.maximum(norms, 1e-9) * bones[:, None]

    lengths = getattr(optimizer, "orientation_lengths", None)
    if lengths is not None:
        # Only the last block - the DIP-to-tip vectors - and only their length.
        # Direction is what carries the posture; magnitude here is an accident
        # of whose hand it is.
        tail = reference[-len(lengths):]
        norms = np.linalg.norm(tail, axis=1, keepdims=True)
        scale = max(getattr(optimizer, "scaling", 1.0), 1e-6)
        reference[-len(lengths):] = (tail / np.maximum(norms, 1e-9)
                                     * lengths[:, None] / scale)
    return reference


def solve_frame(retargeting, points):
    """One frame of keypoints to one joint vector, in the ASKED-FOR order.

    `retarget()` returns a vector in PINOCCHIO's joint order, not the order
    requested: it builds it as `qpos[idx_pin2target] = solution`. Those two
    orders happen to coincide for this URDF, which is luck, not a contract -
    reordering by the same index keeps it correct if a re-export ever shuffles
    the joints. Getting this wrong puts the ring finger's angle on the thumb,
    silently.
    """
    optimizer = retargeting.optimizer
    reference = reference_vectors(optimizer, np.asarray(points, dtype=float))
    return retargeting.retarget(reference)[optimizer.idx_pin2target]


def build(args):
    """A DexPilot optimizer for the hand described on the command line.

    The config is assembled here rather than read from a YAML file on purpose:
    the joint order is the wire format for every message this process sends,
    and `main.py` already knows it. A file would be a second copy of that order,
    free to drift from the first, and the failure it produces - the ring
    finger's angle applied to the thumb - is silent.
    """
    from dex_retargeting.retargeting_config import RetargetingConfig

    if args.objective == "bones":
        return build_bones(args)

    retargeting = RetargetingConfig.from_dict({
        "type": "DexPilot",
        "urdf_path": args.urdf,
        "wrist_link_name": args.wrist,
        # Order is the contract: DexPilot pairs these with human keypoints
        # 4, 8, 12, 16 - thumb, index, middle, ring.
        "finger_tip_link_names": args.tips.split(","),
        "target_joint_names": args.joints.split(","),
        "scaling_factor": args.scaling,
        "low_pass_alpha": args.alpha,
    }).build()

    if args.orientation:
        oriented(retargeting, args.distals.split(","), args.tips.split(","))
    if args.open_init:
        # Start from the open hand, not the midpoint of every joint's limits.
        # For a [-20, 110] joint that midpoint is +45 degrees, and a joint with
        # little authority over the fingertip never gets pulled back out of it.
        retargeting.last_qpos = np.zeros(
            len(retargeting.optimizer.target_joint_names), dtype=np.float32)
    return retargeting


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--joints", required=True,
                        help="comma-separated, in the order angles come back")
    parser.add_argument("--tips", required=True,
                        help="comma-separated tip links: thumb,index,middle,ring")
    parser.add_argument("--objective", default="bones",
                        choices=("bones", "dexpilot"),
                        help="bones matches every bone's DIRECTION plus three "
                             "pinch distances; dexpilot matches fingertip "
                             "positions with a hysteretic pinch snap. Measured "
                             "on a real recording, bones reproduces the "
                             "operator's posture five times more closely at "
                             "the same contact accuracy")
    parser.add_argument("--posture", type=int, default=2,
                        help="how much the bone directions outweigh the pinch "
                             "distances. 1 favours contact, 4 favours looking "
                             "like your hand; see bone_vectors for the measured "
                             "trade")
    parser.add_argument("--wrist", default="palm_base_link")
    parser.add_argument("--distals", default="",
                        help="one distal link per fingertip, same order; adds "
                             "the DIP-to-tip orientation term")
    parser.add_argument("--no-orientation", dest="orientation",
                        action="store_false",
                        help="drop the orientation term - the fingers will "
                             "hook rather than curl; for comparison only")
    parser.add_argument("--midpoint-init", dest="open_init",
                        action="store_false",
                        help="start joints at their limit midpoints, as the "
                             "library does, rather than at the open hand")
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="the solver's own low-pass, 1.0 being none. The "
                             "shipped default of 0.2 is a third filter in "
                             "series with the sensor filter and the output "
                             "rate limit - 36 ms of lag only this mode pays. "
                             "Turning it off entirely raised the worst "
                             "frame-to-frame jump from 10 to 27 degrees, so "
                             "0.5 splits it. Neither end is obviously right")
    parser.add_argument("--bind", type=int, default=9872)
    parser.add_argument("--send-to", type=int, default=9873)
    args = parser.parse_args(argv)

    if args.objective == "bones":
        args.orientation = False       # every bone is already a direction
    elif args.orientation and not args.distals:
        args.orientation = False
    try:
        retargeting = build(args)
    # Broad on purpose: a missing package, an unreadable URDF and a link name
    # that is not in it all fail here, and all of them are the operator's to
    # fix. The parent reads this line; a traceback would be worse.
    except Exception as failure:
        print(f"[keyvector] cannot start: {failure}", flush=True)
        return 1

    optimizer = retargeting.optimizer
    origin, task = optimizer.target_link_human_indices
    joints = list(optimizer.target_joint_names)
    shape = (f"{getattr(optimizer, 'direction_count', 0)} bone directions + "
             f"{len(origin) - getattr(optimizer, 'direction_count', 0)} pinch"
             if args.objective == "bones" else
             f"{len(origin)} keyvectors"
             + (" (with DIP-to-tip orientation)" if args.orientation else ""))
    print(f"[keyvector] ready: {len(joints)} joints, {shape}, "
          f"listening on {args.bind}", flush=True)

    inbound = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    inbound.bind((HOST, args.bind))
    inbound.settimeout(0.5)
    outbound = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        try:
            data, _ = inbound.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            message = json.loads(data.decode())
        except (ValueError, UnicodeDecodeError):
            continue
        if message.get("type") == "bye":
            break
        if message.get("type") == "config":
            # Retune the size ratio without a restart. The operator's hand
            # length is only known once a hand has been seen, which is after
            # this process has to have started.
            if "scaling" in message:
                optimizer.scaling = float(message["scaling"])
            continue
        if message.get("type") != "keypoints":
            continue

        points = np.asarray(message["points"], dtype=float)
        if points.shape != (21, 3):
            continue

        started = time.perf_counter()
        try:
            q = solve_frame(retargeting, points)
        # A solve that throws must not take the bridge down: the operator can
        # switch modes, and a dead bridge looks like a frozen hand.
        except Exception as failure:
            outbound.sendto(json.dumps({"type": "error", "seq": message.get("seq"),
                                        "error": str(failure)[:200]}).encode(),
                            (HOST, args.send_to))
            continue
        elapsed = (time.perf_counter() - started) * 1000.0

        outbound.sendto(json.dumps({
            "type": "solution",
            "seq": message.get("seq"),
            "q": [float(v) for v in q],
            "ms": round(elapsed, 3),
        }).encode(), (HOST, args.send_to))

    inbound.close()
    outbound.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
