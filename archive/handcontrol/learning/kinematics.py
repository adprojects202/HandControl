"""Differentiable forward kinematics for the whole finger chain, in torch.

One FK for everything learned. The student's loss needs fingertips; the
teacher's objective needs every phalanx origin, because a bone-direction term
is the difference of two link positions. Both come from here, walked off the
URDF the same way `HandModel.link_pose` walks it - so a re-export that moves a
joint moves it in both, and there is no second copy of the chain to drift.

Positions are metres in the palm frame, exactly as `link_pose` reports them,
so anything measured against the numpy path can be checked against this one
to the precision of float32.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..robot.model import FINGERS

# The four link origins a bone-direction objective needs, base to tip. The
# three bones are the differences between consecutive entries.
CHAIN = ("proximal", "middle", "distal", "tip")


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch),
                              np.sin(pitch), np.cos(yaw), np.sin(yaw))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def axis_angle(axis, angle):
    """Rodrigues, batched over `angle`. `axis` is one fixed (3,)."""
    axis = axis / axis.norm().clamp_min(1e-12)
    x, y, z = axis
    zero = torch.zeros_like(angle)
    cross = torch.stack([
        torch.stack([zero, -z.expand_as(angle), y.expand_as(angle)], -1),
        torch.stack([z.expand_as(angle), zero, -x.expand_as(angle)], -1),
        torch.stack([-y.expand_as(angle), x.expand_as(angle), zero], -1)], -2)
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    sine, cosine = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    return eye + sine * cross + (1.0 - cosine) * (cross @ cross)


class ChainFK(nn.Module):
    """(batch, joints) -> (batch, fingers, 4 links, 3): every phalanx origin.

    Walked from each fingertip back to the palm at construction, then replayed
    forward at call time. Joints the model has but the joint vector does not
    name are treated as fixed at zero, which is what the URDF says they are.

    Two views of the same walk. `forward` gives the four named links a
    bone-direction term wants, in CHAIN order, the same for every digit.
    `origins` gives every MOVABLE joint's frame in chain order plus the tip -
    five per digit - which is where each collision body starts and ends; the
    thumb's order differs from the fingers' (proximal, adduction, middle,
    distal against adduction, proximal, middle, distal), and `bodies` names
    which link each origin begins.
    """

    def __init__(self, model, joint_names, fingers=FINGERS):
        super().__init__()
        self.fingers = list(fingers)
        self.joint_names = list(joint_names)
        self.steps = []                 # per finger: [(joint index or -1, chain slot or -1, origin slot or -1)]
        self.bodies = {}                # per finger: the link each origin starts
        xyz, rpy, axes = [], [], []
        for finger in self.fingers:
            chain, link = [], f"{finger}_tip_link"
            while link in model.by_child:
                chain.append(model.by_child[link])
                link = model.by_child[link].parent
            chain.reverse()
            wanted = [f"{finger}_{name}_link" for name in CHAIN]
            steps, bodies = [], []
            for joint in chain:
                xyz.append(joint.xyz)
                rpy.append(rpy_matrix(joint.rpy))
                axes.append(joint.axis if joint.movable
                            else np.array([0.0, 0.0, 1.0]))
                origin = -1
                if joint.movable or joint.child == f"{finger}_tip_link":
                    origin = len(bodies)
                    bodies.append(joint.child)
                steps.append((self.joint_names.index(joint.name)
                              if joint.movable and joint.name in self.joint_names
                              else -1,
                              wanted.index(joint.child)
                              if joint.child in wanted else -1,
                              origin))
            self.steps.append(steps)
            self.bodies[finger] = bodies
        self.origins_per_digit = len(self.bodies[self.fingers[0]])
        assert all(len(b) == self.origins_per_digit for b in self.bodies.values()), \
            "every digit is expected to carry the same number of movable joints"
        self.register_buffer("xyz", torch.tensor(np.array(xyz), dtype=torch.float32))
        self.register_buffer("rpy", torch.tensor(np.array(rpy), dtype=torch.float32))
        self.register_buffer("axis", torch.tensor(np.array(axes), dtype=torch.float32))

    def walk(self, q):
        """One pass: named links (b, F, 4, 3), joint origins (b, F, 5, 3) and
        the rotation of each body's frame (b, F, 5, 3, 3)."""
        batch, links_out, origins_out, frames_out, step = q.shape[0], [], [], [], 0
        for steps in self.steps:
            position = q.new_zeros(batch, 3)
            rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(batch, 3, 3)
            links = [None] * len(CHAIN)
            origins = [None] * self.origins_per_digit
            frames = [None] * self.origins_per_digit
            for joint, slot, origin in steps:
                local = self.rpy[step].expand(batch, 3, 3)
                if joint >= 0:
                    local = local @ axis_angle(self.axis[step], q[:, joint])
                position = position + torch.einsum("bij,j->bi", rotation, self.xyz[step])
                rotation = rotation @ local
                if slot >= 0:
                    links[slot] = position
                if origin >= 0:
                    origins[origin] = position
                    frames[origin] = rotation
                step += 1
            links_out.append(torch.stack(links, dim=1))
            origins_out.append(torch.stack(origins, dim=1))
            frames_out.append(torch.stack(frames, dim=1))
        return (torch.stack(links_out, dim=1), torch.stack(origins_out, dim=1),
                torch.stack(frames_out, dim=1))

    def forward(self, q):
        return self.walk(q)[0]


