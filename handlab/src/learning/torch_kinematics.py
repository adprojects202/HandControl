"""The robot hand's 16 points from its joints in PyTorch - differentiable
and batched, for losses on where the fingers are (the motion space, and what
reads intent into it). The same numbers as handmotion.robot_points through
MuJoCo's kinematics (scripts/selftest.py checks them).

    kinematics = TorchKinematics(hand.model, hand.chain, hand.palm)
    points = kinematics(q)        # (..., J) rad -> (..., 16, 3) palm frame, palm lengths

The hand sim's bodies each turn on at most one hinge about their own origin
(as MuJoCo composes them: the body's frame from its parent's, then the
hinge's turn about its axis, from qpos0); anything else is refused.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch


class TorchKinematics(torch.nn.Module):
    def __init__(self, model, chain, palm):
        super().__init__()
        if (model.jnt_type != mujoco.mjtJoint.mjJNT_HINGE).any() or np.abs(model.jnt_pos).max() > 0:
            raise ValueError("expects hinges about their bodies' origins")
        if len(set(model.jnt_bodyid.tolist())) != model.njnt:
            raise ValueError("expects at most one joint a body")
        self.parent = model.body_parentid.tolist()
        self.joint = {int(body): j for j, body in enumerate(model.jnt_bodyid)}
        rotations = np.zeros((model.nbody, 9))
        for body in range(model.nbody):
            mujoco.mju_quat2Mat(rotations[body], model.body_quat[body])
        wrist, axes, length = palm
        self.register_buffer("body_pos", torch.tensor(model.body_pos, dtype=torch.float32))
        self.register_buffer("body_rot", torch.tensor(rotations.reshape(-1, 3, 3), dtype=torch.float32))
        self.register_buffer("axis", torch.tensor(model.jnt_axis, dtype=torch.float32))
        self.register_buffer("qpos0", torch.tensor(model.qpos0[model.jnt_qposadr], dtype=torch.float32))
        self.register_buffer("qadr", torch.tensor(model.jnt_qposadr, dtype=torch.long))
        self.register_buffer("chain_body", torch.tensor([b for b, _ in chain], dtype=torch.long))
        self.register_buffer("chain_local", torch.tensor(np.array([p for _, p in chain]), dtype=torch.float32))
        self.register_buffer("wrist", torch.tensor(wrist, dtype=torch.float32))
        self.register_buffer("axes", torch.tensor(axes, dtype=torch.float32))
        self.length = float(length)

    def forward(self, q):
        """Joints (..., J) in the model's qpos order -> the 16 points (..., 16, 3)."""
        batch = q.shape[:-1]
        eye = torch.eye(3, device=q.device).expand(*batch, 3, 3)
        rot = [eye]
        pos = [torch.zeros(*batch, 3, device=q.device)]
        for body in range(1, len(self.parent)):
            parent = self.parent[body]
            p = pos[parent] + (rot[parent] @ self.body_pos[body].unsqueeze(-1)).squeeze(-1)
            r = rot[parent] @ self.body_rot[body]
            j = self.joint.get(body)
            if j is not None:
                r = r @ _turn(self.axis[j], q[..., self.qadr[j]] - self.qpos0[j])
            rot.append(r)
            pos.append(p)
        rot, pos = torch.stack(rot, -3), torch.stack(pos, -2)
        world = pos[..., self.chain_body, :] + (
            rot[..., self.chain_body, :, :] @ self.chain_local.unsqueeze(-1)).squeeze(-1)
        return (world - self.wrist) @ self.axes / self.length


def _turn(axis, angle):
    """Rotations (..., 3, 3) by `angle` (...) about the unit `axis` (3,)."""
    k = torch.zeros(3, 3, device=axis.device)
    k[0, 1], k[0, 2], k[1, 2] = -axis[2], axis[1], -axis[0]
    k = k - k.T
    s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    return torch.eye(3, device=axis.device) + s * k + (1 - c) * (k @ k)
