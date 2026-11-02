"""Fingers inside each other, as the hand sim's own collision geometry has it
(its convex meshes, MuJoCo's contacts and filters): how deep, and how the
joints would move to undo it - so that what the motion space makes stays
clear of itself (motion_space.py), with the same geometry the sims use.
Touching is fine; only depth beyond INSIDE costs.

    contacts = HandContacts(hand)                 # hand_library.HandModel
    cost = contacts.cost(q)                       # torch (N, J) -> (N,), differentiable
    share = contacts.share_inside(q)              # NumPy (N, J) -> poses with fingers inside

The cost of a pose: half the sum, over its contacts deeper than INSIDE, of
the excess depth squared (m^2). Its gradient comes from each contact's
normal and the contact point's Jacobian on the two bodies (mj_jac): the
first-order change of the contact's distance as the joints move - exact for
the geometry, as MuJoCo's own constraint solver uses it. Computed on the
CPU, ~45 us a pose.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

INSIDE = 0.002                # m: fingers deeper in each other than this


class HandContacts:
    def __init__(self, hand):
        self.model, self.data = hand.model, hand.data
        self._jacobians = np.zeros((2, 3, self.model.nv))

    def depths(self, q):
        """(cost (m^2), its gradient (J,)) for one pose."""
        model, data = self.model, self.data
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)         # the motion axes mj_jac reads
        mujoco.mj_collision(model, data)
        cost, gradient = 0.0, np.zeros(model.nv)
        for contact in data.contact[:data.ncon]:
            excess = -contact.dist - INSIDE
            if excess <= 0:
                continue
            first, second = self._jacobians
            mujoco.mj_jac(model, data, first, None, contact.pos, model.geom_bodyid[contact.geom1])
            mujoco.mj_jac(model, data, second, None, contact.pos, model.geom_bodyid[contact.geom2])
            # The distance grows as the second geom moves along the normal
            # (which points from the first to the second), relative to the first.
            closer = -contact.frame[:3] @ (second - first)
            cost += 0.5 * excess ** 2
            gradient += excess * closer
        return cost, gradient

    def cost(self, q):
        """Differentiable cost (N,) of poses (N, J) (any device)."""
        return _Cost.apply(q, self)

    def share_inside(self, q):
        """The share of poses (N, J) with fingers more than INSIDE deep in each other."""
        return float(np.mean([self.depths(row)[0] > 0 for row in np.asarray(q, float)]))


class _Cost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, contacts):
        rows = q.detach().cpu().double().numpy()
        out = [contacts.depths(row) for row in rows]
        ctx.save_for_backward(torch.as_tensor(np.array([g for _, g in out]), dtype=q.dtype,
                                              device=q.device))
        return torch.as_tensor([c for c, _ in out], dtype=q.dtype, device=q.device)

    @staticmethod
    def backward(ctx, grad_out):
        (gradient,) = ctx.saved_tensors
        return grad_out[:, None] * gradient, None
