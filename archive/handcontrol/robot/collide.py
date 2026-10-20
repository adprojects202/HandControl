"""Self-collision for the h4nd_left, and why it cannot just ask the simulator.

MuJoCo resolves self-contact properly now, and mink's IK steers around it, so
this is no longer the only thing standing between the fingers. It stays because
the REAL hand has no physics to save it: a contact pushes back in the
simulator, and the same joint targets go down the bus to sixteen servos that
will simply pull. This is the last check before that happens.

So collisions are checked here instead, geometrically, against the same meshes,
with two decisions that make the check mean something on this particular model:

  * PENETRATION DEPTH, not distance. Adjacent fingers abut at 0.00 mm on an
    open hand - measured, on the raw meshes, not the convex hulls - so any
    distance-based rule forbids the rest pose. Depth separates cleanly:
    0.80 mm worst case on an open hand against 8.44 mm with the index finger
    swung 17 degrees into its neighbour.

  * PER-PAIR BASELINE. Each pair's depth at the neutral pose is measured once
    and subtracted, so what is reported is how much deeper than as-built a pose
    has driven that pair. A pair that ships 0.8 mm inside itself starts at
    zero, like every other.

Pairs are chosen, not listed: two links are worth checking when at least one
movable joint lies between them, which drops the palm against the finger
mounts (rigidly connected, so their distance is a constant) and every
parent-child pair sharing a joint. Links of the same digit are dropped too -
the export overlaps them along the whole chain, and the joint limits do not let
a finger reach itself.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

# Deeper than this past as-built counts as a collision. Picked from the gap
# between what the hand must be able to do and what it must not, measured:
#
#     open hand, worst pair             0.80 mm   (as built, baselines to 0)
#     three fingers curled to 80 deg    0.07 mm
#     fist preset, thumb on index       2.82 mm   must be ALLOWED
#     thumb/index pinch, tips 0.6 mm    3.90 mm   must be ALLOWED
#     index swung 17 deg into neighbour 7.72 mm   must be REFUSED
#     index tip driven into the palm   11.30 mm   must be REFUSED
#
# 6 mm sits in the middle of that gap with 2.1 mm of headroom either side.
# Contact is not the enemy - a hand that cannot touch itself cannot pinch or
# grip. Passing THROUGH is. Tighten it with --collision-margin if the hand is
# being asked to do something more delicate than this.
MARGIN = 0.0060


def _digit(link_name):
    return link_name.split("_")[0]


class SelfCollision:
    """Does this joint vector put the hand through itself?"""

    def __init__(self, model, urdf, margin=MARGIN):
        import fcl
        import trimesh

        self.model = model
        self.names = model.joint_names()
        self.margin = margin
        self._fcl = fcl

        self.shapes = {}
        self.bounds = {}
        for link in ET.parse(urdf).getroot().findall("link"):
            mesh = link.find("collision/geometry/mesh")
            if mesh is None:
                continue
            geometry = trimesh.load(mesh.get("filename"), force="mesh")
            shape = fcl.BVHModel()
            shape.beginModel(len(geometry.faces), len(geometry.vertices))
            shape.addSubModel(np.asarray(geometry.vertices),
                              np.asarray(geometry.faces))
            shape.endModel()
            self.shapes[link.get("name")] = shape

            self.bounds[link.get("name")] = np.asarray(geometry.bounds)

        # ONE POSE PER LINK, cached in `poses` and shared by every pair that
        # names it. The naive version recomputed two chains for every pair, so
        # a 160-pair check cost 320 tree walks and 10.7 ms - too slow to sit
        # in a control loop. Twenty-one walks instead of 320 is where the time
        # went; each is still a walk to the root.
        self.order = list(self.shapes)

        self.pairs = self._choose_pairs()
        neutral = np.zeros(len(self.names))
        poses = self.poses(neutral)
        self.baseline = {pair: self._depth(pair, poses) for pair in self.pairs}

    # ---------------------------------------------------------------- setup
    def _choose_pairs(self):
        """Link pairs whose separation can actually change, across digits."""
        movers = {link: set(self.model.chain_to(link)) for link in self.shapes}
        links = sorted(self.shapes)
        pairs = []
        for i, a in enumerate(links):
            for b in links[i + 1:]:
                if _digit(a) == _digit(b):
                    continue
                if not movers[a] ^ movers[b]:
                    continue          # rigidly connected: distance is a constant
                pairs.append((a, b))
        return pairs

    # ------------------------------------------------------------- geometry
    def poses(self, q):
        """Every collision link's world transform, computed once each and
        reused by every pair that mentions it - see `__init__`."""
        return {link: self.model.link_pose(link, q, self.names)
                for link in self.order}

    def _box(self, link, pose):
        """World AABB of a link, from its local one. Loose, and that is fine -
        it only has to be cheap enough to skip the pairs that cannot touch."""
        position, rotation = pose
        low, high = self.bounds[link]
        corners = np.array([[x, y, z] for x in (low[0], high[0])
                            for y in (low[1], high[1]) for z in (low[2], high[2])])
        world = corners @ rotation.T + position
        return world.min(axis=0), world.max(axis=0)

    def _depth(self, pair, poses):
        a, b = pair
        pa, ra = poses[a]
        pb, rb = poses[b]
        request = self._fcl.CollisionRequest(enable_contact=True,
                                             num_max_contacts=4)
        result = self._fcl.CollisionResult()
        self._fcl.collide(
            self._fcl.CollisionObject(self.shapes[a],
                                      self._fcl.Transform(ra, pa)),
            self._fcl.CollisionObject(self.shapes[b],
                                      self._fcl.Transform(rb, pb)),
            request, result)
        if not result.contacts:
            return 0.0
        return max(c.penetration_depth for c in result.contacts)

    def worst(self, q):
        """(metres past as-built, which pair). Zero when the hand is clear."""
        poses = self.poses(q)
        boxes = {link: self._box(link, poses[link]) for link in self.order}
        deepest, culprit = 0.0, None
        for pair in self.pairs:
            (alo, ahi), (blo, bhi) = boxes[pair[0]], boxes[pair[1]]
            if not (np.all(alo <= bhi) and np.all(blo <= ahi)):
                continue              # boxes apart: the meshes cannot touch
            excess = self._depth(pair, poses) - self.baseline[pair]
            if excess > deepest:
                deepest, culprit = excess, pair
        return deepest, culprit

    def free(self, q):
        """Clear of itself past the margin. Under 2 ms on a clear pose: the
        bounding boxes skip most pairs before fcl sees them. A blocked
        `clamp` bisects through this and costs about eight of them."""
        return self.worst(q)[0] <= self.margin

    # ---------------------------------------------------------------- clamp
    def clamp(self, start, target, steps=7):
        """The furthest point along start -> target that is still clear.

        Bisection rather than a solver: the caller already has a pose it knows
        is safe, the requested one is a straight line away in joint space, and
        the answer only has to be conservative. Seven halvings put it within
        1% of the boundary, which on a finger is well under a millimetre.

        Returns (q, blocked). `blocked` says the target itself was refused, so
        a caller can tell "moved as far as it could" from "did as it was told".
        """
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        if self.free(target):
            return target, False
        if not self.free(start):
            return start, True        # already stuck; do not make it worse

        low, high = 0.0, 1.0          # low is known clear, high is known bad
        for _ in range(steps):
            middle = 0.5 * (low + high)
            if self.free(start + middle * (target - start)):
                low = middle
            else:
                high = middle
        return start + low * (target - start), True
