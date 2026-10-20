"""The hand's own links as oriented boxes, so an objective can keep them apart.

WHY A PROXY. The teacher is an optimiser through a differentiable FK, and
the student is trained through the same FK. Neither can call a mesh collider
inside a gradient step - the meshes live in MuJoCo and fcl, and a solve is
600 steps over a few thousand frames. What both can differentiate is a
separating-axis test between two boxes. So every collision body becomes the
box its mesh fits inside, carried by that body's own frame, and the gap
between two such boxes, minus one offset per pair, stands in for the distance
between the two meshes. (It began as capsules, one segment per body, which is
where the word survives in a few names.)

THE OFFSET IS MEASURED, PER PAIR, OFF THE MESHES. A servo block is not a
cylinder, and adjacent fingers sit a millimetre apart at rest, so one radius
per link would be wrong on the pairs that matter most. Instead, for every
pair, random poses are swept - uniform ones and pinch-like ones - and MuJoCo's
signed mesh distance is compared with the box gap; the median
difference near contact is the pair's offset, and the spread of the
difference is the proxy's honesty, reported by `calibrate` and checked by
`selftest`. The result is cached beside the generated model, because it
belongs to the meshes and not to any session.

WHAT IT MEASURED on this hand, the day it was built. Before any term used
it, MuJoCo's own contacts said the teacher's labels put the hand through
itself on 42% of the recorded frames and 95% of the synthetic ones: the
pinch drove tip ORIGINS together and the tip meshes overlapped by ten
millimetres, and the sampler crossed neighbouring fingers freely. See
`objective.py` for the two terms that read this, and `robot/collide.py`
for the mesh-exact guard that still stands between any command and the
servos.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..paths import BUILD

# Every body spans consecutive joint origins along its digit; four bodies
# per digit (see `ChainFK.bodies`). The palm is the fifth thing, fixed at
# the origin: on the synthetic set, where fists and curls go further than
# the operator's did, it was 22 of the 37 points of penetration left after
# the digit pairs were handled, and the operator's own labels never touch
# it (0.1% of frames).
BODIES = 4
PALM = "palm_base_link"
# A body that is inside the palm's box AT REST - the mounts the digits hang
# from, the thumb's most of all at 30 mm - has no offset that can make its
# pair mean anything, and charged anyway it bent the whole hand: finger
# posture on a thumb-tap run went from 4.7 to 13.6 degrees. Pairs deeper
# than this at the zero pose are left out, measured by MuJoCo when the
# proxy is calibrated.
PALM_REST_MM = -2.0

# A SERVO LINK IS A BOX, NOT A CAPSULE. Calibrated as capsules, the
# thumb-to-fingertip pair came out at 13.8 mm of segment distance when the
# pads meet and 24 when the sides do - which face meets which is the whole
# difference - and the thumb against the knuckle blocks scattered by 4 to
# 5 mm. So every body is an oriented box, its mesh bounds in its own frame,
# and a pair's gap is the separating-axis gap between the two boxes: over
# the fifteen candidate axes (each box's three face normals and their nine
# cross products), the largest projected separation, negative by the least
# overlap when they intersect. Exact for face-on contact, a lower bound
# elsewhere, and it sees the edge-on crossings that sampled points missed -
# a curl at speed put the thumb 13 mm through the index with no corner of
# either box inside the other. The per-pair offset that remains is the
# box's own error against the mesh, and it is measured.

# How near to contact a sample has to be to vote on a pair's offset, mm.
# The proxy only has to be right where a term can fire.
NEAR_MM = 15.0
# Per pair, the samples inside NEAR_MM it takes to trust the fit. A pair the
# sweep never brought close borrows the median offset of pairs it did.
ENOUGH = 40


class Proxy(nn.Module):
    """Signed surface gaps between every cross-digit pair of bodies, mm.

    `gaps(origins, frames)` takes `ChainFK.walk`'s origins and frames and
    returns (b, pairs); negative is penetration. `tip_gaps` is the thumb's
    distal body against each finger's, in `geometry.THUMB_ROW` order - the
    pinch, measured between surfaces rather than between tip origins.
    """

    def __init__(self, fk, offsets_mm, boxes, palm_pairs=None):
        super().__init__()
        self.fingers = list(fk.fingers)
        self.pairs = pairs_for(self.fingers)
        self.palm_pairs = ([tuple(p) for p in palm_pairs] if palm_pairs is not None
                           else palm_pairs_for(self.fingers))
        self.names = names_for(fk, self.palm_pairs)
        index = torch.tensor([[self.fingers.index(a), i, self.fingers.index(b), j]
                              for a, i, b, j in self.pairs])
        self.register_buffer("index", index)
        self.register_buffer("palm_index", torch.tensor(
            [[self.fingers.index(f), k] for f, k in self.palm_pairs]))
        self.register_buffer("offset", torch.tensor(np.asarray(offsets_mm), dtype=torch.float32))
        assert len(self.offset) == len(self.names), "one offset per pair"
        self.register_buffer("palm_centre", torch.tensor(boxes[PALM][0], dtype=torch.float32))
        self.register_buffer("palm_half", torch.tensor(boxes[PALM][1], dtype=torch.float32))
        # Every body's box in its own frame: (F, BODIES, 3) centre and half
        # extents, metres.
        self.register_buffer("box_centre", torch.tensor(np.array(
            [[boxes[fk.bodies[f][k]][0] for k in range(BODIES)] for f in self.fingers]),
            dtype=torch.float32))
        self.register_buffer("box_half", torch.tensor(np.array(
            [[boxes[fk.bodies[f][k]][1] for k in range(BODIES)] for f in self.fingers]),
            dtype=torch.float32))
        self.tip_rows = [self.pairs.index((f, BODIES - 1, "thumb", BODIES - 1))
                         for f in ("index", "middle", "ring")]
        # Which neighbouring-finger pair each column belongs to - 0 for
        # index against middle, 1 for middle against ring, -1 for the rest -
        # so an objective can ask those pairs for clearance rather than mere
        # non-penetration. See `objective.CLEAR_MM`.
        neighbours = {("index", "middle"): 0, ("middle", "ring"): 1}
        self.register_buffer("adjacent", torch.tensor(
            [neighbours.get((a, b), -1) for a, _, b, _ in self.pairs]
            + [-1] * len(self.palm_pairs)))

    def _boxes(self, origins, frames):
        """World centre (b, F, BODIES, 3) of every body's box; the rotations
        are the frames and the half extents are fixed."""
        return origins[:, :, :BODIES] + torch.einsum(
            "bfkij,fkj->bfki", frames[:, :, :BODIES], self.box_centre)

    @staticmethod
    def box_gap(ca, ra, ha, cb, rb, hb, eps=1e-9):
        """Separating-axis gap between two oriented boxes, metres, batched:
        centres (b, 3), rotations (b, 3, 3), half extents (b, 3). Positive
        when apart, negative by the least overlap when not."""
        axes_a = ra.transpose(1, 2)                                 # (b, 3, 3) rows = face normals
        axes_b = rb.transpose(1, 2)
        crosses = torch.cross(axes_a[:, :, None, :].expand(-1, 3, 3, 3),
                              axes_b[:, None, :, :].expand(-1, 3, 3, 3), dim=-1).reshape(-1, 9, 3)
        # A smooth norm: the plain one has no gradient at zero length, and a
        # pair of parallel face normals puts a zero there on every frame.
        #
        # THE GUARD READS THE RAW LENGTH, NOT THE PADDED ONE. It used to test
        # the padded `length`, which for a zero cross product is sqrt(eps) =
        # 3.2e-5 - comfortably past any small threshold - so the guard never
        # fired, the zero axis stayed in, and it contributed a gap of exactly
        # zero (a zero axis has no separation and no reach). `max` then picked
        # that zero over every true negative, and the whole function returned
        # 0.0 for ANY overlap however deep: two 40 mm boxes driven concentric
        # read "just touching". Nine of the fifteen axes are cross products of
        # parallel face normals whenever two boxes share an axis direction,
        # which two links on different fingers do through most of a curl -
        # their hinges are parallel - so this was the common case, not the
        # corner. What it cost: the collision term saw a constant instead of a
        # depth wherever it was already too late, so it had no gradient to
        # push a finger back out, and `Proxy.penetration` under-reported.
        square = (crosses * crosses).sum(-1, keepdim=True)
        length = square.add(eps).sqrt()
        crosses = crosses / length
        axes = torch.cat([axes_a, axes_b, crosses], dim=1)          # (b, 15, 3)
        valid = torch.cat([torch.ones_like(square[:, :6, 0], dtype=torch.bool),
                           square[:, :, 0] > 1e-6], dim=1)
        centre = (cb - ca)[:, None, :]
        separation = (centre * axes).sum(-1).abs()
        reach_a = (torch.einsum("bkj,bij->bki", axes, ra.transpose(1, 2)).abs()
                   * ha[:, None, :]).sum(-1)
        reach_b = (torch.einsum("bkj,bij->bki", axes, rb.transpose(1, 2)).abs()
                   * hb[:, None, :]).sum(-1)
        gap = separation - reach_a - reach_b
        # Degenerate axes are dropped from the maximum, not counted as zero.
        gap = torch.where(valid, gap, torch.full_like(gap, -1e9))
        return gap.max(dim=1).values

    def gaps(self, origins, frames):
        """`ChainFK.walk` origins and frames -> (b, pairs) mm, signed. Digit
        pairs first, in `pairs` order, then every body against the palm."""
        centres = self._boxes(origins, frames)
        a, i, b, j = self.index.T
        batch = origins.shape[0]
        # Every pair at once: (b * pairs) boxes against boxes.
        ca = centres[:, a, i].reshape(-1, 3)
        cb = centres[:, b, j].reshape(-1, 3)
        ra = frames[:, a, i].reshape(-1, 3, 3)
        rb = frames[:, b, j].reshape(-1, 3, 3)
        ha = self.box_half[a, i].expand(batch, -1, -1).reshape(-1, 3)
        hb = self.box_half[b, j].expand(batch, -1, -1).reshape(-1, 3)
        digit = self.box_gap(ca, ra, ha, cb, rb, hb).reshape(batch, len(self.pairs))
        f, k = self.palm_index.T
        count = batch * len(self.palm_pairs)
        palm = self.box_gap(
            centres[:, f, k].reshape(-1, 3), frames[:, f, k].reshape(-1, 3, 3),
            self.box_half[f, k].expand(batch, -1, -1).reshape(-1, 3),
            self.palm_centre.expand(count, 3),
            torch.eye(3, device=origins.device, dtype=origins.dtype).expand(count, 3, 3),
            self.palm_half.expand(count, 3)).reshape(batch, len(self.palm_pairs))
        return torch.cat([digit, palm], dim=1) * 1000.0 - self.offset

    def tip_gaps(self, origins, frames):
        """The pinch: thumb distal against each finger's distal, mm, in
        `geometry.THUMB_ROW` order."""
        return self.gaps(origins, frames)[:, self.tip_rows]

    def penetration(self, origins, frames, tolerance=0.0):
        """(b, pairs) mm past `tolerance` into each other; zero when clear."""
        return torch.relu(-self.gaps(origins, frames) - tolerance)


def pairs_for(fingers):
    """Every body of one digit against every body of every other digit,
    digits in FK order: (digit a, body i, digit b, body j)."""
    return [(a, i, b, j) for a, b in combinations(fingers, 2)
            for i in range(BODIES) for j in range(BODIES)]


def palm_pairs_for(fingers):
    """Every body against the palm: (digit, body)."""
    return [(f, k) for f in fingers for k in range(BODIES)]


def names_for(fk, palm_pairs=None):
    """The pair names a proxy for this FK carries, in gap-column order."""
    palm_pairs = palm_pairs_for(fk.fingers) if palm_pairs is None else palm_pairs
    return ([f"{fk.bodies[a][i]} x {fk.bodies[b][j]}" for a, i, b, j in pairs_for(fk.fingers)]
            + [f"{fk.bodies[f][k]} x {PALM}" for f, k in palm_pairs])


def geom_pairs(mj, fk, palm_pairs=None):
    """MuJoCo geoms behind every proxy pair, in gap-column order. The palm
    is the geom the exporter named `palm` - the palm_base_link mesh, fused
    into the world body along with the finger mounts, which are not it."""
    import mujoco

    by_body = {}
    for g in range(mj.ngeom):
        by_body.setdefault(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY,
                                             mj.geom_bodyid[g]), []).append(g)
    palm = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "palm")]
    assert palm[0] >= 0, "the exported model has no geom named palm"
    palm_pairs = palm_pairs_for(fk.fingers) if palm_pairs is None else palm_pairs
    return ([(by_body[fk.bodies[a][i]], by_body[fk.bodies[b][j]])
             for a, i, b, j in pairs_for(fk.fingers)]
            + [(by_body[fk.bodies[f][k]], palm) for f, k in palm_pairs])


def mesh_boxes(urdf):
    """Every collision mesh's axis-aligned bounds in its link's frame:
    {link: (centre, half extents)} metres. The URDF places each collision
    mesh at its link origin with no rotation, so the mesh's own bounds are
    the box."""
    import xml.etree.ElementTree as ET

    import trimesh

    out = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        mesh = link.find("collision/geometry/mesh")
        if mesh is None:
            continue
        bounds = np.asarray(trimesh.load(mesh.get("filename"), force="mesh").bounds)
        out[link.get("name")] = ((bounds[0] + bounds[1]) / 2.0, (bounds[1] - bounds[0]) / 2.0)
    return out


# ----------------------------------------------------------------- objects
# THINGS IN THE HAND. A capsule or a sphere in the palm frame, present on a
# range of frames, so the teacher's objective can be asked two things a
# hand alone cannot answer: do not pass through it, and where the operator
# has closed on it, close on it too. Direction-preserving retargeting keeps
# the operator's finger curvature and still closes this hand on nothing -
# its joint axes sit at the back of links 21 mm thick - so the object has to
# be in the objective for the labels to hold it.
OBJECT_SAMPLES = 9          # points along a capsule's axis the box test sees


def point_box_distance(points, centre, rotation, half):
    """Signed distance from world points (b, n, 3) to an oriented box with
    world centre (b, 3), rotation (b, 3, 3) and half extents (b, 3): positive
    outside, negative inside by the least escape, metres."""
    local = torch.einsum("bji,bnj->bni", rotation, points - centre[:, None])
    excess = local.abs() - half[:, None]
    outside = torch.relu(excess).norm(dim=-1)
    inside = excess.max(dim=-1).values.clamp(max=0.0)
    return outside + inside


class Objects(nn.Module):
    """Primitives in the palm frame: (kind, p0, p1, radius) each - a sphere
    has p0 == p1 - and a (frames,) mask of when each is there.

    `gaps(origins, frames, proxy, present)` -> (b, bodies, objects) mm, the
    surface gap between every body's box and every object, +large where the
    object is absent on that frame so no term ever reads it."""

    def __init__(self, items):
        super().__init__()
        self.items = list(items)
        self.register_buffer("p0", torch.tensor([i["p0"] for i in self.items], dtype=torch.float32).reshape(-1, 3))
        self.register_buffer("p1", torch.tensor([i["p1"] for i in self.items], dtype=torch.float32).reshape(-1, 3))
        self.register_buffer("radius", torch.tensor([i["radius"] for i in self.items], dtype=torch.float32))
        self.register_buffer("fractions", torch.linspace(0.0, 1.0, OBJECT_SAMPLES))

    def __len__(self):
        return len(self.items)

    def presence(self, start, end, device=None):
        """(end - start, objects) float mask for a run's frames."""
        mask = torch.zeros(end - start, len(self.items), device=device)
        for k, item in enumerate(self.items):
            a, b = item.get("frames", (0, 10 ** 9))
            lo, hi = max(a, start) - start, min(b, end) - start
            if hi > lo:
                mask[lo:hi, k] = 1.0
        return mask

    def gaps(self, origins, frames, proxy, present=None, shift=None):
        """`shift` (objects, 3) metres moves each object - the solve's free
        variable, see `objective.solve`."""
        batch = origins.shape[0]
        centres = proxy._boxes(origins, frames)                     # (b, F, BODIES, 3)
        out = []
        for k in range(len(self.items)):
            p0 = self.p0[k] if shift is None else self.p0[k] + shift[k]
            p1 = self.p1[k] if shift is None else self.p1[k] + shift[k]
            points = p0 + (p1 - p0) * self.fractions[:, None]      # (S, 3)
            gaps = []
            for fi in range(len(proxy.fingers)):
                for body in range(BODIES):
                    d = point_box_distance(points[None].expand(batch, -1, -1), centres[:, fi, body],
                                           frames[:, fi, body], proxy.box_half[fi, body].expand(batch, 3))
                    gaps.append(d.min(dim=1).values - self.radius[k])
            out.append(torch.stack(gaps, dim=1))                    # (b, bodies)
        gaps = torch.stack(out, dim=2) * 1000.0                     # (b, bodies, objects) mm
        if present is not None:
            gaps = gaps + (1.0 - present)[:, None, :] * 1000.0
        return gaps


    @classmethod
    def load(cls, path):
        """`objects.json` beside a pair file, or None when there is none."""
        path = Path(path)
        if not path.exists():
            return None
        items = json.loads(path.read_text())
        return cls(items) if items else None


# ------------------------------------------------------------- calibration
def mujoco_gaps(mj, data, qadr, geoms, q, distmax=0.05):
    """MuJoCo's signed mesh distance for every pair at one pose, mm."""
    import mujoco

    data.qpos[qadr] = q
    mujoco.mj_kinematics(mj, data)
    fromto = np.zeros(6)
    out = np.empty(len(geoms))
    for k, (ga, gb) in enumerate(geoms):
        best = distmax
        for a in ga:
            for b in gb:
                best = min(best, mujoco.mj_geomDistance(mj, data, a, b, distmax, fromto))
        out[k] = best * 1000.0
    return out


def tips_close(fk, lower, upper, count=150000, within_mm=20.0, seed=1):
    """Random poses whose thumb tip comes within `within_mm` of a fingertip:
    the pinch-like poses a uniform draw is short of. Cheap, FK only."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(lower, upper, size=(count, len(lower))).astype(np.float32)
    with torch.no_grad():
        tips = fk(torch.tensor(q, device=fk.xyz.device))[:, :, -1]
        thumb = fk.fingers.index("thumb")
        others = [k for k, f in enumerate(fk.fingers) if f != "thumb"]
        near = ((tips[:, thumb, None] - tips[:, others]).norm(dim=-1) * 1000.0
                < within_mm).any(dim=1).cpu().numpy()
    return q[near]


def calibrate(fk, lower, upper, hand="left", samples=8000, seed=0):
    """Fit one offset per pair against MuJoCo, return (offsets, boxes, report).

    The sweep is uniform over the joint box, plus `tips_close` poses so the
    thumb-to-fingertip pairs are fitted where a pinch lives. Offsets are
    fitted on the samples within NEAR_MM of contact, which is the only place
    a term reads them.
    """
    import mujoco

    from ..robot.mjcf import ensure_mjcf
    from ..robot.model import ensure_urdf

    boxes = mesh_boxes(ensure_urdf(hand, None))
    mj = mujoco.MjModel.from_xml_path(str(ensure_mjcf(hand)))
    data = mujoco.MjData(mj)
    qadr = np.array([mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in fk.joint_names])
    # Which bodies may be charged against the palm: the ones clear of it at
    # rest - see PALM_REST_MM.
    every = palm_pairs_for(fk.fingers)
    rest = mujoco_gaps(mj, data, qadr, geom_pairs(mj, fk, every), np.zeros(len(fk.joint_names)))
    palm_pairs = [pair for pair, gap in zip(every, rest[len(pairs_for(fk.fingers)):])
                  if gap > PALM_REST_MM]
    geoms = geom_pairs(mj, fk, palm_pairs)
    names = names_for(fk, palm_pairs)

    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    q = np.concatenate([rng.uniform(lower, upper, size=(samples, len(lower))),
                        tips_close(fk, lower, upper, seed=seed + 1)])
    # Raw proxy distances with every offset at zero; the fit is the offset.
    zero = Proxy(fk, np.zeros(len(names)), boxes, palm_pairs).to(fk.xyz.device)
    with torch.no_grad():
        _, origins, frames = fk.walk(torch.tensor(q, dtype=torch.float32, device=fk.xyz.device))
        segment = zero.gaps(origins, frames).cpu().numpy()
    truth = np.stack([mujoco_gaps(mj, data, qadr, geoms, row) for row in q])

    offsets, spread, count = np.zeros(len(names)), np.full(len(names), np.nan), np.zeros(len(names), int)
    for k in range(len(names)):
        near = np.abs(truth[:, k]) < NEAR_MM
        count[k] = int(near.sum())
        if count[k] >= ENOUGH:
            residual = segment[near, k] - truth[near, k]
            offsets[k] = float(np.median(residual))
            spread[k] = float(np.median(np.abs(residual - offsets[k])))
    borrowed = count < ENOUGH
    if borrowed.any():
        offsets[borrowed] = float(np.median(offsets[~borrowed]))
    report = {"pairs": names, "offset_mm": offsets.round(3).tolist(),
              "spread_mm": [None if np.isnan(v) else round(float(v), 3) for v in spread],
              "near_samples": count.tolist(), "borrowed": int(borrowed.sum()),
              "samples": int(len(q)), "palm_pairs": [list(p) for p in palm_pairs],
              "palm_rest_mm": {f"{fk.bodies[f][k]}": round(float(g), 2)
                               for (f, k), g in zip(every, rest[len(pairs_for(fk.fingers)):])},
              "boxes": {link: [c.round(5).tolist(), h.round(5).tolist()]
                        for link, (c, h) in boxes.items()}}
    return offsets, boxes, report, palm_pairs


class Census:
    """The truth the proxy is checked against: MuJoCo's own contacts.

    Poses are set kinematically - no physics to push the links apart - and
    every contact deeper than `deeper_than_mm` counts. What `Teacher.measure`
    reports as penetration comes from here, not from the proxy, so the
    number cannot flatter the term that optimised it.
    """

    def __init__(self, joint_names, hand="left", objects=None):
        import mujoco

        from ..robot.mjcf import ensure_mjcf

        self.mujoco = mujoco
        xml = Path(ensure_mjcf(hand)).read_text()
        self.objects = [] if objects is None else list(objects.items)
        if self.objects:
            # The objects as real geoms, so the contacts are MuJoCo's and not
            # the proxy's. Capsules; a sphere is a capsule of no length.
            geoms = ""
            for k, item in enumerate(self.objects):
                p0, p1, r = item["p0"], item["p1"], item["radius"]
                if np.allclose(p0, p1):
                    geoms += (f'<geom name="object{k}" type="sphere" pos="{p0[0]} {p0[1]} {p0[2]}" '
                              f'size="{r}" rgba="0.85 0.65 0.2 1"/>')
                else:
                    geoms += (f'<geom name="object{k}" type="capsule" fromto="{p0[0]} {p0[1]} {p0[2]} '
                              f'{p1[0]} {p1[1]} {p1[2]}" size="{r}" rgba="0.85 0.65 0.2 1"/>')
            xml = xml.replace("<worldbody>", "<worldbody>" + geoms, 1)
        self.mj = mujoco.MjModel.from_xml_string(xml)
        self.object_geoms = {mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_GEOM, f"object{k}")
                             for k in range(len(self.objects))}
        self.data = mujoco.MjData(self.mj)
        self.qadr = np.array([self.mj.jnt_qposadr[mujoco.mj_name2id(
            self.mj, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names])
        # PAST AS-BUILT, the way `robot/collide.py` counts it: neighbouring
        # fingers ship 0.9 mm inside each other on the convex hulls, and a
        # pose is charged only for what it adds to that.
        self.baseline = {}
        for depth, pair in self._contacts(np.zeros(len(joint_names))):
            self.baseline[pair] = max(self.baseline.get(pair, 0.0), depth)

    def body(self, geom):
        return self.mujoco.mj_id2name(self.mj, self.mujoco.mjtObj.mjOBJ_BODY,
                                      self.mj.geom_bodyid[geom])

    def _contacts(self, q):
        """Self-contacts only; object contacts go through `against_objects`."""
        self.data.qpos[self.qadr] = q
        self.mujoco.mj_forward(self.mj, self.data)
        out = []
        for k in range(self.data.ncon):
            contact = self.data.contact[k]
            if contact.geom1 in self.object_geoms or contact.geom2 in self.object_geoms:
                continue
            out.append((-contact.dist * 1000.0,
                        tuple(sorted((self.body(contact.geom1), self.body(contact.geom2))))))
        return out

    def against_objects(self, q, touching_mm=1.5):
        """(deepest penetration into any object, mm; digit links touching one)."""
        self.data.qpos[self.qadr] = q
        self.mujoco.mj_forward(self.mj, self.data)
        deepest, touching = 0.0, set()
        for k in range(self.data.ncon):
            contact = self.data.contact[k]
            if contact.geom1 in self.object_geoms:
                other = contact.geom2
            elif contact.geom2 in self.object_geoms:
                other = contact.geom1
            else:
                continue
            if self.mj.geom_bodyid[other] == 0:
                continue                      # the palm is where it rests
            deepest = max(deepest, -contact.dist * 1000.0)
            if contact.dist < touching_mm * 1e-3:
                touching.add(self.body(other))
        return deepest, touching

    def objects_survey(self, q, present=None):
        """Over many poses: mean and worst penetration into the objects and
        the mean count of links on them, on the frames where one is there."""
        rows = range(len(q)) if present is None else [i for i in range(len(q)) if present[i]]
        pens, links = [], []
        for i in rows:
            deepest, touching = self.against_objects(q[i])
            pens.append(deepest); links.append(len(touching))
        if not pens:
            return {"object_penetration_mm": None, "object_worst_mm": None, "object_links": None}
        return {"object_penetration_mm": round(float(np.mean(pens)), 2),
                "object_worst_mm": round(float(np.max(pens)), 1),
                "object_links": round(float(np.mean(links)), 2)}

    def worst(self, q):
        """(deepest penetration past as-built, mm, body pair) at one pose;
        (0, None) if clear."""
        deepest, pair = 0.0, None
        for depth, bodies in self._contacts(q):
            depth -= self.baseline.get(bodies, 0.0)
            if depth > deepest:
                deepest, pair = depth, bodies
        return deepest, pair

    def survey(self, q, deeper_than_mm=3.0):
        """Over many poses: the share deeper than the threshold, the worst,
        and which pairs did it."""
        from collections import Counter

        worst, pairs, hits = 0.0, Counter(), 0
        for row in q:
            depth, pair = self.worst(row)
            worst = max(worst, depth)
            if depth > deeper_than_mm:
                hits += 1
                pairs[pair] += 1
        return {"penetration_pct": round(100.0 * hits / max(len(q), 1), 1),
                "worst_penetration_mm": round(worst, 1),
                "pairs": [(f"{a} x {b}", n) for (a, b), n in pairs.most_common(3)]}


# ------------------------------------------------------------------ repair
# REPAIR, NOT REJECT. A regressor at four degrees of error on a hand whose
# fingers sit a millimetre apart will pass through a neighbour between two
# clear poses however good its labels are - the clearance is smaller than
# the precision. So the last thing a command goes through is a few steps of
# the penetration penalty on the proxy: the offending joints move, the way
# that clears, and nothing else does. Differentiable, so training can run
# through it and learn to emit poses that need none of it.
#
# THE STEP LANDS ON THE BOUNDARY. A fixed-rate gradient step overshot: seven
# millimetres of penetration became 3.4 degrees on four joints at once, which
# carried the tip fifteen millimetres and left every repaired pinch five to
# six millimetres SHORT of the skin - and a student trained through that
# learned to hover there. The penalty is a sum of squares of near-linear
# gaps, so the step along its gradient that zeroes it is 2f / |g|^2 - one
# Gauss-Newton step - damped a little for the curvature and cleaned up by
# the next. Measured on the old student's own collisions: the same 0.0% of
# frames past 3 mm, and the pinch lands at the tolerance instead of outside.
REPAIR_STEPS = 6
REPAIR_DAMPING = 0.9    # of the step that would zero the linearised penalty
REPAIR_MAX_RAD = 0.10   # per step, per joint: a repair is a nudge, not a solve


def repair(fk, proxy, q, tolerance=0.0, steps=REPAIR_STEPS,
           damping=REPAIR_DAMPING):
    """Move `q` (b, joints) the least it needs to clear the proxy past
    `tolerance`. Frames already clear are returned untouched.

    STRAIGHT-THROUGH when `q` carries a graph: each step's shift is taken as
    a constant and the result stays differentiable in `q` through the
    identity, so a loss on the repaired pose trains the network as if the
    repair were a fixed offset. The exact second-order version was tried
    and produced NaN through the box test's degenerate axes on the first
    batch. `train.losses` can run this way, but ships with
    `train.REPAIR_STEPS` at zero - see the note there; live, `Mimic.repair`
    calls this with no graph at all."""
    trained = q.requires_grad
    out = q
    for _ in range(steps):
        held = out if trained else out.detach().requires_grad_(True)
        _, origins, frames = fk.walk(held)
        penetration = proxy.penetration(origins, frames, tolerance)
        f = (penetration ** 2).sum(dim=1)                     # (b,)
        if not bool(f.detach().max() > 0.0):
            break
        grad, = torch.autograd.grad(f.sum(), held, retain_graph=trained)
        norm = (grad * grad).sum(dim=1).clamp_min(1e-12)      # (b,)
        alpha = damping * 2.0 * f.detach() / norm
        step = (alpha[:, None] * grad).clamp(-REPAIR_MAX_RAD, REPAIR_MAX_RAD).detach()
        out = held - step
    return out if trained else out.detach()


# Bumped whenever `Proxy.box_gap` changes what it returns, so a cache fitted
# against the old arithmetic is not reused: the offsets are a fit BETWEEN this
# function and MuJoCo, and they only mean anything alongside the version of it
# they were measured against. 2: the degenerate-axis guard, which had made
# every overlap read as exactly zero.
GAP_VERSION = 2


def cache_path(hand):
    return BUILD / f"collision_{hand}.json"


def for_hand(fk, lower, upper, hand="left", force=False):
    """The calibrated proxy for this hand, from the cache or by measuring."""
    from ..robot.model import ensure_urdf

    path = cache_path(hand)
    stamp = Path(ensure_urdf(hand, None)).stat().st_mtime
    if path.exists() and not force:
        stored = json.loads(path.read_text())
        palm_pairs = [tuple(p) for p in stored.get("palm_pairs", [])]
        if stored.get("urdf_mtime") == stamp and stored.get("gap_version") == GAP_VERSION and \
                stored.get("pairs") == names_for(fk, palm_pairs):
            boxes = {link: (np.array(c), np.array(h)) for link, (c, h) in stored["boxes"].items()}
            return Proxy(fk, stored["offset_mm"], boxes, palm_pairs).to(fk.xyz.device)
    offsets, boxes, report, palm_pairs = calibrate(fk, lower, upper, hand)
    report["urdf_mtime"] = stamp
    report["gap_version"] = GAP_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    return Proxy(fk, offsets, boxes, palm_pairs).to(fk.xyz.device)
