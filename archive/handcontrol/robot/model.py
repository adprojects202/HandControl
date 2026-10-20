"""The h4nd_left kinematic model, read straight from the URDF.

Joint names, limits, chains and fingertip offsets are all derived from
`h4nd_left/h4nd_left_genesis.urdf` and the STL meshes it points at. Change it
and this follows.

Four facts about the vendored model shape everything in this file:

  * Every joint declares effort="0" velocity="0". Taken literally that means a
    motor that cannot move, so the simulator substitutes real XL330 numbers.
  * A distal link's origin sits on its own joint axis, so the link frame does
    not move when that joint rotates. Aiming IK at `index_distal_link` would
    therefore ignore the fingertip joint, so `ensure_ik_urdf` writes a derived
    URDF with a fixed `<finger>_tip_link` at the actual tip of each distal
    mesh, and IK targets that instead.
  * The joint NAMES do not describe the motion on the thumb: its `proximal`
    joint swings across the palm and its `adduction` joint flexes. Everything
    that asks "is this a spread or a flexion joint" must go through
    `joint_role`, never through the name. See ROLES.
  * The exported spread limits are 130 degrees on index and ring against 40 on
    the middle finger, whose bracket is identical - a copied limit. See
    `set_spread_limit`.

Nothing here is hardcoded from a spec sheet except `ROLES`, which cannot be
derived and says so.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import re

import numpy as np

from ..paths import BUILD, URDF_PACKAGE

URDF = URDF_PACKAGE / "h4nd_left_genesis.urdf"       # relative mesh paths, LF

# GENERATED FROM THE VENDORED EXPORT, and only in two ways: LF line endings,
# and `filename="meshes/..."` in place of the `package://` URIs, so the file
# opens without a ROS package path. Nothing else about it is rewritten.
#
# It used to also correct `ring_dstial_link` - a transposition the export
# carried on the ring's distal link and its mesh file alike. That is fixed at
# source now, in the vendored URDF and by renaming the STL, so the two names
# agree and this copy is a straight transcription again. See
# `world/h4nd_left/README.md`.
MIRRORED_URDF = BUILD / "h4nd_right_mirrored.urdf"       # the same hand, other chirality
MIRRORED_MESHES = BUILD / "meshes_mirrored"
IK_VENDORED = BUILD / "h4nd_left_ik.urdf"       # the package as exported: the LEFT hand
IK_MIRRORED = BUILD / "h4nd_right_ik.urdf"      # reflected through XZ: the right

# WHICH LABEL GETS THE MIRROR.
#
# THIS INVERTED when the model was replaced, and the disagreement it used to
# record is gone. The first export was a RIGHT hand: every geometric test said
# so, `left` therefore meant the MIRRORED copy, and the label and the
# arithmetic disagreed on purpose because the label names the hand on the
# bench. `h4nd_left` is modelled as the left hand directly - measured, its
# (forward, normal, thumbward) triple is left-handed - so the vendored copy IS
# the bench hand and the mirror now makes the right one. Label and geometry
# finally agree, and this table says the plain thing.
MIRRORED_FOR = {"left": False, "right": True}
HANDS = {"left": IK_VENDORED, "right": IK_MIRRORED}
DEFAULT_HAND = "left"                                # what is actually on the bench

FINGERS = ("index", "middle", "ring", "thumb")
TIP_LINKS = {f: f"{f}_tip_link" for f in FINGERS}
DISTAL_LINKS = {f: f"{f}_distal_link" for f in FINGERS}

# XL330-M288: stall torque ~0.52 Nm. The URDF declares effort="0" velocity="0"
# for every joint, which taken literally is a motor that cannot move, so real
# numbers are substituted at build time.
#
# Rotor inertia reflected through the gearbox, per joint. Not a numerical
# fudge: the links weigh 9 grams with inertias near 1e-6, and a position
# actuator stiff enough to hold one is stiff enough to make the integrator
# diverge - at REST, with contacts disabled, inside 4 ms. A 288:1 XL330
# reflects far more inertia than the link has, so this is the honest model.
# Measured: 0.001 takes a fist from 294 degrees of error to 9, and the
# residual joint velocity to zero.
ARMATURE = 0.001

# What the IK solver is given. `damping` is mink's DampingTask cost.
# Defaults, not limits: `HandSim.set_ik` moves them at runtime and the TUNING
# tab has them on sliders.
# `max_solver_iters` is a CAP, not a count: mink stops as soon as the tip is
# inside `pos_tol`, which warm started is one or two steps. The cap only binds
# on a cold solve from far away, and 20 was nowhere near enough for one - it
# left a standing target 38 mm out.
IK = {"damping": 0.01, "max_solver_iters": 60, "pos_tol": 5e-4}

# Tuned for MuJoCo, and it needs no fudge. Under Genesis these numbers had to
# lie: a slider-sized step saturated the 0.52 Nm clip for its whole travel, so
# the only way to get a 0.26 s rise was to pretend the servo was ten times
# stronger than it is. MuJoCo reaches 0.10 s at the XL330's REAL stall torque.
#
# Measured, 0 -> 57.3 degrees on index_proximal, armature 0.001:
#
#     clip    KP    KV   overshoot   rise to 90%
#     0.5     10   0.5     +0.07 deg    0.13 s
#     0.5     20   1.0     +0.05 deg    0.13 s
#     0.5     30   1.0     +0.04 deg    0.10 s    <- these
#     0.5     30   2.0     +0.85 deg    0.16 s
#     0.5     50   2.0     +0.45 deg    0.11 s
#     5.0     40   4.0    +20.05 deg    0.16 s    (the Genesis-era values)
#
# The last row is why gains do not travel between engines: those numbers were
# right for Genesis and are badly wrong here.
EFFORT = 0.5           # Nm - the XL330-M288's stall, no longer an approximation
KP = 30.0
KV = 1.0


def _floats(text, default=(0.0, 0.0, 0.0)):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0

    @property
    def movable(self):
        return self.type in ("revolute", "prismatic", "continuous")


@dataclass
class HandModel:
    """Joints, chains and limits for one URDF."""

    path: Path
    joints: dict = field(default_factory=dict)
    by_child: dict = field(default_factory=dict)
    root_link: str = ""

    @classmethod
    def load(cls, path=URDF):
        path = Path(path)
        root = ET.parse(path).getroot()
        model = cls(path=path)
        children = set()
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            limit = element.find("limit")
            joint = Joint(
                name=element.get("name"),
                type=element.get("type"),
                parent=element.find("parent").get("link"),
                child=element.find("child").get("link"),
                xyz=_floats(origin.get("xyz") if origin is not None else None),
                rpy=_floats(origin.get("rpy") if origin is not None else None),
                axis=_floats(axis.get("xyz") if axis is not None else None,
                             (0.0, 0.0, 1.0)),
                lower=float(limit.get("lower")) if limit is not None else 0.0,
                upper=float(limit.get("upper")) if limit is not None else 0.0,
            )
            model.joints[joint.name] = joint
            model.by_child[joint.child] = joint
            children.add(joint.child)
        links = [l.get("name") for l in root.findall("link")]
        model.root_link = next(l for l in links if l not in children)
        return model

    # ------------------------------------------------------------- chains
    def chain_to(self, link):
        """Movable joints from the root down to `link`, base first.

        Walks parents rather than assuming an order: the thumb runs
        proximal -> adduction -> middle -> distal while the fingers run
        adduction -> proximal -> middle -> distal, and only the URDF knows that.
        """
        names = []
        while link in self.by_child:
            joint = self.by_child[link]
            if joint.movable:
                names.append(joint.name)
            link = joint.parent
        return list(reversed(names))

    def finger_chains(self):
        return {f: self.chain_to(DISTAL_LINKS[f]) for f in FINGERS}

    def joint_names(self):
        """All movable joints, finger by finger, base to tip."""
        out = []
        for chain in self.finger_chains().values():
            out += [n for n in chain if n not in out]
        return out

    def limits(self, names=None):
        names = names or self.joint_names()
        lower = np.array([self.joints[n].lower for n in names])
        upper = np.array([self.joints[n].upper for n in names])
        return lower, upper

    # ------------------------------------------------- forward kinematics
    def link_pose(self, link, q, names=None):
        """(position, rotation) of `link` in root coordinates for joint vector q.

        Plain numpy, no simulator - used to seed IK, to place the generated tip
        frames, and to sanity-check what the simulator reports.
        """
        names = names or self.joint_names()
        angles = dict(zip(names, np.asarray(q, dtype=float)))
        transforms = []
        while link in self.by_child:
            joint = self.by_child[link]
            transforms.append((joint, angles.get(joint.name, 0.0)))
            link = joint.parent
        position = np.zeros(3)
        rotation = np.eye(3)
        for joint, angle in reversed(transforms):
            local = _rpy(joint.rpy)
            if joint.movable:
                local = local @ _axis_angle(joint.axis, angle)
            position = position + rotation @ joint.xyz
            rotation = rotation @ local
        return position, rotation


def _rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def R_to_rpy(R):
    """Inverse of `_rpy`: URDF's fixed-axis roll-pitch-yaw, Rz Ry Rx."""
    if abs(R[2, 0]) > 1.0 - 1e-9:              # pitch at +/-90, roll and yaw merge
        pitch = -math.copysign(math.pi / 2, R[2, 0])
        return np.array([0.0, pitch, math.atan2(-R[0, 1], R[1, 1])])
    return np.array([math.atan2(R[2, 1], R[2, 2]),
                     -math.asin(max(-1.0, min(1.0, R[2, 0]))),
                     math.atan2(R[1, 0], R[0, 0])])


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return np.array([
        [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
        [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
        [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
    ])


# ------------------------------------------------------------------ mirror
# Reflection through the XZ plane. The vendored `h4nd_left` URDF is a LEFT
# hand: fingers along +Z (index, middle and ring tips at z = +149 mm at rest)
# and the thumb along -Y (tip at y = -148), with the fingers themselves
# spanning y = -28 to +23. Measured from the URDF, not asserted.
#
# The comment this replaces said "thumb at +Y" and named `h4nd_2`, and both
# were wrong after the hand was swapped: h4nd_2 was a RIGHT hand that had to
# be mirrored to be used, which is the opposite of the package here.
#
# This IS what `--hand left` uses, and left is the default - see MIRRORED_FOR
# above for why the naming and the arithmetic disagree. `--hand right` is the
# vendored package untouched. Either way the vendored files are only ever read.
MIRROR = np.diag([1.0, -1.0, 1.0])


def _mirror_rpy(rpy):
    """M R M, which is a rotation again because det(M)^2 = 1."""
    return R_to_rpy(MIRROR @ _rpy(rpy) @ MIRROR)


def _mirror_axis(axis):
    """A joint axis is a pseudovector: it reflects to -M a, not M a.

    M R(a, t) M = R(-M a, t), so the mirrored joint takes the SAME angle to
    reach the mirrored pose. That is what keeps every limit, every calibration
    sign and every recorded pose valid across the mirror.
    """
    return -(MIRROR @ np.asarray(axis, dtype=float))


def _set(element, attribute, vector, fmt="{:.9g}"):
    if element is not None:
        element.set(attribute, " ".join(fmt.format(float(v)) for v in vector))


def mirror_meshes(source, out_dir):
    """Reflected copies of every STL, with the winding put back the right way."""
    import trimesh

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = np.eye(4)
    transform[:3, :3] = MIRROR
    written = {}
    for stl in sorted((Path(source).parent / "meshes").glob("*.STL")):
        mesh = trimesh.load(stl, force="mesh")
        mesh.apply_transform(transform)
        if mesh.volume < 0:
            # A reflection turns every triangle inside out. Left uncorrected the
            # surface renders black and inertia comes out negative.
            mesh.invert()
        target = out_dir / stl.name
        mesh.export(target)
        written[stl.name] = target
    return written


def mirror_urdf(source=URDF, target=MIRRORED_URDF, force=False):
    """Write the opposite-chirality hand: same joint names, same limits."""
    target = Path(target)
    if target.exists() and not force and \
            target.stat().st_mtime >= Path(source).stat().st_mtime:
        return target

    meshes = mirror_meshes(source, MIRRORED_MESHES)
    tree = ET.parse(source)
    root = tree.getroot()
    # `_mirrored`, not `_right`: which LABEL this chirality answers to is
    # decided by MIRRORED_FOR above, and it is `left`. Naming the file's robot
    # after the other one is how that disagreement gets read back in wrong.
    root.set("name", (root.get("name") or "h4nd_left") + "_mirrored")

    for joint in root.findall("joint"):
        origin = joint.find("origin")
        if origin is not None:
            _set(origin, "xyz", MIRROR @ _floats(origin.get("xyz")))
            _set(origin, "rpy", _mirror_rpy(_floats(origin.get("rpy"))))
        axis = joint.find("axis")
        if axis is not None and np.linalg.norm(_floats(axis.get("xyz"))) > 0:
            _set(axis, "xyz", _mirror_axis(_floats(axis.get("xyz"))), "{:g}")

    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            _set(origin, "xyz", MIRROR @ _floats(origin.get("xyz")))
            _set(origin, "rpy", _mirror_rpy(_floats(origin.get("rpy"))))
            inertia = inertial.find("inertia")
            if inertia is not None:
                # I -> M I M leaves the diagonal alone and flips the two
                # products of inertia that carry a single y.
                for name in ("ixy", "iyz"):
                    inertia.set(name, f"{-float(inertia.get(name)):.9g}")
        for part in list(link.findall("visual")) + list(link.findall("collision")):
            origin = part.find("origin")
            _set(origin, "xyz", MIRROR @ _floats(origin.get("xyz")))
            _set(origin, "rpy", _mirror_rpy(_floats(origin.get("rpy"))))
            mesh = part.find("geometry/mesh")
            if mesh is not None:
                name = Path(mesh.get("filename")).name
                mesh.set("filename", str(meshes[name]))

    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return target


def set_spread_limit(path, degrees):
    """Clamp the three fingers' spread joints to +/- `degrees`.

    The exported spans are 130 degrees for index and ring - the same span as
    every flexion joint, which is what a copied limit looks like - against 40
    for the middle finger, whose bracket and servo are identical. 110 degrees
    of finger spread is not a thing a hand does, and left in place it is what
    the IK will happily use to reach a target. The real number is whatever the
    hardware does; `hw watch` measures it.

    Selected by ROLE, and the thumb is deliberately left out: its spread is
    `thumb_proximal`, a 95-degree swing across the palm that is nothing like a
    finger's sideways splay, and clamping it to a finger's range would be
    wrong. Matching on the name would have excluded the thumb by accident
    instead of on purpose, and got the fingers right only because their names
    happen to agree with their roles - the same coincidence that hid the
    inverted thumb signs for three rounds.
    """
    path = Path(path)
    tree = ET.parse(path)
    limit = math.radians(abs(degrees))
    changed = False
    for joint in tree.getroot().findall("joint"):
        name = joint.get("name")
        if joint_role(name) != "spread" or name.startswith("thumb"):
            continue
        element = joint.find("limit")
        if element is None:
            continue
        if not (abs(float(element.get("lower")) + limit) < 1e-9
                and abs(float(element.get("upper")) - limit) < 1e-9):
            element.set("lower", f"{-limit:.9g}")
            element.set("upper", f"{limit:.9g}")
            changed = True
    if changed:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def ensure_urdf(hand=DEFAULT_HAND, spread_limit=None, force=False):
    """The URDF to actually simulate: chosen chirality, with fingertip frames."""
    if hand not in HANDS:
        raise ValueError(f"hand must be 'left' or 'right', not {hand!r}")
    source = mirror_urdf(force=force) if MIRRORED_FOR[hand] else URDF
    target = HANDS[hand]
    if spread_limit is not None:
        target = target.with_name(f"{target.stem}_s{int(spread_limit)}.urdf")
    path = ensure_ik_urdf(source, target, force=force)
    if spread_limit is not None:
        set_spread_limit(path, spread_limit)
    return path


# --------------------------------------------------------------- tip frames
def tip_offsets(path=URDF, band=0.002):
    """Where each fingertip sits in its distal link's own frame.

    Measured, not assumed. The link origin is the joint, so the link extends
    along whichever axis reaches furthest from it - +Z for the fingers, +Y for
    the thumb, neither written down anywhere. Take that axis, then average the
    vertices within `band` of its far end: the single furthest vertex is a
    corner of the fingertip, the average of the end face is its middle.
    """
    import trimesh

    path = Path(path)
    root = ET.parse(path).getroot()
    meshes = {}
    for link in root.findall("link"):
        collision = link.find("collision/geometry/mesh")
        if collision is not None:
            meshes[link.get("name")] = collision.get("filename")

    offsets = {}
    for finger, link in DISTAL_LINKS.items():
        mesh = trimesh.load(path.parent / meshes[link], force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=float)
        axis = int(np.abs(vertices).max(axis=0).argmax())
        column = vertices[:, axis]
        sign = 1.0 if column.max() >= -column.min() else -1.0
        far = (column * sign).max()
        offsets[finger] = vertices[(column * sign) >= far - band].mean(axis=0)
    return offsets


def _absolute_meshes(text, source):
    """Point every mesh at a full path.

    Mesh filenames resolve against the directory of the URDF that names them,
    and the generated copy does not live in that directory. Rewriting to
    absolute paths is what makes build/ a safe place to put it. ROS-style
    package:// URIs are resolved against the package root, one level above.
    """
    source = Path(source).resolve()
    # `package://<pkg>/meshes/...` is relative to the PACKAGE ROOT, which is
    # not always the directory the URDF sits in - a vendored export puts its
    # urdf in `<pkg>/urdf/` and its meshes in `<pkg>/meshes/`. Find the
    # ancestor that actually holds `meshes/` rather than assuming one level.
    root = next((d for d in (source.parent, *source.parents)
                 if (d / "meshes").is_dir()), source.parent)
    text = re.sub(r'filename="package://[^/]+/', f'filename="{root}/', text)
    return text.replace('filename="meshes/', f'filename="{source.parent}/meshes/')


def ensure_ik_urdf(source, target, force=False):
    """Write a copy of `source` with a `<finger>_tip_link` at each fingertip.

    Both paths are required. They used to default to the vendored URDF and one
    fixed output name, which quietly produced an un-mirrored model under a
    filename that said nothing about its chirality - the one mistake this
    module can make that nothing downstream would catch. Call `ensure_urdf`.

    The tip links are massless fixed children of the distal links, so they add
    a frame to aim IK at and change no dynamics. Load this file with
    merge_fixed_links=False or the frames are merged away again.

    Generated, never edited by hand: delete build/ and it comes back.
    """
    target = Path(target)
    if target.exists() and not force and \
            target.stat().st_mtime >= Path(source).stat().st_mtime:
        return target

    offsets = tip_offsets(source)
    text = Path(source).read_text()
    addition = []
    for finger, offset in offsets.items():
        x, y, z = offset
        addition.append(f"""
  <link name="{finger}_tip_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="1E-09" />
      <inertia ixx="1E-12" ixy="0" ixz="0" iyy="1E-12" iyz="0" izz="1E-12" />
    </inertial>
  </link>
  <joint name="{finger}_tip_joint" type="fixed">
    <origin xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="0 0 0" />
    <parent link="{DISTAL_LINKS[finger]}" />
    <child link="{finger}_tip_link" />
  </joint>""")

    text = _absolute_meshes(text, source)
    marker = text.rindex("</robot>")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text[:marker] + "\n".join(addition) + "\n" + text[marker:])
    return target


# What each joint DOES, which is what a control should be labelled with.
#
# For the fingers the names would do: adduction spreads, proximal/middle/distal
# flex. The thumb's names do not describe its motion. At the zero pose a 0.4 rad
# probe moves the thumb tip by (+54.6, +20.5, +11.8) mm on `thumb_proximal` -
# a big swing across the palm, which is what anyone looking at the hand calls
# the spread - and by (-2.2, +3.2, +44.9) mm on `thumb_adduction`, which mostly
# lifts along the fingers, like a knuckle.
#
# This is a table rather than something derived because the thumb cannot be
# derived: all four of its axes sit obliquely (axis . distal_axis runs 0.68,
# 0.72, 0.71, 1.00, so none is the clean perpendicular the fingers have), and
# an odd-one-out test on the tip motions splits 0.757 against 0.740 - a
# coin toss. The hardware settles it, and the hardware agrees with the motion
# rather than the name.
ROLES = ("spread", "knuckle", "middle", "tip")
SEGMENT_ROLE = {"adduction": "spread", "proximal": "knuckle",
                "middle": "middle", "distal": "tip"}
THUMB_ROLE = {"proximal": "spread", "adduction": "knuckle",
              "middle": "middle", "distal": "tip"}


def joint_role(joint_name):
    """spread / knuckle / middle / tip for a joint, by what it moves."""
    digit, segment = joint_name.split("_")[0], joint_name.split("_")[1]
    table = THUMB_ROLE if digit == "thumb" else SEGMENT_ROLE
    return table.get(segment, segment)


def roles_in_order(model, finger):
    """That finger's joints as spread, knuckle, middle, tip."""
    by_role = {joint_role(n): n for n in model.finger_chains()[finger]}
    return [(role, by_role[role]) for role in ROLES if role in by_role]


def spread_signs(model, names=None, probe=0.3):
    """Per joint: does a positive angle move that fingertip AWAY from the middle
    finger (+1) or toward it (-1)?

    All three finger adduction joints share one axis, so a positive angle
    swings index, middle and ring the same way through space. That is fine for
    the model and wrong for a person: on the index it means spreading out, on
    the ring it means closing in, and two mechanically identical fingers end up
    with controls that behave oppositely.

    Derived from the geometry rather than written down, so it stays right
    across the mirror and any future URDF. Only joints whose motion is
    dominantly sideways count - the thumb's adduction mostly lifts, so it is
    left alone rather than being given a meaning it does not have.
    """
    names = names or model.joint_names()
    zero = np.zeros(len(names))
    middle_y = model.link_pose(TIP_LINKS["middle"], zero, names)[0][1]

    signs = {}
    for finger in FINGERS:
        base = model.link_pose(TIP_LINKS[finger], zero, names)[0]
        outward = np.sign(base[1] - middle_y)
        for name in model.finger_chains()[finger]:
            q = zero.copy()
            q[names.index(name)] = probe
            delta = model.link_pose(TIP_LINKS[finger], q, names)[0] - base
            sideways = abs(delta[1]) > max(abs(delta[0]), abs(delta[2]))
            if not sideways or outward == 0:
                signs[name] = 1
            else:
                signs[name] = int(np.sign(delta[1]) * outward)
    return signs


# ------------------------------------------------------------------ poses
# Joint angles in radians, keyed by joint NAME - the one place that is still
# fine, because these are hand-written for this hand rather than computed.
# They were checked by role afterwards: `fist` reads thumb spread 1.4, knuckle
# 0.8, middle 0.6, tip 0.6, which is a thumb folding across a closed hand.
# Hand-written starting points, not fitted to anything - a curl is "every
# flexion joint bent the same way".
def _curl(fingers, amount):
    return {f"{f}_{s}_joint": amount
            for f in fingers for s in ("proximal", "middle", "distal")}


POSES = {
    "open": {},
    "fist": {**_curl(("index", "middle", "ring"), 1.4),
             "thumb_proximal_joint": 1.4, "thumb_adduction_joint": 0.8,
             "thumb_middle_joint": 0.6, "thumb_distal_joint": 0.6},
    "point": {**_curl(("middle", "ring"), 1.4),
              "thumb_proximal_joint": 1.2, "thumb_adduction_joint": 0.6},
    "spread": {"index_adduction_joint": 0.35, "ring_adduction_joint": -0.35,
               "thumb_proximal_joint": 1.0},
    "half": {**_curl(FINGERS[:3], 0.7), "thumb_proximal_joint": 0.7,
             "thumb_middle_joint": 0.4, "thumb_distal_joint": 0.4},
}


def pose_vector(pose, names):
    """dict{joint_name: angle} -> full joint vector, missing joints at zero."""
    q = np.zeros(len(names))
    index = {n: i for i, n in enumerate(names)}
    for name, value in pose.items():
        name = name if name.endswith("_joint") else f"{name}_joint"
        if name in index:
            q[index[name]] = value
    return q
