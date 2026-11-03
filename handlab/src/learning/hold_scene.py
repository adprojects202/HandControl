"""The scene the physics policy (step 4) holds objects in: the task sim's
hand (sim/task.py hand_spec - its servos, self-collision and mount) and one
free object that can be any of many. Every object's mesh is in the model;
which one the object is - its variant - is set per world: in MuJoCo Warp's
batched model fields (geom_dataid points the object's geom at a mesh, as
mjlab's heterogeneous worlds do), or in a CPU model in place (`use`).

    spec, parts = build({name: points}, friction)   # Mac: the scene, to MJCF
    fields = variant_fields(spec, model)          # either side, from the spec
    use(model, data, fields, variant, mass)       # a CPU model, one object

A variant's fields are those of the object compiled alone from its own
mesh (as mjlab checks against a full compile): where its geom sits on the
body (MuJoCo moves a mesh to its centre of mass), its bounds, and its
inertia at DENSITY - which `use` and the GPU scale to the mass drawn.
Meshes are given as points; MuJoCo collides (and weighs) their convex hull.

Each of the hand's bodies has a distance sensor to the object (in the
order of `parts`, at the start of sensordata): the gap, SENSOR_CUTOFF at
most; under CONTACT_M it touches. `parts` names each body's finger - the
palm (bodies no joint moves) or a joint group.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..sim.task import TIMESTEP, hand_spec
from ..world import FLOOR_BIT, HAND_BIT

OBJECT = "object"
PREFIX = "object_"               # the objects' meshes' names (the robot has its own)
ANCHOR = "anchor"
WELD = "hold"
DENSITY = 1000.0                 # kg/m^3: only the inertia's shape; the mass is drawn
SENSOR_CUTOFF = 0.01             # m
CONTACT_M = 0.003
FINGERS = ("palm", "thumb", "index", "middle", "ring")
GEOM_FIELDS = ("geom_dataid", "geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb")
BODY_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_subtreemass")


def build(meshes, friction):
    """The scene with every mesh of `meshes` ({name: points (V, 3) m, in
    the object's frame}), the object as the first; the object's sliding
    friction `friction`. Returns (spec, parts): parts the finger of each
    gap sensor's body, in FINGERS."""
    spec, robot, _, _ = hand_spec()
    for name, points in meshes.items():
        spec.add_mesh(name=PREFIX + name, uservert=np.asarray(points, float).ravel())
    body = spec.worldbody.add_body(name=OBJECT, pos=[0, 0, 0.5])
    body.add_freejoint(name=OBJECT)
    body.add_geom(name=OBJECT, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=PREFIX + next(iter(meshes)),
                  density=DENSITY, rgba=[0.85, 0.55, 0.3, 1], contype=FLOOR_BIT,
                  conaffinity=FLOOR_BIT | HAND_BIT, priority=1, condim=4,
                  friction=[friction, 0.005, 0.0001], solref=[2.0 * TIMESTEP, 1.0])
    spec.worldbody.add_body(name=ANCHOR, mocap=True, pos=[0, 0, 0.5])
    weld = spec.add_equality(name=WELD, type=mujoco.mjtEq.mjEQ_WELD, name1=OBJECT, name2=ANCHOR,
                             objtype=mujoco.mjtObj.mjOBJ_BODY)
    weld.data[:11] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]    # as task._add_object: they coincide
    parts = []
    for name, finger in _hand_bodies(spec, robot):
        spec.add_sensor(name=f"{name}_gap", type=mujoco.mjtSensor.mjSENS_GEOMDIST,
                        objtype=mujoco.mjtObj.mjOBJ_BODY, objname=name,
                        reftype=mujoco.mjtObj.mjOBJ_GEOM, refname=OBJECT, cutoff=SENSOR_CUTOFF)
        parts.append(finger)
    return spec, parts


def _hand_bodies(spec, robot):
    """[(body name, finger)] for every body with a hand geom: the joint
    group of the joint that moves it, or the palm."""
    model = spec.compile()
    group_of = {name: joint.group for name, joint in zip(robot.names, robot.joints)}
    out = []
    for body in range(1, model.nbody):
        geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
        if not any(model.geom_contype[g] == HAND_BIT for g in geoms):
            continue
        finger, at = "palm", body
        while at and finger == "palm":
            joints = range(model.body_jntadr[at], model.body_jntadr[at] + model.body_jntnum[at])
            names = [model.joint(j).name for j in joints if model.joint(j).name in group_of]
            finger = group_of[names[0]] if names else "palm"
            at = int(model.body_parentid[at])
        if finger not in FINGERS:
            raise RuntimeError(f"the robot's joint group {finger!r} is not one of {FINGERS}")
        out.append((model.body(body).name, finger))
    return out


def variant_fields(spec, model):
    """{field: (variants, ...)} for GEOM_FIELDS and BODY_FIELDS - the
    object's geom's and body's, each object mesh of `spec` in turn the
    object - and "names" (variants,): the objects as `build` was given
    them, in its order. `model`: `spec` compiled, for its mesh ids."""
    names, rows, ids = [], [], []
    for mesh in spec.meshes:
        if not mesh.name.startswith(PREFIX):
            continue
        alone = mujoco.MjSpec()
        alone.add_mesh(name=mesh.name, uservert=mesh.uservert)
        body = alone.worldbody.add_body(name=OBJECT)
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, density=DENSITY)
        single = alone.compile()
        row = {field: getattr(single, field)[0].copy() for field in GEOM_FIELDS}
        row.update({field: getattr(single, field)[1].copy() for field in BODY_FIELDS})
        names.append(mesh.name[len(PREFIX):])
        ids.append(model.mesh(mesh.name).id)
        rows.append(row)
    fields = {field: np.stack([row[field] for row in rows]) for field in GEOM_FIELDS + BODY_FIELDS}
    fields["geom_dataid"] = np.array(ids)              # the mesh in the scene, not alone
    fields["names"] = np.array(names)
    return fields


def scaled(fields, variants, masses):
    """The object's GEOM_FIELDS and BODY_FIELDS rows for `variants` (n,)
    at `masses` (n,) kg: the inertia scaled with the mass."""
    out = {field: fields[field][variants] for field in GEOM_FIELDS + BODY_FIELDS}
    ratio = masses / fields["body_mass"][variants]
    out["body_mass"] = masses.astype(fields["body_mass"].dtype)
    out["body_subtreemass"] = masses.astype(fields["body_subtreemass"].dtype)
    out["body_inertia"] = fields["body_inertia"][variants] * ratio[:, None]
    return out


def use(model, data, fields, variant, mass):
    """Make the object of a CPU model variant `variant`, `mass` kg."""
    geom, body = model.geom(OBJECT).id, model.body(OBJECT).id
    rows = scaled(fields, np.array([variant]), np.array([mass], float))
    for field in GEOM_FIELDS:
        getattr(model, field)[geom] = rows[field][0]
    for field in BODY_FIELDS:
        getattr(model, field)[body] = rows[field][0]
    mujoco.mj_setConst(model, data)

