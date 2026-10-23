"""The task sim: the robot hand with motors, a free object, gravity - the
place a task is tried before the real hand.

    task = Task.build(goal_first_pose)     # from settings.json
    task.model, task.data                  # MuJoCo, ready to step
    task.object_pose(data)                 # (position, rotation) in the mount frame
    task.self_overlap(data)                # (m, link pair): fingers into each other

The display world (src/world.py) poses its hands; here the hand is
driven - each joint by a position servo - and the object is free, so what
the fingers do to it is physics. It is built from the same settings as the
world - the same robot, mount and orientation (so the same gravity), the
calibration object's size, mass and friction.

The hand collides with itself, as the real one does: a finger cannot pass
through another finger, the palm or itself. Every link collides as its CAD
mesh's convex hull (MuJoCo's mesh collision). MuJoCo already ignores a link
against the link it hangs from - but not when that one is fixed to the world,
as the palm and the finger bases are here; so each such pair that touches at
rest (a finger's first link against the base and palm it is mounted in) is
excluded, as MuJoCo would have. So are neighbouring fingers' proximal and
middle links, whose hulls already overlap at rest by up to 1 mm (the servos
sit almost touching): colliding, they would push the fingers apart at rest.
A finger still cannot pass through its neighbour - their other links (the
fingertips, the side-swing links, each against the neighbour's next link)
still collide; measured, forcing the fingers into each other leaves no pair
more than ~1 mm deeper than at rest. (A contact gap or a negative pair margin
would let those 4 pairs collide past their rest overlap; MuJoCo 3.12 ignores
gap, and with mesh hulls a negative margin makes contact start sooner, not
later - both measured.) All of it is found from the CAD at build time
(`_self_collision`), not listed by hand.

The servos are XL330-M288s as their spec sheet gives them, not as measured
on this hand: 0.52 N m stall torque at 1.47 A, so `hardware.current_limit_ma`
caps each joint's torque (300 mA: 0.11 N m); a position loop that reaches
that cap SATURATE_RAD from its target; reflected rotor inertia (armature),
damping and gear friction of a small geared servo. Good enough to ask whether
a hand of this shape and strength can do a task; not a model of this one
hand's wiring and drag.

The object starts held: a weld (equality) pins it to where the task starts
while the fingers close on it; `release(data)` lets it go.

Each finger's last link has a distance sensor to the object (`tips`, in that
order at the start of sensordata): the gap between them, negative when they
press into each other, GAP_CUTOFF at most.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .. import robot as robot_module, settings
from ..world import FLOOR_BIT, HAND_BIT, _compose, _scene, mount_quat

STALL_TORQUE_NM, STALL_CURRENT_A = 0.52, 1.47     # XL330-M288, 5 V
SATURATE_RAD = 0.2       # position error at which the servo gives its full torque
ARMATURE = 5e-4          # kg m^2: rotor inertia times the 288:1 gear, squared
DAMPING_RATIO = 0.6
GEAR_FRICTION_NM = 0.005
TIMESTEP = 0.002
OBJECT = "object"
WELD = "hold"
GAP_CUTOFF = 0.1         # m


@dataclass
class Task:
    model: mujoco.MjModel
    data: mujoco.MjData
    mount: tuple                 # (position, quaternion) of the hand mount, world
    object_qpos: int             # where the object's free joint sits in qpos
    max_torque: float            # N m, per joint
    tips: tuple                  # each finger's last link, as its gap sensor's order
    link_pairs: tuple            # hand geom pairs that can meet, and their overlap at rest
    spec: mujoco.MjSpec          # what it was compiled from (spec.to_xml(): the model as MJCF)

    @classmethod
    def build(cls, start):
        """The task with the object at `start` - (position m, quaternion w x y
        z) in the hand mount's frame - held there until released."""
        robot = robot_module.load()
        world_config, obj = settings.section("world"), settings.section("object")
        mount = (np.array(world_config["hand_position"], float),
                 mount_quat(world_config["hand_orientation"]))
        max_torque = (STALL_TORQUE_NM * settings.get("hardware.current_limit_ma") / 1000
                      / STALL_CURRENT_A)
        spec = mujoco.MjSpec()
        spec.modelname = "task"
        spec.option.timestep = TIMESTEP
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        _scene(spec)
        _add_hand(spec, robot, mount, max_torque)
        position, quat = _compose(mount, *start)
        _add_object(spec, obj, position, quat)
        tips = _fingertips(robot)
        for tip in tips:
            spec.add_sensor(name=f"{tip}_gap", type=mujoco.mjtSensor.mjSENS_GEOMDIST,
                            objtype=mujoco.mjtObj.mjOBJ_BODY, objname=tip,
                            reftype=mujoco.mjtObj.mjOBJ_GEOM, refname=OBJECT,
                            cutoff=GAP_CUTOFF)
        _self_collision(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        joint = model.joint(OBJECT)
        return cls(model, data, mount, int(joint.qposadr[0]), max_torque, tips,
                   _link_pairs(model, data), spec)

    def reset(self):
        """Back to the start: the object held where it starts, the hand at
        zero, time 0 - the weld on again, in the model and the data."""
        self.model.eq_active0[self.model.equality(WELD).id] = 1
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def release(self, data):
        """Let go of the object: the weld off, in `data` and in the model -
        MuJoCo's batched rollouts restore positions and velocities but take
        whether a constraint is on from the model."""
        weld = self.model.equality(WELD).id
        data.eq_active[weld] = 0
        self.model.eq_active0[weld] = 0

    def self_overlap(self, data):
        """How far (m) any two of the hand's links are inside each other
        beyond how far they are at rest - every pair that can meet, the
        excluded neighbours too - and which pair."""
        worst, which = 0.0, None
        for first, second, rest in self.link_pairs:
            extra = -mujoco.mj_geomDistance(self.model, data, first, second, 0.02, None) - rest
            if extra > worst:
                worst, which = extra, (self.model.geom(first).name, self.model.geom(second).name)
        return worst, which

    def object_pose(self, qpos):
        """The object's (position, rotation) in the mount frame, from a qpos."""
        position = qpos[self.object_qpos:self.object_qpos + 3]
        quat = qpos[self.object_qpos + 3:self.object_qpos + 7]
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, self.mount[1])
        local, rotated = np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(local, position - self.mount[0], inverse)
        mujoco.mju_mulQuat(rotated, inverse, quat)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, rotated)
        return local, rotation.reshape(3, 3)


def _fingertips(robot):
    """Each joint group's last link: the body its deepest joint moves."""
    model = robot.spec().compile()

    def depth(body):
        count = 0
        while body:
            body, count = int(model.body_parentid[body]), count + 1
        return count

    tips = []
    for indices in robot.groups().values():
        bodies = [int(model.jnt_bodyid[model.joint(robot.names[i]).id]) for i in indices]
        tips.append(model.body(max(bodies, key=depth)).name)
    return tuple(tips)


def _add_hand(spec, robot, mount, max_torque):
    """The robot, fixed on the mount, a position servo on every joint. Its
    geoms touch the object and each other, and are named after their links
    (`_self_collision` refers to them)."""
    hand = robot.spec()
    for index, geom in enumerate(hand.geoms):
        geom.contype = geom.conaffinity = HAND_BIT
        geom.name = f"{geom.parent.name}_{index}"
        # As stiff as the object's contacts (and MuJoCo allows at this
        # timestep): at its default 0.02 s, a link pressed by a servo sank
        # 4-7 mm into another.
        geom.solref = [2.0 * TIMESTEP, 1.0]
    frame = spec.worldbody.add_frame(pos=mount[0], quat=mount[1])
    child = hand.worldbody.first_body()
    while child is not None:
        following = hand.worldbody.next_body(child)
        frame.attach_body(child, "", "")
        child = following
    kp = max_torque / SATURATE_RAD
    kv = 2 * DAMPING_RATIO * np.sqrt(kp * ARMATURE)
    for joint, low, high in zip(robot.names, robot.lower, robot.upper):
        spec.joint(joint).armature = ARMATURE
        spec.joint(joint).frictionloss = GEAR_FRICTION_NM
        servo = spec.add_actuator(name=joint, target=joint,
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
        servo.set_to_position(kp=kp, kv=kv)
        servo.ctrlrange = [low, high]
        servo.ctrllimited = True
        servo.forcerange = [-max_torque, max_torque]
        servo.forcelimited = True


def _self_collision(spec):
    """Exclusions for the hand's links that touch at rest (see the module
    docstring), from the model compiled as it stands."""
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)                 # every joint at zero
    hand = [g for g in range(model.ngeom) if model.geom_contype[g] == HAND_BIT
            and model.geom_conaffinity[g] == HAND_BIT]
    weld = model.body_weldid
    excluded = set()
    for i, first in enumerate(hand):
        for second in hand[i + 1:]:
            one, two = (int(model.geom_bodyid[g]) for g in (first, second))
            if weld[one] == weld[two] or _filtered(model, one, two):
                continue
            overlap = -mujoco.mj_geomDistance(model, data, first, second, 0.01, None)
            if overlap <= 0:
                continue
            bodies = tuple(sorted((model.body(one).name, model.body(two).name)))
            if bodies not in excluded:
                spec.add_exclude(bodyname1=bodies[0], bodyname2=bodies[1])
                excluded.add(bodies)


def _mounted(model, one, two):
    """Whether one body is fixed to the world and the other hangs from it."""
    weld, parent = model.body_weldid, model.body_parentid
    return (weld[one] == 0 and weld[parent[two]] == 0) or (weld[two] == 0 and weld[parent[one]] == 0)


def _link_pairs(model, data):
    """((geom, geom, overlap at rest m), ...) for every two hand links that
    can meet: not the same rigid part, not parent and child, not mounted."""
    hand = [g for g in range(model.ngeom) if model.geom_contype[g] == HAND_BIT
            and model.geom_conaffinity[g] == HAND_BIT]
    pairs = []
    for i, first in enumerate(hand):
        for second in hand[i + 1:]:
            one, two = (int(model.geom_bodyid[g]) for g in (first, second))
            if model.body_weldid[one] == model.body_weldid[two] or \
                    _filtered(model, one, two) or _mounted(model, one, two):
                continue
            rest = min(0.0, mujoco.mj_geomDistance(model, data, first, second, 0.02, None))
            pairs.append((first, second, -rest))
    return tuple(pairs)


def _filtered(model, one, two):
    """Whether MuJoCo already leaves two bodies out of collisions as parent
    and child (its filterparent rule, which spares bodies fixed to the world)."""
    weld = model.body_weldid
    first, second = weld[one], weld[two]
    if not first or not second:
        return False
    return first == weld[model.body_parentid[second]] or second == weld[model.body_parentid[first]]


def _add_object(spec, obj, position, quat):
    """The calibration object, free, held by a weld to where it starts."""
    body = spec.worldbody.add_body(name=OBJECT, pos=position, quat=quat)
    body.add_freejoint(name=OBJECT)
    body.add_geom(name=OBJECT, type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=np.array(obj["size_mm"]) / 2000.0, mass=obj["mass_g"] / 1000.0,
                  rgba=obj["rgba"], contype=FLOOR_BIT, conaffinity=FLOOR_BIT | HAND_BIT,
                  priority=1, condim=4, friction=[obj["friction"], 0.005, 0.0001],
                  solref=[2.0 * TIMESTEP, 1.0])
    anchor = spec.worldbody.add_body(name="anchor", mocap=True, pos=position, quat=quat)
    weld = spec.add_equality(name=WELD, type=mujoco.mjtEq.mjEQ_WELD, name1=OBJECT,
                             name2="anchor", objtype=mujoco.mjtObj.mjOBJ_BODY)
    # anchor (3), relative position (3), relative quaternion (4), torque scale:
    # the two coincide. MjSpec's default put the anchor 1 m off along y.
    weld.data[:11] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    del anchor
