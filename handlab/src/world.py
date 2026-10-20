"""THE simulation world. There is one, and it is built the same way every time.

Whoever opens the simulator - main.py, a script, a test - calls `World()` and
gets the same scene from the same settings.json. Nothing about what is in the
world is a parameter.

Contents:

    encoder hand   opaque white; the real robot, posed from its encoders
    command hand   translucent purple; the joint command sent to the robot
    object         the calibration box, when `object.enabled`

Both hands stand on one mount, turned by `world.hand_orientation`. They are
KINEMATIC: posed by writing their joint angles, never driven by physics, and
gravity-compensated. One opaque, one translucent: see the colour settings for
why.

LIVE EDITS. `apply()` takes the world and object settings into the running
model in place - the mount (each hand's base is a mocap body, so its pose is
simulation state), the object's placement, mode, weight and friction. Showing or hiding
the object, or changing its size, needs `build()`, which the viewer shows as a
brief reload: Studio's renderer reads geometry only when a model loads, and in
MuJoCo 3.12 it crashes if geoms it has drawn are hidden by group.

Who touches whom - MuJoCo collides two geoms when either one's contype shares
a bit with the other's conaffinity:

    object   contype 1, conaffinity 3    hits the floor and the real hand
    floor    contype 0, conaffinity 1    stops the object only
    encoder  contype 2, conaffinity 0    pushes the object only - never itself
    command  contype 0, conaffinity 0    a ghost: touches nothing

FRICTION AND STIFFNESS. MuJoCo gives a contact the larger friction of its two
geoms - so with the hand at MuJoCo's default 1.0, a lower object value would
be ignored. The object's geom has a higher `priority`, which makes its
friction and contact stiffness the ones every contact it makes uses:
`object.friction` is the friction of the object against the hand and the
floor. Its contacts are as stiff as MuJoCo allows at this timestep (solref
time constant = 2 x timestep, MuJoCo's stability limit). Measured, a 20 g box
resting on the palm: 0.08 mm overlap and 0.8 mm/s creep, against 0.89 mm and
3.2 mm/s at MuJoCo's default 0.02 s.

The real hand's links overlap each other where the CAD export joins them, so
it must not collide with itself. And it is replayed, not simulated - but
MuJoCo contacts are soft constraints that correct penetration at a rate set
by their solref, whatever the masses, so contact with the object does nudge
the replayed fingers during a physics step (0.3 deg in a step with the
fingers 22 mm deep in a fixed box). `hold()` writes every replayed pose back
after each physics advance, so nothing shown or read ever carries that.

Each hand and the object have their own geom group (GROUPS). Do not hide
them by group while the window is open - see LIVE EDITS.
"""

from __future__ import annotations

import mujoco
import numpy as np

from . import robot as robot_module
from . import settings

HANDS = ("encoder", "command")
# Geom groups. MuJoCo draws groups 0-2 by default and hides 3-5, so
# everything that should be seen stays in 0-2.
GROUPS = {"object": 0, "encoder": 1, "command": 2}
OBJECT = "object"

# The hand mount's rotation (x, y, z Euler, degrees) for each orientation. In
# the hand's own frame the palm faces +x, the fingers run along +z and the
# thumb sticks out along -y; gravity is always -z in the world.
ORIENTATIONS = {
    "upright": (0, 0, 0),           # fingers up, palm forward (+x)
    "palm up": (0, -90, 0),         # palm to the sky, fingers to -x
    "palm down": (0, 90, 0),        # palm to the floor, fingers to +x
    "fingers down": (180, 0, 0),    # hanging, palm still forward
    "thumb up": (-90, 0, 0),        # thumb to the sky, fingers to +y
    "thumb down": (90, 0, 0),       # thumb to the floor, fingers to -y
}
assert tuple(ORIENTATIONS) == settings.CHOICES["world.hand_orientation"]

# Collision bits; see the table in the module docstring.
FLOOR_BIT, HAND_BIT = 1, 2


class World:
    def __init__(self, robot=None):
        self.robot = robot or robot_module.load()
        self.build()

    # ------------------------------------------------------------ building
    def build(self):
        """Make the model from settings.json. Hands start at zero; the object
        starts where it is placed."""
        obj = settings.section("object")
        self.size_mm, self.has_object = tuple(obj["size_mm"]), obj["enabled"]
        self.spec, self._bases = _build(self.robot, settings.section("world"), obj)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._poses = {}                # the last pose written to each hand
        self._qadr, self._vadr, self._mocap = {}, {}, {}
        for hand in HANDS:
            ids = [self.model.joint(f"{hand}/{name}").id for name in self.robot.names]
            self._qadr[hand] = self.model.jnt_qposadr[ids]
            self._vadr[hand] = self.model.jnt_dofadr[ids]
            self._mocap[hand] = [self.model.body_mocapid[self.model.body(name).id]
                                 for name in self._bases[hand]]
        if self.has_object:
            joint = self.model.joint(OBJECT)
            self._object_qadr, self._object_vadr = joint.qposadr[0], joint.dofadr[0]
            self._object_body = self.model.body(OBJECT).id
            self._object_geom = self.model.geom(OBJECT).id
        self.object, self.placed = None, None
        return self.apply()

    def apply(self):
        """Take the world and object settings into the running model. In
        place, except showing, hiding or resizing the object, which rebuild."""
        if settings.get("object.enabled") != self.has_object or \
                (self.has_object and tuple(settings.get("object.size_mm")) != self.size_mm):
            return self.build()
        previous = (self.object, self.placed)
        self.config, self.object = settings.section("world"), settings.section("object")
        self.mount = (np.array(self.config["hand_position"]),
                      mount_quat(self.config["hand_orientation"]))
        self.placed = _compose(self.mount, np.array(self.object["position_mm"]) / 1000.0,
                               np.array([1.0, 0.0, 0.0, 0.0]))

        if self.has_object:
            self._set_mass(self.object["mass_g"] / 1000.0)
            # Sliding friction; MuJoCo reads it whenever it makes a contact.
            self.model.geom_friction[self._object_geom, 0] = self.object["friction"]
            old, old_place = previous
            if old is None or old["mode"] != self.object["mode"] \
                    or any(not np.allclose(a, b) for a, b in zip(old_place, self.placed)):
                self.reset_object()
        self.hold()
        return self

    def reset_object(self):
        """Put the object back where it is placed, at rest."""
        if not self.has_object:
            return
        position, quat = self.placed
        self.data.qpos[self._object_qadr:self._object_qadr + 7] = np.r_[position, quat]
        self.data.qvel[self._object_vadr:self._object_vadr + 6] = 0.0

    def _set_mass(self, mass):
        """New mass, and a solid box's inertia to match. mj_setConst then
        refreshes what MuJoCo derives from them - around a saved copy of the
        state, so nothing that is moving gets reset."""
        if self.model.body_mass[self._object_body] == mass:
            return
        x, y, z = np.array(self.size_mm) / 1000.0
        self.model.body_mass[self._object_body] = mass
        self.model.body_inertia[self._object_body] = mass / 12.0 * np.array(
            [y * y + z * z, x * x + z * z, x * x + y * y])
        kind = mujoco.mjtState.mjSTATE_FULLPHYSICS
        state = np.empty(mujoco.mj_stateSize(self.model, kind))
        mujoco.mj_getState(self.model, self.data, state, kind)
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state, kind)

    # --------------------------------------------------------------- poses
    def set_pose(self, hand, q):
        """Pose one hand. `q` is in URDF joint order, radians."""
        self._poses[hand] = self.robot.clamp(q)
        self.data.qpos[self._qadr[hand]] = self._poses[hand]
        self.data.qvel[self._vadr[hand]] = 0.0

    def hold(self):
        """Write every replayed pose back - the hands' joints and mount, and a
        fixed object - after physics has stepped, so contact cannot leave any
        of them anywhere but where it was put."""
        for hand, q in self._poses.items():
            self.data.qpos[self._qadr[hand]] = q
            self.data.qvel[self._vadr[hand]] = 0.0
        for hand in HANDS:
            for mocap, (local_pos, local_quat) in zip(self._mocap[hand],
                                                      self._bases[hand].values()):
                self.data.mocap_pos[mocap], self.data.mocap_quat[mocap] = \
                    _compose(self.mount, local_pos, local_quat)
        if self.object["mode"] == "fixed":
            self.reset_object()
        mujoco.mj_kinematics(self.model, self.data)

    def pose(self, hand):
        return np.array(self.data.qpos[self._qadr[hand]])

    def object_position(self):
        """The object's centre in world metres, or None without one."""
        if not self.has_object:
            return None
        return np.array(self.data.xpos[self._object_body])

    def geom_ids(self, hand):
        """Ids of every geom belonging to one hand."""
        return [i for i in range(self.model.ngeom)
                if self.model.body(self.model.geom_bodyid[i]).name.startswith(f"{hand}/")]

    def hand_center(self):
        return hand_center(self.model, self.data)


def hand_center(model, data):
    """Middle of the mounted hand, world metres - what a camera looks at.
    Takes the model and data so the viewer can ask about its own copy."""
    bodies = [i for i in range(model.nbody) if model.body(i).name.startswith("encoder/")]
    return np.asarray(data.xipos[bodies]).mean(axis=0)


def mount_quat(orientation):
    """The hand mount's rotation as a MuJoCo quaternion (w, x, y, z)."""
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.radians(ORIENTATIONS[orientation]), "xyz")
    return quat


def _compose(mount, local_pos, local_quat):
    """A pose given in the mount's frame, in world coordinates."""
    mount_pos, mount_q = mount
    position, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_rotVecQuat(position, np.asarray(local_pos, dtype=float), mount_q)
    mujoco.mju_mulQuat(quat, mount_q, np.asarray(local_quat, dtype=float))
    return mount_pos + position, quat


# --------------------------------------------------------------------- build
def _build(robot, config, obj):
    """The spec, and each hand's base bodies' poses in the mount frame."""
    world = mujoco.MjSpec()
    world.modelname = "handlab"
    world.option.timestep = config["timestep"]
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    _scene(world)
    bases = {hand: _add_hand(world, robot, hand, rgba=config[f"{hand}_rgba"],
                             emission=config[f"{hand}_emission"])
             for hand in HANDS}
    if obj["enabled"]:
        _add_object(world, obj, config["timestep"])
    return world, bases


def _add_hand(world, robot, hand, rgba, emission):
    """One kinematic copy of the robot, every name prefixed with `hand/`.
    Its base bodies become mocap bodies - MuJoCo's kinematic bodies - so the
    mount is a pose in the simulation state, set by `World.hold`. Returns
    {base body name: (position, quaternion) in the mount frame}."""
    spec = robot.spec()
    # Defined on the hand, not the world, so it is prefixed along with it.
    spec.add_material(name="skin", rgba=rgba, emission=emission)
    touches = hand == "encoder"     # only the real hand handles the object
    for geom in spec.geoms:
        geom.material = "skin"
        geom.rgba = rgba
        geom.group = GROUPS[hand]
        geom.contype = HAND_BIT if touches else 0
        geom.conaffinity = 0
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    bases = {}
    frame = world.worldbody.add_frame()
    for body in list(_children(spec.worldbody)):
        bases[f"{hand}/{body.name}"] = (np.array(body.pos), np.array(body.quat))
        frame.attach_body(body, f"{hand}/", "")
    for name in bases:
        world.body(name).mocap = True
    return bases


def _add_object(world, obj, timestep):
    """The calibration box: a free body, so it can be fixed or floating
    without a rebuild (fixed = held by `World.hold`). Placed and weighed by
    `World.apply`."""
    body = world.worldbody.add_body(name=OBJECT)
    body.add_freejoint(name=OBJECT)
    body.add_geom(name=OBJECT, type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=np.array(obj["size_mm"]) / 2000.0,      # half-lengths, m
                  mass=obj["mass_g"] / 1000.0, rgba=obj["rgba"],
                  group=GROUPS["object"], contype=FLOOR_BIT,
                  conaffinity=FLOOR_BIT | HAND_BIT,
                  # its friction and stiffness win in every contact it makes
                  priority=1, friction=[obj["friction"], 0.005, 0.0001],
                  solref=[2.0 * timestep, 1.0])


def _children(body):
    """Direct child bodies of a spec body."""
    child = body.first_body()
    while child is not None:
        yield child
        child = body.next_body(child)


def _scene(world):
    """Light, sky and a floor. The floor stops the object and nothing else."""
    visual = world.visual
    visual.headlight.ambient = [0.45, 0.45, 0.5]
    visual.headlight.diffuse = [0.5, 0.5, 0.5]
    visual.headlight.specular = [0.2, 0.2, 0.2]
    visual.global_.offwidth = 1920
    visual.global_.offheight = 1080

    world.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=[0.24, 0.30, 0.40], rgb2=[0.05, 0.06, 0.09],
                      width=512, height=512)
    world.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                      rgb1=[0.20, 0.22, 0.27], rgb2=[0.26, 0.29, 0.35],
                      width=512, height=512)
    floor = world.add_material(name="floor", texrepeat=[8, 8], reflectance=0.05)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[1, 1, 0.05], material="floor",
                             contype=0, conaffinity=FLOOR_BIT)
    for index, (x, y) in enumerate(((0.4, -0.4), (-0.4, 0.4))):
        world.worldbody.add_light(name=f"key{index}", pos=[x, y, 1.4],
                                  dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5],
                                  specular=[0.1, 0.1, 0.1])
