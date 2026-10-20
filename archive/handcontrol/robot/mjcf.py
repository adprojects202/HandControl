"""A MuJoCo copy of the hand, for the mink IK solver.

Two models, both generated from the one URDF: the SIMULATION model that
`sim.py` steps, and a kinematics-only one the IK solves against. They are
separate because the simulation model carries a second, ghost copy of every
joint for the encoder hand, and the solver must not be able to spend those.

MuJoCo reads the URDF directly - joints, ranges, meshes and all - but its
importer fuses fixed joints, which costs exactly the two things the solver
needs:

  * The `<finger>_tip_link` frames disappear. They are fixed children of the
    distal links and they are the whole reason the IK aims true: a distal
    link's own origin sits on its joint axis and does not move when that joint
    rotates. They come back here as MuJoCo SITES, which survive fusing.

  * Geoms arrive unnamed, and `mink.CollisionAvoidanceLimit` is addressed by
    name. They are named after the link they belong to.

So the pipeline is URDF -> MuJoCo's own MJCF export -> these two additions,
cached in build/. Regenerated whenever the URDF is newer.
"""

from __future__ import annotations

import numpy as np

import xml.etree.ElementTree as ET
from pathlib import Path

from .model import BUILD, DEFAULT_HAND, FINGERS, ensure_urdf


def tip_sites(urdf):
    """finger -> (parent body, offset) for each `<finger>_tip_joint`.

    Read off the generated URDF rather than measured again: `ensure_urdf` has
    already put those frames where the fingertips are, and measuring twice is
    two chances to disagree.
    """
    root = ET.parse(urdf).getroot()
    sites = {}
    for joint in root.findall("joint"):
        name = joint.get("name") or ""
        if not name.endswith("_tip_joint"):
            continue
        finger = name[: -len("_tip_joint")]
        origin = joint.find("origin")
        sites[finger] = (joint.find("parent").get("link"),
                         origin.get("xyz") if origin is not None else "0 0 0")
    return sites


def ensure_mjcf(hand=DEFAULT_HAND, spread_limit=None, force=False):
    """The MuJoCo model to solve against. Generated; delete build/ and it returns."""
    import mujoco

    urdf = ensure_urdf(hand, spread_limit)
    target = BUILD / f"{Path(urdf).stem}.mjcf.xml"
    if target.exists() and not force and \
            target.stat().st_mtime >= Path(urdf).stat().st_mtime:
        return target

    model = mujoco.MjModel.from_xml_path(str(urdf))
    exported = target.with_suffix(".raw.xml")
    mujoco.mj_saveLastXML(str(exported), model)

    tree = ET.parse(exported)
    root = tree.getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}

    # `worldbody` as well as every `body`: the palm and the four finger mounts
    # are fixed to the root, so MuJoCo fuses them into the world and their
    # geoms hang directly off worldbody. Skipping it leaves the palm out of
    # every collision pair, which is the one obstacle a fingertip is most
    # likely to be driven into.
    for body in list(root.iter("worldbody")) + list(root.iter("body")):
        for index, geom in enumerate(body.findall("geom")):
            if geom.get("name"):
                continue
            stem = body.get("name") or "palm"
            geom.set("name", stem if index == 0 else f"{stem}_{index}")

    for finger, (parent, offset) in tip_sites(urdf).items():
        body = bodies.get(parent)
        if body is None:
            continue
        site = ET.SubElement(body, "site")
        site.set("name", f"{finger}_tip")
        site.set("pos", offset)
        site.set("size", "0.002")

    target.parent.mkdir(parents=True, exist_ok=True)
    # NOT `target` as the scratch file. This step has to load the tree in
    # MuJoCo to find out which bodies ship overlapping, and writing that
    # exclusion-less tree to the final path left a fresh-dated, VALID-looking
    # cache behind if anything went wrong before the real write - a Ctrl-C
    # during the first build is enough. The next run would read it, skip the
    # rebuild, and produce NaN in QACC within four milliseconds with the hand
    # still at its rest pose, pointing at nothing.
    scratch = target.with_suffix(".probe.xml")
    _exclude_as_built_contacts(root, scratch)

    # Written beside the target and moved into place, so the cache either has
    # the finished model or has nothing at all.
    staged = target.with_suffix(".partial.xml")
    tree.write(staged, encoding="utf-8", xml_declaration=False)
    staged.replace(target)
    scratch.unlink(missing_ok=True)
    exported.unlink(missing_ok=True)
    return target


def _exclude_as_built_contacts(root, scratch):
    """Turn off contact between the body pairs that ship interpenetrating.

    The SolidWorks export overlaps neighbouring links around their shared
    joints - 19 to 25 mm deep between the palm and the finger mounts. MuJoCo
    resolves that on the first step and the simulation diverges: NaN in QACC
    within 4 ms, with the hand still at its rest pose.

    Which pairs are found by asking, at the neutral pose, rather than listed:
    the set depends on the meshes, and the meshes are generated. Everything not
    excluded still collides normally, so a finger driven into its neighbour is
    still a contact - only the overlap the model was born with is ignored.
    """
    import mujoco

    written = ET.ElementTree(root)
    written.write(scratch, encoding="utf-8", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(scratch))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pairs = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        first = model.geom_bodyid[contact.geom1]
        second = model.geom_bodyid[contact.geom2]
        names = tuple(sorted(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            for body in (first, second)))
        if names[0] != names[1]:
            pairs.add(names)

    if not pairs:
        return
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for first, second in sorted(pairs):
        exclude = ET.SubElement(contact, "exclude")
        exclude.set("body1", first)
        exclude.set("body2", second)


GHOST = "ghost_"
GHOST_RGBA = "1.0 0.62 0.28 0.55"

# Extra posed copies of the hand, beyond the encoder ghost. Each is a prefix, a
# colour, and how far to the side it sits. They were built for the replay
# viewer, which has several mappings to show at once and only ever HAD two
# hands (the live lanes and the gallery use them too now) -
# and the second one is spoken for: amber is the real robot's encoders
# everywhere else in this project, so it stays the encoders here too rather
# than being borrowed for whichever comparison is on screen.
#
# Posed by writing qpos, never simulated and never actuated - the same
# treatment the ghost already gets, for the same reason.
TEACHER = "teacher_"
STUDENT = "student_"
LIVE = "live_"
FUTURE = "future_"
TEACHER_RGBA = "0.36 0.78 0.50 1.0"
# Saturated purple. The paler violet it replaces shared its red channel
# with the live lane's grey and sat 0.30 away from it - the two hands a
# viewer most needs to separate.
STUDENT_RGBA = "0.60 0.28 0.95 1.0"
# Slate, and deliberately NOT amber. Amber is the real robot's encoder hand
# everywhere in this project, and lending it to whichever mapping happens to be
# on screen makes the one colour mean two things - so the live solver gets its
# own and GHOST_RGBA stays spoken for even when no hardware is attached.
# Neutral grey, on purpose. It was a blue slate, which sat 0.29 from the
# student's violet in colour distance - the two closest lanes on screen, and
# the two most worth telling apart. Grey competes with nothing.
LIVE_RGBA = "0.66 0.68 0.70 1.0"
# Coral for the student's forward head. It was teal, and teal was a bad
# choice: it carries the same 0.74 green channel the teacher's green does, and
# under this scene's lighting the two lanes read as the same colour. The
# palette has to be separable on screen, not in the hex.
#
# Warm and red-ish because nothing else here is - the skeleton is blue, the
# teacher green, the student violet, the live solver slate, and amber is
# spoken for by the encoders.
FUTURE_RGBA = "0.95 0.36 0.42 1.0"


def _scene(root, floor_z):
    """Lights, a floor and a sky, because a bare URDF import renders black.

    MuJoCo adds a headlight and nothing else to a model that declares no
    visual assets, so the hand arrives unlit against an empty background. None
    of this touches the dynamics - the floor is a visual plane far below the
    hand, with collision off, so it cannot become something to trip over.
    """
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.45 0.45 0.5")
    headlight.set("diffuse", "0.5 0.5 0.5")
    headlight.set("specular", "0.2 0.2 0.2")
    ET.SubElement(visual, "rgba").set("haze", "0.15 0.18 0.24 1")
    # The default offscreen buffer is 640x480, which is smaller than any
    # sensible screenshot and silently refuses one.
    buffer = ET.SubElement(visual, "global")
    buffer.set("offwidth", "1920")
    buffer.set("offheight", "1080")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    sky = ET.SubElement(asset, "texture")
    sky.set("type", "skybox"); sky.set("builtin", "gradient")
    sky.set("rgb1", "0.24 0.30 0.40"); sky.set("rgb2", "0.05 0.06 0.09")
    sky.set("width", "512"); sky.set("height", "512")
    grid = ET.SubElement(asset, "texture")
    grid.set("name", "grid"); grid.set("type", "2d"); grid.set("builtin", "checker")
    grid.set("rgb1", "0.20 0.22 0.27"); grid.set("rgb2", "0.26 0.29 0.35")
    grid.set("width", "512"); grid.set("height", "512")
    material = ET.SubElement(asset, "material")
    material.set("name", "floor"); material.set("texture", "grid")
    material.set("texrepeat", "8 8"); material.set("reflectance", "0.05")

    world = root.find("worldbody")
    floor = ET.SubElement(world, "geom")
    floor.set("name", "floor"); floor.set("type", "plane")
    floor.set("size", "1 1 0.05"); floor.set("material", "floor")
    floor.set("pos", f"0 0 {floor_z}")
    floor.set("contype", "0"); floor.set("conaffinity", "0")
    for index, (px, py) in enumerate(((0.4, -0.4), (-0.4, 0.4))):
        light = ET.SubElement(world, "light")
        light.set("name", f"key{index}")
        light.set("pos", f"{px} {py} {floor_z + 1.2}")
        light.set("dir", "0 0 -1")
        light.set("diffuse", "0.5 0.5 0.5")
        light.set("specular", "0.1 0.1 0.1")


def _prefixed(element, prefix):
    """A deep copy of a body subtree with every name prefixed.

    MuJoCo has no notion of two instances of one model in a scene, so the
    encoder hand is a second copy of the body tree. Mesh assets are global and
    stay shared - only the names that must be unique get the prefix.
    """
    import copy

    clone = copy.deepcopy(element)
    for node in clone.iter():
        name = node.get("name")
        if name is not None:
            node.set("name", prefix + name)
    return clone


def ensure_sim_mjcf(hand=DEFAULT_HAND, spread_limit=None,
                    physics=None, scene=None, force=False):
    """The model the simulator runs: the kinematic one, plus what makes it move.

    Takes the same `Physics` and `Scene` objects `HandSim` holds, rather than
    the dozen values unpacked out of them - see `robot/options.py`. That is
    what keeps a knob from having one spelling here and another there.

    Adds, on top of `ensure_mjcf`:

      * ARMATURE on every joint. Not cosmetic - without it the simulation
        diverges at rest, contacts or no contacts: these links weigh 9 grams
        with inertias around 1e-6, and a position actuator stiff enough to hold
        them is stiff enough to make the integrator explode. It is also the
        honest model, since a 288:1 gearbox reflects far more rotor inertia to
        the joint than the link itself has.
      * A position actuator per joint, force-clipped like the servo.
      * The encoder hand, as a prefixed copy of the body tree with collision
        turned off and no actuators. It is posed by writing qpos, never
        simulated.
    """
    from .model import EFFORT, KP, KV
    from .options import Physics, Scene

    physics = Physics() if physics is None else physics
    scene = Scene() if scene is None else scene
    kp = KP if physics.kp is None else physics.kp
    kv = KV if physics.kv is None else physics.kv
    effort = EFFORT if physics.effort is None else physics.effort
    armature, position = physics.armature, scene.position
    ghost, ghost_offset, main_rgba = scene.ghost, scene.ghost_offset, scene.main_rgba

    base = ensure_mjcf(hand, spread_limit, force=force)
    target = base.with_name(base.name.replace(".mjcf.xml", ".sim.xml"))
    tree = ET.parse(base)
    root = tree.getroot()

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    # Without this the hand shimmers on the spot: a position actuator stiff
    # enough to hold a 9 gram link overshoots every step and reverses, giving a
    # two-step limit cycle - 0.06 degrees of travel, but 0.5 rad/s of joint
    # velocity at 500 Hz, and very visible. `implicitfast` integrates the
    # actuator damping implicitly and takes the resting velocity to exactly
    # zero, with no cost to the rise time.
    option.set("integrator", "implicitfast")

    for joint in root.iter("joint"):
        joint.set("armature", str(armature))

    world = root.find("worldbody")
    x, y, z = position
    for body in world.findall("body"):
        body.set("pos", " ".join(
            str(a + b) for a, b in zip(
                [float(v) for v in (body.get("pos") or "0 0 0").split()],
                (x, y, z))))
    for geom in world.findall("geom"):
        geom.set("pos", " ".join(
            str(a + b) for a, b in zip(
                [float(v) for v in (geom.get("pos") or "0 0 0").split()],
                (x, y, z))))

    # One entry per posed copy: (prefix, colour, sideways offset in metres).
    # `ghost` alone keeps the historical single encoder hand, so every existing
    # caller is unchanged.
    # The encoder hand is a posed copy like any other, so it is APPENDED to
    # whatever else was asked for rather than replaced by it. Written the other
    # way round - `copies` standing in for the ghost - passing comparison lanes
    # silently deleted the amber hand, which is the one lane that shows the
    # real robot.
    copies = list(scene.copies)
    if ghost and not any(prefix == GHOST for prefix, _, _ in copies):
        copies.append((GHOST, GHOST_RGBA, ghost_offset))

    if copies:
        side = 1.0 if hand == "right" else -1.0

        # Snapshot the REAL bodies before adding any of our own. A copy's palm
        # is appended to worldbody, so a list taken afterwards would include it
        # and clone it again - `ghost_ghost_palm`, shifted twice, a second
        # orange palm floating past the first.
        originals = list(world.findall("body"))
        original_geoms = list(world.findall("geom"))

        for prefix, rgba, offset in copies:
            # A scalar is a sideways offset, which is what every lane layout
            # passes. A pair is (forward, sideways), which is what a grid of
            # hands needs - the gallery places sixteen of them.
            forward, sideways = (0.0, float(offset)) if np.isscalar(offset) \
                else (float(offset[0]), float(offset[1]))
            shift = (x + forward, y + side * sideways, z)

            # The palm first, and separately, because it is not a body. MuJoCo
            # fuses the fixed palm and finger mounts into the world, so their
            # geoms hang off `worldbody` directly - cloning only `body`
            # children gave a copy with four fingers and no palm to hold them.
            palm = ET.SubElement(world, "body")
            palm.set("name", prefix + "palm")
            palm.set("pos", " ".join(str(a - b)
                                     for a, b in zip(shift, (x, y, z))))
            for geom in original_geoms:
                clone = _prefixed(geom, prefix)
                clone.set("contype", "0")
                clone.set("conaffinity", "0")
                clone.set("rgba", rgba)
                palm.append(clone)

            for body in originals:
                clone = _prefixed(body, prefix)
                clone.set("pos", " ".join(
                    str(a - b + c) for a, b, c in zip(
                        [float(v) for v in (clone.get("pos") or "0 0 0").split()],
                        (x, y, z), shift)))
                for geom in clone.iter("geom"):
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
                    geom.set("rgba", rgba)
                world.append(clone)

    if main_rgba is not None:
        # Recolour the DRIVEN hand. The replay viewer stands several mappings
        # side by side and names each one by colour, and the driven hand was
        # the only one it could not name - it renders in the mesh's own grey
        # while every posed copy takes a colour. A legend that says "green" of
        # a grey hand is worse than no legend.
        #
        # Applied after the copies are cloned, so it recolours the original and
        # not the copies, which have already taken their own.
        prefixes = tuple(p for p, _, _ in copies)
        for body in world.iter("body"):
            if (body.get("name") or "").startswith(prefixes):
                continue
            for geom in body.findall("geom"):
                geom.set("rgba", main_rgba)
        for geom in world.findall("geom"):
            if not (geom.get("name") or "").startswith(prefixes):
                geom.set("rgba", main_rgba)

    _scene(root, floor_z=0.0)

    actuator = ET.SubElement(root, "actuator")
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name is None or any(name.startswith(p) for p, _, _ in copies):
            continue          # posed copies are written to, never driven
        motor = ET.SubElement(actuator, "position")
        motor.set("name", name)
        motor.set("joint", name)
        motor.set("kp", str(kp))
        motor.set("kv", str(kv))
        motor.set("forcerange", f"{-effort} {effort}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=False)
    return target


def collision_geoms(model):
    """Geom names grouped by digit, for building collision pairs.

    The palm and the finger mounts come back under "palm": they are one rigid
    lump, so a pair drawn inside it can never change distance and is only work.
    """
    import mujoco

    groups = {finger: [] for finger in FINGERS}
    groups["palm"] = []
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if not name:
            continue
        digit = name.split("_")[0]
        groups.setdefault(digit if digit in groups else "palm", []).append(name)
    return groups
