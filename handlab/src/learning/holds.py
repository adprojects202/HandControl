"""Holds (step 4's data): GRAB's people holding an object in one hand, put
onto the robot hand with the object in it - what the physics policy
(gpu_holds.py) learns to keep hold of, and is tested on.

    python scripts/build_holds.py        # build(): ~4 min
    holds = load()

A hold is a span of a take where one hand holds the object alone: its
thumb and another part on it (GRAB's contact labels; gaps of up to GAP
frames filled), the other hand off it, and the object's lowest point
ABOVE_TABLE_M over the table - held up, not resting on it. Spans of
MIN_HOLD_S or more are kept.

Onto the robot: the object goes where it is in the person's palm frame
(handmotion.palm_frame) into the robot's palm frame, the person's metres
times the robot's palm length over theirs (each person's median), its mesh
scaled alike - each object, person and side a variant of the scene
(hold_scene.py). A right hand is mirrored onto the robot's left as in_palm
mirrors it, and so its object. Gravity stays where it was in the person's
palm frame, frame by frame: as their hand turned, the robot's palm turns
under it. The robot's motion is the take's retargeted one (grab.py); what
the camera gives of it, the reader's (reader.py) on the tracker's errors
(tracker_noise.py, from a camera of grab.views), DRAWS times.

Episodes start every START_EVERY_S along a hold with MIN_HOLD_S left, and
last EPISODE_S at most. Each start is made on the CPU: the hand at the
retargeted pose, the object pinned where it is; each finger GRAB has on the
object closed until it meets it (along the gap's gradient, the sim's own
distances) and pressed SQUEEZE_RAD on; SETTLE_S of physics. The state then,
the object still pinned, is the episode's start; `reachable` if the thumb
and another finger met the object. Placed by the person's grasp, the
robot's fingers - other lengths, other proportions - do not reach it by
themselves: with the retargeted pose as it is, nothing touches.

Let go, the object falls from nearly all of them: the fingers press, but
not against each other - the pin held what they pushed. Measured, CHECK_S
on with the servos on the retargeted motion (`drift_mm`), and on it plus
the closing (`grip_drift_mm`): how far the object went. A grip that holds
is what the physics policy has to find (gpu_holds.py).

    data/learning/intent/holds.npz
        per frame (F) of the takes held in: joints (F, J) rad, retargeted;
        position (F, 3) m and quat (F, 4), where the object is to be in the
        scene; gravity (F, 3) m/s^2 in the scene; fingers (F, 5) bool,
        GRAB's contacts by hold_scene.FINGERS; read (DRAWS, F, J) rad and
        meant (DRAWS, F, reader.INTENTS) float16, the reader's
        per episode (E): first (frame), length, variant, subject, test (one
        of the people never fitted on), qpos (E, nq), qvel (E, nv), grip
        (E, J) the closing (servo order), reachable, drift_mm,
        grip_drift_mm
        variant_names (V,), meshes (scene.xml's), scene (MJCF), parts,
        joint_names (the frames' joints, in order), rate, made
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from multiprocessing import get_context

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .. import robot as robot_module, settings
from . import grab, hold_scene
from . import motion_space as space_module
from .hand_library import robot_chain
from .handmotion import RATE, in_palm, palm_frame
from .intent_pairs import FOLDER, WORKERS

HOLDS = FOLDER / "holds.npz"
GAP = 2                          # frames of a hold whose contact labels drop out
ABOVE_TABLE_M = 0.005
MIN_HOLD_S = 1.0
EPISODE_S = 8.0
START_EVERY_S = 0.5
SETTLE_S = 0.3
CHECK_S = 1.0
DRAWS = 3
MASS_KG = (0.02, 0.10)           # drawn per world; settled at the middle
REACH_M = 0.05                   # a finger further from the object than this is not closed
TOUCH_M = 0.0005
MAX_STEP_RAD = 0.1
SQUEEZE_RAD = 0.2                # past touching: the servo presses with all its torque
GRAVITY = np.array([0.0, 0.0, -9.81])        # GRAB's z is up
MIRROR = np.diag([1.0, -1.0, 1.0])           # in_palm's: y -> -y
TEST = space_module.GRAB_PEOPLE["grab held out"]


def build(progress=lambda done, total, text="": None):
    """Every hold of GRAB onto the robot; writes holds.npz, returns a
    summary."""
    from . import reader as reader_module
    from .motion_space import MotionSpace
    from .tracker_noise import TrackerNoise

    hands, labels = dict(np.load(grab.HANDS)), grab.load()
    views, meshes = dict(np.load(grab.VIEWS)), np.load(grab.MESHES)
    if not labels:
        raise RuntimeError("no GRAB robot motion - run scripts/build_grab.py")
    take = hands["take"]
    edges = np.r_[0, np.flatnonzero(np.diff(take)) + 1, len(take)]
    hulls = {name: _hull(meshes[f"{name}_vertices"]) for name in np.unique(hands["object"])}
    table = _hull(meshes[f"{grab.TABLE}_vertices"])
    _, (wrist_r, axes_r, length_r) = robot_chain()
    palm = {}                                    # each person's median palm length, m
    for subject in np.unique(hands["subject"]):
        rows = np.isin(take, np.flatnonzero(hands["subject"] == subject))
        palm[subject] = float(np.median(np.linalg.norm(
            hands["joints"][rows, :, 9] - hands["joints"][rows, :, 0], axis=-1)))

    frames = {key: [] for key in ("joints", "position", "quat", "gravity", "fingers", "take", "hand")}
    episodes = {key: [] for key in ("first", "length", "variant", "subject")}
    clips, variants, points = [], {}, {}
    count = 0
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        t = take[a]
        subject, name = str(hands["subject"][t]), str(hands["object"][t])
        rotation = Rotation.from_rotvec(hands["object_pose"][a:b, :3]).as_matrix()
        lowest = (np.einsum("fij,vj->fvi", rotation, hulls[name])[..., 2].min(1)
                  + hands["object_pose"][a:b, 5])
        top = (np.einsum("fij,vj->fvi", Rotation.from_rotvec(hands["table_pose"][a:b, :3]).as_matrix(),
                         table)[..., 2].max(1) + hands["table_pose"][a:b, 5])
        for hand, side in enumerate(grab.SIDES):
            spans = _holds(hands["contact"][a:b], hand, lowest > top + ABOVE_TABLE_M)
            if not spans:
                continue
            key = (name, subject, side)
            if key not in variants:
                variants[key] = len(variants)
                scale = length_r / palm[subject]
                points[f"{name}_{subject}_{side.lower()}"] = scale * hulls[name] @ (
                    MIRROR if side == "Right" else np.eye(3)).T
            placed = _onto_robot(hands["joints"][a:b, hand], side, hands["object_pose"][a:b],
                                 (wrist_r, axes_r), length_r / palm[subject])
            for key2, value in zip(("position", "quat", "gravity"), placed):
                frames[key2].append(value)
            frames["joints"].append(labels["joints"][a:b, hand].astype(np.float32))
            frames["fingers"].append(_fingers(hands["contact"][a:b, hand]))
            frames["take"].append(np.full(b - a, t))
            frames["hand"].append(np.full(b - a, hand))
            clips.append((a, b, hand, side))
            for start, end in spans:
                for first in range(start, end - round(MIN_HOLD_S * RATE) + 1,
                                   round(START_EVERY_S * RATE)):
                    episodes["first"].append(count + first)
                    episodes["length"].append(min(end - first, round(EPISODE_S * RATE)))
                    episodes["variant"].append(variants[key])
                    episodes["subject"].append(subject)
            count += b - a
        progress(k + 1, len(edges) - 1, f"holds: {hands['name'][t]}")
    frames = {key: np.concatenate(value) for key, value in frames.items()}
    episodes = {key: np.array(value) for key, value in episodes.items()}

    progress(0, 1, "the reader on the tracker's errors")
    space = MotionSpace(space_module.load())
    reader = reader_module.Reader(reader_module.load(), space)
    noise = TrackerNoise()
    rng = np.random.default_rng(0)
    read = np.zeros((DRAWS, count, labels["joints"].shape[-1]), np.float16)
    meant = np.zeros((DRAWS, count, reader_module.INTENTS), np.float16)
    for draw in range(DRAWS):
        seen, found, truth = [], [], []
        for a, b, hand, side in clips:
            person = in_palm(hands["joints"][a:b, hand].astype(float), side).astype(np.float32)
            camera = rng.integers(grab.CAMERAS)
            s, f = noise.corrupt(person[None], noise.neighbours(person)[None],
                                 views["hidden"][a:b, hand, camera][None], rng)
            seen.append(s[0])
            found.append(f[0])
            truth.append(labels["joints"][a:b, hand].astype(np.float32))
        made, likely = reader_module._follow_all(reader.net, reader.space, seen, found, truth)
        read[draw] = np.concatenate(made)
        meant[draw] = np.concatenate(likely)
        progress(draw + 1, DRAWS, "the reader on the tracker's errors")

    friction = settings.section("object")["friction"]
    spec, parts = hold_scene.build(points, friction)
    xml = spec.to_xml().replace("  <default>\n    <default/>\n  </default>\n", "", 1)
    jobs = [(int(v), *(frames[key][f:f + round(CHECK_S * RATE) + 1]
                       for key in ("joints", "position", "quat", "gravity", "fingers")))
            for f, v in zip(episodes["first"], episodes["variant"])]
    settled = []
    with get_context("spawn").Pool(WORKERS, initializer=_start, initargs=(points, friction)) as pool:
        for k, done in enumerate(pool.imap(_settle, jobs, chunksize=16)):
            settled.append(done)
            if k % 200 == 0 or k == len(jobs) - 1:
                progress(k + 1, len(jobs), "settling the starts")
    model = spec.compile()
    qpos, qvel, grip, reachable, drifts = zip(*settled)
    drifts = 1000 * np.array(drifts, np.float32)
    episodes.update(qpos=np.array(qpos, np.float32), qvel=np.array(qvel, np.float32),
                    grip=np.array(grip, np.float32), reachable=np.array(reachable),
                    drift_mm=drifts[:, 0], grip_drift_mm=drifts[:, 1],
                    test=np.isin(episodes["subject"], TEST))
    np.savez_compressed(
        HOLDS, **frames, read=read, meant=meant, **episodes,
        variant_names=np.array(list(points)), scene=np.array(xml), parts=np.array(parts),
        joint_names=np.array([model.joint(j).name for j in range(model.njnt)
                         if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]),
        rate=np.array(RATE), made=np.array(datetime.now().isoformat(timespec="seconds")))
    return report(load())


def load():
    return dict(np.load(HOLDS)) if HOLDS.exists() else {}


def export_bundle(folder):
    """What the GPU trains on (gpu_holds.py), into `folder`: holds.npz -
    the scene's MJCF in it - and bundle.json."""
    if not HOLDS.exists():
        raise RuntimeError("no holds - run scripts/build_holds.py")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(HOLDS, folder / HOLDS.name)
    made = str(np.load(HOLDS)["made"])
    (folder / "bundle.json").write_text(json.dumps({
        "mujoco": mujoco.__version__, "library_made": made,
        "made": datetime.now().isoformat(timespec="seconds")}, indent=2))
    return folder


def report(holds):
    """How many holds and how much of each kept with no policy."""
    lines = []
    for name, test in (("fitted people", False), ("people never fitted", True)):
        rows = holds["test"] == test
        replays = []
        for how, key in (("the retargeted motion", "drift_mm"), ("it plus the start's grip", "grip_drift_mm")):
            drift = holds[key][rows]
            replays.append(f"{how}: within 20 mm {np.mean(drift < 20):.0%}, 50 mm {np.mean(drift < 50):.0%}")
        lines.append(f"{name}: {rows.sum()} holds ({np.mean(holds['reachable'][rows]):.0%} reachable), "
                     f"{len(np.unique(holds['variant'][rows]))} objects (by person and side); the object after {CHECK_S:g} s with no policy, the servos on "
                     + "; on ".join(replays))
    return "\n".join(lines)


def _hull(vertices):
    """A mesh's convex hull's points: what MuJoCo collides and weighs."""
    vertices = np.asarray(vertices, float)
    return vertices[ConvexHull(vertices).vertices]


def _holds(contact, hand, lifted):
    """[(start, end)] frames where `hand` holds the object alone and up."""
    thumb = contact[:, hand, 1:4].any(1)
    other = contact[:, hand, [0, *range(4, 16)]].any(1)
    held = thumb & other & ~contact[:, 1 - hand].any(1) & lifted
    held = _filled(held, GAP)
    edges = np.flatnonzero(np.diff(np.r_[0, held.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])
            if b - a >= round(MIN_HOLD_S * RATE)]


def _filled(flags, gap):
    """`flags` with runs of up to `gap` False between Trues made True."""
    out = flags.copy()
    edges = np.flatnonzero(np.diff(np.r_[1, flags.astype(int), 1]))
    for a, b in zip(edges[::2], edges[1::2]):
        if 0 < a and b < len(flags) and b - a <= gap:
            out[a:b] = True
    return out


def _fingers(contact):
    """GRAB's 16 parts' contacts (T, 16) -> by hold_scene.FINGERS (T, 5)."""
    groups = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    return np.stack([contact[:, g].any(1) for g in groups], 1)


def _onto_robot(world, side, object_pose, robot_palm, scale):
    """A hand's landmarks (T, 21, 3) and its object's poses (T, 6: rotation
    vector, position) -> the object's position (T, 3) and quaternion (T,
    4, w x y z) in the scene, and gravity (T, 3) - all as they were in
    the person's palm frame, in the robot's (wrist, axes) - the object's
    distances times `scale`."""
    wrist, axes, _ = palm_frame(world.astype(float), side)
    mirror = MIRROR if side == "Right" else np.eye(3)
    into = robot_palm[1] @ mirror @ np.swapaxes(axes, 1, 2)          # (T, 3, 3)
    position = robot_palm[0] + scale * np.einsum("tij,tj->ti", into, object_pose[:, 3:] - wrist)
    rotation = into @ Rotation.from_rotvec(object_pose[:, :3]).as_matrix() @ mirror
    quat = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    return (position.astype(np.float32), quat.astype(np.float32),
            np.einsum("tij,j->ti", into, GRAVITY).astype(np.float32))


_scene = None


def _start(points, friction):
    global _scene
    spec, parts = hold_scene.build(points, friction)
    model = spec.compile()
    robot = robot_module.load()
    joints = {finger: [int(model.jnt_qposadr[model.joint(robot.names[i]).id]) for i in indices]
              for finger, indices in robot.groups().items()}
    geoms = {finger: [] for finger in hold_scene.FINGERS}
    for sensor, finger in enumerate(parts):
        body = model.body(model.sensor(sensor).name[:-len("_gap")]).id
        geoms[finger] += [g for g in range(model.body_geomadr[body],
                                           model.body_geomadr[body] + model.body_geomnum[body])
                          if model.geom_contype[g]]
    _scene = (model, mujoco.MjData(model), hold_scene.variant_fields(spec, model), joints, geoms)


def _settle(job):
    """One episode's start (see the docstring) -> (qpos, qvel) still
    pinned, the grip given (J,) - the servo targets less the retargeted
    pose - whether the thumb and another finger reached the object, and
    how far (m) the object went, let go, with the servos on the retargeted
    motion alone and on it plus that grip."""
    variant, joints, position, quat, gravity, fingers = job
    model, data, fields, finger_joints, finger_geoms = _scene
    hold_scene.use(model, data, fields, variant, float(np.mean(MASS_KG)))
    hinge = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    obj = model.jnt_qposadr[model.joint(hold_scene.OBJECT).id]

    def placed():
        mujoco.mj_resetData(model, data)
        data.qpos[hinge] = joints[0][hinge]
        data.qpos[obj:obj + 3], data.qpos[obj + 3:obj + 7] = position[0], quat[0]
        data.mocap_pos[0], data.mocap_quat[0] = position[0], quat[0]
        model.opt.gravity[:] = gravity[0]

    placed()
    target = data.qpos[:model.nu].copy()          # qpos order
    reached = {}
    for k, finger in enumerate(hold_scene.FINGERS):
        if finger in finger_joints and fingers[0, k]:
            target[finger_joints[finger]], reached[finger] = _closed(
                model, data, finger_joints[finger], finger_geoms[finger], SQUEEZE_RAD)
    grip = (target - joints[0])[hinge]            # the servos' order
    reachable = reached.get("thumb", False) and sum(reached.values()) >= 2
    data.eq_active[model.equality(hold_scene.WELD).id] = 1
    data.ctrl[:] = joints[0][hinge] + grip
    for _ in range(round(SETTLE_S / model.opt.timestep)):
        mujoco.mj_step(model, data)
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    data.eq_active[model.equality(hold_scene.WELD).id] = 0
    drifts = []
    for held in (np.zeros_like(grip), grip):
        data.qpos[:], data.qvel[:] = qpos, qvel
        data.qacc_warmstart[:] = 0
        for frame in range(1, len(joints)):
            data.ctrl[:] = joints[frame][hinge] + held
            model.opt.gravity[:] = gravity[frame]
            for _ in range(round(1.0 / (RATE * model.opt.timestep))):
                mujoco.mj_step(model, data)
        drift = np.linalg.norm(data.qpos[obj:obj + 3] - position[-1])
        drifts.append(float(drift if np.isfinite(drift) else np.inf))
    return qpos, qvel, grip, reachable, drifts


def _closed(model, data, joints, geoms, squeeze, steps=30):
    """A finger's joints (qpos indices) moved from where `data` has them
    until one of its `geoms` meets the object - each step along the
    gap's gradient, as far as the gap says - and then `squeeze` rad
    further that way: (the servo targets that press it on, whether it
    met the object). The finger's joints in `data` are left closed."""
    obj = model.geom(hold_scene.OBJECT).id
    low, high = model.jnt_range[[_joint_of(model, q) for q in joints]].T

    def gap():
        mujoco.mj_kinematics(model, data)
        return min(mujoco.mj_geomDistance(model, data, g, obj, REACH_M, None) for g in geoms)

    way = np.zeros(len(joints))
    for _ in range(steps):
        now = gap()
        if now >= REACH_M:                        # out of reach
            break
        slope = np.zeros(len(joints))
        for i, q in enumerate(joints):
            data.qpos[q] += 1e-3
            slope[i] = (gap() - now) / 1e-3
            data.qpos[q] -= 1e-3
        size = np.linalg.norm(slope)
        if size < 1e-6:                           # nothing moves it nearer
            break
        way = -slope / size
        if now < TOUCH_M:
            break
        data.qpos[joints] = np.clip(data.qpos[joints] + way * min(MAX_STEP_RAD, now / size), low, high)
    reached = gap() < TOUCH_M
    return np.clip(data.qpos[joints] + squeeze * way, low, high), reached


def _joint_of(model, qpos):
    return int(np.flatnonzero(model.jnt_qposadr == qpos)[0])
