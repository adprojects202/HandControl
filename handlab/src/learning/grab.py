"""GRAB's people handling objects, put onto the robot hand: real motion -
people's own timing, grasps, holds and rests - for the motion space
(motion_space.py); the objects themselves come later.

    ../.venv-wilor/bin/python wilor/grab.py   # GRAB's hands, their 21 points (~1 min)
    python scripts/build_grab.py              # label(): onto the robot (~20 min)
    python scripts/build_grab_views.py        # views(): what a camera loses (~10 s)
    labels = load()
    takes = clips(labels, ("s1", "s2"))       # [(T, J) rad] a hand of a take each

Each hand of each take is retargeted frame by frame (human_retarget.
HumanRetarget), each frame solved from the last and held towards it - one
motion, not poses one by one. Where the thumb and a fingertip both touch the
object (GRAB's contact labels on their last segments) the hand is holding:
the object is between them, so no touch is made. Right hands go as
FreiHAND's do (mirrored onto the robot's left hand), left hands as they are.

What a camera would see of the hands (`views`, for the reader's tracker -
tracker_noise.py): CAMERAS places per take, each fixed as a camera on a
stand, CAMERA_M from the take's hands, in front of the person (within
AZIMUTH_DEG of their forward) and above (ELEVATION_DEG); from each, which
of the hands' 21 points the object or the table hides - a ray from the
camera meeting their meshes (MuJoCo's, triangle by triangle) before the
point, by more than SURFACE_M (a fingertip on the object is not behind it).
The other hand and the person's body are not counted.

    data/learning/intent/grab_robot.npz
        joints (N, 2, J) rad: right then left hand, the robot's settled pose;
        touching (N, 2, 3) bool: the robot's thumb on the index, middle,
        ring; holding (N, 2, 3) bool; take (N,) as grab_hands.npz; subject,
        object, intent, name (takes,); rate; made
    data/learning/intent/grab_views.npz
        hidden (N, 2, CAMERAS, 21) bool; camera (takes, CAMERAS, 3) m;
        take (N,); made
"""

from __future__ import annotations

from datetime import datetime
from multiprocessing import get_context

import mujoco
import numpy as np

from .handmotion import RATE
from .human_retarget import FINGERS, HumanRetarget
from .intent_pairs import FOLDER, WORKERS

HANDS = FOLDER / "grab_hands.npz"
LABELS = FOLDER / "grab_robot.npz"
MESHES = FOLDER / "grab_meshes.npz"
VIEWS = FOLDER / "grab_views.npz"
SIDES = ("Right", "Left")
PARTS = ("palm", "thumb1", "thumb2", "thumb3", "index1", "index2", "index3", "middle1", "middle2",
         "middle3", "ring1", "ring2", "ring3", "pinky1", "pinky2", "pinky3")    # as wilor/grab.py's
THUMB_TIP, FINGER_TIPS = 3, (6, 9, 12)       # in PARTS: the last segments
JUMP = 20                                    # degrees in a frame: a jump, reported
TABLE = "table"
CAMERAS = 4                                  # per take: places a camera could be
CAMERA_M = (0.4, 0.9)                        # from the take's hands
AZIMUTH_DEG = 75                             # either side of the person's forward
ELEVATION_DEG = (0, 60)                      # above the hands
SURFACE_M = 0.005                            # a point this close behind a surface is on it
VIEW_PARTS = {"palm": [0, 1, 5, 9, 13, 17], "thumb": [2, 3, 4], "index": [6, 7, 8],
              "middle": [10, 11, 12], "ring": [14, 15, 16], "little": [18, 19, 20]}


def label(progress=lambda done, total, text="": None):
    """Retarget every take's hands; writes grab_robot.npz, returns a summary."""
    if not HANDS.exists():
        raise RuntimeError("no GRAB hands - run ../.venv-wilor/bin/python wilor/grab.py")
    hands = dict(np.load(HANDS))         # read once: an npz reads an array at every lookup
    if int(hands["rate"]) != RATE:
        raise RuntimeError(f"GRAB hands at {int(hands['rate'])} Hz, the motion space's {RATE}")
    take = hands["take"]
    edges = np.r_[0, np.flatnonzero(np.diff(take)) + 1, len(take)]
    jobs = [(hands["joints"][a:b], hands["contact"][a:b]) for a, b in zip(edges[:-1], edges[1:])]
    out = {"joints": [], "touching": [], "holding": []}
    with get_context("spawn").Pool(WORKERS, initializer=_start) as pool:
        for k, done in enumerate(pool.imap(_retarget, jobs)):
            for key, value in zip(out, done):
                out[key].append(value)
            progress(k + 1, len(jobs), str(hands["name"][take[edges[k]]]))
    arrays = {key: np.concatenate(value) for key, value in out.items()}
    np.savez(LABELS, **arrays, take=take, **{k: hands[k] for k in ("subject", "object", "intent", "name")},
             rate=hands["rate"], made=np.array(datetime.now().isoformat(timespec="seconds")))
    return report(arrays["joints"], arrays["touching"], arrays["holding"], take)


def views(progress=lambda done, total, text="": None, seed=0):
    """For CAMERAS places a camera could be in each take, which of each
    hand's 21 points the object or the table hides; writes grab_views.npz,
    returns a summary."""
    if not MESHES.exists():
        raise RuntimeError("no GRAB meshes - run ../.venv-wilor/bin/python wilor/grab.py")
    hands = dict(np.load(HANDS))
    rng = np.random.default_rng(seed)
    take = hands["take"]
    edges = np.r_[0, np.flatnonzero(np.diff(take)) + 1, len(take)]
    jobs = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = take[a]
        jobs.append((hands["joints"][a:b], hands["object_pose"][a:b], hands["table_pose"][a:b],
                     str(hands["object"][k]), _cameras(hands["joints"][a:b], hands["facing"][k], rng)))
    hidden, cameras = [], []
    with get_context("spawn").Pool(WORKERS, initializer=_start_views) as pool:
        for k, done in enumerate(pool.imap(_hidden, jobs)):
            hidden.append(done)
            cameras.append(jobs[k][4])
            progress(k + 1, len(jobs), str(hands["name"][take[edges[k]]]))
    hidden = np.concatenate(hidden)
    np.savez(VIEWS, hidden=hidden, camera=np.array(cameras, np.float32), take=take,
             made=np.array(datetime.now().isoformat(timespec="seconds")))
    return report_views(hidden)


def report_views(hidden):
    """How much of each finger a camera loses behind the object or table."""
    lines = []
    for hand, side in enumerate(SIDES):
        seen = hidden[:, hand]
        lines.append(f"{side}: hidden " + ", ".join(
            f"{name} {seen[..., points].mean():.1%}" for name, points in VIEW_PARTS.items())
            + f"; most of the hand hidden {np.mean(seen.mean(-1) > 0.5):.1%} of views")
    return "\n".join(lines)


def load():
    return dict(np.load(LABELS)) if LABELS.exists() else {}


def clips(labels, subjects):
    """Each hand of each take of `subjects`, its joints (T, J) float32."""
    take = labels["take"]
    edges = np.r_[0, np.flatnonzero(np.diff(take)) + 1, len(take)]
    return [labels["joints"][a:b, hand].astype(np.float32)
            for a, b in zip(edges[:-1], edges[1:]) if labels["subject"][take[a]] in subjects
            for hand in range(len(SIDES))]


def report(joints, touching, holding, take):
    """How often each hand touches and holds, and how often it jumps."""
    lines = [f"{len(np.unique(take))} takes, {len(take)} frames at {RATE} Hz"]
    same = take[1:] == take[:-1]
    for hand, side in enumerate(SIDES):
        step = np.degrees(np.abs(np.diff(joints[:, hand], axis=0))).max(1)[same]
        lines.append(f"{side}: touching " + ", ".join(
            f"{f} {touching[:, hand, k].mean():.1%}" for k, f in enumerate(FINGERS))
            + "; holding " + ", ".join(f"{f} {holding[:, hand, k].mean():.1%}" for k, f in enumerate(FINGERS))
            + f"; steps over {JUMP} degrees {np.mean(step > JUMP):.1%}")
    return "\n".join(lines)


_worker = None


def _start():
    global _worker
    _worker = HumanRetarget()


def _cameras(joints, facing, rng):
    """CAMERAS places (CAMERAS, 3) m: around the take's hands, in front of the
    person, above them - fixed for the take, as a camera on a stand is."""
    centre = joints[:, :, 0].reshape(-1, 3).mean(0)
    forward = np.arctan2(facing[1], facing[0])
    turn = forward + np.radians(rng.uniform(-AZIMUTH_DEG, AZIMUTH_DEG, CAMERAS))
    up = np.radians(rng.uniform(*ELEVATION_DEG, CAMERAS))
    way = np.stack([np.cos(up) * np.cos(turn), np.cos(up) * np.sin(turn), np.sin(up)], 1)
    return centre + way * rng.uniform(*CAMERA_M, CAMERAS)[:, None]


_scene = None


def _start_views():
    global _scene
    _scene = _Scene()


class _Scene:
    """An object and the table as MuJoCo meshes (its rays meet their
    triangles, not hulls), placed frame by frame."""

    def __init__(self):
        self.meshes = np.load(MESHES)
        self.models = {}

    def model(self, name):
        if name not in self.models:
            spec = mujoco.MjSpec()
            spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
            for thing in (name, TABLE):
                spec.add_mesh(name=thing, uservert=self.meshes[f"{thing}_vertices"].astype(float).ravel(),
                              userface=self.meshes[f"{thing}_faces"].ravel())
                spec.worldbody.add_body(name=thing, mocap=True).add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH, meshname=thing, contype=0, conaffinity=0)
            model = spec.compile()
            self.models[name] = (model, mujoco.MjData(model))
        return self.models[name]


def _hidden(job):
    """One take's (T, 2, CAMERAS, 21) bool: a point hidden from a camera."""
    joints, object_pose, table_pose, name, cameras = job
    model, data = _scene.model(name)
    hidden = np.zeros((len(joints), len(SIDES), CAMERAS, 21), bool)
    count = len(SIDES) * 21
    distance, geom = np.zeros(count), np.zeros(count, np.int32)
    for frame in range(len(joints)):
        for body, pose in enumerate((object_pose[frame], table_pose[frame])):
            data.mocap_pos[body] = pose[3:]
            mujoco.mju_axisAngle2Quat(data.mocap_quat[body], pose[:3] / max(np.linalg.norm(pose[:3]), 1e-12),
                                      np.linalg.norm(pose[:3]))
        mujoco.mj_kinematics(model, data)
        points = joints[frame].reshape(-1, 3).astype(float)
        for c, camera in enumerate(cameras):
            way = points - camera
            far = np.linalg.norm(way, axis=1)
            mujoco.mj_multiRay(model, data, camera.astype(float), (way / far[:, None]).ravel(), None, True,
                               -1, geom, distance, None, count, far.max() + 1)
            hidden[frame, :, c] = ((distance >= 0) & (distance < far - SURFACE_M)).reshape(len(SIDES), 21)
    return hidden


def _retarget(job):
    """One take's hands (T, 2, 21, 3), contact (T, 2, 16) -> joints, touching, holding."""
    world, contact = job
    holding = contact[:, :, THUMB_TIP:THUMB_TIP + 1] & contact[:, :, FINGER_TIPS]
    joints = np.zeros((len(world), len(SIDES), _worker.hand.model.nq), np.float32)
    touching = np.zeros((len(world), len(SIDES), len(FINGERS)), bool)
    for hand, side in enumerate(SIDES):
        last = None
        for frame in range(len(world)):
            last, touching[frame, hand] = _worker(world[frame, hand], side, holding[frame, hand], last)
            joints[frame, hand] = last
    return joints, touching, holding
