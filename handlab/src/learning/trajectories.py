"""The trajectory library the tracker learns from: object motions in the hand
mount's frame, each from a start the hand can grip, each tried by the planner.

    build(progress)        # -> data/learning/library.npz
    library = load()

Starts. Every recording sent to sim (a current sim.npz, left hand) gives one:
where its motion begins. Around each, START_JITTER more, shifted up to
START_SHIFT_M and turned up to START_TURN about a random axis. For each the
planner searches a grip (Planner.grasp); a start whose best grip lets the
object slip more than GRIP_SLIP_M or turn more than GRIP_TURN is dropped. An
episode starts from its state: the grip closed on the held object for HOLD
seconds, then the object let go.

Motions. From each start, MOTIONS_PER_START generated ones, half of each
kind. Moves: up to SEGMENTS of them, each a turn of TURN_RANGE about a
random axis through the object's centre plus a shift to a point within
SHIFT_M of the start, over SEGMENT_S seconds with a minimum-jerk profile,
with PAUSE_S still between. Wobbles: WOBBLE_WAVES sine waves of WOBBLE_HZ
on random axes, added - a turn of up to WOBBLE_TURN and a shift of up to
SHIFT_M, back and forth - eased in over RAMP_S, for WOBBLE_S seconds. A
person's motion is mostly the second kind: the red-cube recordings turn at
up to 95-150 deg/s and reverse 9-13 times in 4 s (smoothed over 0.25 s,
which takes out only ~1 degree of tracking jitter). Plus the recordings' own
motions, marked recorded: the tracker trains on generated motions only, so
a recording tests it on a motion it has never seen. `motion_source` says
which kind ("moves", "wobble") or which recording.

Feasibility. The planner (sim/planner.py) follows every motion from its
start's grip. Feasible: it held the object throughout, its median position
error stayed within FEASIBLE_POSITION_M, and its median turn error within
FEASIBLE_TURN or within FEASIBLE_SHARE of the turn had the object stayed put
(it followed at least that far). The planner is a lower bound on what the
hand can do - the tracker may do better - so this only drops what it could
not follow at all.

    library.npz
        start_position (S, 3), start_quat (S, 4) w x y z: in the mount frame
        start_grip (S, J) rad; start_slip (S,) m, start_turn (S,) rad
        start_state (S, nstate): MuJoCo full physics state, just let go
        start_source (S,): the recording it came from
        motion_start (M,), motion_length (M,): frames at RATE Hz
        motion_position (M, L, 3), motion_quat (M, L, 4): after its length,
            the last pose again
        motion_recorded (M,) bool, motion_source (M,)
        motion_feasible (M,) bool; motion_position_error, motion_turn_error,
            motion_still_turn (M,): the planner's medians, and the turn had
            the object stayed put
        rate, made
"""

from __future__ import annotations

from datetime import datetime

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..paths import ROOT
from ..processing.files import SIM, stale
from ..sim.planner import HOLD, Goal, Planner
from ..sim.task import Task

LIBRARY = ROOT / "data" / "learning" / "library.npz"
RATE = 20                       # Hz: frames of a motion, and the tracker's steps
START_JITTER = 5
START_SHIFT_M = 0.008
START_TURN = np.radians(20)
GRIP_SLIP_M, GRIP_TURN = 0.008, np.radians(10)
MOTIONS_PER_START = 24
SEGMENTS = 3
TURN_RANGE = np.radians([10, 45])
SHIFT_M = 0.025
SEGMENT_S = (0.3, 1.2)
PAUSE_S = (0.0, 0.6)
WOBBLE_WAVES = 3
WOBBLE_HZ = (0.3, 1.5)
WOBBLE_TURN = np.radians([10, 35])
WOBBLE_S = (3.0, 6.0)
RAMP_S = 0.5
FEASIBLE_POSITION_M = 0.015
FEASIBLE_TURN = np.radians(12)
FEASIBLE_SHARE = 0.5
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


def build(progress=lambda done, total, text="": None, seed=0):
    """Make the library from the recordings; returns a summary."""
    rng = np.random.default_rng(seed)
    recorded = _recorded()
    if not recorded:
        raise ValueError("no recording has a current sim.npz (Real to Sim: fit 3D)")
    poses = []
    for source, positions, quats in recorded:
        poses.append((source, positions[0], quats[0]))
        for _ in range(START_JITTER):
            poses.append((source, *_jitter(positions[0], quats[0], rng)))
    starts, motions = [], []
    total = len(poses) * (1 + MOTIONS_PER_START) + len(recorded)
    done = 0
    for source, position, quat in poses:
        task = Task.build((position, quat))
        planner = Planner(task, seed=int(rng.integers(1 << 31)))
        progress(done, total, f"start {len(starts) + 1}: searching a grip")
        grip = planner.grasp()
        done += 1
        if grip["slip_m"] > GRIP_SLIP_M or grip["turn_rad"] > GRIP_TURN:
            done += MOTIONS_PER_START
            continue
        start = len(starts)
        starts.append({"position": position, "quat": quat, "grip": grip["targets"],
                       "slip": grip["slip_m"], "turn": grip["turn_rad"],
                       "state": let_go(task, grip["targets"]), "source": source})
        own = [m for m in recorded if m[0] == source and np.allclose(m[1][0], position)]
        tries = [(m[1], m[2], True, source) for m in own]
        for index in range(MOTIONS_PER_START):
            kind, generate = (("moves", _generate) if index % 2 == 0 else ("wobble", _wobble))
            tries.append((*generate(position, quat, rng), False, kind))
        for positions, quats, is_recorded, origin in tries:
            progress(done, total, f"start {start + 1}: planning motion {len(motions) + 1}")
            motions.append({"start": start, "position": positions, "quat": quats,
                            "recorded": is_recorded, "source": origin,
                            **_try(task, planner, grip, positions, quats)})
            done += 1
    if not starts:
        raise ValueError("no start could be gripped")
    return _save(starts, motions, tried=len(poses))


def load(path=LIBRARY):
    """The library as a dict of arrays, or {} before it is built."""
    if not path.exists():
        return {}
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


# ------------------------------------------------------------------ motions
def _recorded():
    """[(folder, positions (L, 3), quats (L, 4))] from each recording that
    has a current sim.npz made with the left hand."""
    from ..recording import Episode, episodes     # here: training needs no recordings
    out = []
    for folder in episodes():
        motion = motion_of(Episode(folder))
        if motion is not None:
            out.append((str(folder), *motion))
    return out


def motion_of(episode):
    """A recording's motion sent to sim, at RATE: (positions (L, 3), quats
    (L, 4)) in the mount frame - or None without a current sim.npz made with
    the left hand."""
    sim = episode.arrays(SIM)
    if not sim or SIM in stale(episode) or str(sim["hand"]) != "Left" \
            or sim["valid"].sum() < 2:
        return None
    goal = Goal(sim["time"].astype(float), sim["position"].astype(float),
                sim["quat"].astype(float), sim["valid"], start=0.0)
    positions, rotations = goal.at(np.arange(int(goal.end * RATE) + 1) / RATE)
    return positions, _quats(rotations)


def _jitter(position, quat, rng):
    """A start near (position, quat): shifted and turned a little."""
    shift = _in_ball(rng) * START_SHIFT_M
    turn = Rotation.from_rotvec(_axis(rng) * rng.uniform(0, START_TURN))
    rotation = turn * Rotation.from_quat(quat[[1, 2, 3, 0]])
    return position + shift, rotation.as_quat()[[3, 0, 1, 2]]


def _generate(position, quat, rng):
    """A random smooth motion from (position, quat): (positions, quats) at RATE."""
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    at, positions, rotations = position, [position], [rotation]

    def still(seconds):
        for _ in range(round(seconds * RATE)):
            positions.append(at)
            rotations.append(rotation)

    for _ in range(int(rng.integers(1, SEGMENTS + 1))):
        still(rng.uniform(*PAUSE_S))
        frames = max(2, round(rng.uniform(*SEGMENT_S) * RATE))
        turn = _axis(rng) * rng.uniform(*TURN_RANGE)
        target = position + _in_ball(rng) * SHIFT_M
        for frame in range(1, frames + 1):
            s = _minimum_jerk(frame / frames)
            positions.append(at + (target - at) * s)
            rotations.append(Rotation.from_rotvec(turn * s) * rotation)
        at, rotation = target, Rotation.from_rotvec(turn) * rotation
    still(rng.uniform(*PAUSE_S))
    return np.array(positions), _quats(Rotation.concatenate(rotations).as_matrix())


def _wobble(position, quat, rng):
    """Back and forth: sine waves added, eased in. (positions, quats) at RATE."""
    times = np.arange(round(rng.uniform(*WOBBLE_S) * RATE) + 1) / RATE
    ease = _minimum_jerk(np.clip(times / RAMP_S, 0.0, 1.0))[:, None]

    def waves(largest):
        # Amplitudes sum to `largest`: it is never exceeded, and the speed
        # stays within 2 pi WOBBLE_HZ[1] largest.
        share = rng.dirichlet(np.ones(WOBBLE_WAVES))
        out = np.zeros((len(times), 3))
        for weight in share:
            frequency, phase = rng.uniform(*WOBBLE_HZ), rng.uniform(0, 2 * np.pi)
            out += np.outer(np.sin(2 * np.pi * frequency * times + phase), _axis(rng)) \
                * weight * largest
        return out * ease

    turns = waves(rng.uniform(*WOBBLE_TURN))
    positions = position + waves(rng.uniform(0, SHIFT_M))
    rotations = Rotation.from_rotvec(turns) * Rotation.from_quat(quat[[1, 2, 3, 0]])
    return positions, _quats(rotations.as_matrix())


def _minimum_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def _axis(rng):
    axis = rng.normal(size=3)
    return axis / np.linalg.norm(axis)


def _in_ball(rng):
    """A point uniformly inside the unit ball."""
    return _axis(rng) * rng.uniform() ** (1 / 3)


# ------------------------------------------------------------------ the planner
def let_go(task, grip):
    """The full physics state just after the grip closed on the held object
    and it was let go, at time 0."""
    task.reset()
    task.data.ctrl[:] = grip
    while task.data.time < HOLD:
        mujoco.mj_step(task.model, task.data)
    task.release(task.data)
    state = np.zeros(mujoco.mj_stateSize(task.model, FULL))
    mujoco.mj_getState(task.model, task.data, state, FULL)
    state[0] = 0.0
    return state


def _try(task, planner, grip, positions, quats):
    """The planner following one motion from the start's grip."""
    task.reset()
    times = np.arange(len(positions)) / RATE
    goal = Goal(times, positions, quats, np.ones(len(positions), bool), start=HOLD)
    _, rows = planner.track(goal, grip=grip)
    following = rows["time"] >= goal.start
    position_error = float(np.median(rows["position_error"][following]))
    turn_error = float(np.median(rows["turn_error"][following]))
    _, rotations = goal.at(rows["time"][following])
    still_turn = float(np.median(planner.turn(rotations, rotations[0])))
    held = not (rows["position_error"][following] > 0.05).any()
    return {"feasible": held and position_error <= FEASIBLE_POSITION_M
            and (turn_error <= FEASIBLE_TURN or turn_error <= FEASIBLE_SHARE * still_turn),
            "position_error": position_error, "turn_error": turn_error,
            "still_turn": still_turn}


def _save(starts, motions, tried):
    length = max(len(m["position"]) for m in motions)

    def padded(key):
        return np.stack([np.concatenate([m[key], np.repeat(m[key][-1:], length - len(m[key]),
                                                           axis=0)]) for m in motions])

    arrays = {
        "start_position": np.array([s["position"] for s in starts]),
        "start_quat": np.array([s["quat"] for s in starts]),
        "start_grip": np.array([s["grip"] for s in starts]),
        "start_slip": np.array([s["slip"] for s in starts]),
        "start_turn": np.array([s["turn"] for s in starts]),
        "start_state": np.array([s["state"] for s in starts]),
        "start_source": np.array([s["source"] for s in starts]),
        "motion_start": np.array([m["start"] for m in motions]),
        "motion_length": np.array([len(m["position"]) for m in motions]),
        "motion_position": padded("position"), "motion_quat": padded("quat"),
        "motion_recorded": np.array([m["recorded"] for m in motions]),
        "motion_source": np.array([m["source"] for m in motions]),
        "motion_feasible": np.array([m["feasible"] for m in motions]),
        "motion_position_error": np.array([m["position_error"] for m in motions]),
        "motion_turn_error": np.array([m["turn_error"] for m in motions]),
        "motion_still_turn": np.array([m["still_turn"] for m in motions]),
        "rate": np.array(RATE), "made": np.array(datetime.now().isoformat(timespec="seconds")),
    }
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIBRARY.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(LIBRARY)
    generated = ~arrays["motion_recorded"]
    kinds = {kind: int(((arrays["motion_source"] == kind) & arrays["motion_feasible"]).sum())
             for kind in ("moves", "wobble")}
    return {"starts": len(starts), "starts_tried": tried,
            "generated": int(generated.sum()),
            "generated_feasible": int((generated & arrays["motion_feasible"]).sum()),
            "feasible_by_kind": kinds,
            "recorded": int((~generated).sum()),
            "recorded_feasible": int((~generated & arrays["motion_feasible"]).sum())}


def _quats(rotations):
    """MuJoCo quaternions (N, 4) w x y z from rotation matrices (N, 3, 3)."""
    return Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
