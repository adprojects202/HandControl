"""Plan in sim: can the robot hand do what the recording shows? -> plan.npz

    plan(episode, progress)

Needs sim.npz (Send to sim), made from the current 3D fit. Builds the task
sim (sim/task.py: the robot with its servos, the object free, gravity) with
the object where the recording's motion starts, searches for a grip that
holds it there, and follows the motion with predictive sampling
(sim/planner.py). No learning - it says whether this hand, mounted as the
World section has it, can hold the object and move it the way the person did.

To judge the following, the summary also gives the error had the object
just stayed where it started: following that is no better than holding
still has not moved it.

    plan.npz   time (T,) s of sim time, a row every planner period; the
               recording's motion starts at `hold` (the grip closes before)
               qpos, ctrl (T, J) rad: the hand's joints and servo targets
               position (T, 3) m, quat (T, 4) w x y z: the object as the sim
               moved it, in the hand mount's frame (like sim.npz)
               goal_position, goal_quat: where sim.npz wanted it then
               position_error (T,) m, turn_error (T,) rad; gaps (T, F) m:
               each fingertip to the object (tips (F,) names them)
               grip (J,) rad, grip_slip m, grip_turn rad: the grip found
               still_position_error, still_turn_error (T,): had the object
               stayed where it started
               dropped (T,) bool: the object more than DROP_M from its goal
               hold, made; orientation (the mount it was planned for)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

from .. import settings
from ..sim.planner import DROP_M, HOLD, Goal, Planner
from ..sim.task import Task
from .files import PLAN
from .transfer import current_sim



def plan(episode, progress=lambda done, total, text="": None):
    """Grip and follow the recording's motion in the task sim; returns a summary."""
    sim = current_sim(episode, progress)
    if str(sim["hand"]) != "Left":
        raise ValueError("recorded with the right hand: not yet mirrored for the robot's "
                         "left hand")
    valid = sim["valid"]
    first = int(np.flatnonzero(valid)[0])
    position, quat = sim["position"].astype(float), sim["quat"].astype(float)
    progress(0, 1, "building the task sim")
    task = Task.build((position[first], quat[first]))
    goal = Goal(sim["time"].astype(float), position, quat, valid, start=HOLD)
    planner = Planner(task)
    grip, rows = planner.track(goal, progress=progress)

    times = rows["time"]
    goal_position, goal_rotation = goal.at(times)
    start_position, start_rotation = goal.at([0.0])
    still_position = np.linalg.norm(goal_position - start_position[0], axis=1)
    still_turn = planner.turn(goal_rotation, start_rotation[0])
    dropped = rows["position_error"] > DROP_M
    episode.save_arrays(PLAN, {
        "time": times.astype(np.float32), "hold": np.array(goal.start),
        "qpos": rows["qpos"].astype(np.float32), "ctrl": rows["ctrl"].astype(np.float32),
        "position": rows["position"].astype(np.float32),
        "quat": _quats(rows["rotation"]).astype(np.float32),
        "goal_position": goal_position.astype(np.float32),
        "goal_quat": _quats(goal_rotation).astype(np.float32),
        "position_error": rows["position_error"].astype(np.float32),
        "turn_error": rows["turn_error"].astype(np.float32),
        "gaps": rows["gaps"].astype(np.float32), "tips": np.array(task.tips),
        "grip": grip["targets"].astype(np.float32), "grip_slip": np.array(grip["slip_m"]),
        "grip_turn": np.array(grip["turn_rad"]),
        "still_position_error": still_position.astype(np.float32),
        "still_turn_error": still_turn.astype(np.float32),
        "dropped": dropped,
        "orientation": np.array(settings.get("world.hand_orientation")),
        "made": np.array(datetime.now().isoformat(timespec="seconds"))})
    return summary(episode.arrays(PLAN))


def summary(plan):
    """What a plan.npz says, in numbers: {grip_slip_mm, grip_turn_deg,
    dropped_at (s of the motion, or None), position_mm and turn_deg (median,
    max) while following, and the same had the object stayed still}."""
    following = plan["time"] >= plan["hold"]
    drops = np.flatnonzero(plan["dropped"] & following)

    def spread(values, scale):
        values = values[following] * scale
        return float(np.median(values)), float(values.max())

    return {"grip_slip_mm": 1000 * float(plan["grip_slip"]),
            "grip_turn_deg": float(np.degrees(plan["grip_turn"])),
            "dropped_at": (float(plan["time"][drops[0]] - plan["hold"]) if len(drops)
                           else None),
            "position_mm": spread(plan["position_error"], 1000),
            "turn_deg": spread(plan["turn_error"], np.degrees(1)),
            "still_position_mm": spread(plan["still_position_error"], 1000),
            "still_turn_deg": spread(plan["still_turn_error"], np.degrees(1))}


def _quats(rotations):
    """MuJoCo quaternions (N, 4) w x y z from rotation matrices (N, 3, 3)."""
    return Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
