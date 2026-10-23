"""Run the trained policy: it follows this recording's motion
in the task sim - no retraining, only a grip where the motion starts.
-> track.npz

    track(episode, progress)

Needs sim.npz (current, left hand) and a trained tracker: the run
`learning.run` names, else the newest with a checkpoint (Learning panel). The grip:
plan.npz's when it is current and planned for this mount, else the planner
searches one (about 10 s). From the grip closed on the held object and let
go, the policy follows the motion, nominal physics, no sensor noise.

    track.npz   as plan.npz (processing/plan.py), so step 6's summary and the
                plan stage read it alike: time (T,) s since the object was
                let go (hold 0), qpos, position, quat, goal_position,
                goal_quat, position_error, turn_error, still_position_error,
                still_turn_error, dropped, grip, grip_slip, grip_turn,
                orientation, made; plus run (the run folder) and checkpoint
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .. import settings
from ..learning.train import chosen_run, follow, latest_checkpoint
from ..learning.trajectories import RATE, let_go, motion_of
from ..sim.planner import Planner, _quat_matrices
from ..sim.symmetry import box_symmetries
from ..sim.task import OBJECT, Task
from .files import PLAN, SIM, TRACK, stale
from .plan import summary


def track(episode, progress=lambda done, total, text="": None):
    """Follow the recording's motion with the trained tracker; returns a summary."""
    if not episode.arrays(SIM):
        raise ValueError("send to sim first")
    if SIM in stale(episode):
        raise ValueError(stale(episode)[SIM])
    motion = motion_of(episode)
    if motion is None:
        raise ValueError("recorded with the right hand: not yet mirrored for the robot's left hand")
    run = chosen_run()
    if run is None:
        raise ValueError("no trained policy yet (Learning panel > Train)")
    positions, quats = motion
    orientation = settings.get("world.hand_orientation")
    progress(0, 3, "building the task sim")
    task = Task.build((positions[0], quats[0]))
    plan = episode.arrays(PLAN)
    if plan and PLAN not in stale(episode) and str(plan["orientation"]) == orientation:
        grip = {"targets": plan["grip"].astype(float), "slip_m": float(plan["grip_slip"]),
                "turn_rad": float(plan["grip_turn"])}
    else:
        progress(1, 3, "searching a grip")
        grip = Planner(task).grasp()
    progress(2, 3, "following the motion")
    library = _one_motion(task, grip, positions, quats)
    checkpoint = latest_checkpoint(run)
    result = follow(run, library, [0], checkpoint)[0]
    rows = len(result["joints"])
    # Had the object stayed put: the turn to each goal, least over the
    # object's symmetries, as step 6 measures it.
    symmetries = np.array(box_symmetries(task.model.geom(OBJECT).size * 2000))
    relative = _quat_matrices(quats[0]).T @ _quat_matrices(result["goal_quat"])
    traces = np.einsum("nij,sji->ns", relative, symmetries).max(axis=1)
    still_turn = np.arccos(np.clip((traces - 1) / 2, -1, 1))
    episode.save_arrays(TRACK, {
        "time": (np.arange(1, rows + 1) / RATE).astype(np.float32), "hold": np.array(0.0),
        "qpos": result["joints"].astype(np.float32),
        "position": result["position"].astype(np.float32),
        "quat": result["quat"].astype(np.float32),
        "goal_position": result["goal_position"].astype(np.float32),
        "goal_quat": result["goal_quat"].astype(np.float32),
        "position_error": result["position_error"].astype(np.float32),
        "turn_error": result["turn_error"].astype(np.float32),
        "still_position_error": np.linalg.norm(result["goal_position"] - positions[0],
                                               axis=1).astype(np.float32),
        "still_turn_error": still_turn.astype(np.float32),
        "dropped": np.r_[np.zeros(rows - 1, bool), result["dropped"]],
        "grip": grip["targets"].astype(np.float32), "grip_slip": np.array(grip["slip_m"]),
        "grip_turn": np.array(grip["turn_rad"]), "orientation": np.array(orientation),
        "run": np.array(str(run)), "checkpoint": np.array(checkpoint.name),
        "made": np.array(datetime.now().isoformat(timespec="seconds"))})
    progress(3, 3, "done")
    return summary(episode.arrays(TRACK))


def _one_motion(task, grip, positions, quats):
    """A library of one start and one motion, as env.TrackEnv reads it."""
    return {"start_position": positions[:1], "start_quat": quats[:1],
            "start_grip": grip["targets"][None], "start_state": let_go(task, grip["targets"])[None],
            "motion_start": np.array([0]), "motion_length": np.array([len(positions)]),
            "motion_position": positions[None], "motion_quat": quats[None],
            "motion_recorded": np.array([True]), "motion_feasible": np.array([True])}
