"""Can the hand do the task? A grip first, then predictive sampling.

    planner = Planner(task)
    grip, result = planner.track(goal, progress=...)

No learning: if this cannot make the hand do the task, the task or the hand
needs changing before training is worth it. Two parts, both trying many
candidates at once in MuJoCo (mujoco.rollout, on every core):

Grip (`grasp`). The object starts where the goal starts, held by the task's
weld. A candidate is one servo target per joint: the fingers close on the
held object for CLOSE seconds, then it is let go for LET_GO seconds. It costs
how far the object slips and turns meanwhile, plus how far each fingertip
stayed from it (which leads the search to the object before any candidate
holds). The cross-entropy method: draw GRASP_SAMPLES around a mean, move the
mean to the best GRASP_ELITE of them, repeat GRASP_ROUNDS times. Starting
from the middle of every joint's range - the fingers half closed - keeps the
first round near the object.

Following (`track`). The sim closes on the object with the grip, held, until
the goal starts, then lets go. From then on, every PERIOD the planner draws
SAMPLES motions - servo targets at KNOTS times over the next HORIZON, the
last plan shifted on plus random changes (SIGMA) - runs them all forward with
the object free, and keeps the one whose object best follows the goal. The
first plan starts from the grip.

Cost per step of a motion, averaged: (position error / POSITION_SCALE)^2 +
(turn error / TURN_SCALE)^2, the turn taken least over the object's
symmetries (a cube: 24), plus DROP_COST once the object is DROP_M from its
goal, plus (fingertip gap / GAP_SCALE)^2 for each finger; plus EFFORT for
each rad^2 the targets change between knots.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from mujoco import rollout
from scipy.spatial.transform import Rotation, Slerp

from .symmetry import box_symmetries
from .task import WELD

PERIOD = 0.04             # s between plans
HORIZON = 0.4             # s each plan looks ahead
KNOTS = 4
SAMPLES = 128
SIGMA = 0.12              # rad, spread of the random changes
POSITION_SCALE = 0.01     # m that costs 1
TURN_SCALE = np.radians(15)
GAP_SCALE = 0.01          # m between a fingertip and the object that costs 1
DROP_M, DROP_COST = 0.05, 100.0
EFFORT = 0.01
HOLD = 1.0                # s the object is held before the goal starts
GRASP_SAMPLES = 512
GRASP_ROUNDS = 20
GRASP_ELITE = 0.05        # share of the candidates the next round centres on
GRASP_MIN_SIGMA = 0.02    # rad: the search never narrows below this
CLOSE, LET_GO = 0.3, 0.5  # s held while a candidate closes, then free
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


class Goal:
    """Where the object should be: a recording's poses in the mount frame,
    over its frame times; held at the first pose until `start`."""

    def __init__(self, times, positions, quats, valid, start=HOLD):
        times, positions, quats = times[valid], positions[valid], quats[valid]
        self.start, self.end = start, start + float(times[-1] - times[0])
        self.times = times - times[0]
        self.positions = positions
        self.turns = Slerp(self.times, Rotation.from_quat(quats[:, [1, 2, 3, 0]]))

    def at(self, t):
        """(positions (T, 3), rotations (T, 3, 3)) at sim times `t`."""
        local = np.clip(np.asarray(t, float) - self.start, 0.0, self.times[-1])
        positions = np.stack([np.interp(local, self.times, self.positions[:, k])
                              for k in range(3)], axis=-1)
        return positions, self.turns(local).as_matrix()


class Planner:
    def __init__(self, task, threads=None, seed=0):
        self.task = task
        model = task.model
        self.weld = model.equality(WELD).id
        self.symmetries = np.array(box_symmetries(model.geom("object").size * 2000))
        self.steps = round(HORIZON / model.opt.timestep)
        self.knot_times = np.linspace(0.0, HORIZON, KNOTS)
        self.step_times = np.arange(1, self.steps + 1) * model.opt.timestep
        self.low, self.high = model.actuator_ctrlrange.T
        self.rng = np.random.default_rng(seed)
        self.nominal = None
        self.datas = [mujoco.MjData(model) for _ in range(threads or max(1, os.cpu_count() - 2))]
        mount_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(mount_rotation, task.mount[1])
        self.mount_rotation = mount_rotation.reshape(3, 3)

    # ------------------------------------------------------------ the grip
    def grasp(self, progress=lambda done, total, text="": None):
        """Servo targets (nu,) that hold the object where the task starts,
        and how well: {"targets", "slip_m", "turn_rad", "gaps_m"} of the best."""
        model, data = self.task.model, self.task.data
        state = self._state(data)
        start_position, start_rotation = self.task.object_pose(data.qpos)
        mean, sigma = (self.low + self.high) / 2, (self.high - self.low) / 4
        best = None
        for round_ in range(GRASP_ROUNDS):
            targets = np.clip(mean + sigma * self.rng.standard_normal((GRASP_SAMPLES, model.nu)),
                              self.low, self.high)
            targets[0] = mean
            held, sensors = self._run(np.tile(state, (GRASP_SAMPLES, 1)), targets, CLOSE,
                                      weld=True)
            free, _ = self._run(held[:, -1], targets, LET_GO, weld=False)
            positions, rotations = self._object_poses(free)
            slip = np.linalg.norm(positions - start_position, axis=-1).mean(axis=1)
            turn = self.turn(start_rotation, rotations[:, -1])
            gaps = np.maximum(sensors[:, -1, :len(self.task.tips)], 0.0)
            costs = ((slip / POSITION_SCALE) ** 2 + (turn / TURN_SCALE) ** 2
                     + ((gaps / GAP_SCALE) ** 2).sum(axis=1))
            order = np.argsort(costs)
            elite = targets[order[:max(2, int(GRASP_ELITE * GRASP_SAMPLES))]]
            mean, sigma = elite.mean(axis=0), np.maximum(elite.std(axis=0), GRASP_MIN_SIGMA)
            top = order[0]
            if best is None or costs[top] < best[0]:
                best = (costs[top], targets[top], slip[top], turn[top], gaps[top])
            progress(round_ + 1, GRASP_ROUNDS,
                     f"grip: slips {1000 * best[2]:.0f} mm, turns {np.degrees(best[3]):.0f} deg")
        return {"targets": best[1], "slip_m": float(best[2]), "turn_rad": float(best[3]),
                "gaps_m": best[4]}

    # ------------------------------------------------------------ one plan
    def plan(self, data, goal, since=PERIOD):
        """The best motion from `data`'s state: knot targets (KNOTS, nu).
        It starts from the last plan, moved on `since` seconds (the time
        since it was made), or from holding `data.ctrl` when there is none."""
        if self.nominal is None:
            self.nominal = np.tile(data.ctrl, (KNOTS, 1))
        else:
            self.nominal = np.stack([np.interp(self.knot_times + since, self.knot_times,
                                               self.nominal[:, j])
                                     for j in range(self.nominal.shape[1])], axis=1)
        samples = self.nominal + self.rng.normal(0.0, SIGMA, (SAMPLES,) + self.nominal.shape)
        samples[0] = self.nominal
        samples = np.clip(samples, self.low, self.high)
        states, sensors = self._run(np.tile(self._state(data), (SAMPLES, 1)),
                                    self._spread(samples), weld=False)
        costs = self._costs(states, sensors, goal, data.time, samples)
        self.nominal = samples[int(np.argmin(costs))]
        return self.nominal, float(costs.min())

    def control(self, since):
        """The current plan's targets `since` seconds after it was made."""
        return np.array([np.interp(since, self.knot_times, self.nominal[:, j])
                         for j in range(self.nominal.shape[1])])

    def _spread(self, samples):
        """Knot targets -> per-step controls (SAMPLES, steps, nu)."""
        weights = np.clip(np.interp(self.step_times, self.knot_times,
                                    np.arange(KNOTS)), 0, KNOTS - 1)
        below = np.floor(weights).astype(int)
        above = np.minimum(below + 1, KNOTS - 1)
        share = (weights - below)[None, :, None]
        return samples[:, below] * (1 - share) + samples[:, above] * share

    def _costs(self, states, sensors, goal, now, samples):
        positions, rotations = self._object_poses(states)
        target_positions, target_rotations = goal.at(now + self.step_times)
        position_error = np.linalg.norm(positions - target_positions, axis=-1)
        turn_error = self.turn(target_rotations, rotations)
        gaps = np.maximum(sensors[..., :len(self.task.tips)], 0.0)
        step_cost = ((position_error / POSITION_SCALE) ** 2 + (turn_error / TURN_SCALE) ** 2
                     + DROP_COST * (position_error > DROP_M)
                     + ((gaps / GAP_SCALE) ** 2).sum(axis=-1))
        effort = EFFORT * (np.diff(samples, axis=1) ** 2).sum(axis=(1, 2))
        return step_cost.mean(axis=1) + effort

    # ------------------------------------------------------------ closed loop
    def track(self, goal, duration=None, grip=None,
              progress=lambda done, total, text="": None):
        """Grip (searched unless given), hold until the goal starts, then run
        the sim under the planner to `duration` (default: the goal's end).
        Returns the grip and what happened, a row every PERIOD."""
        task, data, model = self.task, self.task.data, self.task.model
        if grip is None:
            grip = self.grasp(progress)
        end = goal.end if duration is None else duration
        per_plan = round(PERIOD / model.opt.timestep)
        plans = int(np.ceil(end / PERIOD))
        rows = {"time": [], "ctrl": [], "qpos": [], "position": [], "rotation": [],
                "position_error": [], "turn_error": [], "gaps": [], "cost": []}
        self.nominal = np.tile(grip["targets"], (KNOTS, 1))
        for plan in range(plans):
            if data.time < goal.start:              # held while the grip closes
                cost = np.nan
                for _ in range(per_plan):
                    data.ctrl[:] = grip["targets"]
                    mujoco.mj_step(model, data)
                if data.time >= goal.start:
                    task.release(data)
            else:
                _, cost = self.plan(data, goal)
                for step in range(per_plan):
                    data.ctrl[:] = self.control(step * model.opt.timestep)
                    mujoco.mj_step(model, data)
            position, rotation = task.object_pose(data.qpos)
            target_position, target_rotation = goal.at([data.time])
            rows["time"].append(data.time)
            rows["ctrl"].append(data.ctrl.copy())
            rows["qpos"].append(data.qpos[:model.nu].copy())
            rows["position"].append(position)
            rows["rotation"].append(rotation)
            rows["position_error"].append(float(np.linalg.norm(position - target_position[0])))
            rows["turn_error"].append(float(self.turn(target_rotation[0], rotation)))
            rows["gaps"].append(data.sensordata[:len(task.tips)].copy())
            rows["cost"].append(cost)
            progress(plan + 1, plans, f"{data.time:.2f} s of {plans * PERIOD:.2f}")
        return grip, {key: np.array(value) for key, value in rows.items()}

    # ------------------------------------------------------------ helpers
    def _state(self, data):
        state = np.zeros(mujoco.mj_stateSize(self.task.model, FULL))
        mujoco.mj_getState(self.task.model, data, state, FULL)
        return state

    def _run(self, states, controls, seconds=None, weld=False):
        """Batched rollouts from `states` (n, nstate): `controls` either per
        step (n, steps, nu) or held (n, nu) for `seconds`. The weld is on in
        them or not: rollouts take that from the model, not the state.
        Returns (states, sensordata), each per step."""
        model = self.task.model
        if controls.ndim == 2:
            controls = np.repeat(controls[:, None], round(seconds / model.opt.timestep), axis=1)
        model.eq_active0[self.weld] = int(weld)
        try:
            return rollout.rollout(model, self.datas, states, controls, persistent_pool=True)
        finally:
            model.eq_active0[self.weld] = 0

    def _object_poses(self, states):
        """The object's positions (..., 3) and rotations (..., 3, 3) in the
        mount frame, from rollout states (..., nstate)."""
        at = 1 + self.task.object_qpos                 # a state starts with its time
        positions = (states[..., at:at + 3] - self.task.mount[0]) @ self.mount_rotation
        rotations = self.mount_rotation.T @ _quat_matrices(states[..., at + 3:at + 7])
        return positions, rotations

    def turn(self, targets, rotations):
        """Angle (rad) from each target rotation to each rotation, the least
        over the object's symmetries; both (..., 3, 3), broadcast."""
        relative = np.swapaxes(targets, -1, -2) @ rotations
        traces = np.einsum("...ij,sji->...s", relative, self.symmetries).max(axis=-1)
        return np.arccos(np.clip((traces - 1) / 2, -1, 1))


def _quat_matrices(quats):
    """Rotation matrices (..., 3, 3) from MuJoCo quaternions (..., 4) w x y z."""
    w, x, y, z = np.moveaxis(quats / np.linalg.norm(quats, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
