"""The tracker's environment: many copies of the task sim at once, each
following one motion from the library - for rsl_rl's PPO (its VecEnv).

    env = TrackEnv(library, envs=2048)
    obs = env.get_observations()
    obs, rewards, dones, extras = env.step(actions)

A step is 1/RATE s. The policy's action (-1..1 per joint) moves each servo
target by up to ACTION_SCALE rad; then the physics runs SUBSTEPS steps -
every env at once, on every core (mujoco.rollout). An episode is one motion:
it starts from its start's let-go state (the grip closed, the object just
released) and ends at the motion's last frame (a time-out), or once the
object is DROP_M from where the motion has it (a drop).

Observations. "policy" - what the real hand will have, noisy: the joint
angles now and a step ago, the servo targets, the object's position and
rotation (a camera's estimate), and the goal at each of FUTURE_S ahead as an
offset and a turn from the object, all in the hand mount's frame. The goal's
rotation is taken as its equivalent nearest the object's: a cube looks the
same 24 ways, so only the nearest needs reaching. "privileged" - the critic
only, in sim: the object's velocity, the fingertip gaps, and this env's
friction, mass and servo strength.

Reward per step: exp(-position error / POSITION_SCALE) + exp(-turn error /
TURN_SCALE), less TARGET_COST per rad^2 the targets moved and GAP_COST for
fingertips off the object (the mean gap over GAP_SCALE, at most 1); a drop
costs DROP_COST.

Randomized per env when `randomize` (VARIANTS models, drawn once): the
object's friction and mass, the servos' torque (and gain). Per step: sensor
noise, and a lag of up to LAG_S before a new target reaches the servos.
"""

from __future__ import annotations

import copy
import os

import mujoco
import numpy as np
import torch
from mujoco import rollout
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from ..sim.planner import _quat_matrices
from ..sim.symmetry import box_symmetries
from ..sim.task import OBJECT, Task
from .trajectories import RATE

ACTION_SCALE = 0.1             # rad per step at most
FUTURE_S = (0.05, 0.25, 0.5, 1.0)
POSITION_SCALE = 0.01          # m
TURN_SCALE = np.radians(12)
TARGET_COST = 0.5
GAP_COST, GAP_SCALE = 0.1, 0.01
DROP_M, DROP_COST = 0.05, 10.0
VARIANTS = 64
FRICTION_SCALE = (0.7, 1.4)
MASS_SCALE = (0.7, 1.3)
TORQUE_SCALE = (0.8, 1.2)
LAG_S = 0.02
JOINT_NOISE = 0.01             # rad
POSITION_NOISE = 0.002         # m
TURN_NOISE = np.radians(2)
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


class TrackEnv(VecEnv):
    def __init__(self, library, envs=2048, motions=None, randomize=True, seed=0,
                 threads=None, cycle=False):
        """`motions`: library indices to draw episodes from - by default every
        feasible generated one; drawn at random, or with `cycle` env i always
        runs motions[i % len(motions)]. Without `randomize`, nominal physics
        and no noise or lag (for evaluation)."""
        self.library = library
        self.motions = (np.flatnonzero(library["motion_feasible"] & ~library["motion_recorded"])
                        if motions is None else np.asarray(motions))
        if not len(self.motions):
            raise ValueError("no motions to train on")
        self.task = Task.build((library["start_position"][0], library["start_quat"][0]))
        model = self.task.model
        model.eq_active0[:] = 0                 # episodes start already let go
        self.randomize, self.cycle = randomize, cycle
        self.rng = np.random.default_rng(seed)
        variants, scales = self._variants(model)
        self.models = [variants[i % len(variants)] for i in range(envs)]
        self.scales = scales[np.arange(envs) % len(variants)]
        self.datas = [mujoco.MjData(model) for _ in range(threads or max(1, os.cpu_count() - 2))]

        self.num_envs, self.num_actions = envs, model.nu
        self.max_episode_length = int(library["motion_length"].max())
        self.episode_length_buf = torch.zeros(envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {"envs": envs, "rate": RATE, "randomize": randomize,
                    "motions": len(self.motions)}

        self.substeps = round(1.0 / (RATE * model.opt.timestep))
        self.low, self.high = model.actuator_ctrlrange.T
        self.joint_qpos = 1 + model.jnt_qposadr[model.actuator_trnid[:, 0]]   # in a state
        self.object_qpos = 1 + self.task.object_qpos
        self.object_qvel = 1 + model.nq + int(model.joint(OBJECT).dofadr[0])
        self.tips = len(self.task.tips)
        self.future = np.maximum(1, np.round(np.array(FUTURE_S) * RATE).astype(int))
        self.symmetries = np.array(box_symmetries(model.geom(OBJECT).size * 2000))
        mount = np.zeros(9)
        mujoco.mju_quat2Mat(mount, self.task.mount[1])
        self.mount_rotation = mount.reshape(3, 3)

        nstate = mujoco.mj_stateSize(model, FULL)
        self.state = np.zeros((envs, nstate))
        self.targets = np.zeros((envs, model.nu))
        self.previous_joints = np.zeros((envs, model.nu))
        self.motion = np.zeros(envs, int)
        self.steps = np.zeros(envs, int)
        self._states = np.empty((envs, self.substeps, nstate))
        self._sensors = np.empty((envs, self.substeps, model.nsensordata))
        self.gaps = np.zeros((envs, self.tips))
        self._reset(np.arange(envs))
        self._obs = self._observe()

    # ------------------------------------------------------------ VecEnv
    def get_observations(self):
        return self._obs

    def step(self, actions):
        action = np.clip(actions.detach().cpu().numpy().astype(float), -1.0, 1.0)
        previous = self.targets
        self.targets = np.clip(previous + ACTION_SCALE * action, self.low, self.high)
        control = np.repeat(self.targets[:, None], self.substeps, axis=1)
        if self.randomize:                       # the new target arrives late
            lag = self.rng.integers(0, round(LAG_S * RATE * self.substeps) + 1, self.num_envs)
            late = np.arange(self.substeps)[None, :] < lag[:, None]
            control[late] = np.repeat(previous[:, None], self.substeps, axis=1)[late]
        self.previous_joints = self.state[:, self.joint_qpos]
        rollout.rollout(self.models, self.datas, self.state, control, nstep=self.substeps,
                        skip_checks=True,
                        state=self._states, sensordata=self._sensors, persistent_pool=True)
        self.state = self._states[:, -1].copy()
        self.gaps = self._sensors[:, -1, :self.tips]
        self.steps += 1
        self.episode_length_buf += 1

        position_error, turn_error = self._errors()
        dropped = position_error > DROP_M
        timed_out = ~dropped & (self.steps >= self.library["motion_length"][self.motion] - 1)
        gap = np.clip(np.maximum(self.gaps, 0.0).mean(axis=1) / GAP_SCALE, 0.0, 1.0)
        reward = (np.exp(-position_error / POSITION_SCALE) + np.exp(-turn_error / TURN_SCALE)
                  - TARGET_COST * ((self.targets - previous) ** 2).sum(axis=1)
                  - GAP_COST * gap - DROP_COST * dropped)
        log = {"/track/position_mm": float(1000 * position_error.mean()),
               "/track/turn_deg": float(np.degrees(turn_error).mean()),
               "/track/drops_per_step": float(dropped.mean()),
               "/track/fingertip_gap_mm": float(1000 * np.maximum(self.gaps, 0).mean())}
        done = dropped | timed_out
        if done.any():
            self._reset(np.flatnonzero(done))
        self._obs = self._observe()
        return (self._obs, torch.as_tensor(reward, dtype=torch.float32),
                torch.as_tensor(done), {"time_outs": torch.as_tensor(timed_out), "log": log,
                                        "errors": np.stack([position_error, turn_error], 1)})

    # ------------------------------------------------------------ episodes
    def restart(self):
        """Every env back to the start of a motion (its own, with `cycle`)."""
        self._reset(np.arange(self.num_envs))
        self._obs = self._observe()
        return self._obs

    def _reset(self, envs):
        motion = (self.motions[envs % len(self.motions)] if self.cycle
                  else self.rng.choice(self.motions, len(envs)))
        start = self.library["motion_start"][motion]
        self.motion[envs] = motion
        self.steps[envs] = 0
        self.episode_length_buf[envs] = 0
        self.state[envs] = self.library["start_state"][start]
        self.targets[envs] = self.library["start_grip"][start]
        self.previous_joints[envs] = self.state[envs][:, self.joint_qpos]
        self.gaps[envs] = 0.0

    def _goal(self, offset=0):
        """The motion's (positions (N, 3), rotations (N, 3, 3)) `offset`
        frames after now, held at its last frame."""
        frame = np.minimum(self.steps + offset, self.library["motion_length"][self.motion] - 1)
        positions = self.library["motion_position"][self.motion, frame]
        quats = self.library["motion_quat"][self.motion, frame]
        return positions, _quat_matrices(quats)

    def _object(self, state=None):
        """The object's positions (N, 3) and rotations (N, 3, 3), mount frame."""
        state = self.state if state is None else state
        at = self.object_qpos
        positions = (state[:, at:at + 3] - self.task.mount[0]) @ self.mount_rotation
        rotations = self.mount_rotation.T @ _quat_matrices(state[:, at + 3:at + 7])
        return positions, rotations

    def _errors(self):
        positions, rotations = self._object()
        goal_positions, goal_rotations = self._goal()
        _, turn = self._nearest(rotations, goal_rotations[:, None])
        return np.linalg.norm(positions - goal_positions, axis=1), turn[:, 0]

    def _nearest(self, rotations, goals):
        """For goals (N, K, 3, 3): each one's equivalent nearest `rotations`
        (N, 3, 3) under the object's symmetries, and the angle to it."""
        relative = np.einsum("nji,nkjl->nkil", rotations, goals)
        traces = np.einsum("nkij,sji->nks", relative, self.symmetries)
        best = traces.argmax(axis=-1)
        angle = np.arccos(np.clip((np.take_along_axis(traces, best[..., None], -1)[..., 0] - 1)
                                  / 2, -1, 1))
        return goals @ self.symmetries[best], angle

    # ------------------------------------------------------------ observing
    def _observe(self):
        envs = self.num_envs
        joints = self.state[:, self.joint_qpos]
        previous = self.previous_joints
        positions, rotations = self._object()
        if self.randomize:
            joints = joints + self.rng.normal(0, JOINT_NOISE, joints.shape)
            previous = previous + self.rng.normal(0, JOINT_NOISE, previous.shape)
            positions = positions + self.rng.normal(0, POSITION_NOISE, positions.shape)
            wobble = self.rng.normal(0, TURN_NOISE / np.sqrt(3), (envs, 3))
            rotations = _rotvec_matrices(wobble) @ rotations
        goals = [self._goal(offset) for offset in self.future]
        goal_positions = np.stack([g[0] for g in goals], axis=1)            # (N, K, 3)
        goal_rotations = np.stack([g[1] for g in goals], axis=1)            # (N, K, 3, 3)
        nearest, _ = self._nearest(rotations, goal_rotations)
        turns = nearest @ np.swapaxes(rotations, -1, -2)[:, None]           # object -> goal
        policy = np.concatenate([
            joints, previous, self.targets, positions, _six(rotations),
            (goal_positions - positions[:, None]).reshape(envs, -1),
            _six(turns).reshape(envs, -1)], axis=1)
        velocity = self.state[:, self.object_qvel:self.object_qvel + 6]
        privileged = np.concatenate([velocity, self.gaps, self.scales], axis=1)
        return TensorDict({"policy": torch.as_tensor(policy, dtype=torch.float32),
                           "privileged": torch.as_tensor(privileged, dtype=torch.float32)},
                          batch_size=[envs])

    # ------------------------------------------------------------ physics
    def _variants(self, model):
        """VARIANTS copies of the model - friction, mass and servo strength
        drawn per copy - and their scales (VARIANTS, 3); one nominal copy
        without `randomize`."""
        if not self.randomize:
            return [model], np.ones((1, 3))
        geom, body = model.geom(OBJECT).id, model.body(OBJECT).id
        variants, scales = [], []
        for _ in range(VARIANTS):
            variant = copy.deepcopy(model)
            friction, mass, torque = (self.rng.uniform(*FRICTION_SCALE),
                                      self.rng.uniform(*MASS_SCALE),
                                      self.rng.uniform(*TORQUE_SCALE))
            variant.geom_friction[geom, 0] *= friction
            variant.body_mass[body] *= mass
            variant.body_inertia[body] *= mass
            variant.actuator_forcerange *= torque
            variant.actuator_gainprm[:, 0] *= torque          # kp
            variant.actuator_biasprm[:, 1] *= torque          # -kp
            variant.actuator_biasprm[:, 2] *= np.sqrt(torque)  # -kv: same damping ratio
            mujoco.mj_setConst(variant, mujoco.MjData(variant))
            variants.append(variant)
            scales.append((friction, mass, torque))
        return variants, np.array(scales)


def _six(rotations):
    """The first two columns of rotations (..., 3, 3), flattened: (..., 6)."""
    return np.concatenate([rotations[..., :, 0], rotations[..., :, 1]], axis=-1)


def _rotvec_matrices(vectors):
    """Rotation matrices (N, 3, 3) from rotation vectors (N, 3) (Rodrigues)."""
    angle = np.linalg.norm(vectors, axis=1, keepdims=True)
    axis = vectors / np.maximum(angle, 1e-12)
    x, y, z = axis.T
    zero = np.zeros_like(x)
    k = np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1),
                  np.stack([-y, x, zero], -1)], -2)
    angle = angle[:, :, None]
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
