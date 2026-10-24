"""The trained hand-motion policy (gpu_hand.py) on this Mac: following a
person's hand in the hand sim, on the CPU.

    run = follow(features, folder)      # features (T, FEATURES) at RATE

    env = CpuHandEnv(features)          # or, live: one shape, aimed again each step
    policy = load_policy(env, folder)
    env.aim(shape); obs, *_ = env.step(policy(env.get_observations()))

The policy sees what it saw on the GPU (handmotion.observation): the joints,
the joints a step before, the servo targets, the robot's shape now, and the
shapes FUTURE_STEPS ahead less the shape now - nominal physics, no noise.
The hand starts at rest. Live (learning/live.py) only the shape now is
known: aimed at it, every shape ahead is that one.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from ..sim.task import hand_spec
from .handmotion import (ACTION_SCALE, FUTURE_STEPS, RATE, errors, observation, robot_chain,
                         robot_points, shape)


class CpuHandEnv(VecEnv):
    """One hand sim following `features` (T, FEATURES) at RATE - rsl_rl's
    VecEnv, so a trained policy loads onto it."""

    def __init__(self, features):
        self.model = hand_spec()[0].compile()
        self.data = mujoco.MjData(self.model)
        model = self.model
        self.chain, self.palm = robot_chain(model)
        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        self.num_envs, self.num_actions = 1, model.nu
        self.max_episode_length = len(features)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {"envs": 1, "rate": RATE}
        self.substeps = round(1.0 / (RATE * model.opt.timestep))
        self.low, self.high = model.actuator_ctrlrange.T
        self.joint_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.joint_dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.joint_names = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
        self.targets = np.zeros(model.nu)
        self.previous_joints = np.zeros(model.nu)
        self.steps = 0
        mujoco.mj_forward(model, self.data)
        self._obs = self._observe()

    def get_observations(self):
        return self._obs

    def step(self, actions):
        action = np.clip(actions.detach().cpu().numpy()[0].astype(float), -1.0, 1.0)
        self.targets = np.clip(self.targets + ACTION_SCALE * action, self.low, self.high)
        self.previous_joints = self.data.qpos[self.joint_qpos].copy()
        self.data.ctrl[:] = self.targets
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        mujoco.mj_kinematics(self.model, self.data)      # the links where qpos has them
        self.steps += 1
        self.episode_length_buf += 1
        done = torch.tensor([self.steps >= self.max_episode_length - 1])
        self._obs = self._observe()
        return self._obs, torch.zeros(1), done, {"time_outs": done, "log": {}}

    def aim(self, shape_):
        """Follow one shape (FEATURES,) from now on - every shape ahead the
        same - as live, where only the shape now is known."""
        self.features = torch.as_tensor(np.asarray(shape_), dtype=torch.float32)[None]
        self.max_episode_length = 1
        self._obs = self._observe()

    def joints(self):
        return self.data.qpos[self.joint_qpos].copy()

    def shape_now(self):
        """The robot's shape now (1, FEATURES)."""
        points = robot_points(self.model, self.data, self.chain, self.palm)
        return torch.as_tensor(shape(points), dtype=torch.float32)[None]

    def goal(self, offset=0):
        return self.features[min(self.steps + offset, self.max_episode_length - 1)][None]

    def _observe(self):
        tensor = lambda array: torch.as_tensor(array, dtype=torch.float32)[None]  # noqa: E731
        goals = torch.stack([self.goal(offset) for offset in FUTURE_STEPS], dim=1)
        policy = observation(tensor(self.joints()), tensor(self.previous_joints),
                             tensor(self.targets), self.shape_now(), goals)
        privileged = torch.cat([tensor(self.data.qvel[self.joint_dof]), torch.ones((1, 1))],
                               dim=1)
        return TensorDict({"policy": policy, "privileged": privileged}, batch_size=[1])


def load_policy(env, folder, checkpoint=None):
    """Run `folder`'s policy (its last checkpoint, or `checkpoint`), for
    `env`: observations in, actions out."""
    from rsl_rl.runners import OnPolicyRunner

    from .train import CONFIG, latest_checkpoint

    runner = OnPolicyRunner(env, copy.deepcopy(CONFIG), log_dir=None, device="cpu")
    # map_location: a run trained on the GPU saved its tensors there.
    runner.load(str(checkpoint or latest_checkpoint(folder)), load_cfg={"actor": True},
                strict=True, map_location="cpu")
    return runner.get_inference_policy()


def follow(features, folder, checkpoint=None):
    """The policy of run `folder` following `features` (T, FEATURES) at
    RATE from rest. Returns {joints (T, J) rad - frame 0 at rest - and
    joint_names; angle, tip, pinch (T,): the shape's errors (rad, palm
    lengths), and still_angle, still_tip, still_pinch: holding the rest
    pose's; palm_length m}."""
    env = CpuHandEnv(features)
    policy = load_policy(env, folder, checkpoint)
    rest = env.shape_now()
    joints, rows, still = [env.joints()], [errors(rest, env.goal())], [errors(rest, env.goal())]
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(len(features) - 1):
            obs, _, _, _ = env.step(policy(obs))
            joints.append(env.joints())
            rows.append(errors(env.shape_now(), env.goal()))
            still.append(errors(rest, env.goal()))
    stack = lambda measured, i: np.array([float(m[i][0]) for m in measured])  # noqa: E731
    return {"joints": np.array(joints), "joint_names": env.joint_names,
            "angle": stack(rows, 0), "tip": stack(rows, 1), "pinch": stack(rows, 2),
            "still_angle": stack(still, 0), "still_tip": stack(still, 1),
            "still_pinch": stack(still, 2), "palm_length": env.palm[2]}
