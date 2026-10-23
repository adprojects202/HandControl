"""Hand motion on an NVIDIA GPU: the robot hand alone (no object) learning
to take the shape of a hand as it moves (handmotion.py) - thousands of hand
sims stepped at once by MuJoCo Warp (gpu.Worlds), PPO (rsl_rl) on the GPU.

    env = GpuHandEnv(bundle, envs=8192)           # rsl_rl's VecEnv
    train(bundle, folder, envs, iterations, config, writer)
    results = evaluate(bundle, folder, checkpoint, config)

An episode follows one clip of the hand library: each step (1/RATE s) the
action moves the servo targets (at most ACTION_SCALE rad), then the hand sim
runs to the next frame, and the reward is how close the robot's shape is to
the clip's shape there:

    exp(-segment angle / ANGLE_SCALE)          each finger's segments' mean
                                               angle to the person's: how
                                               each joint bends
    + TIP_WEIGHT exp(-tip error / TIP_SCALE)   the fingertips, palm lengths
    + PINCH_WEIGHT exp(-pinch error / PINCH_SCALE)   thumb tip to each tip
    - TARGET_COST |target change|^2            moving at all costs a little
    - SMOOTH_COST |change of target change|^2  jerking costs more

A random clip starts in its first pose (or, one in REST_SHARE, at rest); a
recorded one at rest. Physics and observations are randomized as the
tracker's are (env.py): servo strength, the target arriving late, joint
noise.

Observations. "policy": joints, the joints one step before, the servo
targets, the robot's shape now (FEATURES), and the shape to take
FUTURE_STEPS steps ahead, each less the shape now. "privileged" (the critic
only): joint speeds and the servo strength drawn.

The bundle (handmotion.export_bundle, made on the Mac): hand_library.npz,
the hand sim as MJCF (hand.xml) and bundle.json - where the robot's 16
points sit (body names, points in the bodies' frames) and its palm frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from . import env as cpu
from .gpu import LINESEARCH_ITERATIONS, SOLVER_ITERATIONS, Worlds
from .handmotion import FEATURES, LIBRARY, RATE, load, shape

ACTION_SCALE = cpu.ACTION_SCALE
FUTURE_STEPS = (1, 3, 6)
ANGLE_SCALE = 0.3                # rad
TIP_SCALE = 0.1                  # palm lengths
PINCH_SCALE = 0.08               # palm lengths
TIP_WEIGHT = 0.5
PINCH_WEIGHT = 0.5
TARGET_COST = 0.5
SMOOTH_COST = 2.0
REST_SHARE = 0.2
SEGMENTS = 12
SETTINGS = ("ACTION_SCALE", "FUTURE_STEPS", "ANGLE_SCALE", "TIP_SCALE", "PINCH_SCALE",
            "TIP_WEIGHT", "PINCH_WEIGHT", "TARGET_COST", "SMOOTH_COST", "REST_SHARE")


class GpuHandEnv(VecEnv):
    def __init__(self, bundle, envs=8192, clips=None, randomize=True, seed=0, cycle=False,
                 device="cuda:0", iterations=SOLVER_ITERATIONS,
                 linesearch=LINESEARCH_ITERATIONS):
        """`clips`: library indices to draw episodes from (default: all);
        drawn at random, or with `cycle` env i always follows clips[i %
        len(clips)]. Without `randomize`, nominal physics, no noise or lag,
        every start as the clip's (for evaluation)."""
        bundle = Path(bundle)
        library = load(bundle / LIBRARY.name)
        meta = json.loads((bundle / "bundle.json").read_text())
        self.library = library
        self.clips = np.arange(len(library["length"])) if clips is None else np.asarray(clips)
        self.randomize, self.cycle = randomize, cycle
        self.worlds = Worlds((bundle / "hand.xml").read_text(), envs, device, seed, randomize,
                             None, iterations, linesearch)
        self.device, self.generator = self.worlds.device, self.worlds.generator
        mjm = self.mjm = self.worlds.mjm

        self.num_envs, self.num_actions = envs, mjm.nu
        self.max_episode_length = int(library["length"].max())
        self.episode_length_buf = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.cfg = {"envs": envs, "rate": RATE, "randomize": randomize,
                    "clips": len(self.clips), "device": str(self.device)}

        tensor = lambda array, dtype=torch.float32: torch.as_tensor(  # noqa: E731
            np.asarray(array), dtype=dtype, device=self.device)
        self.substeps = round(1.0 / (RATE * mjm.opt.timestep))
        self.low, self.high = tensor(mjm.actuator_ctrlrange[:, 0]), tensor(mjm.actuator_ctrlrange[:, 1])
        self.joint_qpos = tensor(mjm.jnt_qposadr[mjm.actuator_trnid[:, 0]], torch.long)
        self.joint_dof = tensor(mjm.jnt_dofadr[mjm.actuator_trnid[:, 0]], torch.long)
        hinges = [mjm.joint(name).id for name in meta["joints"]]
        self.start_qpos = tensor(mjm.jnt_qposadr[hinges], torch.long)
        self.bodies = tensor([mjm.body(name).id for name in meta["bodies"]], torch.long)
        self.local = tensor(meta["points"])
        self.wrist, self.axes = tensor(meta["wrist"]), tensor(meta["axes"])
        self.palm_length = float(meta["palm_length"])
        self.features = tensor(library["features"])
        self.length = tensor(library["length"], torch.long)
        start = np.asarray(library["start"], np.float32)
        self.has_start = tensor(np.isfinite(start).all(axis=1), torch.bool)
        self.start = tensor(np.nan_to_num(start))
        self.future = tensor(FUTURE_STEPS, torch.long)
        self.clip_choices = tensor(self.clips, torch.long)

        worlds = self.worlds
        self.qpos, self.qvel, self.ctrl = worlds.qpos, worlds.qvel, worlds.ctrl
        self.time, self.warmstart = worlds.time, worlds.warmstart
        self.targets = torch.zeros((envs, mjm.nu), device=self.device)
        self.change = torch.zeros_like(self.targets)
        self.previous_joints = torch.zeros_like(self.targets)
        self.clip = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.restart()

    # ------------------------------------------------------------ VecEnv
    def get_observations(self):
        return self._obs

    def step(self, actions):
        action = torch.clamp(actions.to(self.device, torch.float32), -1.0, 1.0)
        previous = self.targets
        self.targets = torch.clamp(previous + ACTION_SCALE * action, self.low, self.high)
        change = self.targets - previous
        if self.randomize:                       # the new target arrives late
            top = round(cpu.LAG_S * RATE * self.substeps) + 1
            lag = torch.randint(0, top, (self.num_envs,), generator=self.generator,
                                device=self.device)
        else:
            lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_joints = self.qpos[:, self.joint_qpos].clone()
        for substep in range(self.substeps):
            self.ctrl.copy_(torch.where((substep < lag)[:, None], previous, self.targets))
            self.worlds.step()
        unstable = self.worlds.unstable()
        if unstable.any():                       # ends: kept finite until reset
            self.qpos[unstable] = 0.0
            self.qvel[unstable] = 0.0
        self.worlds.kinematics()
        self.steps += 1
        self.episode_length_buf += 1
        self.worlds.counted()

        angle, tip, pinch = errors(self.robot_shape(), self._goal())
        reward = (torch.exp(-angle / ANGLE_SCALE) + TIP_WEIGHT * torch.exp(-tip / TIP_SCALE)
                  + PINCH_WEIGHT * torch.exp(-pinch / PINCH_SCALE)
                  - TARGET_COST * (change ** 2).sum(dim=1)
                  - SMOOTH_COST * ((change - self.change) ** 2).sum(dim=1))
        self.change = change
        timed_out = self.steps >= self.length[self.clip] - 1
        log = {"/hand/angle_deg": torch.rad2deg(angle).mean(),
               "/hand/tip_mm": 1000 * self.palm_length * tip.mean(),
               "/hand/pinch_mm": 1000 * self.palm_length * pinch.mean(),
               "/hand/unstable": unstable.float().sum()}
        rows = torch.stack([angle, tip, pinch], dim=1)
        done = timed_out | unstable
        if done.any():
            self._reset(torch.nonzero(done).squeeze(1))
            self.worlds.kinematics()
        self._obs = self._observe()
        return self._obs, reward, done, {"time_outs": timed_out, "log": log, "errors": rows}

    # ------------------------------------------------------------ episodes
    def restart(self):
        self._reset(torch.arange(self.num_envs, device=self.device))
        self.worlds.kinematics()
        self._obs = self._observe()
        return self._obs

    def _reset(self, envs):
        if self.cycle:
            clip = self.clip_choices[envs % len(self.clip_choices)]
        else:
            pick = torch.randint(0, len(self.clip_choices), (len(envs),),
                                 generator=self.generator, device=self.device)
            clip = self.clip_choices[pick]
        start = self.has_start[clip]
        if self.randomize:
            rest = torch.rand(len(envs), generator=self.generator, device=self.device) < REST_SHARE
            start = start & ~rest
        joints = torch.where(start[:, None], self.start[clip], 0.0)
        self.clip[envs] = clip
        self.steps[envs] = 0
        self.episode_length_buf[envs] = 0
        qpos = torch.zeros((len(envs), self.qpos.shape[1]), device=self.device)
        qpos[:, self.start_qpos] = joints
        self.qpos[envs] = qpos
        self.qvel[envs] = 0.0
        self.warmstart[envs] = 0.0
        self.time[envs] = 0.0
        self.targets[envs] = qpos[:, self.joint_qpos]
        self.ctrl[envs] = qpos[:, self.joint_qpos]
        self.change[envs] = 0.0
        self.previous_joints[envs] = qpos[:, self.joint_qpos]

    def _goal(self, offset=0):
        frame = torch.minimum(self.steps + offset, self.length[self.clip] - 1)
        return self.features[self.clip, frame]

    def robot_shape(self):
        """The robot's shapes now (envs, FEATURES), from where the links are."""
        rotations = self.worlds.xmat[:, self.bodies].reshape(self.num_envs, -1, 3, 3)
        world = self.worlds.xpos[:, self.bodies] + torch.einsum("npij,pj->npi", rotations,
                                                                self.local)
        return shape((world - self.wrist) @ self.axes / self.palm_length)

    # ------------------------------------------------------------ observing
    def _observe(self):
        joints = self.qpos[:, self.joint_qpos]
        previous = self.previous_joints
        if self.randomize:
            joints = joints + self._noise(joints.shape, cpu.JOINT_NOISE)
            previous = previous + self._noise(previous.shape, cpu.JOINT_NOISE)
        now = self.robot_shape()
        goals = torch.stack([self._goal(int(offset)) for offset in self.future], dim=1)
        policy = observation(joints, previous, self.targets, now, goals)
        privileged = torch.cat([self.qvel[:, self.joint_dof], self.worlds.scales[:, 2:]], dim=1)
        return TensorDict({"policy": policy.float(), "privileged": privileged.float()},
                          batch_size=[self.num_envs], device=self.device)

    def _noise(self, shape_, scale):
        return torch.randn(shape_, generator=self.generator, device=self.device) * scale


def observation(joints, previous, targets, now, goals):
    """The policy's input (n, 3 J + FEATURES (1 + len(FUTURE_STEPS))): what
    the robot reads, its shape now, and the shapes ahead (n, steps, FEATURES)
    less the shape now - the same wherever the hand runs."""
    return torch.cat([joints, previous, targets, now,
                      (goals - now[:, None]).reshape(len(now), -1)], dim=1)


def errors(robot, goal):
    """(segment angle rad, tip error, pinch error - palm lengths), each (n,),
    between shapes (n, FEATURES)."""
    directions = SEGMENTS * 3
    dot = (robot[:, :directions].reshape(-1, SEGMENTS, 3)
           * goal[:, :directions].reshape(-1, SEGMENTS, 3)).sum(dim=-1)
    angle = torch.arccos(torch.clamp(dot, -1.0, 1.0)).mean(dim=1)
    tips = directions + 12
    tip = torch.linalg.norm((robot[:, directions:tips] - goal[:, directions:tips]).reshape(-1, 4, 3),
                            dim=-1).mean(dim=1)
    pinch = (robot[:, tips:FEATURES] - goal[:, tips:FEATURES]).abs().mean(dim=1)
    return angle, tip, pinch


def train(bundle, folder, envs, iterations, config, writer_config, device="cuda:0",
          physics=None):
    """PPO on the GPU; checkpoints and progress into `folder`."""
    from rsl_rl.runners import OnPolicyRunner

    env = GpuHandEnv(bundle, envs, device=device, **(physics or {}))
    cfg = json.loads(json.dumps(config))
    cfg["logger"] = writer_config
    OnPolicyRunner(env, cfg, log_dir=str(folder), device=str(env.device)).learn(iterations)
    return env


def evaluate(bundle, folder, checkpoint, config, device="cuda:0", physics=None):
    """The trained policy on every clip once, nominal physics, no noise: per
    clip {clip, source, angle_deg, tip_mm, pinch_mm (medians), and the same
    for keeping still - the start pose held - for scale}."""
    from rsl_rl.runners import OnPolicyRunner

    library = load(Path(bundle) / LIBRARY.name)
    clips = np.arange(len(library["length"]))
    env = GpuHandEnv(bundle, envs=len(clips), clips=clips, randomize=False, cycle=True,
                     device=device, **(physics or {}))
    cfg = json.loads(json.dumps(config))
    cfg["logger"] = "tensorboard"
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=str(env.device))
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy()
    still = env.robot_shape()                       # every clip's start, before moving
    count = len(clips)
    finished = torch.zeros(count, dtype=torch.bool, device=env.device)
    rows, still_rows = [[] for _ in range(count)], [[] for _ in range(count)]
    obs = env.get_observations()
    with torch.inference_mode():
        while not finished.all():
            goal = env._goal(1)
            obs, _, done, extras = env.step(policy(obs))
            kept = torch.stack(errors(still, goal), dim=1).cpu().numpy()
            measured = extras["errors"].cpu().numpy()
            for i in torch.nonzero(~finished).squeeze(1).tolist():
                rows[i].append(measured[i])
                still_rows[i].append(kept[i])
            finished |= done
    scale = 1000 * env.palm_length
    results = []
    for i, clip in enumerate(clips):
        moved, kept = np.median(rows[i], axis=0), np.median(still_rows[i], axis=0)
        results.append({
            "clip": int(clip), "source": str(library["source"][clip]),
            "recorded": bool(library["source"][clip] != "random"),
            "angle_deg": float(np.degrees(moved[0])), "tip_mm": float(scale * moved[1]),
            "pinch_mm": float(scale * moved[2]),
            "still_angle_deg": float(np.degrees(kept[0])), "still_tip_mm": float(scale * kept[1]),
            "still_pinch_mm": float(scale * kept[2])})
    return results
