"""Hand motion on an NVIDIA GPU: the robot hand alone (no object) learning
to take the shape of a hand as it moves (handmotion.py) - thousands of hand
sims stepped at once by MuJoCo Warp (gpu.Worlds), PPO (rsl_rl) on the GPU.

    env = GpuHandEnv(bundle, envs=8192)           # rsl_rl's VecEnv
    train(bundle, folder, envs, iterations, config, writer)
    results = evaluate(bundle, folder, checkpoint, config)

An episode follows one clip of the hand library: each step (1/RATE s) the
action moves the servo targets (at most handmotion.ACTION_SCALE rad), then the hand sim
runs to the next frame, and the reward is how close the robot's shape is to
the clip's shape there:

    exp(-segment angle / ANGLE_SCALE)          each finger's segments' mean
                                               angle to the person's: how
                                               each joint bends
    + TIP_WEIGHT exp(-tip error / TIP_SCALE)   the fingertips, palm lengths
    + PINCH_WEIGHT exp(-pinch error / PINCH_SCALE)   thumb tip to each tip
    + TOUCH_WEIGHT exp(-touch error / TOUCH_SCALE)   the pinches the clip has
                                               touching (handmotion.touch_error):
                                               tips that should meet, meet
    - TARGET_COST |target change|^2            moving at all costs a little
    - SMOOTH_COST |change of target change|^2  jerking costs more

A generated clip starts in its first pose (or, one in REST_SHARE, at rest);
a recorded one at rest. Physics and observations are randomized as the
tracker's are (env.py): servo strength, the target arriving late, joint
noise.

Observations. "policy": handmotion.observation - the same wherever the
hand runs (hand_follow.py on the Mac). "privileged" (the critic only):
joint speeds and the servo strength drawn.

GOALS AS THE CAMERA GIVES THEM (bundle.json "goals": "camera skeletons";
learning.hand_goals). Live and on recordings the shapes to follow come from
the camera, which never sees a pinch as touching; so the goals the policy
SEES are the camera's, while its reward is still the robot pose meant:

    generated clips  the robot shape through the camera model
                     (camera_model.py, fitted to Mimic sessions): its mean,
                     plus its spread times BIAS_SHARE of a draw kept for the
                     episode (a hand off the same way for a while) and
                     JITTER_SHARE of a fresh one each step
    Mimic clips      your hand's shape as the camera saw it, against the
                     pose you were copying (mimic.clips) - appended to the
                     library's, drawn for MIMIC_SHARE of the episodes (by
                     count they would be under 1%); the held-out ones only
                     tested on
    recorded clips   not trained on: they have no robot pose to reward

Without "camera skeletons", every goal seen is the clip's own shape.

The bundle (hand_library.export_bundle, made on the Mac): hand_library.npz,
the hand sim as MJCF (hand.xml) and bundle.json - its joints, where the
robot's 16 points sit (body names, points in the bodies' frames) and its
palm frame. The library holds generated clips as key poses: their shapes
are made here, through this hand.xml's kinematics (hand_library.features).
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
from .camera_model import CameraModel
from .hand_library import LIBRARY, features, load
from .handmotion import (ACTION_SCALE, FEATURES, FUTURE_STEPS, RATE, Kinematics, errors,
                         observation, shape, touch_error)

ANGLE_SCALE = 0.3                # rad
TIP_SCALE = 0.1                  # palm lengths
PINCH_SCALE = 0.08               # palm lengths
TIP_WEIGHT = 0.5
PINCH_WEIGHT = 0.5
TOUCH_SCALE = 0.03               # palm lengths (~4 mm)
TOUCH_WEIGHT = 1.0
TARGET_COST = 0.5
SMOOTH_COST = 2.0
REST_SHARE = 0.2
BIAS_SHARE, JITTER_SHARE = 0.8, 0.6   # of the camera model's spread
MIMIC_JITTER = 0.01                   # palm lengths, on Mimic goals
CAMERA = "camera skeletons"
MIMIC_SHARE = 0.25                    # of episodes, on Mimic clips (with camera skeletons)
SETTINGS = ("ACTION_SCALE", "FUTURE_STEPS", "ANGLE_SCALE", "TIP_SCALE", "PINCH_SCALE",
            "TIP_WEIGHT", "PINCH_WEIGHT", "TOUCH_SCALE", "TOUCH_WEIGHT", "TARGET_COST",
            "SMOOTH_COST", "REST_SHARE", "BIAS_SHARE", "JITTER_SHARE", "MIMIC_JITTER",
            "MIMIC_SHARE")


class GpuHandEnv(VecEnv):
    def __init__(self, bundle, envs=8192, clips=None, randomize=True, seed=0, cycle=False,
                 device="cuda:0", iterations=SOLVER_ITERATIONS,
                 linesearch=LINESEARCH_ITERATIONS):
        """`clips`: clip indices to draw episodes from - the library's, then
        the Mimic clips' (default: all trained on: see GOALS); drawn at
        random, or with `cycle` env i always follows clips[i % len(clips)].
        Without `randomize`, nominal physics, no noise or lag, every start as
        the clip's (for evaluation)."""
        bundle = Path(bundle)
        library = load(bundle / LIBRARY.name)
        meta = json.loads((bundle / "bundle.json").read_text())
        self.library = library
        self.camera_goals = meta.get("goals") == CAMERA
        mimic = (dict(np.load(bundle / "mimic_clips.npz")) if self.camera_goals else
                 {"first": np.zeros(1, int), "held_out": np.zeros(0, bool)})
        self.mimic = mimic
        self.generated = len(library["length"])
        mimic_lengths = np.diff(mimic["first"])
        lengths = np.concatenate([library["length"], mimic_lengths])
        if clips is None:
            trained = ~library["recorded"] if self.camera_goals else np.ones(self.generated, bool)
            clips = np.concatenate([np.flatnonzero(trained),
                                    self.generated + np.flatnonzero(~mimic["held_out"])])
        self.clips = np.asarray(clips)
        self.randomize, self.cycle = randomize, cycle
        self.worlds = Worlds((bundle / "hand.xml").read_text(), envs, device, seed, randomize,
                             None, iterations, linesearch)
        self.device, self.generator = self.worlds.device, self.worlds.generator
        mjm = self.mjm = self.worlds.mjm

        self.num_envs, self.num_actions = envs, mjm.nu
        self.max_episode_length = int(lengths.max())
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
        chain = [(mjm.body(name).id, np.array(point))
                 for name, point in zip(meta["bodies"], meta["points"])]
        palm = (np.array(meta["wrist"]), np.array(meta["axes"]), self.palm_length)
        shapes = features(library, Kinematics(mjm, chain, palm, meta["joints"]))
        longest = int(lengths.max())
        self.features = tensor(np.concatenate([
            _padded(shapes, longest), _clips(mimic.get("target"), mimic["first"], longest)]))
        self.seen = tensor(_clips(mimic.get("seen"), mimic["first"], longest))
        self.length = tensor(lengths, torch.long)
        self.camera = (CameraModel(dict(np.load(bundle / "camera_model.npz")), self.device)
                       if self.camera_goals else None)
        start = np.concatenate([np.asarray(library["start"], np.float32),
                                np.full((len(mimic_lengths), mjm.nq), np.nan, np.float32)])
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
        self.bias = torch.zeros((envs, FEATURES), device=self.device)   # the camera's, per episode
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

        robot, goal = self.robot_shape(), self._goal()
        angle, tip, pinch = errors(robot, goal)
        touch, touching = touch_error(robot, goal)
        reward = (torch.exp(-angle / ANGLE_SCALE) + TIP_WEIGHT * torch.exp(-tip / TIP_SCALE)
                  + PINCH_WEIGHT * torch.exp(-pinch / PINCH_SCALE)
                  + TOUCH_WEIGHT * torch.exp(-touch / TOUCH_SCALE)
                  - TARGET_COST * (change ** 2).sum(dim=1)
                  - SMOOTH_COST * ((change - self.change) ** 2).sum(dim=1))
        self.change = change
        timed_out = self.steps >= self.length[self.clip] - 1
        log = {"/hand/angle_deg": torch.rad2deg(angle).mean(),
               "/hand/tip_mm": 1000 * self.palm_length * tip.mean(),
               "/hand/pinch_mm": 1000 * self.palm_length * pinch.mean(),
               "/hand/touch_mm": 1000 * self.palm_length * touch.sum()
               / torch.clamp(touching.sum(), min=1),
               "/hand/unstable": unstable.float().sum()}
        rows = torch.stack([angle, tip, pinch, touch, touching.float()], dim=1)
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
            clip = self._draw(self.clip_choices[self.clip_choices < self.generated], len(envs))
            mimic = self.clip_choices[self.clip_choices >= self.generated]
            if len(mimic):
                chosen = torch.rand(len(envs), generator=self.generator,
                                    device=self.device) < MIMIC_SHARE
                clip = torch.where(chosen, self._draw(mimic, len(envs)), clip)
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
        self.bias[envs] = self._noise((len(envs), FEATURES), 1.0)

    def _draw(self, choices, count):
        """`count` clips drawn from `choices` at random."""
        pick = torch.randint(0, len(choices), (count,), generator=self.generator,
                             device=self.device)
        return choices[pick]

    def _goal(self, offset=0):
        """The shape the reward asks for, `offset` steps on."""
        frame = torch.minimum(self.steps + offset, self.length[self.clip] - 1)
        return self.features[self.clip, frame]

    def _seen(self, offset=0):
        """The goal as the policy sees it, `offset` steps on (see GOALS)."""
        goal = self._goal(offset)
        if not self.camera_goals:
            return goal
        mean, spread = self.camera(goal)
        if self.randomize:
            mean = mean + spread * (BIAS_SHARE * self.bias
                                    + JITTER_SHARE * self._noise(goal.shape, 1.0))
        mimic = self.clip >= self.generated
        if not mimic.any():
            return mean
        frame = torch.minimum(self.steps + offset, self.length[self.clip] - 1)
        recorded = self.seen[torch.clamp(self.clip - self.generated, min=0), frame]
        if self.randomize:
            recorded = recorded + self._noise(goal.shape, MIMIC_JITTER)
        return torch.where(mimic[:, None], recorded, mean)

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
        goals = torch.stack([self._seen(int(offset)) for offset in self.future], dim=1)
        policy = observation(joints, previous, self.targets, now, goals)
        privileged = torch.cat([self.qvel[:, self.joint_dof], self.worlds.scales[:, 2:]], dim=1)
        return TensorDict({"policy": policy.float(), "privileged": privileged.float()},
                          batch_size=[self.num_envs], device=self.device)

    def _noise(self, shape_, scale):
        return torch.randn(shape_, generator=self.generator, device=self.device) * scale


def _padded(shapes, length):
    """Clips (C, L, F) padded with their last to `length`."""
    return np.concatenate([shapes, np.repeat(shapes[:, -1:], length - shapes.shape[1], axis=1)],
                          axis=1)


def _clips(rows, first, length):
    """Concatenated clips (rows, first: each clip's first row) as (K, length,
    FEATURES), each padded with its last."""
    if rows is None:
        return np.zeros((0, length, FEATURES), np.float32)
    out = np.empty((len(first) - 1, length, FEATURES), np.float32)
    for k, (a, b) in enumerate(zip(first[:-1], first[1:])):
        out[k, :b - a], out[k, b - a:] = rows[a:b], rows[b - 1]
    return out


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
    """The trained policy on every clip once - the library's and the Mimic
    clips - nominal physics, no camera noise: per clip {clip, kind, source,
    recorded, held_out (a Mimic clip only tested on), angle_deg, tip_mm,
    pinch_mm, touch_mm (medians; touch_mm over the frames the clip has
    fingertips touching, None without any), and the same had the hand kept
    still - its start pose held - for scale}."""
    from rsl_rl.runners import OnPolicyRunner

    library = load(Path(bundle) / LIBRARY.name)
    meta = json.loads((Path(bundle) / "bundle.json").read_text())
    mimic = (dict(np.load(Path(bundle) / "mimic_clips.npz")) if meta.get("goals") == CAMERA
             else {"held_out": np.zeros(0, bool)})
    generated = len(library["length"])
    clips = np.arange(generated + len(mimic["held_out"]))
    env = GpuHandEnv(bundle, envs=len(clips), clips=clips, randomize=False, cycle=True,
                     device=device, **(physics or {}))
    cfg = json.loads(json.dumps(config))
    cfg["logger"] = "tensorboard"
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=str(env.device))
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy()
    still = env.robot_shape()                       # every clip's start, before moving
    finished = torch.zeros(len(clips), dtype=torch.bool, device=env.device)
    moved, kept, active = [], [], []                # per step: (clips, 5), (clips,)
    obs = env.get_observations()
    with torch.inference_mode():
        while not finished.all():
            goal = env._goal(1)
            obs, _, done, extras = env.step(policy(obs))
            touch, touching = touch_error(still, goal)
            kept.append(torch.stack([*errors(still, goal), touch, touching.float()], dim=1).cpu())
            moved.append(extras["errors"].cpu())
            active.append((~finished).cpu())
            finished |= done
    moved, kept = torch.stack(moved).numpy(), torch.stack(kept).numpy()
    active = torch.stack(active).numpy()
    scale = 1000 * env.palm_length
    results = []
    for i, clip in enumerate(clips):
        rows, still_rows = moved[active[:, i], i], kept[active[:, i], i]
        touched = rows[:, 4] > 0
        median, still_median = np.median(rows, axis=0), np.median(still_rows, axis=0)
        if clip < generated:
            kind, source = str(library["kind"][clip]), str(library["source"][clip])
            recorded, held_out = bool(library["recorded"][clip]), False
        else:
            kind = source = "mimic"
            recorded, held_out = False, bool(mimic["held_out"][clip - generated])
        results.append({
            "clip": int(clip), "kind": kind, "source": source, "recorded": recorded,
            "held_out": held_out,
            "angle_deg": float(np.degrees(median[0])), "tip_mm": float(scale * median[1]),
            "pinch_mm": float(scale * median[2]),
            "touch_mm": float(scale * np.median(rows[touched, 3])) if touched.any() else None,
            "still_angle_deg": float(np.degrees(still_median[0])),
            "still_tip_mm": float(scale * still_median[1]),
            "still_pinch_mm": float(scale * still_median[2]),
            "still_touch_mm": (float(scale * np.median(still_rows[touched, 3]))
                               if touched.any() else None)})
    return results
