"""The physics policy (step 4) on an NVIDIA GPU: the robot hand doing what a
person's hand does, from what the reader makes of it - keeping hold of
GRAB's objects, and closing the thumb on a fingertip where the person does -
thousands of hand sims, most with an object each, stepped at once by MuJoCo
Warp (gpu.Worlds), PPO (rsl_rl) on the GPU.

    env = GpuHoldEnv(bundle, envs=8192)           # rsl_rl's VecEnv
    train(bundle, folder, envs, iterations, config, writer)
    results = evaluate(bundle, folder, checkpoint, config)

An episode is one of holds.py's: a person holding an object in one hand,
on the robot hand (the object where it was in their palm, gravity turning
as their palm did), from a start where the fingers have closed on it. Each
step (1/RATE s) the policy's action is a correction to the reader's pose
for that frame - the servo targets are the reader's joints plus up to
RESIDUAL_RAD - and the hand sim runs to the next frame. Every world has
its own object (hold_scene.py: the object's mesh, bounds and inertia in
MuJoCo Warp's batched model fields), mass and friction, kept for the run;
its episodes are those with that object.

The object starts supported: a force that carries its weight and holds it
where it should be (a spring and damper on its position and turn, scaled
by its mass and inertia) fades out over ASSIST_S - as a person hands an
object over and lets go once the hand has it. The fingers pressing where
the start's closing left them do not hold (holds.py measures it); a grip
that does is the policy's to find before the support is gone.

Its reward, each step:

    HOLD_WEIGHT                                  still holding
    + exp(-object position error / POSITION_SCALE)    where the person's
    + ROTATION_WEIGHT exp(-turn error / ROTATION_SCALE)   object was
    + JOINT_WEIGHT exp(-joint error / JOINT_SCALE)    the person's motion
                                                 (retargeted)
    + CONTACT_WEIGHT  of the fingers on the object in GRAB, the share on it
    - RESIDUAL_COST |correction|^2 - CHANGE_COST |target change|^2

An object further than DROP_M from where it should be is dropped: the
episode ends.

FREE_SHARE of the worlds play holds.py's free clips instead: no object
(its geom points at no mesh, held out of the way), the person's hand
moving through FreiHAND's poses, half of them the thumb on a fingertip -
which the reader means (its contacts) but whose thumb it leaves short.
Their reward:

    FREE_JOINT_WEIGHT exp(-joint error / JOINT_SCALE)
    + PINCH_WEIGHT exp(-pinch error / PINCH_SCALE)     the thumb tip's
                                                 distance to each fingertip
                                                 as the person's (robot's
                                                 retargeted), palm lengths
    + TOUCH_WEIGHT exp(-touch error / TOUCH_SCALE)    those the person has
                                                 touching (under TOUCH_PL),
                                                 where there are any
    - the same costs Physics, the target arriving late and the joints read are
randomized as the tracker's are (env.py).

Observations. "policy" - what the real hand has: its joints (read with
noise) now and a step before, its servo targets, the reader's joints for
the last READ_FRAMES frames and the contacts it means now, and gravity in
the palm frame (the mount's, on the real hand) - never whether there is an
object. "privileged" (the critic only): the object's pose and speed
against where it should be, the true joints and pinches, the support left,
each hand body's gap to the object, whether the clip is free, the physics
drawn.

The bundle (holds.export_bundle, made on the Mac): holds.npz - the scene
as MJCF among its arrays - and bundle.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from . import env as cpu
from . import hold_scene
from .gpu import LINESEARCH_ITERATIONS, SOLVER_ITERATIONS, Worlds
from .handmotion import TOUCH_PL

RATE = 20
RESIDUAL_RAD = 0.5
READ_FRAMES = 3
ASSIST_S = 0.5
ASSIST_STIFFNESS = 3000.0        # 1/s^2: per unit mass (and inertia) - 55 rad/s, stiffer than the fingers
ASSIST_DAMPING = 110.0           # 1/s: critical
DROP_M = 0.05
HOLD_WEIGHT = 1.0
POSITION_SCALE = 0.02            # m
ROTATION_WEIGHT = 0.5
ROTATION_SCALE = 0.5             # rad
JOINT_WEIGHT = 0.5
JOINT_SCALE = 0.3                # rad, the mean over joints
CONTACT_WEIGHT = 0.5
RESIDUAL_COST = 0.05
CHANGE_COST = 0.5
MASS_KG = (0.02, 0.10)
FREE_SHARE = 0.5                 # of the worlds, on free clips
FREE_JOINT_WEIGHT = 1.0
PINCH_WEIGHT = 1.0
PINCH_SCALE = 0.08               # palm lengths, the mean over the three
TOUCH_WEIGHT = 1.0
TOUCH_SCALE = 0.03               # palm lengths (~4 mm)
TIPS = {"thumb": 3, "index": 7, "middle": 11, "ring": 15}   # in robot_chain's order
SETTINGS = ("RESIDUAL_RAD", "READ_FRAMES", "ASSIST_S", "ASSIST_STIFFNESS", "ASSIST_DAMPING",
            "DROP_M", "HOLD_WEIGHT", "POSITION_SCALE", "ROTATION_WEIGHT", "ROTATION_SCALE",
            "JOINT_WEIGHT", "JOINT_SCALE", "CONTACT_WEIGHT", "RESIDUAL_COST", "CHANGE_COST",
            "MASS_KG", "FREE_SHARE", "FREE_JOINT_WEIGHT", "PINCH_WEIGHT", "PINCH_SCALE",
            "TOUCH_WEIGHT", "TOUCH_SCALE")
MODES = ("policy", "reader", "reader and grip")   # what moves the hand (evaluate)


class GpuHoldEnv(VecEnv):
    def __init__(self, bundle, envs=8192, episodes=None, randomize=True, seed=0, cycle=False,
                 device="cuda:0", iterations=SOLVER_ITERATIONS,
                 linesearch=LINESEARCH_ITERATIONS):
        """`episodes`: which of holds.npz's to play (default: the reachable
        holds of the people fitted on and the free clips of FreiHAND's
        training poses); with `cycle` world i plays episodes[i % len(episodes)]
        - its object the episode's - else FREE_SHARE of the worlds play the
        free clips, and each other world's object is drawn as often as the
        holds have it, its episodes drawn from those with it. Without
        `randomize`: nominal physics, no noise or lag, the reader's first
        draw."""
        bundle = Path(bundle)
        holds = dict(np.load(bundle / "holds.npz"))
        self.holds = holds
        if episodes is None:
            episodes = np.flatnonzero(holds["reachable"] & ~holds["test"])
        self.episodes = np.asarray(episodes)
        self.randomize, self.cycle, self.mode = randomize, cycle, "policy"
        self.worlds = Worlds(str(holds["scene"]), envs, device, seed, randomize, None,
                             iterations, linesearch)
        self.device, self.generator = self.worlds.device, self.worlds.generator
        mjm = self.mjm = self.worlds.mjm
        rng = self.worlds.rng

        self.num_envs, self.num_actions = envs, mjm.nu
        self.max_episode_length = int(holds["length"].max())
        self.episode_length_buf = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.cfg = {"envs": envs, "rate": RATE, "randomize": randomize,
                    "episodes": len(self.episodes), "device": str(self.device)}
        tensor = lambda array, dtype=torch.float32: torch.as_tensor(  # noqa: E731
            np.asarray(array), dtype=dtype, device=self.device)

        # each world's object, mass and friction
        if cycle:
            world_episode = self.episodes[np.arange(envs) % len(self.episodes)]
        else:
            free = holds["variant"][self.episodes] < 0
            share = FREE_SHARE if free.any() and (~free).any() else float(free.all())
            kinds = np.arange(envs) < round(share * envs)                  # a free world
            world_episode = np.where(kinds, rng.choice(self.episodes[free] if free.any() else self.episodes, envs),
                                     rng.choice(self.episodes[~free] if (~free).any() else self.episodes, envs))
        variant = holds["variant"][world_episode]
        mass = (rng.uniform(*MASS_KG, envs) if randomize else np.full(envs, np.mean(MASS_KG)))
        friction = rng.uniform(*cpu.FRICTION_SCALE, envs) if randomize else np.ones(envs)
        self._objects(variant, mass, friction)
        self.variant = tensor(variant, torch.long)
        self.free = self.variant < 0
        self.world_episode = tensor(world_episode, torch.long)
        # pool v + 1: the episodes of object v; pool 0: the free clips
        pools = [self.episodes[holds["variant"][self.episodes] == v] for v in range(-1, variant.max() + 1)]
        self.pool_size = tensor([len(p) for p in pools], torch.long)
        self.pool = tensor(np.stack([np.pad(p, (0, max(map(len, pools)) - len(p))) for p in pools]),
                           torch.long)

        self.substeps = round(1.0 / (RATE * mjm.opt.timestep))
        self.low, self.high = tensor(mjm.actuator_ctrlrange[:, 0]), tensor(mjm.actuator_ctrlrange[:, 1])
        self.joint_qpos = tensor(mjm.jnt_qposadr[mjm.actuator_trnid[:, 0]], torch.long)
        self.joint_dof = tensor(mjm.jnt_dofadr[mjm.actuator_trnid[:, 0]], torch.long)
        joint = mjm.joint(hold_scene.OBJECT)
        self.object_qpos, self.object_dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.object_body = mjm.body(hold_scene.OBJECT).id
        self.parts = len(holds["parts"])
        self.finger_of = tensor([list(hold_scene.FINGERS).index(p) for p in holds["parts"]], torch.long)
        from .hand_library import robot_chain
        chain, (wrist, axes, palm_length) = robot_chain(mjm)
        self.palm_axes, self.palm_length = tensor(axes), float(palm_length)
        self.tip_bodies = tensor([chain[i][0] for i in TIPS.values()], torch.long)
        self.tip_points = tensor(np.array([chain[i][1] for i in TIPS.values()]))

        # the holds' frames, on the device
        self.first, self.length = tensor(holds["first"], torch.long), tensor(holds["length"], torch.long)
        self.start_qpos, self.start_qvel = tensor(holds["qpos"]), tensor(holds["qvel"])
        self.start_grip = tensor(holds["grip"])
        hinge = mjm.jnt_qposadr[mjm.actuator_trnid[:, 0]]       # frames hold joints in qpos order
        self.truth = tensor(holds["joints"][:, hinge])
        self.read = tensor(holds["read"][:, :, hinge].astype(np.float32))
        self.meant = tensor(holds["meant"].astype(np.float32))
        self.target_position, self.target_quat = tensor(holds["position"]), tensor(holds["quat"])
        self.gravity_frames = tensor(holds["gravity"])
        self.fingers = tensor(holds["fingers"], torch.bool)
        self.truth_pinch = tensor(holds["pinch"])
        block = np.r_[True, (np.diff(holds["take"]) != 0) | (np.diff(holds["hand"]) != 0)]
        self.block_start = tensor(np.maximum.accumulate(np.where(block, np.arange(len(block)), 0)),
                                  torch.long)

        worlds = self.worlds
        self.qpos, self.qvel, self.ctrl = worlds.qpos, worlds.qvel, worlds.ctrl
        self.sensordata, self.time, self.warmstart = worlds.sensordata, worlds.time, worlds.warmstart
        self.xfrc = wp.to_torch(worlds.data.xfrc_applied)
        self.gravity = wp.to_torch(worlds.model.opt.gravity)
        self.body_mass = wp.to_torch(worlds.model.body_mass)[:, self.object_body]
        self.body_inertia = wp.to_torch(worlds.model.body_inertia)[:, self.object_body]
        self.inertial = _quat_matrices(wp.to_torch(worlds.model.body_iquat)[:, self.object_body])
        self.targets = torch.zeros((envs, mjm.nu), device=self.device)
        self.change = torch.zeros_like(self.targets)
        self.residual = torch.zeros_like(self.targets)
        self.previous_joints = torch.zeros_like(self.targets)
        self.episode = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.draw = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.restart()

    def _objects(self, variant, mass, friction):
        """Each world's object variant, mass and friction into MuJoCo Warp's
        batched model fields; gravity made per world."""
        mjm, model, worlds = self.mjm, self.worlds.model, self.worlds
        fields = hold_scene.variant_fields(mujoco.MjSpec.from_string(_with_assets(str(self.holds["scene"]))),
                                           mjm)
        rows = hold_scene.scaled(fields, np.maximum(variant, 0), mass)
        rows["geom_dataid"] = np.where(variant < 0, -1, rows["geom_dataid"])    # no object
        geom, body = mjm.geom(hold_scene.OBJECT).id, self.object_body_id()
        for names, index in ((hold_scene.GEOM_FIELDS, geom), (hold_scene.BODY_FIELDS, body)):
            for field in names:
                current = getattr(model, field)
                values = np.repeat(np.asarray(getattr(mjm, field))[None], len(variant), axis=0)
                values[:, index] = rows[field].reshape(values[:, index].shape)
                width = wp.types.type_size(current.dtype)         # a vector's length, or 1
                shape = (len(variant),) + current.shape[1:] + ((width,) if width > 1 else ())
                setattr(model, field, wp.array(values.reshape(shape), dtype=current.dtype,
                                               device=worlds.wp_device))
        geom_friction = np.repeat(mjm.geom_friction[None], len(variant), axis=0)
        geom_friction[:, geom, 0] *= friction
        model.geom_friction = wp.array(geom_friction, dtype=model.geom_friction.dtype,
                                       device=worlds.wp_device)
        model.opt.gravity = wp.array(np.repeat(mjm.opt.gravity[None], len(variant), axis=0),
                                     dtype=wp.vec3, device=worlds.wp_device)
        import mujoco_warp as mjw
        mjw.set_const(model, worlds.data)
        scales = worlds.scales.clone()
        scales[:, 0] = torch.as_tensor(friction, device=self.device)
        scales[:, 1] = torch.as_tensor(mass, device=self.device)
        worlds.scales = scales

    def object_body_id(self):
        return self.mjm.body(hold_scene.OBJECT).id

    # ------------------------------------------------------------ VecEnv
    def get_observations(self):
        return self._obs

    def step(self, actions):
        action = torch.clamp(actions.to(self.device, torch.float32), -1.0, 1.0)
        frame = self._frame()
        read = self.read[self.draw, frame]
        if self.mode == "reader":
            action = torch.zeros_like(action)
        elif self.mode == "reader and grip":
            action = torch.clamp(self.start_grip[self.episode] / RESIDUAL_RAD, -1.0, 1.0)
        previous = self.targets
        self.targets = torch.clamp(read + RESIDUAL_RAD * action, self.low, self.high)
        change = self.targets - previous
        self.residual = action
        if self.randomize:                       # the new target arrives late
            top = round(cpu.LAG_S * RATE * self.substeps) + 1
            lag = torch.randint(0, top, (self.num_envs,), generator=self.generator,
                                device=self.device)
        else:
            lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_joints = self.qpos[:, self.joint_qpos].clone()
        self.gravity.copy_(self.gravity_frames[frame])
        for substep in range(self.substeps):
            self.ctrl.copy_(torch.where((substep < lag)[:, None], previous, self.targets))
            self._assist(frame + 1, (self.steps * self.substeps + substep) / (RATE * self.substeps))
            self.worlds.step()
        unstable = self.worlds.unstable()
        if unstable.any():                       # ends: kept finite until reset
            self.qpos[unstable] = 0.0
            self.qvel[unstable] = 0.0
        self.worlds.kinematics()
        self.steps += 1
        self.episode_length_buf += 1
        self.worlds.counted()

        frame = self._frame()
        position, turn, joints, contact = self.errors(frame)
        pinch, wanted = self.pinch(), self.truth_pinch[frame]
        touching = wanted < TOUCH_PL
        off = (pinch - wanted).abs()
        touch = (off * touching).sum(dim=1) / torch.clamp(touching.sum(dim=1), min=1)
        held = (HOLD_WEIGHT + torch.exp(-position / POSITION_SCALE)
                + ROTATION_WEIGHT * torch.exp(-turn / ROTATION_SCALE)
                + JOINT_WEIGHT * torch.exp(-joints / JOINT_SCALE)
                + CONTACT_WEIGHT * contact)
        free = (FREE_JOINT_WEIGHT * torch.exp(-joints / JOINT_SCALE)
                + PINCH_WEIGHT * torch.exp(-off.mean(dim=1) / PINCH_SCALE)
                + TOUCH_WEIGHT * torch.exp(-touch / TOUCH_SCALE) * touching.any(dim=1))
        reward = (torch.where(self.free, free, held)
                  - RESIDUAL_COST * (action ** 2).sum(dim=1)
                  - CHANGE_COST * (change ** 2).sum(dim=1))
        self.change = change
        dropped = ((position > DROP_M) & ~self.free) | unstable
        reward = torch.where(dropped, 0.0, reward)
        timed_out = (self.steps >= self.length[self.episode] - 1) & ~dropped
        supported = self._support() > 0
        made = touching & (pinch < TOUCH_PL)
        hold, loose = ~self.free, self.free & touching.any(dim=1)
        log = {"/hold/position_mm": 1000 * position[hold].mean(), "/hold/turn_deg": torch.rad2deg(turn[hold]).mean(),
               "/hold/joint_deg": torch.rad2deg(joints[hold]).mean(), "/hold/contact": contact[hold].mean(),
               "/hold/dropped": (dropped & ~supported & hold).float().sum(),
               "/hold/unstable": unstable.float().sum(),
               "/pinch/joint_deg": torch.rad2deg(joints[self.free]).mean(),
               "/pinch/touch_mm": 1000 * self.palm_length * touch[loose].mean(),
               "/pinch/made": made[self.free].float().sum() / torch.clamp(touching[self.free].float().sum(), min=1)}
        log = {key: value for key, value in log.items() if torch.isfinite(value)}
        rows = torch.stack([position, turn, joints, contact, dropped.float(), supported.float(),
                            self.free.float(), touching.sum(dim=1).float(), made.sum(dim=1).float(),
                            (~touching).sum(dim=1).float(), (~touching & (pinch < TOUCH_PL)).sum(dim=1).float()],
                           dim=1)
        done = timed_out | dropped
        if done.any():
            self._reset(torch.nonzero(done).squeeze(1))
            self.worlds.kinematics()
        self._obs = self._observe()
        return self._obs, reward, done, {"time_outs": timed_out, "log": log, "errors": rows}

    def errors(self, frame):
        """(position m, turn rad, mean joint rad, share of GRAB's fingers
        on the object that are on it) each (envs,), at `frame`."""
        position = torch.linalg.norm(self.qpos[:, self.object_qpos:self.object_qpos + 3]
                                     - self.target_position[frame], dim=1)
        quat = self.qpos[:, self.object_qpos + 3:self.object_qpos + 7]
        dot = (quat * self.target_quat[frame]).sum(dim=1).abs() / torch.clamp(
            torch.linalg.norm(quat, dim=1), min=1e-9)
        turn = 2 * torch.acos(torch.clamp(dot, max=1.0))
        joints = (self.qpos[:, self.joint_qpos] - self.truth[frame]).abs().mean(dim=1)
        touching = torch.zeros((self.num_envs, len(hold_scene.FINGERS)), device=self.device)
        touching.index_reduce_(1, self.finger_of,
                               (self.sensordata[:, :self.parts] < hold_scene.CONTACT_M).float(),
                               "amax", include_self=True)
        wanted = self.fingers[frame].float()
        contact = (touching * wanted).sum(dim=1) / torch.clamp(wanted.sum(dim=1), min=1)
        return position, turn, joints, contact

    def pinch(self):
        """The robot's thumb tip to its index, middle and ring tips (envs,
        3), palm lengths."""
        rotations = self.worlds.xmat[:, self.tip_bodies].reshape(self.num_envs, -1, 3, 3)
        tips = self.worlds.xpos[:, self.tip_bodies] + torch.einsum("npij,pj->npi", rotations, self.tip_points)
        return torch.linalg.norm(tips[:, 1:] - tips[:, :1], dim=-1) / self.palm_length

    # ------------------------------------------------------------ the support
    def _support(self):
        """How much of the object's weight is carried for it, (envs,): 1 at
        the start, 0 from ASSIST_S on."""
        return torch.clamp(1.0 - self.steps / (ASSIST_S * RATE), min=0.0)

    def _assist(self, frame, seconds):
        """The support's force and torque on the object, into xfrc_applied:
        its weight carried, and a spring and damper towards where it should
        be at `frame` - all times what is left of the support at `seconds`
        into the episode."""
        left = torch.where(self.free, 1.0, torch.clamp(1.0 - seconds / ASSIST_S, min=0.0))[:, None]
        frame = torch.minimum(frame, self.first[self.episode] + self.length[self.episode] - 1)
        position = self.qpos[:, self.object_qpos:self.object_qpos + 3]
        quat = self.qpos[:, self.object_qpos + 3:self.object_qpos + 7]
        velocity = self.qvel[:, self.object_dof:self.object_dof + 3]
        spin = self.qvel[:, self.object_dof + 3:self.object_dof + 6]        # in the object's frame
        mass = self.body_mass[:, None]
        force = mass * (-self.gravity + ASSIST_STIFFNESS * (self.target_position[frame] - position)
                        - ASSIST_DAMPING * velocity)
        # About the object's principal axes, where MuJoCo's inertia is diagonal.
        axes = _quat_matrices(quat) @ self.inertial
        error = _rotvec(_quat_multiply(self.target_quat[frame], _quat_conjugate(quat)))  # world
        local = torch.einsum("nji,nj->ni", axes, error)
        spin = torch.einsum("nji,nj->ni", self.inertial, spin)
        torque = torch.einsum("nij,nj->ni", axes, self.body_inertia
                              * (ASSIST_STIFFNESS * local - ASSIST_DAMPING * spin))
        self.xfrc[:, self.object_body] = left * torch.cat([force, torque], dim=1)

    # ------------------------------------------------------------ episodes
    def restart(self):
        self._reset(torch.arange(self.num_envs, device=self.device))
        self.worlds.kinematics()
        self._obs = self._observe()
        return self._obs

    def _reset(self, envs):
        if self.cycle:
            episode = self.world_episode[envs]
        else:
            pool = self.variant[envs] + 1
            pick = (torch.rand(len(envs), generator=self.generator, device=self.device)
                    * self.pool_size[pool]).long()
            episode = self.pool[pool, pick]
        self.episode[envs] = episode
        self.draw[envs] = (torch.randint(0, self.read.shape[0], (len(envs),), generator=self.generator,
                                         device=self.device) if self.randomize else 0)
        self.steps[envs] = 0
        self.episode_length_buf[envs] = 0
        self.qpos[envs] = self.start_qpos[episode]
        self.qvel[envs] = self.start_qvel[episode]
        self.warmstart[envs] = 0.0
        self.time[envs] = 0.0
        self.xfrc[envs] = 0.0
        self.targets[envs] = self.truth[self.first[episode]] + self.start_grip[episode]
        self.ctrl[envs] = self.targets[envs]
        self.change[envs] = 0.0
        self.residual[envs] = 0.0
        self.previous_joints[envs] = self.qpos[envs][:, self.joint_qpos]
        self.gravity[envs] = self.gravity_frames[self.first[episode]]

    def _frame(self):
        """Each world's frame of the holds now."""
        return self.first[self.episode] + torch.minimum(self.steps, self.length[self.episode] - 1)

    # ------------------------------------------------------------ observing
    def _observe(self):
        frame = self._frame()
        joints = self.qpos[:, self.joint_qpos]
        previous = self.previous_joints
        if self.randomize:
            joints = joints + self._noise(joints.shape, cpu.JOINT_NOISE)
            previous = previous + self._noise(previous.shape, cpu.JOINT_NOISE)
        back = torch.maximum(frame[:, None] - torch.arange(READ_FRAMES, device=self.device),
                             self.block_start[frame][:, None])
        read = self.read[self.draw[:, None], back].flatten(1)
        down = self.gravity @ self.palm_axes / torch.clamp(
            torch.linalg.norm(self.gravity, dim=1, keepdim=True), min=1e-9)
        policy = torch.cat([joints, previous, self.targets, read, self.meant[self.draw, frame], down], dim=1)
        position = self.qpos[:, self.object_qpos:self.object_qpos + 3]
        quat = self.qpos[:, self.object_qpos + 3:self.object_qpos + 7]
        privileged = torch.cat([
            (position - self.target_position[frame]) @ self.palm_axes,
            _six(_quat_matrices(quat)), _six(_quat_matrices(self.target_quat[frame])),
            self.qvel[:, self.object_dof:self.object_dof + 6], self.truth[frame],
            self.truth_pinch[frame], self.qvel[:, self.joint_dof], self._support()[:, None],
            self.free[:, None].float(), self.sensordata[:, :self.parts], self.worlds.scales], dim=1)
        return TensorDict({"policy": policy.float(), "privileged": privileged.float()},
                          batch_size=[self.num_envs], device=self.device)

    def _noise(self, shape_, scale):
        return torch.randn(shape_, generator=self.generator, device=self.device) * scale


def _with_assets(xml):
    """The scene's MJCF with the robot's meshes found where gpu.Worlds finds them."""
    from .gpu import ASSETS
    return xml.replace('<compiler angle="radian"',
                       f'<compiler angle="radian" meshdir="{ASSETS.as_posix()}"', 1)


def _quat_matrices(quats):
    quats = quats / torch.clamp(torch.linalg.norm(quats, dim=-1, keepdim=True), min=1e-9)
    w, x, y, z = quats.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _quat_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _quat_conjugate(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _rotvec(q):
    """Quaternions (n, 4) -> rotation vectors (n, 3), the short way round."""
    q = q * torch.where(q[:, :1] < 0, -1.0, 1.0)
    vector = q[:, 1:]
    size = torch.linalg.norm(vector, dim=1, keepdim=True)
    angle = 2 * torch.atan2(size, q[:, :1])
    return vector * torch.where(size > 1e-9, angle / torch.clamp(size, min=1e-9), 2.0)


def _six(rotations):
    return torch.cat([rotations[..., :, 0], rotations[..., :, 1]], dim=-1)


def train(bundle, folder, envs, iterations, config, writer_config, device="cuda:0",
          physics=None):
    """PPO on the GPU; checkpoints and progress into `folder`."""
    from rsl_rl.runners import OnPolicyRunner

    env = GpuHoldEnv(bundle, envs, device=device, **(physics or {}))
    cfg = json.loads(json.dumps(config))
    cfg["logger"] = writer_config
    OnPolicyRunner(env, cfg, log_dir=str(folder), device=str(env.device)).learn(iterations)
    return env


def evaluate(bundle, folder, checkpoint, config, device="cuda:0", physics=None):
    """Each set once - the reachable holds of the people never fitted on and
    as many of the fitted people's, the free clips of FreiHAND's evaluation
    poses and as many of its training poses' - nominal physics, the
    reader's first draw, the hand moved by the trained policy, by the
    reader's joints alone, and (holds) by them plus the start's closing
    held: {set: {mode: ...}}. Holds: held (share kept to their end), held_s
    (mean seconds kept after the support), position_mm, turn_deg, joint_deg
    (medians of the frames held), contact (mean). Free clips: joint_deg
    (median), touches made (of the person's thumb-on-fingertip frames, the
    share the robot's thumb touches too) and false (of the rest, the share
    it touches anyway)."""
    from rsl_rl.runners import OnPolicyRunner

    holds = dict(np.load(Path(bundle) / "holds.npz"))
    rng = np.random.default_rng(0)
    sets = {}
    for kind, chosen in (("holds", holds["reachable"] & (holds["variant"] >= 0)),
                         ("free clips", holds["variant"] < 0)):
        test = np.flatnonzero(chosen & holds["test"])
        fitted = np.flatnonzero(chosen & ~holds["test"])
        sets[f"{kind}, never trained on"] = test
        sets[f"{kind}, trained on"] = rng.choice(fitted, size=min(len(test), len(fitted)), replace=False)
    out = {}
    for name, episodes in sets.items():
        free = name.startswith("free")
        env = GpuHoldEnv(bundle, envs=len(episodes), episodes=episodes, randomize=False, cycle=True,
                         device=device, **(physics or {}))
        cfg = json.loads(json.dumps(config))
        cfg["logger"] = "tensorboard"
        runner = OnPolicyRunner(env, cfg, log_dir=None, device=str(env.device))
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True)
        policy = runner.get_inference_policy()
        out[name] = {}
        for mode in MODES[:2] if free else MODES:
            env.mode = mode
            with torch.inference_mode():
                obs = env.restart()
                finished = torch.zeros(len(episodes), dtype=torch.bool, device=env.device)
                dropped = torch.zeros_like(finished)
                rows, active = [], []
                while not finished.all():
                    obs, _, done, extras = env.step(policy(obs))
                    rows.append(extras["errors"].cpu())
                    active.append((~finished).cpu())
                    new = done & ~finished
                    dropped |= new & ~extras["time_outs"]
                    finished |= new
            rows, active = torch.stack(rows).numpy(), torch.stack(active).numpy()
            if free:
                out[name][mode] = {
                    "episodes": int(len(episodes)),
                    "joint_deg": float(np.degrees(np.median(rows[..., 2][active]))),
                    "touches_made": float(rows[..., 8][active].sum() / max(rows[..., 7][active].sum(), 1)),
                    "touches_false": float(rows[..., 10][active].sum() / max(rows[..., 9][active].sum(), 1))}
                continue
            kept = active & (rows[..., 4] == 0)
            after = kept & (rows[..., 5] == 0)
            out[name][mode] = {
                "episodes": int(len(episodes)), "held": float(1 - dropped.float().mean()),
                "held_s": float(after.sum(0).mean() / RATE),
                "position_mm": float(1000 * np.median(rows[..., 0][kept])),
                "turn_deg": float(np.degrees(np.median(rows[..., 1][kept]))),
                "joint_deg": float(np.degrees(np.median(rows[..., 2][kept]))),
                "contact": float(rows[..., 3][kept].mean())}
    return out
