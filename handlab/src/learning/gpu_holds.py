"""The physics policy (step 4) on an NVIDIA GPU: the robot hand keeping hold
of GRAB's objects while it follows what the reader makes of a person's hand
- thousands of hand sims with an object each, stepped at once by MuJoCo
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
episode ends. Physics, the target arriving late and the joints read are
randomized as the tracker's are (env.py).

Observations. "policy" - what the real hand has: its joints (read with
noise) now and a step before, its servo targets, the reader's joints for
the last READ_FRAMES frames and the contacts it means now, and gravity in
the palm frame (the mount's, on the real hand). "privileged" (the critic
only): the object's pose and speed against where it should be, the true
joints, the support left, each hand body's gap to the object, the physics
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
SETTINGS = ("RESIDUAL_RAD", "READ_FRAMES", "ASSIST_S", "ASSIST_STIFFNESS", "ASSIST_DAMPING",
            "DROP_M", "HOLD_WEIGHT", "POSITION_SCALE", "ROTATION_WEIGHT", "ROTATION_SCALE",
            "JOINT_WEIGHT", "JOINT_SCALE", "CONTACT_WEIGHT", "RESIDUAL_COST", "CHANGE_COST",
            "MASS_KG")
MODES = ("policy", "reader", "reader and grip")   # what moves the hand (evaluate)


class GpuHoldEnv(VecEnv):
    def __init__(self, bundle, envs=8192, episodes=None, randomize=True, seed=0, cycle=False,
                 device="cuda:0", iterations=SOLVER_ITERATIONS,
                 linesearch=LINESEARCH_ITERATIONS):
        """`episodes`: which of holds.npz's to play (default: the reachable
        ones of the people fitted on); with `cycle` world i plays
        episodes[i % len(episodes)] - its object the episode's - else each
        world's object is drawn as often as the episodes have it, and its
        episodes drawn from those with it. Without `randomize`: nominal
        physics, no noise or lag, the reader's first draw."""
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
            world_episode = self.episodes[rng.integers(len(self.episodes), size=envs)]
        variant = holds["variant"][world_episode]
        mass = (rng.uniform(*MASS_KG, envs) if randomize else np.full(envs, np.mean(MASS_KG)))
        friction = rng.uniform(*cpu.FRICTION_SCALE, envs) if randomize else np.ones(envs)
        self._objects(variant, mass, friction)
        self.variant = tensor(variant, torch.long)
        self.world_episode = tensor(world_episode, torch.long)
        pools = [self.episodes[holds["variant"][self.episodes] == v] for v in range(variant.max() + 1)]
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
        _, (wrist, axes, _) = robot_chain(mjm)
        self.palm_axes = tensor(axes)

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
        rows = hold_scene.scaled(fields, variant, mass)
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
        reward = (HOLD_WEIGHT + torch.exp(-position / POSITION_SCALE)
                  + ROTATION_WEIGHT * torch.exp(-turn / ROTATION_SCALE)
                  + JOINT_WEIGHT * torch.exp(-joints / JOINT_SCALE)
                  + CONTACT_WEIGHT * contact
                  - RESIDUAL_COST * (action ** 2).sum(dim=1)
                  - CHANGE_COST * (change ** 2).sum(dim=1))
        self.change = change
        dropped = (position > DROP_M) | unstable
        reward = torch.where(dropped, 0.0, reward)
        timed_out = (self.steps >= self.length[self.episode] - 1) & ~dropped
        supported = self._support() > 0
        log = {"/hold/position_mm": 1000 * position.mean(), "/hold/turn_deg": torch.rad2deg(turn).mean(),
               "/hold/joint_deg": torch.rad2deg(joints).mean(), "/hold/contact": contact.mean(),
               "/hold/dropped": (dropped & ~supported).float().sum(),
               "/hold/unstable": unstable.float().sum()}
        rows = torch.stack([position, turn, joints, contact, dropped.float(), supported.float()], dim=1)
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
        left = torch.clamp(1.0 - seconds / ASSIST_S, min=0.0)[:, None]
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
            variant = self.variant[envs]
            pick = (torch.rand(len(envs), generator=self.generator, device=self.device)
                    * self.pool_size[variant]).long()
            episode = self.pool[variant, pick]
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
            self.qvel[:, self.joint_dof], self._support()[:, None],
            self.sensordata[:, :self.parts], self.worlds.scales], dim=1)
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
    """The reachable episodes of the people never fitted on, and as many of
    the fitted people's, each once, nominal physics, the reader's first
    draw - moved by the trained policy, by the reader's joints alone, and
    by them plus the start's closing held: {"test" | "fitted": {mode:
    {held (share of episodes kept to their end), held_s (mean seconds
    kept after the support), position_mm, turn_deg, joint_deg (medians of
    the frames held), contact (mean), episodes}}}."""
    from rsl_rl.runners import OnPolicyRunner

    holds = dict(np.load(Path(bundle) / "holds.npz"))
    reachable = holds["reachable"]
    test = np.flatnonzero(reachable & holds["test"])
    fitted = np.flatnonzero(reachable & ~holds["test"])
    fitted = np.random.default_rng(0).choice(fitted, size=min(len(test), len(fitted)), replace=False)
    out = {}
    for name, episodes in (("test", test), ("fitted", fitted)):
        env = GpuHoldEnv(bundle, envs=len(episodes), episodes=episodes, randomize=False, cycle=True,
                         device=device, **(physics or {}))
        cfg = json.loads(json.dumps(config))
        cfg["logger"] = "tensorboard"
        runner = OnPolicyRunner(env, cfg, log_dir=None, device=str(env.device))
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True)
        policy = runner.get_inference_policy()
        out[name] = {}
        for mode in MODES:
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
