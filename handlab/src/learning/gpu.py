"""The tracker on an NVIDIA GPU: env.TrackEnv's twin, thousands of task sims
stepped at once by MuJoCo Warp, observations and rewards in PyTorch on the
GPU, PPO (rsl_rl) on the GPU too.

    env = GpuTrackEnv(bundle, envs=8192)          # rsl_rl's VecEnv
    folder = train(bundle, folder, envs, iterations, writer)
    results = evaluate(bundle, folder)            # train.evaluate's format

Same observations, actions, rewards, episodes, randomization and noise as
TrackEnv - so a policy trained here runs there unchanged: Test, Runs and
Recordings step 7 on the Mac. Only PPO: the planner (the planner_teacher
method) still runs on a CPU.

The bundle (remote.export_bundle, made on the Mac): library.npz, the task
sim as MJCF (task.xml - the very model the Mac compiles, self-collision and
sensors included) and bundle.json (the hand mount). It is loaded with this
machine's MuJoCo (3.17, which MuJoCo Warp needs), its meshes found in the
repository's assets. Measured on the Mac (Warp on the CPU): 10 steps from a
grip agree with MuJoCo 3.12 to 3e-7 in every joint, and the fingertip gaps
to 0.01 mm.

A step is TrackEnv's: the action moves the servo targets, then SUBSTEPS
MuJoCo steps - each one CUDA graph on a stream of MuJoCo Warp's own (a graph
cannot be captured on PyTorch's default stream), the two streams waiting on
each other around it, so PyTorch's writes land before the step and its
reads come after. Randomization is per world, in MuJoCo Warp's batched
model fields (friction, the object's mass and inertia, servo gain, damping
and torque) with set_const after. Every CHECK_EVERY steps the worlds'
overflow flags are read: a contact or constraint buffer too small stops the
run with which one (CONTACTS, CONSTRAINTS).

The contact solver gets at most SOLVER_ITERATIONS (and LINESEARCH_ITERATIONS)
here, against MuJoCo's 100 (50): on the GPU all worlds pay for the slowest,
and on the CPU a grip converges in 2. A world whose physics goes non-finite
ends like a drop and starts over; the log counts them (/track/unstable).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from ..sim.symmetry import box_symmetries
from . import env as cpu
from .trajectories import LIBRARY, RATE, load

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "h4nd_left"
CONTACTS = 96                    # per world
CONSTRAINTS = 800                # per world
CHECK_EVERY = 50
SOLVER_ITERATIONS = 10
LINESEARCH_ITERATIONS = 10
SERIOUS = (mjw.OverflowType.NEFC | mjw.OverflowType.NARROWPHASE
           | mjw.OverflowType.BROADPHASE | mjw.OverflowType.CCD)


class GpuTrackEnv(VecEnv):
    def __init__(self, bundle, envs=8192, motions=None, randomize=True, seed=0, cycle=False,
                 device="cuda:0", iterations=SOLVER_ITERATIONS,
                 linesearch=LINESEARCH_ITERATIONS):
        """As env.TrackEnv; `bundle` is the folder remote.export_bundle made."""
        bundle = Path(bundle)
        library = load(bundle / LIBRARY.name)
        meta = json.loads((bundle / "bundle.json").read_text())
        self.library = library
        self.motions = (np.flatnonzero(library["motion_feasible"] & ~library["motion_recorded"])
                        if motions is None else np.asarray(motions))
        self.device = torch.device(device)
        self.wp_device = "cpu" if self.device.type == "cpu" else f"cuda:{self.device.index or 0}"
        wp.set_device(self.wp_device)
        self.randomize, self.cycle = randomize, cycle
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        xml = (bundle / "task.xml").read_text().replace(
            '<compiler angle="radian"', f'<compiler angle="radian" meshdir="{ASSETS.as_posix()}"', 1)
        mjm = mujoco.MjModel.from_xml_string(xml)
        mjm.eq_active0[:] = 0                          # episodes start let go
        mjm.opt.iterations, mjm.opt.ls_iterations = iterations, linesearch
        self.mjm = mjm
        self.model = mjw.put_model(mjm)
        self.data = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=envs, nconmax=CONTACTS,
                                 njmax=CONSTRAINTS)
        self.scales = torch.ones((envs, 3), device=self.device)
        if randomize:
            self._randomize(envs)

        self.num_envs, self.num_actions = envs, mjm.nu
        self.max_episode_length = int(library["motion_length"].max())
        self.episode_length_buf = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.cfg = {"envs": envs, "rate": RATE, "randomize": randomize,
                    "motions": len(self.motions), "device": str(self.device)}

        tensor = lambda array, dtype=torch.float32: torch.as_tensor(  # noqa: E731
            np.asarray(array), dtype=dtype, device=self.device)
        self.substeps = round(1.0 / (RATE * mjm.opt.timestep))
        self.low, self.high = tensor(mjm.actuator_ctrlrange[:, 0]), tensor(mjm.actuator_ctrlrange[:, 1])
        self.joint_qpos = tensor(mjm.jnt_qposadr[mjm.actuator_trnid[:, 0]], torch.long)
        self.object_qpos = int(mjm.jnt_qposadr[mjm.joint(cpu.OBJECT).id])
        self.object_qvel = int(mjm.jnt_dofadr[mjm.joint(cpu.OBJECT).id])
        self.tips = int(meta["tips"])
        self.future = tensor(np.maximum(1, np.round(np.array(cpu.FUTURE_S) * RATE)), torch.long)
        self.symmetries = tensor(np.array(box_symmetries(mjm.geom(cpu.OBJECT).size * 2000)))
        self.mount_position = tensor(meta["mount_position"])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.array(meta["mount_quat"]))
        self.mount_rotation = tensor(rotation.reshape(3, 3))
        self.motion_position = tensor(library["motion_position"])
        self.motion_quat = tensor(library["motion_quat"])
        self.motion_length = tensor(library["motion_length"], torch.long)
        self.motion_start = tensor(library["motion_start"], torch.long)
        self.start_qpos = tensor(library["start_state"][:, 1:1 + mjm.nq])
        self.start_qvel = tensor(library["start_state"][:, 1 + mjm.nq:1 + mjm.nq + mjm.nv])
        self.start_grip = tensor(library["start_grip"])
        self.motion_choices = tensor(self.motions, torch.long)

        # PyTorch views of MuJoCo Warp's arrays: written in place, read in place.
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.sensordata = wp.to_torch(self.data.sensordata)
        self.time = wp.to_torch(self.data.time)
        self.warmstart = wp.to_torch(self.data.qacc_warmstart)
        self.targets = torch.zeros((envs, mjm.nu), device=self.device)
        self.previous_joints = torch.zeros_like(self.targets)
        self.motion = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(envs, dtype=torch.long, device=self.device)
        self.gaps = torch.zeros((envs, self.tips), device=self.device)
        self._count = 0
        self._graph = None
        if self.device.type == "cuda":
            self.physics_stream = wp.Stream(self.wp_device)
            self.physics_torch = wp.stream_to_torch(self.physics_stream)
        wp.synchronize()
        self.restart()

    # ------------------------------------------------------------ VecEnv
    def get_observations(self):
        return self._obs

    def step(self, actions):
        action = torch.clamp(actions.to(self.device, torch.float32), -1.0, 1.0)
        previous = self.targets
        self.targets = torch.clamp(previous + cpu.ACTION_SCALE * action, self.low, self.high)
        if self.randomize:                       # the new target arrives late
            top = round(cpu.LAG_S * RATE * self.substeps) + 1
            lag = torch.randint(0, top, (self.num_envs,), generator=self.generator,
                                device=self.device)
        else:
            lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_joints = self.qpos[:, self.joint_qpos].clone()
        for substep in range(self.substeps):
            self.ctrl.copy_(torch.where((substep < lag)[:, None], previous, self.targets))
            self._physics_step()
        unstable = ~(torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1))
        if unstable.any():                       # ends like a drop: kept finite until reset
            self.qpos[unstable] = self.start_qpos[self.motion_start[self.motion[unstable]]]
            self.qvel[unstable] = 0.0
        self.gaps = torch.nan_to_num(self.sensordata[:, :self.tips].clone())
        self.steps += 1
        self.episode_length_buf += 1
        self._count += 1
        if self._count % CHECK_EVERY == 0:
            self._check_overflow()

        position_error, turn_error = self._errors()
        dropped = (position_error > cpu.DROP_M) | unstable
        timed_out = ~dropped & (self.steps >= self.motion_length[self.motion] - 1)
        gap = torch.clamp(torch.clamp(self.gaps, min=0.0).mean(dim=1) / cpu.GAP_SCALE, 0.0, 1.0)
        reward = (torch.exp(-position_error / cpu.POSITION_SCALE)
                  + torch.exp(-turn_error / cpu.TURN_SCALE)
                  - cpu.TARGET_COST * ((self.targets - previous) ** 2).sum(dim=1)
                  - cpu.GAP_COST * gap - cpu.DROP_COST * dropped.float())
        log = {"/track/position_mm": 1000 * position_error.mean(),
               "/track/turn_deg": torch.rad2deg(turn_error).mean(),
               "/track/drops_per_step": dropped.float().mean(),
               "/track/fingertip_gap_mm": 1000 * torch.clamp(self.gaps, min=0).mean(),
               "/track/unstable": unstable.float().sum()}
        errors = torch.stack([position_error, turn_error], dim=1)
        done = dropped | timed_out
        if done.any():
            self._reset(torch.nonzero(done).squeeze(1))
        self._obs = self._observe()
        return self._obs, reward, done, {"time_outs": timed_out, "log": log, "errors": errors}

    # ------------------------------------------------------------ episodes
    def restart(self):
        self._reset(torch.arange(self.num_envs, device=self.device))
        self._obs = self._observe()
        return self._obs

    def _reset(self, envs):
        if self.cycle:
            motion = self.motion_choices[envs % len(self.motion_choices)]
        else:
            pick = torch.randint(0, len(self.motion_choices), (len(envs),),
                                 generator=self.generator, device=self.device)
            motion = self.motion_choices[pick]
        start = self.motion_start[motion]
        self.motion[envs] = motion
        self.steps[envs] = 0
        self.episode_length_buf[envs] = 0
        self.qpos[envs] = self.start_qpos[start]
        self.qvel[envs] = self.start_qvel[start]
        self.warmstart[envs] = 0.0
        self.time[envs] = 0.0
        self.targets[envs] = self.start_grip[start]
        self.ctrl[envs] = self.start_grip[start]
        self.previous_joints[envs] = self.start_qpos[start][:, self.joint_qpos]
        self.gaps[envs] = 0.0

    def _goal(self, offset=0):
        frame = torch.minimum(self.steps + offset, self.motion_length[self.motion] - 1)
        return (self.motion_position[self.motion, frame],
                _quat_matrices(self.motion_quat[self.motion, frame]))

    def _object(self):
        at = self.object_qpos
        positions = (self.qpos[:, at:at + 3] - self.mount_position) @ self.mount_rotation
        rotations = self.mount_rotation.T @ _quat_matrices(self.qpos[:, at + 3:at + 7])
        return positions, rotations

    def _errors(self):
        positions, rotations = self._object()
        goal_positions, goal_rotations = self._goal()
        _, turn = self._nearest(rotations, goal_rotations[:, None])
        return torch.linalg.norm(positions - goal_positions, dim=1), turn[:, 0]

    def _nearest(self, rotations, goals):
        relative = torch.einsum("nji,nkjl->nkil", rotations, goals)
        traces = torch.einsum("nkij,sji->nks", relative, self.symmetries)
        best = traces.argmax(dim=-1)
        top = torch.gather(traces, -1, best[..., None])[..., 0]
        angle = torch.arccos(torch.clamp((top - 1) / 2, -1.0, 1.0))
        return goals @ self.symmetries[best], angle

    # ------------------------------------------------------------ observing
    def _observe(self):
        envs = self.num_envs
        joints = self.qpos[:, self.joint_qpos]
        previous = self.previous_joints
        positions, rotations = self._object()
        if self.randomize:
            joints = joints + self._noise(joints.shape, cpu.JOINT_NOISE)
            previous = previous + self._noise(previous.shape, cpu.JOINT_NOISE)
            positions = positions + self._noise(positions.shape, cpu.POSITION_NOISE)
            wobble = self._noise((envs, 3), cpu.TURN_NOISE / np.sqrt(3))
            rotations = _rotvec_matrices(wobble) @ rotations
        goals = [self._goal(int(offset)) for offset in self.future]
        goal_positions = torch.stack([g[0] for g in goals], dim=1)
        goal_rotations = torch.stack([g[1] for g in goals], dim=1)
        nearest, _ = self._nearest(rotations, goal_rotations)
        turns = nearest @ rotations.transpose(-1, -2)[:, None]
        policy = torch.cat([joints, previous, self.targets, positions, _six(rotations),
                            (goal_positions - positions[:, None]).reshape(envs, -1),
                            _six(turns).reshape(envs, -1)], dim=1)
        velocity = self.qvel[:, self.object_qvel:self.object_qvel + 6]
        privileged = torch.cat([velocity, self.gaps, self.scales], dim=1)
        return TensorDict({"policy": policy.float(), "privileged": privileged.float()},
                          batch_size=[envs], device=self.device)

    def _noise(self, shape, scale):
        return torch.randn(shape, generator=self.generator, device=self.device) * scale

    # ------------------------------------------------------------ physics
    def _physics_step(self):
        """One MuJoCo step of every world, after PyTorch's work so far and
        before its next."""
        if self.device.type != "cuda":
            mjw.step(self.model, self.data)
            return
        main = torch.cuda.current_stream(self.device)
        self.physics_torch.wait_stream(main)
        with wp.ScopedStream(self.physics_stream):
            if self._graph is None:                 # the first step: compile, then capture
                mjw.step(self.model, self.data)
                with wp.ScopedCapture() as capture:
                    mjw.step(self.model, self.data)
                self._graph = capture.graph
            else:
                wp.capture_launch(self._graph)
        main.wait_stream(self.physics_torch)

    def _check_overflow(self):
        flags = int(np.bitwise_or.reduce(self.data.overflow.numpy()))
        if flags & SERIOUS:
            raise RuntimeError(f"MuJoCo Warp ran out of room ({mjw.OverflowType(flags & SERIOUS)!r}):"
                               " raise gpu.CONTACTS / gpu.CONSTRAINTS")

    def _randomize(self, envs):
        """Per world: the object's friction and mass, the servos' strength -
        as TrackEnv draws them - into MuJoCo Warp's batched model fields."""
        mjm, model = self.mjm, self.model
        geom, body = mjm.geom(cpu.OBJECT).id, mjm.body(cpu.OBJECT).id
        friction = self.rng.uniform(*cpu.FRICTION_SCALE, envs)
        mass = self.rng.uniform(*cpu.MASS_SCALE, envs)
        torque = self.rng.uniform(*cpu.TORQUE_SCALE, envs)

        def batched(field, values):
            current = getattr(model, field)
            setattr(model, field, wp.array(values, dtype=current.dtype, device=self.wp_device))

        geom_friction = np.repeat(mjm.geom_friction[None], envs, axis=0)
        geom_friction[:, geom, 0] *= friction
        batched("geom_friction", geom_friction)
        body_mass = np.repeat(mjm.body_mass[None], envs, axis=0)
        body_mass[:, body] *= mass
        batched("body_mass", body_mass)
        body_inertia = np.repeat(mjm.body_inertia[None], envs, axis=0)
        body_inertia[:, body] *= mass[:, None]
        batched("body_inertia", body_inertia)
        gain = np.repeat(mjm.actuator_gainprm[None], envs, axis=0)
        bias = np.repeat(mjm.actuator_biasprm[None], envs, axis=0)
        force = np.repeat(mjm.actuator_forcerange[None], envs, axis=0)
        gain[:, :, 0] *= torque[:, None]                     # kp
        bias[:, :, 1] *= torque[:, None]                     # -kp
        bias[:, :, 2] *= np.sqrt(torque)[:, None]            # -kv: same damping ratio
        force *= torque[:, None, None]
        batched("actuator_gainprm", gain)
        batched("actuator_biasprm", bias)
        batched("actuator_forcerange", force)
        mjw.set_const(model, self.data)
        self.scales = torch.as_tensor(np.column_stack([friction, mass, torque]),
                                      dtype=torch.float32, device=self.device)


def train(bundle, folder, envs, iterations, config, writer_config, device="cuda:0",
          physics=None):
    """PPO on the GPU; checkpoints and progress into `folder`. `physics`:
    GpuTrackEnv's solver settings ({"iterations", "linesearch"})."""
    from rsl_rl.runners import OnPolicyRunner

    env = GpuTrackEnv(bundle, envs, device=device, **(physics or {}))
    cfg = json.loads(json.dumps(config))
    cfg["logger"] = writer_config
    OnPolicyRunner(env, cfg, log_dir=str(folder), device=str(env.device)).learn(iterations)
    return env


def evaluate(bundle, folder, checkpoint, config, device="cuda:0", physics=None):
    """train.evaluate on the GPU: every recorded and feasible motion once,
    nominal physics, no noise; the same rows."""
    from rsl_rl.runners import OnPolicyRunner

    library = load(Path(bundle) / LIBRARY.name)
    motions = np.flatnonzero(library["motion_recorded"] | library["motion_feasible"])
    env = GpuTrackEnv(bundle, envs=len(motions), motions=motions, randomize=False, cycle=True,
                      device=device, **(physics or {}))
    cfg = json.loads(json.dumps(config))
    cfg["logger"] = "tensorboard"
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=str(env.device))
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy()
    count = len(motions)
    finished = torch.zeros(count, dtype=torch.bool, device=env.device)
    dropped = torch.zeros_like(finished)
    errors = [[] for _ in range(count)]
    obs = env.get_observations()
    with torch.inference_mode():
        while not finished.all():
            obs, _, done, extras = env.step(policy(obs))
            rows = extras["errors"].cpu().numpy()
            for i in torch.nonzero(~finished).squeeze(1).tolist():
                errors[i].append(rows[i])
            new = done & ~finished
            dropped |= new & ~extras["time_outs"]
            finished |= new
    results = []
    for i, motion in enumerate(motions):
        rows = np.array(errors[i])
        results.append({
            "motion": int(motion), "recorded": bool(library["motion_recorded"][motion]),
            "source": str(library["motion_source"][motion]), "dropped": bool(dropped[i]),
            "position_mm": float(1000 * np.median(rows[:, 0])),
            "turn_deg": float(np.degrees(np.median(rows[:, 1]))),
            "planner_position_mm": float(1000 * library["motion_position_error"][motion]),
            "planner_turn_deg": float(np.degrees(library["motion_turn_error"][motion])),
            "still_turn_deg": float(np.degrees(library["motion_still_turn"][motion]))})
    return results


def _quat_matrices(quats):
    quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
    w, x, y, z = quats.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _six(rotations):
    return torch.cat([rotations[..., :, 0], rotations[..., :, 1]], dim=-1)


def _rotvec_matrices(vectors):
    angle = torch.linalg.norm(vectors, dim=1, keepdim=True)
    axis = vectors / torch.clamp(angle, min=1e-12)
    x, y, z = axis.unbind(1)
    zero = torch.zeros_like(x)
    k = torch.stack([torch.stack([zero, -z, y], -1), torch.stack([z, zero, -x], -1),
                     torch.stack([-y, x, zero], -1)], -2)
    angle = angle[:, :, None]
    return torch.eye(3, device=vectors.device) + torch.sin(angle) * k \
        + (1 - torch.cos(angle)) * k @ k
