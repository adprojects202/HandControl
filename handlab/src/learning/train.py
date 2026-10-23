"""Training the tracker, and trying a trained one.

    folder = train()                     # -> data/learning/runs/<stamp>/
    results = evaluate(folder)           # every recorded and generated motion

`train` runs `learning.method` (METHODS), on this Mac's cores:

    ppo              rsl_rl's PPO - the policy finds its own way by trial
                     and error: `learning.iterations` rounds of STEPS_PER_ENV
                     steps over `learning.envs` copies of the task sim
    planner_teacher  the planner labels and the policy copies (teacher.py):
                     `learning.teacher_rounds` rounds, a copy per motion

A new method is a function here, a name in METHODS and in
settings.CHOICES["learning.method"]; the Learning panel lists it. A run's folder gets run.json (the method, the library it
learned from, the environment's settings), checkpoints (model_<n>.pt), a
TensorBoard log, and progress.jsonl - one line of numbers per round, which
the Learning panel plots (ProgressWriter).

`evaluate` runs a trained policy without noise or randomized physics, one
episode per motion: the recorded motions - never trained on, so this is the
one-shot test - and the generated ones it trained on. For each: whether it
dropped the object, and its median errors next to the planner's. `follow`
gives the whole path of each.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.log_writer import LogWriter
from scipy.spatial.transform import Rotation
from torch.utils.tensorboard import SummaryWriter

from .. import settings
from ..paths import ROOT
from . import background, env as environment, teacher
from .env import TrackEnv
from .trajectories import load

RUNS = ROOT / "data" / "learning" / "runs"
PROGRESS = "progress.jsonl"
STEPS_PER_ENV = 24
CONFIG = {
    "num_steps_per_env": STEPS_PER_ENV,
    "save_interval": 100,
    "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
    "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4,
                  "clip_param": 0.2, "gamma": 0.99, "lam": 0.95, "value_loss_coef": 1.0,
                  "entropy_coef": 0.001, "learning_rate": 1e-3, "max_grad_norm": 1.0,
                  "schedule": "adaptive", "desired_kl": 0.01},
    "actor": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
              "obs_normalization": True,
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.3}},
    "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
               "obs_normalization": True},
    "logger": {"class_name": "src.learning.train:ProgressWriter"},
}
SETTINGS = ("ACTION_SCALE", "FUTURE_S", "POSITION_SCALE", "TURN_SCALE", "TARGET_COST",
            "GAP_COST", "GAP_SCALE", "DROP_M", "DROP_COST", "VARIANTS", "FRICTION_SCALE",
            "MASS_SCALE", "TORQUE_SCALE", "LAG_S", "JOINT_NOISE", "POSITION_NOISE",
            "TURN_NOISE")


def train(library=None, folder=None):
    """Train a tracker by `learning.method`; returns its run folder."""
    config = settings.section("learning")
    library = library or load()
    if not library:
        raise ValueError("build the library first")
    folder = folder or RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    background.report(run=str(folder))
    METHODS[config["method"]](library, folder, config)
    return folder


def _ppo(library, folder, config):
    envs, iterations = config["envs"], config["iterations"]
    env = TrackEnv(library, envs)
    cfg = copy.deepcopy(CONFIG)
    cfg["logger"]["iterations"] = iterations
    _describe(folder, "ppo", library, env, iterations, {"envs": envs, "config": cfg})
    OnPolicyRunner(env, cfg, log_dir=str(folder), device="cpu").learn(iterations)


def _planner_teacher(library, folder, config):
    rounds = config["teacher_rounds"]
    motions = int((library["motion_feasible"] & ~library["motion_recorded"]).sum())
    _describe(folder, "planner_teacher", library, None, rounds,
              {"motions": motions, "config": CONFIG, "teacher": teacher.describe()})
    teacher.teach(library, folder, rounds, CONFIG, ProgressWriter(str(folder), rounds))


def _describe(folder, method, library, env, iterations, extra):
    """run.json: the method, the library it learned from, the environment's
    settings."""
    (folder / "run.json").write_text(json.dumps({
        "method": method, "library_made": str(library["made"]), "iterations": iterations,
        **({"motions": int(len(env.motions))} if env is not None else {}), **extra,
        "env": {name: _plain(getattr(environment, name)) for name in SETTINGS}}, indent=2))


METHODS = {"ppo": _ppo, "planner_teacher": _planner_teacher}


class ProgressWriter(LogWriter):
    """rsl_rl's log writer for our runs: TensorBoard as usual, and each
    round's numbers as a line of progress.jsonl and in the status file."""

    def __init__(self, log_dir, iterations):
        self.board = SummaryWriter(log_dir=log_dir, flush_secs=10)
        self.path = Path(log_dir) / PROGRESS
        self.iterations = iterations
        self.row = None

    def add_scalar(self, tag, scalar_value, global_step):
        self.board.add_scalar(tag, scalar_value, global_step)
        if self.row is not None and self.row["iteration"] != global_step:
            self._flush()
        if self.row is None:
            self.row = {"iteration": int(global_step)}
        self.row[tag] = float(scalar_value)

    def stop(self):
        if self.row is not None:
            self._flush()
        self.board.close()

    def _flush(self):
        with open(self.path, "a") as out:
            out.write(json.dumps(self.row) + "\n")
        done = self.row["iteration"] + 1
        background.report(done=done, total=self.iterations, text=(
            f"round {done} of {self.iterations}: {self.row.get('/track/turn_deg', np.nan):.1f} deg, "
            f"{self.row.get('/track/position_mm', np.nan):.1f} mm off while training"))
        self.row = None


def runs():
    """Run folders, newest first."""
    return sorted((f for f in RUNS.glob("*") if (f / "run.json").exists()), reverse=True) \
        if RUNS.exists() else []


def chosen_run():
    """The run Recordings step 7 uses: `learning.run`, or the newest with a
    checkpoint. None without one."""
    named = settings.get("learning.run")
    trained = [f for f in runs() if any(f.glob("model_*.pt"))]
    for folder in trained:
        if folder.name == named:
            return folder
    return trained[0] if trained else None


def progress(folder):
    """A run's progress.jsonl as a list of dicts (one per round)."""
    path = folder / PROGRESS
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:          # a line still being written
            pass
    return rows


def latest_checkpoint(folder):
    """The run's last model_<iteration>.pt."""
    checkpoints = sorted(folder.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint in {folder}")
    return checkpoints[-1]


def evaluate(folder, checkpoint=None, library=None, motions=None):
    """The policy on each motion (default: every recorded and every feasible
    generated one), one episode each, nominal physics. Returns per motion:
    {motion, recorded, source, dropped, position_mm, turn_deg (medians), the
    planner's position_mm and turn_deg, still_turn_deg}."""
    library = library or load()
    if motions is None:
        motions = np.flatnonzero(library["motion_recorded"] | library["motion_feasible"])
    results = []
    for motion, run in zip(motions, follow(folder, library, motions, checkpoint)):
        results.append({
            "motion": int(motion), "recorded": bool(library["motion_recorded"][motion]),
            "source": str(library["motion_source"][motion]), "dropped": run["dropped"],
            "position_mm": float(1000 * np.median(run["position_error"])),
            "turn_deg": float(np.degrees(np.median(run["turn_error"]))),
            "planner_position_mm": float(1000 * library["motion_position_error"][motion]),
            "planner_turn_deg": float(np.degrees(library["motion_turn_error"][motion])),
            "still_turn_deg": float(np.degrees(library["motion_still_turn"][motion]))})
    return results


def follow(folder, library, motions, checkpoint=None):
    """The trained policy following each motion once from its start, nominal
    physics, no noise. Per motion, a row per step (1/RATE s, the first one
    step after the object was let go): {joints (T, J), position (T, 3),
    quat (T, 4) - the object, mount frame - goal_position, goal_quat,
    position_error (T,) m, turn_error (T,) rad, dropped}."""
    env = TrackEnv(library, envs=len(motions), motions=motions, randomize=False, cycle=True)
    runner = OnPolicyRunner(env, copy.deepcopy(CONFIG), log_dir=None, device="cpu")
    runner.load(str(checkpoint or latest_checkpoint(folder)), load_cfg={"actor": True},
                strict=True)
    policy = runner.get_inference_policy()
    count = len(motions)
    finished, dropped = np.zeros(count, bool), np.zeros(count, bool)
    rows = [[] for _ in range(count)]
    obs = env.get_observations()
    with torch.inference_mode():
        while not finished.all():
            obs, _, done, extras = env.step(policy(obs))
            # env.state already holds the next episode for those done: measure
            # from the rollout's own last state.
            last = env._states[:, -1]
            positions, rotations = env._object(last)
            for i in np.flatnonzero(~finished):
                rows[i].append((last[i, env.joint_qpos], positions[i], _quat(rotations[i]),
                                *extras["errors"][i]))
            new = done.numpy() & ~finished
            dropped |= new & ~extras["time_outs"].numpy()
            finished |= new
    runs = []
    for i, motion in enumerate(motions):
        joints, positions, quats, position_error, turn_error = (np.array(c) for c in zip(*rows[i]))
        frames = np.minimum(np.arange(1, len(joints) + 1), library["motion_length"][motion] - 1)
        runs.append({"joints": joints, "position": positions, "quat": quats,
                     "goal_position": library["motion_position"][motion, frames],
                     "goal_quat": library["motion_quat"][motion, frames],
                     "position_error": position_error, "turn_error": turn_error,
                     "dropped": bool(dropped[i])})
    return runs


def _quat(rotation):
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def _plain(value):
    return value.tolist() if isinstance(value, np.ndarray) else (
        float(value) if isinstance(value, np.floating) else value)
