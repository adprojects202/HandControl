"""The planner teaches the tracker (DAgger): the policy learns to copy the
planner's servo targets, at the states the policy itself gets into.

    teach(library, folder, rounds)       # train.METHODS["planner_teacher"]

Why: by trial and error (PPO) the tracker learned to hold the cube but not
to roll it - those finger motions are too rare to stumble on. The planner
(sim/planner.py) finds them by search at every step, but it is slow and
needs the sim's exact state. So the planner labels, and the policy copies.

A round runs every training motion once, in TrackEnv - noisy sensors and
randomized physics, as the policy sees it. At every step the planner plans
from that env's exact state (continuing its last plan there) and gives the
servo targets it would set a step on: the label, as the policy's action
(the change of target over ACTION_SCALE). With probability BETA[round] the
env takes the label, else the policy's own action - DAgger: the policy then
meets the states its own mistakes lead to, and those get labels too. Every
(observation, label) pair is kept; after each round the policy is fitted to
all of them (mean squared error, EPOCHS passes).

Round 0 has BETA 1: the planner drives, plain imitation. Each round's
checkpoint is rsl_rl's (the actor with its observation normalization), as
PPO's are, so Test, Runs and Real to Sim take it unchanged. The
progress file gets, per round: the error while it ran (the planner's and
the policy's actions mixed by BETA), the fit's loss, and how many labels.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from . import background
from .env import ACTION_SCALE, TrackEnv
from .trajectories import RATE
from ..sim.planner import Goal, Planner

BETA = (1.0, 0.5, 0.25, 0.0)     # per round; later rounds use the last
EPOCHS = 30
BATCH = 4096
LEARNING_RATE = 1e-3
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


def teach(library, folder, rounds, config, writer, seed=0):
    """Train the tracker by the planner's labels over `rounds` rounds;
    `config` is the rsl_rl configuration its actor is built from, `writer`
    a train.ProgressWriter."""
    motions = np.flatnonzero(library["motion_feasible"] & ~library["motion_recorded"])
    env = TrackEnv(library, envs=len(motions), motions=motions, cycle=True, seed=seed)
    runner = OnPolicyRunner(env, copy.deepcopy(config), log_dir=None, device="cpu")
    actor = runner.alg.get_policy()
    optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    planner = Planner(env.task, seed=seed)
    rng = np.random.default_rng(seed)
    seen, labels = [], []
    for round_ in range(rounds):
        beta = BETA[min(round_, len(BETA) - 1)]
        observations, targets, errors = _collect(env, planner, actor, beta, rng,
                                                 round_, rounds)
        seen.append(observations)
        labels.append(targets)
        background.report(text=f"round {round_ + 1} of {rounds}: fitting the policy "
                               f"to {sum(len(s) for s in seen)} labels")
        actor.update_normalization(_obs(torch.as_tensor(observations)))
        loss = _fit(actor, optimizer, torch.as_tensor(np.concatenate(seen)),
                    torch.as_tensor(np.concatenate(labels)), rng)
        runner.save(str(folder / f"model_{round_}.pt"))
        for tag, value in (("/track/position_mm", 1000 * errors[:, 0].mean()),
                           ("/track/turn_deg", np.degrees(errors[:, 1]).mean()),
                           ("/track/drops", errors[:, 2].sum()),
                           ("Teacher/beta", beta), ("Teacher/loss", loss),
                           ("Teacher/labels", sum(len(s) for s in seen))):
            writer.add_scalar(tag, float(value), round_)
    writer.stop()


def _collect(env, planner, actor, beta, rng, round_, rounds):
    """One episode of every motion, the planner labelling each step.
    Returns the policy's observations (n, obs), the labels (n, nu), and
    each step's (position error, turn error, dropped) (n, 3)."""
    model = env.task.model
    data = mujoco.MjData(model)
    data.eq_active[:] = 0
    goals = []
    for motion in env.motions:
        length = int(env.library["motion_length"][motion])
        goals.append(Goal(np.arange(length) / RATE, env.library["motion_position"][motion, :length],
                          env.library["motion_quat"][motion, :length], np.ones(length, bool),
                          start=0.0))
    expected = int(env.library["motion_length"][env.motions].sum())
    nominals = [None] * env.num_envs
    finished = np.zeros(env.num_envs, bool)
    observations, targets, errors = [], [], []
    obs = env.restart()
    while not finished.all():
        active = np.flatnonzero(~finished)
        labels = np.zeros((env.num_envs, env.num_actions))
        for i in active:
            mujoco.mj_setState(model, data, env.state[i], FULL)
            data.ctrl[:] = env.targets[i]
            planner.nominal = nominals[i]
            planner.plan(data, goals[i], since=1.0 / RATE)
            nominals[i] = planner.nominal
            labels[i] = np.clip((planner.control(1.0 / RATE) - env.targets[i]) / ACTION_SCALE,
                                -1.0, 1.0)
        with torch.inference_mode():
            own = actor(obs).numpy()
        observations.append(obs["policy"].numpy()[active])
        targets.append(labels[active])
        planner_drives = rng.random(env.num_envs) < beta
        action = np.where(planner_drives[:, None], labels, own)
        obs, _, done, extras = env.step(torch.as_tensor(action, dtype=torch.float32))
        done = done.numpy()
        dropped = done & ~extras["time_outs"].numpy()
        errors.append(np.column_stack([extras["errors"][active], dropped[active]]))
        finished |= done
        background.report(done=round_ + min(1.0, sum(len(t) for t in targets) / expected),
                          total=rounds, text=f"round {round_ + 1} of {rounds}: the planner "
                                             f"labels step {len(observations)}")
    return (np.concatenate(observations).astype(np.float32),
            np.concatenate(targets).astype(np.float32), np.concatenate(errors))


def _fit(actor, optimizer, observations, labels, rng):
    """EPOCHS passes of mean squared error; returns the last pass's loss."""
    actor.train()
    count = len(observations)
    for _ in range(EPOCHS):
        order = torch.as_tensor(rng.permutation(count))
        total = 0.0
        for start in range(0, count, BATCH):
            batch = order[start:start + BATCH]
            loss = torch.nn.functional.mse_loss(actor(_obs(observations[batch])), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
    actor.eval()
    return total / count


def _obs(policy):
    return TensorDict({"policy": policy}, batch_size=[len(policy)])


def describe():
    """The method's settings, for run.json."""
    return {"beta": list(BETA), "epochs": EPOCHS, "batch": BATCH,
            "learning_rate": LEARNING_RATE}

