"""The reader (step 3 of the plan): what a camera's tracker has seen of a hand
lately -> a point in the motion space (motion_space.py), whose motion the
robot hand makes. The tracker is not trusted: it misreads, jitters, guesses
fingers an object hides and loses the hand; the reader sees its last SEEN
frames and the robot's own last PAST, and the point it gives decodes to a
motion of the robot hand like those the space was fitted to.

    report = fit(progress)                    # -> data/learning/reader.pt
    reader = Reader(load(), MotionSpace(motion_space.load()))
    reader.reset(joints)                      # where the robot is
    q, meant = reader.step(world, side)       # a tracked hand (21, 3) or None -> joints (J,), contacts

Learnt, not searched for: true motions are known - GRAB's people handling
objects and FreiHAND's hands (moved between as the motion space's clips
are), each with the robot's retargeted motion (grab.py, intent_pairs.py) -
and made into what the tracker would give (tracker_noise.py: WiLoR's own
errors on FreiHAND's photos, misreads that last, jitter, points the object
hides from a camera placed about the person, frames lost).

How it reads (`read`): a network gives the robot's next FUTURE poses, and
the motion space's own encoder (frozen) turns them, with the robot's past,
into the point; the decoder makes its motion. A network giving the point
itself, fitted through the decoder, learnt the hand badly: 9.3 degrees off
on people it never saw even given their true hands, where a network from
hand to robot pose alone gets 2.1 and the encoder, given the true motion,
follows it to 1.2 - finding the point was the hard part, and the encoder
already does it. A mixture over points fitted to the encoder's (one mode
chosen) was worse again, 15 degrees and fingers 20-35% inside each other:
nearness in the space is not nearness of the hand.

Each frame of a point's motion is counted as it is used (weight exp(-k /
HORIZON) on frame k): the first is made, those after only expected - and
a person's next 0.4 s is not in what the camera has seen. Counted alike,
they were hedged on, and the motion space's motions being smooth, so was
the first: the hand moved a third as much as the person's (variance), at
half their speed.

And the contacts meant now (`INTENTS`): the thumb on each finger (the true
robot's tips under TOUCH_PL) and each part of the hand on an object
(GRAB's contact labels; FreiHAND's hands have none) - each a yes or no,
fitted as one (cross-entropy), never averaged: a hand hedged between
touching and not is just open, which motion errors alone settle on. For the
physics policy (step 4), whose contacts are its own to make.

Fitted on its own motion: from a true past, ROLL frames on, each frame
the robot's past the frames it made itself (each point's first frame
taken, as `step` does) - as it is used, so its own errors do not run away.

Measured (`measure`) on clips it never saw, followed frame by frame from
start to end: GRAB's people 9-10 and FreiHAND's evaluation hands under the
same kinds of tracker errors, and FreiHAND's evaluation photos as WiLoR
really read them (the odd ones - the even ones set the errors' size); and
the solver that labels the data (human_retarget.HumanRetarget, frame by
frame from the last) given the same tracker's hands, for comparison.

    reader.pt   torch.save: state, config, thresholds (each contact's, calibrate), report, the
                motion space's made
"""

from __future__ import annotations

import time
from datetime import datetime
from multiprocessing import get_context

import numpy as np
import torch

from ..paths import ROOT
from . import grab, hand_library, intent_pairs
from . import motion_space as space_module
from .hand_contacts import HandContacts
from .hand_library import HandModel
from .handmotion import RATE, TOUCH_PL, in_palm
from .human_retarget import TIP, HumanRetarget
from .motion_space import (CLOSE, FUTURE, JOINT_SD, LATENT, PAST, PAST_NOISE, POINT_SD, RELATION_NEAR,
                           RELATION_SD, SPEED_SD, MotionSpace, Relations, _mlp)
from .torch_kinematics import TorchKinematics
from .tracker_noise import NEIGHBOURS, TrackerNoise

MODEL = ROOT / "data" / "learning" / "reader.pt"
SEEN = 10                     # tracker frames the reader sees: 0.5 s
POINTS = 21
ROLL = 8                      # frames fitted on its own motion
HORIZON = 1.0                 # frames: a point's motion counted less the further it looks ahead
PRIOR = 1.0                   # the prior's -log likelihood, times this
STEPS = 20000
BATCH = 256
LEARNING_RATE = 3e-4
SEEDS = (0, 1, 2)
HELD_OUT = 10
FREIHAND_CLIPS = {"freihand": ("training", 3000), "freihand held out": ("evaluation", 300)}
GRAB_PEOPLE = space_module.GRAB_PEOPLE
TEST_ONLY = ("freihand held out", "grab held out")
LEAD = max(SEEN - 1, PAST)    # still frames before a clip
COMPARED = 4                  # every this-th test clip also given to the solver
PHOTO_S = 2.0                 # a photo held this long, the robot from rest
JUMP = 20                     # degrees in a frame: a jump
TOUCHES = ("index", "middle", "ring")
INTENTS = len(TOUCHES) + len(grab.PARTS)
CONTACT = {"palm": [0], "thumb": [1, 2, 3], "index": [4, 5, 6], "middle": [7, 8, 9], "ring": [10, 11, 12],
           "little": [13, 14, 15]}          # grab.PARTS by finger


# ------------------------------------------------------------------ data
class Clip:
    """A true motion: the person's hand (T, 21, 3) in its palm frame, the
    robot's (T, J) rad, points hidden from each camera (T, CAMERAS, 21) or
    None, each pose's FreiHAND neighbours (T, NEIGHBOURS), and which of the
    hand's grab.PARTS touch an object (T, 16) - None where not known."""

    def __init__(self, person, robot, hidden, near, contact=None):
        self.person, self.robot, self.hidden, self.near = person, robot, hidden, near
        self.contact = contact


def clips(noise, seed=0):
    """{source: [Clip]} - GRAB's hands and FreiHAND's, see the docstring."""
    rng = np.random.default_rng(seed)
    out = {}
    for source, (part, count) in FREIHAND_CLIPS.items():
        pairs = dict(np.load(intent_pairs.FOLDER / f"freihand_labels_{part}.npz"))
        rows = np.arange(len(pairs["joints"]))
        if source in TEST_ONLY:
            rows = rows[1::2]                          # the even ones set tracker_noise's size
        person = in_palm(pairs["person"][rows].astype(float), intent_pairs.SIDE).astype(np.float32)
        out[source] = [_pose_clip(person, pairs["joints"][rows], noise, rng) for _ in range(count)]
    hands, labels = dict(np.load(grab.HANDS)), grab.load()
    views = dict(np.load(grab.VIEWS))
    take = labels["take"]
    edges = np.r_[0, np.flatnonzero(np.diff(take)) + 1, len(take)]
    for source, people in GRAB_PEOPLE.items():
        out[source] = []
        for a, b in zip(edges[:-1], edges[1:]):
            if labels["subject"][take[a]] not in people:
                continue
            for hand, side in enumerate(grab.SIDES):
                person = in_palm(hands["joints"][a:b, hand].astype(float), side).astype(np.float32)
                out[source].append(Clip(person, labels["joints"][a:b, hand].astype(np.float32),
                                        views["hidden"][a:b, hand], noise.neighbours(person),
                                        hands["contact"][a:b, hand]))
    return out


def _pose_clip(person, joints, noise, rng):
    """Key poses drawn from a set, both hands timed as the library's random
    clips (minimum jerk between): no object, nothing hidden."""
    keys = rng.integers(len(joints), size=int(rng.integers(*hand_library.POSES)))
    hold = [round(rng.uniform(*hand_library.HOLD_S) * RATE) for _ in keys]
    move = [max(2, round(rng.uniform(*hand_library.MOVE_S) * RATE)) for _ in keys[1:]] + [0]
    moved = hand_library.frames(person[keys].reshape(len(keys), -1), hold, move).reshape(-1, POINTS, 3)
    moved = in_palm(moved, "Left").astype(np.float32)          # its palm frame again, between keys
    return Clip(moved, hand_library.frames(joints[keys], hold, move).astype(np.float32), None,
                noise.neighbours(moved))


class Windows:
    """Every window of a set of clips: SEEN + ROLL - 1 tracker frames, the
    robot's PAST before the first `now` and FUTURE after each of ROLL (each
    clip led in by LEAD copies of its first frame: still before it)."""

    def __init__(self, clip_list):
        parts = {"person": [], "robot": [], "hidden": [], "near": [], "contact": [], "known": []}
        starts, offset = [], 0
        for clip in clip_list:
            length = len(clip.robot)
            hidden = clip.hidden if clip.hidden is not None else np.zeros((length, grab.CAMERAS, POINTS), bool)
            contact = clip.contact if clip.contact is not None else np.zeros((length, len(grab.PARTS)), bool)
            known = np.full(length, clip.contact is not None)
            for key, value in (("person", clip.person), ("robot", clip.robot), ("hidden", hidden),
                               ("near", clip.near), ("contact", contact), ("known", known)):
                parts[key].append(np.concatenate([np.repeat(value[:1], LEAD, 0), value]))
            length = LEAD + len(clip.robot)
            starts.append(offset + np.arange(LEAD, length - ROLL - FUTURE + 2))
            offset += length
        for key, value in parts.items():
            setattr(self, key, np.concatenate(value))
        self.starts = np.concatenate(starts)

    def batch(self, rows, noise, rng):
        """-> seen (B, SEEN + ROLL - 1, 21, 3), found (B, SEEN + ROLL - 1),
        past (B, PAST, J), futures (B, ROLL, FUTURE, J), and at each of ROLL
        frames the object's contact (B, ROLL, 16), known (B, ROLL)."""
        now = self.starts[rows]
        frames = now[:, None] + np.arange(1 - SEEN, ROLL)
        camera = rng.integers(grab.CAMERAS, size=len(rows))
        hidden = self.hidden[frames, camera[:, None]]
        seen, found = noise.corrupt(self.person[frames], self.near[frames], hidden, rng)
        past = self.robot[now[:, None] + np.arange(-PAST, 0)]
        futures = self.robot[now[:, None, None] + np.arange(ROLL)[:, None] + np.arange(FUTURE)]
        at = now[:, None] + np.arange(ROLL)
        return seen, found, past, futures, self.contact[at], self.known[at]


def split(sources):
    """{source: (fitted Windows or None, held-out [Clip])}."""
    out = {}
    for source, clip_list in sources.items():
        held = [c for i, c in enumerate(clip_list) if source in TEST_ONLY or i % HELD_OUT == HELD_OUT - 1]
        fitted = [c for i, c in enumerate(clip_list) if source not in TEST_ONLY and i % HELD_OUT != HELD_OUT - 1]
        out[source] = (Windows(fitted) if fitted else None, held)
    return out


# ------------------------------------------------------------------ the network
class Net(torch.nn.Module):
    """The tracker's last SEEN frames and the robot's last PAST -> the
    robot's next FUTURE poses, moved from its last and kept in its ranges -
    turned into a point by the motion space's own encoder (`read`)."""

    def __init__(self, low, high):
        super().__init__()
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))
        joints = len(low)
        self.read = _mlp(SEEN * (POINTS * 3 + 1) + PAST * joints, FUTURE * joints + INTENTS)

    def forward(self, seen, found, past):
        """seen (B, SEEN, 21, 3), found (B, SEEN), past (B, PAST, J) rad ->
        poses (B, FUTURE, J) rad, and the contacts meant now as logits (B,
        INTENTS): the thumb on each of TOUCHES, then grab.PARTS on an object."""
        scaled = (2 * past - self.low - self.high) / (self.high - self.low)
        x = torch.cat([seen.flatten(-2), found[..., None]], -1).flatten(-2)
        out = self.read(torch.cat([x, scaled.flatten(-2)], -1))
        move = out[..., :-INTENTS].unflatten(-1, (FUTURE, -1))
        poses = torch.clamp(past[..., -1:, :] + move * (self.high - self.low) / 2, self.low, self.high)
        return poses, out[..., -INTENTS:]


def read(net, space, seen, found, past):
    """-> the point (B, LATENT) the reader gives, the poses (B, FUTURE, J)
    it was read from, and the contacts' logits (B, INTENTS)."""
    poses, contacts = net(seen, found, past)
    z, _ = space.encode(past, poses)
    return z, poses, contacts


def losses(net, space, kinematics, relations, seen, found, past, futures, contact, known, noise=PAST_NOISE):
    """The fit's terms, each per window (nats, a mean over the ROLL frames):
    the reader's points over ROLL frames, the robot's past after the first
    its own; then all their motions' errors at once, the poses read, and
    the contacts meant (the thumb's touches from the true robot's tips;
    an object's where known)."""
    past = past + noise * torch.randn_like(past)
    pasts, points_given, made, read_poses, meant = [], [], [], [], []
    for step in range(ROLL):
        z, poses, contacts = read(net, space, seen[:, step:step + SEEN], found[:, step:step + SEEN], past)
        motion = space.decode(past, z)
        pasts.append(past)
        points_given.append(z)
        made.append(motion)
        read_poses.append(poses)
        meant.append(contacts)
        past = torch.cat([past[:, 1:], motion[:, :1].detach()], 1)
    pasts, z, made = torch.stack(pasts, 1), torch.stack(points_given, 1), torch.stack(made, 1)
    read_poses, meant = torch.stack(read_poses, 1), torch.stack(meant, 1)
    points_made, points = kinematics(made), kinematics(futures)
    apart_made, apart = relations(points_made), relations(points)
    near = torch.exp(-apart / RELATION_NEAR)
    speed_made = torch.diff(torch.cat([pasts[:, :, -1:], made], 2), dim=2)
    speed = torch.diff(torch.cat([pasts[:, :, -1:], futures], 2), dim=2)
    prior = space.prior_of(pasts.flatten(0, 1)).log_prob(z.flatten(0, 1)).view(len(z), ROLL)
    # Each frame of a point's motion counted as it will be used: the first
    # is made, those after it are what the robot expects - less and less.
    weight = torch.exp(-torch.arange(FUTURE, device=made.device) / HORIZON)
    frames = lambda x: (x.flatten(3).sum(-1) * weight).sum((1, 2)) / ROLL
    return {"joint": frames((made - futures) ** 2) / (2 * JOINT_SD ** 2),
            "point": frames((points_made - points) ** 2) / (2 * POINT_SD ** 2),
            "relation": frames(near * (apart_made - apart) ** 2) / (2 * RELATION_SD ** 2),
            "speed": frames((speed_made - speed) ** 2) / (2 * SPEED_SD ** 2),
            "pose": frames((read_poses - futures) ** 2) / (2 * JOINT_SD ** 2),
            "prior": -PRIOR * prior.mean(1),
            "touch": bce(meant[..., :len(TOUCHES)], _touching(points[:, :, 0]).float()).sum(-1).mean(1),
            "contact": (bce(meant[..., len(TOUCHES):], contact) * known[..., None]).sum(-1).mean(1)}


def bce(logits, target):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")


def _touching(points):
    """The thumb on each of TOUCHES (..., 3) bool, from the robot's 16 points."""
    return torch.stack([(points[..., TIP[f], :] - points[..., TIP["thumb"], :]).norm(dim=-1) for f in TOUCHES],
                       -1) < TOUCH_PL


# ------------------------------------------------------------------ fitting
def fit(progress=lambda done, total, text="": None, path=MODEL, seeds=SEEDS, steps=STEPS):
    """Fit SEEDS times, each measured on the held-out clips; the first kept."""
    saved = space_module.load()
    if not saved:
        raise RuntimeError("no motion space - fit it (scripts/fit_motion_space.py)")
    hand = HandModel()
    device = space_module._device()
    space = MotionSpace(saved).net.to(device).requires_grad_(False)
    noise = TrackerNoise()
    data = split(clips(noise))
    compared = _solver(data, noise, hand)
    reports, kept = [], None
    for k, seed in enumerate(seeds):
        def told(done, total, text, k=k):
            progress(k * steps + done, len(seeds) * steps, f"seed {seed}: {text}")
        net = _train(data, noise, space, hand, device, seed, steps, told)
        thresholds = calibrate(net, space, data, noise, hand)
        reports.append(measure(net, space, data, noise, hand, compared, thresholds))
        if kept is None:
            kept, kept_thresholds = net, thresholds
    report = space_module._summary(reports, seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in kept.state_dict().items()},
                "config": {"seen": SEEN, "past": PAST, "future": FUTURE, "latent": LATENT,
                           "low": hand.low.tolist(), "high": hand.high.tolist(),
                           "sources": {s: len(h) for s, (_, h) in data.items()}},
                "thresholds": kept_thresholds.tolist(), "report": report, "space_made": saved["made"],
                "made": datetime.now().isoformat(timespec="seconds")}, path)
    return report


def load(path=MODEL):
    return torch.load(path, weights_only=True) if path.exists() else {}


def _train(data, noise, space, hand, device, seed, steps, progress):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fitted = [w for w, _ in data.values() if w is not None]
    share = np.array([len(w.starts) for w in fitted], float)
    share /= share.sum()
    net = Net(hand.low, hand.high).to(device)
    kinematics = TorchKinematics(hand.model, hand.chain, hand.palm).to(device)
    relations = Relations().to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    started = time.monotonic()
    for step in range(steps):
        counts = rng.multinomial(BATCH, share)
        parts = [w.batch(rng.integers(len(w.starts), size=n), noise, rng) for w, n in zip(fitted, counts) if n]
        seen, found, past, futures, contact, known = (
            torch.as_tensor(np.concatenate(x), dtype=torch.float32, device=device) for x in zip(*parts))
        terms = losses(net, space, kinematics, relations, seen, found, past, futures, contact, known)
        loss = sum(v.mean() for v in terms.values())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % 500 == 0 or step == steps - 1:
            text = ", ".join(f"{k} {v.mean().item():.1f}" for k, v in terms.items())
            progress(step + 1, steps, f"{text} ({time.monotonic() - started:.0f} s)")
    return net.eval()


# ------------------------------------------------------------------ following
@torch.no_grad()
def follow(net, space, seen, found, start):
    """The robot's motion (B, T, J) as the reader makes it frame by frame
    from the tracker's hands (B, T, 21, 3), found (B, T), the robot still
    at `start` (B, J) before - each frame the first of its point's motion;
    and the contacts meant, as likelihoods (B, T, INTENTS)."""
    device = next(net.parameters()).device
    seen = torch.as_tensor(seen, dtype=torch.float32, device=device)
    found = torch.as_tensor(found, dtype=torch.float32, device=device)
    count, length = found.shape
    seen = torch.cat([seen.new_zeros(count, SEEN - 1, POINTS, 3), seen], 1)
    found = torch.cat([found.new_zeros(count, SEEN - 1), found], 1)
    past = torch.as_tensor(start, dtype=torch.float32, device=device)[:, None].repeat(1, PAST, 1)
    out, meant = [], []
    for t in range(length):
        z, _, contacts = read(net, space, seen[:, t:t + SEEN], found[:, t:t + SEEN], past)
        made = space.decode(past, z)[:, :1]
        out.append(made)
        meant.append(torch.sigmoid(contacts))
        past = torch.cat([past[:, 1:], made], 1)
    return torch.cat(out, 1).cpu().numpy(), torch.stack(meant, 1).cpu().numpy()


# ------------------------------------------------------------------ measuring
def calibrate(net, space, data, noise, hand, seed=0):
    """Each contact's threshold (INTENTS,): its best F1 on the held-out clips
    of the sources fitted (people and hands it was fitted on, clips it was
    not) under the tracker's errors at their measured size - never the
    test sources'."""
    kinematics = TorchKinematics(hand.model, hand.chain, hand.palm)
    meant, true, known = [], [], []
    for source, (fitted, held) in data.items():
        if fitted is None:
            continue
        truth, seen, found, _ = _tracked(held, noise, seed, 1.0)
        meant += _follow_all(net, space, seen, found, truth)[1]
        with torch.no_grad():
            touch = _touching(kinematics(torch.as_tensor(np.concatenate(truth)))).numpy()
        contact = np.concatenate([c.contact if c.contact is not None else np.zeros((len(c.robot), len(grab.PARTS)), bool)
                                  for c in held])
        true.append(np.concatenate([touch, contact], 1))
        known.append(np.concatenate([np.ones_like(touch), np.concatenate(
            [np.full((len(c.robot), len(grab.PARTS)), c.contact is not None) for c in held])], 1))
    meant, true, known = np.concatenate(meant), np.concatenate(true), np.concatenate(known)
    return np.array([_best_f1(meant[known[:, k], k], true[known[:, k], k]) for k in range(INTENTS)])


def _best_f1(likelihood, true):
    """The threshold with the best F1, of 200 quantiles of `likelihood`."""
    if not true.any():
        return 0.5
    best, threshold = -1.0, 0.5
    for at in np.unique(np.quantile(likelihood, np.linspace(0, 1, 200, endpoint=False))):
        said = likelihood > at
        caught = (said & true).sum()
        f1 = 2 * caught / max(said.sum() + true.sum(), 1)
        if f1 > best:
            best, threshold = f1, float(at)
    return threshold


def measure(net, space, data, noise, hand, compared, thresholds, seed=0):
    """Per held-out source, followed from start to end under the tracker's
    errors at their measured size (and twice it), and on FreiHAND's photos:
    the reader's motion against the true one (scores()) and the contacts it
    means (intents(), at `thresholds`), and the solver's on every
    COMPARED-th clip (`compared`) with the reader's on those."""
    out = {}
    for source, (_, held) in data.items():
        for scale in (1.0, 2.0):
            truth, seen, found, hidden = _tracked(held, noise, seed, scale)
            made, meant = _follow_all(net, space, seen, found, truth)
            name = source if scale == 1.0 else f"{source}, errors x{scale:g}"
            out[name] = {"reader": scores(made, truth, found, hidden, hand),
                         "reader's contacts": intents(meant, truth, [c.contact for c in held], hand, thresholds)}
            if (source, scale) in compared:
                rows = range(0, len(held), COMPARED)
                out[name]["reader, compared clips"] = scores(
                    [made[r] for r in rows], [truth[r] for r in rows], [found[r] for r in rows],
                    [hidden[r] for r in rows], hand)
                out[name]["solver, compared clips"] = compared[(source, scale)]
    out["freihand photos"] = _photos(net, space, hand, compared, thresholds)
    return out


def _tracked(held, noise, seed, scale):
    """Each clip's truth (T, J), the tracker's hands (T, 21, 3), found
    (T,), hidden (T, 21) - from camera 0 - under errors `scale` times the
    measured size."""
    rng = np.random.default_rng(seed)
    truth, seen, found, hidden = [], [], [], []
    for clip in held:
        view = clip.hidden[:, 0] if clip.hidden is not None else np.zeros((len(clip.robot), POINTS), bool)
        s, f = noise.corrupt(clip.person[None], clip.near[None], view[None], rng, scale)
        truth.append(clip.robot)
        seen.append(s[0])
        found.append(f[0])
        hidden.append(view)
    return truth, seen, found, hidden


def _follow_all(net, space, seen, found, truth, batch=512):
    """follow() over clips of any lengths, padded in batches, each from
    its own first true pose."""
    out, meant = [], []
    for a in range(0, len(seen), batch):
        part = slice(a, a + batch)
        length = max(len(s) for s in seen[part])
        pad = lambda x: np.stack([np.concatenate([v, np.repeat(v[-1:] * 0, length - len(v), 0)]) for v in x])
        made, contacts = follow(net, space, pad(seen[part]), pad(found[part]), np.stack([t[0] for t in truth[part]]))
        out += [m[:len(t)] for m, t in zip(made, truth[part])]
        meant += [c[:len(t)] for c, t in zip(contacts, truth[part])]
    return out, meant


def scores(made, truth, found, hidden, hand):
    """A followed motion against the true one, over all frames of all clips
    (lists of (T, J), (T,), (T, 21)):

    joints_deg    mean joint error; points_mm the 16 points' (this robot's
                  palm); close_mm the distances between points of
                  different fingers under CLOSE apart in the truth (contacts
                  of any kind), how far off
    touch         the thumb on each fingertip (tips under TOUCH_PL): of
                  the truth's, the share made (caught), and of frames
                  without, the share made anyway (false)
    speed_deg_s   mean joint speed, and the truth's; jumps the share of
                  frames a joint moves over JUMP degrees, and the truth's
    inside        share of frames with fingers in each other
    hidden, lost  joints_deg on frames with a point hidden, and with no
                  hand found
    """
    kinematics = TorchKinematics(hand.model, hand.chain, hand.palm)
    relations = Relations()
    mm = hand.palm[2] * 1000
    q, t = np.concatenate(made), np.concatenate(truth)
    found, hidden = np.concatenate(found), np.concatenate(hidden).any(-1)
    error = np.degrees(np.abs(q - t)).mean(-1)
    with torch.no_grad():
        points_made, points = kinematics(torch.as_tensor(q)), kinematics(torch.as_tensor(t))
        apart_made, apart = relations(points_made), relations(points)
    close = apart < CLOSE
    tips = lambda p: torch.stack([(p[:, TIP[f]] - p[:, TIP["thumb"]]).norm(dim=-1) for f in TOUCHES], 1).numpy()
    touch_made, touch = tips(points_made) < TOUCH_PL, tips(points) < TOUCH_PL
    step = lambda m: [np.degrees(np.abs(np.diff(x, axis=0))) for x in m]
    steps_made, steps = np.concatenate(step(made)), np.concatenate(step(truth))
    sample = np.random.default_rng(0).choice(len(q), size=min(4000, len(q)), replace=False)
    return {"joints_deg": float(error.mean()),
            "points_mm": float((points_made - points).norm(dim=-1).mean() * mm),
            "close_mm": float((apart_made - apart).abs()[close].mean() * mm),
            "touch": {f: {"caught": _share(touch_made[touch[:, k], k]), "false": _share(touch_made[~touch[:, k], k])}
                      for k, f in enumerate(TOUCHES)},
            "speed_deg_s": float(steps_made.mean() * RATE), "truth_speed_deg_s": float(steps.mean() * RATE),
            "jumps": float((steps_made.max(-1) > JUMP).mean()), "truth_jumps": float((steps.max(-1) > JUMP).mean()),
            "inside": float(HandContacts(hand).share_inside(q[sample])),
            "hidden_joints_deg": float(error[hidden].mean()) if hidden.any() else None,
            "lost_joints_deg": float(error[~found].mean()) if (~found).any() else None,
            "frames": int(len(q))}


def _share(flags):
    return float(flags.mean()) if len(flags) else None


def intents(meant, truth, contact, hand, thresholds):
    """The contacts the reader means (likelihoods over `thresholds`)
    against the truth's, over all frames: the thumb on each of TOUCHES (the
    true robot's tips under TOUCH_PL), and each part of the hand on an
    object (CONTACT groups of grab.PARTS, any of the group; where known) -
    caught and false as in scores(), and auc: how surely a frame with the
    contact is put above one without (1 always, 0.5 a guess)."""
    kinematics = TorchKinematics(hand.model, hand.chain, hand.palm)
    likely = np.concatenate(meant)
    said = likely > thresholds
    with torch.no_grad():
        touch = _touching(kinematics(torch.as_tensor(np.concatenate(truth)))).numpy()
    rate = lambda s, t, l: {"caught": _share(s[t]), "false": _share(s[~t]), "auc": _auc(l, t),
                            "truth": float(t.mean())}
    out = {"touch": {f: rate(said[:, k], touch[:, k], likely[:, k]) for k, f in enumerate(TOUCHES)}}
    known = [c for c in contact if c is not None]
    if known:
        rows = np.concatenate([np.full(len(t), c is not None) for t, c in zip(truth, contact)])
        true = np.concatenate(known)
        on, sure = said[rows, len(TOUCHES):], likely[rows, len(TOUCHES):]
        out["object"] = {name: rate(on[:, parts].any(1), true[:, parts].any(1), sure[:, parts].max(1))
                         for name, parts in CONTACT.items()}
    return out


def _auc(likelihood, true):
    """The area under the ROC curve (Mann-Whitney): None without both kinds."""
    if true.all() or not true.any():
        return None
    ranks = np.empty(len(likelihood))
    ranks[np.argsort(likelihood, kind="stable")] = np.arange(1, len(likelihood) + 1)
    count = true.sum()
    return float((ranks[true].sum() - count * (count + 1) / 2) / (count * (len(true) - count)))


def _photos(net, space, hand, compared, thresholds):
    """FreiHAND's evaluation photos (the odd ones) as WiLoR read them: each
    held PHOTO_S, the robot from rest; its pose then against the label."""
    pairs = intent_pairs.load("evaluation")
    rows = np.arange(len(pairs["joints"]))[1::2]
    rows = rows[pairs["found"][rows].astype(bool)]
    seen = in_palm(pairs["seen"][rows].astype(float), intent_pairs.SIDE).astype(np.float32)
    length = round(PHOTO_S * RATE)
    rest = np.clip(np.zeros(hand.model.nq), hand.low, hand.high)
    made, meant = follow(net, space, np.repeat(seen[:, None], length, 1), np.ones((len(rows), length)),
                         np.repeat(rest[None], len(rows), 0))
    made, meant = made[:, -1], meant[:, -1]
    truth = pairs["joints"][rows].astype(np.float32)
    nothing = np.zeros((len(rows), POINTS), bool)
    one = lambda x: [v[None] for v in x]
    return {"reader": scores(one(made), one(truth), one(np.ones(len(rows), bool)), one(nothing), hand),
            "reader's contacts": intents(one(meant), one(truth), [None] * len(rows), hand, thresholds),
            "solver": compared["photos"]}


# ------------------------------------------------------------------ the solver, for comparison
def _solver(data, noise, hand, seed=0):
    """The labelling solver given the tracker's hands, frame by frame from
    the last (a lost frame holds), on every COMPARED-th held-out clip at
    both error sizes; and on the photos, each solved once from rest."""
    out = {}
    with get_context("spawn").Pool(intent_pairs.WORKERS, initializer=_start) as pool:
        for source, (_, held) in data.items():
            for scale in (1.0, 2.0):
                truth, seen, found, hidden = _tracked(held, noise, seed, scale)
                rows = list(range(0, len(held), COMPARED))
                made = pool.map(_solve_clip, [(seen[r], found[r], truth[r][0]) for r in rows])
                out[(source, scale)] = scores(made, [truth[r] for r in rows], [found[r] for r in rows],
                                              [hidden[r] for r in rows], hand)
        pairs = intent_pairs.load("evaluation")
        rows = np.arange(len(pairs["joints"]))[1::2]
        rows = rows[pairs["found"][rows].astype(bool)]
        seen = in_palm(pairs["seen"][rows].astype(float), intent_pairs.SIDE)
        made = np.array(pool.map(_solve_photo, list(seen), chunksize=64))
    truth = pairs["joints"][rows].astype(np.float32)
    nothing = np.zeros((len(rows), POINTS), bool)
    out["photos"] = scores([made], [truth], [np.ones(len(rows), bool)], [nothing], hand)
    return out


_worker = None


def _start():
    global _worker
    _worker = HumanRetarget()


def _solve_clip(job):
    seen, found, start = job
    out, last = [], np.asarray(start, float)
    for points, ok in zip(seen, found):
        if ok:
            last, _ = _worker(points, "Left", start=last)
        out.append(last)
    return np.array(out, np.float32)


def _solve_photo(points):
    q, _ = _worker(points, "Left")
    return q.astype(np.float32)


# ------------------------------------------------------------------ live
class Reader:
    """The fitted reader, frame by frame, on the CPU: `step` once a frame at
    RATE, with the tracker's hand or None when it has none."""

    def __init__(self, saved, space):
        config = saved["config"]
        if (config["seen"], config["past"], config["future"], config["latent"]) != (SEEN, PAST, FUTURE, LATENT):
            raise RuntimeError("reader.pt was fitted with other sizes - fit it again")
        if saved["space_made"] != space_module.load()["made"]:
            raise RuntimeError("the motion space was fitted again since the reader - fit the reader again")
        self.net = Net(np.array(config["low"]), np.array(config["high"]))
        self.net.load_state_dict(saved["state"])
        self.net.eval()
        self.space = space.net
        self.report = saved["report"]
        self.thresholds = np.array(saved["thresholds"])       # a contact meant over its own
        self.seen = np.zeros((SEEN, POINTS, 3), np.float32)
        self.found = np.zeros(SEEN, np.float32)
        self.reset(np.clip(np.zeros(len(config["low"])), config["low"], config["high"]))

    def reset(self, joints):
        """The robot at `joints` (J,), nothing seen."""
        self.seen[:] = 0
        self.found[:] = 0
        self.past = np.repeat(np.asarray(joints, np.float32)[None], PAST, 0)

    @torch.no_grad()
    def step(self, world, side):
        """The tracker's hand this frame (21, 3) of a `side` hand, or None ->
        the robot's joints (J,) for it, and the contacts it means (INTENTS,)
        likelihoods: the thumb on each of TOUCHES, then grab.PARTS on an
        object."""
        self.seen = np.roll(self.seen, -1, 0)
        self.found = np.roll(self.found, -1, 0)
        self.seen[-1] = 0 if world is None else in_palm(np.asarray(world, float), side)
        self.found[-1] = world is not None
        past = torch.as_tensor(self.past)[None]
        z, _, contacts = read(self.net, self.space, torch.as_tensor(self.seen)[None],
                              torch.as_tensor(self.found)[None], past)
        q = self.space.decode(past, z)[0, 0].numpy()
        self.past = np.concatenate([self.past[1:], q[None]])
        return q, torch.sigmoid(contacts[0]).numpy()
