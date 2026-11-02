"""The motion space: the robot hand's motions as points in a learnt space -
the intent space (step 2 of the plan). What a person means, read from a
camera that sees their hand badly or not at all (fingers behind an object),
is to land here; what the robot is given comes out of it - a motion of the
robot hand like those it was fitted to, never the camera's numbers copied.

    report = fit(progress)                  # -> data/learning/motion_space.pt
    space = MotionSpace(load())
    z = space.encode(past, future)          # (PAST, J), (FUTURE, J) rad -> (LATENT,)
    future = space.decode(past, z)          # the motion a point means
    z = space.sample(past, seed)            # a likely next motion (the prior)
    frames = space.roll(past, count, rng)   # the prior's motions, on and on

A conditional VAE with a learnt prior - as PULSE (Luo et al., ICLR 2024)
and Motion VAEs (Ling et al., SIGGRAPH 2020) - over the robot's 16 joints,
kinematic: the hand-motion policy (gpu_hand.py) makes a motion physical.
A point z is the next FUTURE frames (at RATE) after the PAST ones:

    encoder   past and future -> z (mean, spread)
    prior     past -> where z is likely: the motions that can follow, a
              mixture of PRIOR_MODES (from an open hand a touch, a fist, a
              sign... - one Gaussian over all of them spans the space
              between, where no motion was ever seen)
    decoder   past and z -> future: moves from the last frame, kept inside
              the joints' ranges

Fitted as a VAE whose errors are scaled by how closely each thing is to be
kept (the ..._SD constants): the joints; the hand's 16 points through its
kinematics (torch_kinematics); every distance between points on different
fingers, the close ones counted most (contacts of any kind - fingertips, a
finger's side, an object's surface once objects come in); the speeds; and
the encoder held to the prior (KL). And a motion drawn from the prior must
not put fingers inside each other - the hand sim's own collision geometry
(hand_contacts), on one frame of each: the data alone never says so, and
draws between what it shows went 10-25% inside (to its 0-3%). The past is given with noise
(PAST_NOISE), so rolling the decoder on its own output does not drift.

Sources - clips of joints at RATE (`clips()`):

    library:<kind>      the hand-motion library's generated clips (random,
                        touch, gesture, wave): key poses, minimum-jerk
                        between
    freihand            FreiHAND's training hands retargeted
                        (intent_pairs): key poses people's hands make, moved
                        between as the library's random clips are
    freihand held out   its evaluation hands so - never fitted: does the
                        space hold people's poses it has not seen?
    grab                GRAB's people handling objects (grab.py), both
                        hands of every take of GRAB_PEOPLE's first eight:
                        people's own motion, timing, grasps and rests
    grab held out       its last two people - never fitted

The library's and FreiHAND's motions are key poses joined by minimum jerk;
GRAB's are people's own. Every HELD_OUT-th clip of each source is held out and measured
(`report`), the fit made SEEDS times to see its spread; the first seed's is
kept.

    motion_space.pt   torch.save: state (the network), config (sizes,
                      joint names, ranges), report, made
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import torch

from ..paths import ROOT
from . import grab, hand_library, intent_pairs
from .hand_contacts import HandContacts
from .hand_library import HandModel
from .handmotion import RATE
from .torch_kinematics import TorchKinematics

MODEL = ROOT / "data" / "learning" / "motion_space.pt"
PAST, FUTURE = 4, 8           # frames at RATE: 0.2 s seen, 0.4 s a point means
LATENT = 32
PRIOR_MODES = 8               # the prior: a mixture - the different motions that can follow
HIDDEN = (512, 512, 512)
JOINT_SD = 0.05               # rad
POINT_SD = 0.02               # palm lengths (~2.4 mm)
RELATION_SD = 0.01            # palm lengths, between close points on different fingers
RELATION_NEAR = 0.15          # palm lengths: how close counts most (weight exp(-d / this))
SPEED_SD = 0.01               # rad a frame
PAST_NOISE = 0.02             # rad
KL_RAMP = 0.2                 # share of the fit over which the KL's weight rises to 1
STEPS = 30000
BATCH = 512
LEARNING_RATE = 3e-4
SEEDS = (0, 1, 2)
HELD_OUT = 10
FREIHAND_CLIPS = {"freihand": ("training", 3000), "freihand held out": ("evaluation", 300)}
GRAB_PEOPLE = {"grab": ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"), "grab held out": ("s9", "s10")}
TEST_ONLY = ("freihand held out", "grab held out")
MEASURED = 2000               # held-out windows measured per source
ROLLED = 200                  # of them, rolled on by the prior
ROLL_S = 5.0                  # the prior's own motions, measured
CLOSE = 0.2                   # palm lengths: close points, for the contacts measure
INSIDE_SD = 0.001             # m: fingers' depth in each other beyond hand_contacts.INSIDE


# ------------------------------------------------------------------ data
def clips(seed=0):
    """{source: [clip (T, J) float32 rad]} - see the module docstring."""
    rng = np.random.default_rng(seed)
    library = hand_library.load()
    if not library:
        raise RuntimeError("no hand library - build it (scripts/build_hand_library.py)")
    out = {}
    first = library["key_first"]
    for clip, kind in enumerate(library["kind"]):
        if library["recorded"][clip]:
            continue
        keys = slice(first[clip], first[clip + 1])
        out.setdefault(f"library:{kind}", []).append(hand_library.frames(
            library["key_joints"][keys], library["key_hold"][keys],
            library["key_move"][keys]).astype(np.float32))
    for source, (part, count) in FREIHAND_CLIPS.items():
        path = intent_pairs.FOLDER / f"freihand_labels_{part}.npz"
        if path.exists():
            poses = np.load(path)["joints"]
            out[source] = [_pose_clip(poses, rng) for _ in range(count)]
    labels = grab.load()
    for source, people in GRAB_PEOPLE.items() if labels else ():
        out[source] = grab.clips(labels, people)
    return out


def _pose_clip(poses, rng):
    """Key poses drawn from `poses`, timed as the library's random clips."""
    keys = poses[rng.integers(len(poses), size=int(rng.integers(*hand_library.POSES)))]
    hold = [round(rng.uniform(*hand_library.HOLD_S) * RATE) for _ in keys]
    move = [max(2, round(rng.uniform(*hand_library.MOVE_S) * RATE)) for _ in keys[1:]] + [0]
    return hand_library.frames(keys, hold, move).astype(np.float32)


class Windows:
    """Every PAST + FUTURE window of a set of clips, as one array of frames
    (each clip led in by PAST - 1 copies of its first: still before it)."""

    def __init__(self, clip_list):
        frames, starts = [], []
        offset = 0
        for clip in clip_list:
            padded = np.concatenate([np.repeat(clip[:1], PAST - 1, 0), clip])
            frames.append(padded)
            starts.append(offset + np.arange(len(padded) - PAST - FUTURE + 1))
            offset += len(padded)
        self.frames = np.concatenate(frames)
        self.starts = np.concatenate(starts)

    def batch(self, rows):
        """(past (B, PAST, J), future (B, FUTURE, J)) for window rows."""
        window = self.frames[self.starts[rows][:, None] + np.arange(PAST + FUTURE)]
        return window[:, :PAST], window[:, PAST:]


def split(sources):
    """{source: (fitted Windows or None, held-out Windows)}."""
    out = {}
    for source, clip_list in sources.items():
        held = [c for i, c in enumerate(clip_list)
                if source in TEST_ONLY or i % HELD_OUT == HELD_OUT - 1]
        fitted = [c for i, c in enumerate(clip_list)
                  if source not in TEST_ONLY and i % HELD_OUT != HELD_OUT - 1]
        out[source] = (Windows(fitted) if fitted else None, Windows(held))
    return out


# ------------------------------------------------------------------ the network
class Net(torch.nn.Module):
    def __init__(self, low, high):
        super().__init__()
        joints = len(low)
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))
        self.encoder = _mlp((PAST + FUTURE) * joints, 2 * LATENT)
        self.prior = _mlp(PAST * joints, PRIOR_MODES * (2 * LATENT + 1))
        self.decoder = _mlp(PAST * joints + LATENT, FUTURE * joints)

    def _scaled(self, q):
        return (2 * q - self.low - self.high) / (self.high - self.low)

    def encode(self, past, future):
        """-> (mean, log spread) (..., LATENT)."""
        x = torch.cat([self._scaled(past), self._scaled(future)], -2).flatten(-2)
        return self.encoder(x).chunk(2, -1)

    def prior_of(self, past):
        """-> where z is likely after `past`: a mixture of PRIOR_MODES Gaussians."""
        out = self.prior(self._scaled(past).flatten(-2)).unflatten(-1, (PRIOR_MODES, -1))
        mean, log_sd, logit = out[..., :LATENT], out[..., LATENT:-1], out[..., -1]
        modes = torch.distributions.Independent(
            torch.distributions.Normal(mean, log_sd.clamp(-7, 2).exp()), 1)
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logit), modes)

    def decode(self, past, z):
        """-> future (..., FUTURE, J): moves from the last past frame, clamped
        to the joints' ranges."""
        x = torch.cat([self._scaled(past).flatten(-2), z], -1)
        move = self.decoder(x).unflatten(-1, (FUTURE, -1)) * (self.high - self.low) / 2
        return torch.clamp(past[..., -1:, :] + move, self.low, self.high)


def _mlp(inputs, outputs):
    layers, size = [], inputs
    for hidden in HIDDEN:
        layers += [torch.nn.Linear(size, hidden), torch.nn.ELU()]
        size = hidden
    return torch.nn.Sequential(*layers, torch.nn.Linear(size, outputs))


class Relations(torch.nn.Module):
    """Distances between the 16 points of different fingers (..., 96)."""

    def __init__(self):
        super().__init__()
        finger = np.repeat(np.arange(4), 4)
        a, b = np.triu_indices(16, 1)
        apart = finger[a] != finger[b]
        self.register_buffer("a", torch.as_tensor(a[apart]))
        self.register_buffer("b", torch.as_tensor(b[apart]))

    def forward(self, points):
        return (points[..., self.a, :] - points[..., self.b, :]).norm(dim=-1)


def losses(net, kinematics, relations, past, future, noise=PAST_NOISE, contacts=None):
    """The fit's terms, each per window and summed over its numbers (nats).
    With `contacts` (hand_contacts.HandContacts), "inside" too: a motion drawn
    from the prior, one frame of it, its fingers' depth in each other."""
    seen = past + noise * torch.randn_like(past)
    mean, log_sd = net.encode(seen, future)
    posterior = torch.distributions.Normal(mean, log_sd.clamp(-7, 2).exp())
    z = posterior.rsample()
    made = net.decode(seen, z)
    points_made, points = kinematics(made), kinematics(future)
    apart_made, apart = relations(points_made), relations(points)
    near = torch.exp(-apart / RELATION_NEAR)
    speed_made = torch.diff(torch.cat([past[:, -1:], made], 1), dim=1)
    speed = torch.diff(torch.cat([past[:, -1:], future], 1), dim=1)
    # The KL to a mixture has no closed form: estimated at the z drawn.
    prior = net.prior_of(seen)
    kl = posterior.log_prob(z).sum(-1) - prior.log_prob(z)
    terms = {"joint": ((made - future) ** 2).sum((1, 2)) / (2 * JOINT_SD ** 2),
             "point": ((points_made - points) ** 2).sum((1, 2, 3)) / (2 * POINT_SD ** 2),
             "relation": (near * (apart_made - apart) ** 2).sum((1, 2)) / (2 * RELATION_SD ** 2),
             "speed": ((speed_made - speed) ** 2).sum((1, 2)) / (2 * SPEED_SD ** 2),
             "kl": kl}
    if contacts is not None:
        # Every likely point decodes clear of itself, not only those the
        # data put there: the decoder learns where the prior's draws land.
        drawn = net.decode(seen, prior.sample())
        frame = drawn[torch.arange(len(drawn)), torch.randint(FUTURE, (len(drawn),))]
        terms["inside"] = contacts.cost(frame) / INSIDE_SD ** 2
    return terms


# ------------------------------------------------------------------ fitting
def fit(progress=lambda done, total, text="": None, path=MODEL, seeds=SEEDS, steps=STEPS):
    """Fit SEEDS times, each measured on the held-out clips; the first kept."""
    hand = HandModel()
    device = _device()
    sources = clips()
    data = split(sources)
    reference = _reference(data)
    reports, kept = [], None
    for k, seed in enumerate(seeds):
        def told(done, total, text, k=k):
            progress(k * steps + done, len(seeds) * steps, f"seed {seed}: {text}")
        net = _train(data, hand, device, seed, steps, told)
        reports.append(measure(net, data, hand, reference))
        if kept is None:
            kept = net
    report = _summary(reports, seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in kept.state_dict().items()},
                "config": {"past": PAST, "future": FUTURE, "latent": LATENT, "modes": PRIOR_MODES,
                           "hidden": list(HIDDEN),
                           "rate": RATE, "low": hand.low.tolist(), "high": hand.high.tolist(),
                           "joints": [hand.model.joint(j).name for j in range(hand.model.njnt)],
                           "sources": {s: len(c) for s, c in sources.items()}},
                "report": report, "made": datetime.now().isoformat(timespec="seconds")}, path)
    return report


def load(path=MODEL):
    return torch.load(path, weights_only=True) if path.exists() else {}


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _train(data, hand, device, seed, steps, progress):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fitted = [w for w, _ in data.values() if w is not None]
    # Each source drawn as often as its share of windows - the library's
    # kinds and FreiHAND's poses alike.
    share = np.array([len(w.starts) for w in fitted], float)
    share /= share.sum()
    net = Net(hand.low, hand.high).to(device)
    kinematics = TorchKinematics(hand.model, hand.chain, hand.palm).to(device)
    relations = Relations().to(device)
    contacts = HandContacts(hand)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    started = time.monotonic()
    for step in range(steps):
        counts = rng.multinomial(BATCH, share)
        parts = [w.batch(rng.integers(len(w.starts), size=n)) for w, n in zip(fitted, counts) if n]
        past = torch.as_tensor(np.concatenate([p for p, _ in parts]), device=device)
        future = torch.as_tensor(np.concatenate([f for _, f in parts]), device=device)
        terms = losses(net, kinematics, relations, past, future, contacts=contacts)
        weight = min(1.0, step / (KL_RAMP * steps))
        loss = sum(v.mean() for k, v in terms.items() if k != "kl") + weight * terms["kl"].mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % 500 == 0 or step == steps - 1:
            text = ", ".join(f"{k} {v.mean().item():.1f}" for k, v in terms.items())
            progress(step + 1, steps, f"{text} ({time.monotonic() - started:.0f} s)")
    return net.eval()


# ------------------------------------------------------------------ measuring
def _reference(data, count=100000, seed=0):
    """Fitted poses (count, J) - what a realistic pose is near."""
    rng = np.random.default_rng(seed)
    frames = np.concatenate([w.frames for w, _ in data.values() if w is not None])
    return frames[rng.choice(len(frames), size=min(count, len(frames)), replace=False)]


@torch.no_grad()
def measure(net, data, hand, reference, seed=0):
    """Per source, on held-out windows (millimetres on this robot's palm):

    rebuilt      the encoder's point decoded: joints_deg, points_mm, and
                 close_mm - the distances between points of different
                 fingers that are under CLOSE apart, how far off
    prior        a point drawn from the prior: realism_mm - each pose's
                 distance to the nearest fitted pose, against the held-out
                 truth's own (truth_mm); inside - the share of poses with
                 fingers more than hand_contacts.INSIDE deep in each other (truth_inside
                 for the truth: under 3% for the library, under 0.5% for FreiHAND)
    rolled       the prior's own motions over ROLL_S, from held-out pasts:
                 realism_mm at their end, inside, speed against the truth's
    """
    device = next(net.parameters()).device
    kinematics = TorchKinematics(hand.model, hand.chain, hand.palm).to(device)
    relations = Relations().to(device)
    mm = hand.palm[2] * 1000
    ref = kinematics(torch.as_tensor(reference, device=device)).flatten(-2)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out = {}
    for source, (_, held) in data.items():
        rows = rng.choice(len(held.starts), size=min(MEASURED, len(held.starts)), replace=False)
        past, future = (torch.as_tensor(x, device=device) for x in held.batch(rows))
        mean, _ = net.encode(past, future)
        made = net.decode(past, mean)
        points_made, points = kinematics(made), kinematics(future)
        apart_made, apart = relations(points_made), relations(points)
        close = apart < CLOSE
        drawn = net.decode(past, net.prior_of(past).sample())
        rolled = _roll(net, past[:ROLLED], round(ROLL_S * RATE))
        out[source] = {
            "rebuilt": {"joints_deg": float(torch.rad2deg((made - future).abs()).mean()),
                        "points_mm": float((points_made - points).norm(dim=-1).mean() * mm),
                        "close_mm": float((apart_made - apart).abs()[close].mean() * mm)
                        if close.any() else None},
            "prior": {"realism_mm": _nearest(kinematics(drawn).flatten(-2), ref) * mm,
                      "truth_mm": _nearest(points.flatten(-2), ref) * mm,
                      "inside": _inside(hand, drawn), "truth_inside": _inside(hand, future)},
            "rolled": {"realism_mm": _nearest(kinematics(rolled[:, -1]).flatten(-2), ref) * mm,
                       "inside": _inside(hand, rolled[:, ::5]),
                       "speed_deg_s": float(torch.rad2deg(torch.diff(rolled, dim=1).abs()).mean() * RATE),
                       "truth_speed_deg_s": float(torch.rad2deg(torch.diff(future, dim=1).abs()).mean()
                                                  * RATE)},
            "windows": int(len(held.starts))}
    return out


def _roll(net, past, count):
    """The prior's motions from `past` (B, PAST, J): (B, count, J), a point
    drawn every FUTURE // 2 frames."""
    frames = []
    stride = FUTURE // 2
    while sum(f.shape[1] for f in frames) < count:
        made = net.decode(past, net.prior_of(past).sample())[:, :stride]
        frames.append(made)
        past = torch.cat([past, made], 1)[:, -PAST:]
    return torch.cat(frames, 1)[:, :count]


def _nearest(points, reference, chunk=1000):
    """From each pose's points to the nearest reference pose's: the root
    mean square over the 16 points, palm lengths, averaged over the poses."""
    best = []
    for part in points.reshape(-1, reference.shape[-1]).split(chunk):
        best.append(torch.cdist(part, reference).min(1).values)
    return float(torch.cat(best).mean() / np.sqrt(16))


def _inside(hand, joints):
    """The share of poses with fingers inside each other (hand_contacts)."""
    return HandContacts(hand).share_inside(joints.reshape(-1, joints.shape[-1]).cpu().numpy())


def _summary(reports, seeds):
    """{source: {part: {measure: mean}}, ..., "spread": the same, max - min over seeds}."""
    def walk(items):
        first = items[0]
        if isinstance(first, dict):
            return {k: walk([i[k] for i in items]) for k in first}
        values = [v for v in items if v is not None]
        return (float(np.mean(values)), float(np.ptp(values))) if values else None
    combined = walk(reports)
    mean = _map(combined, lambda v: v[0])
    spread = _map(combined, lambda v: v[1])
    return {"seeds": list(seeds), "mean": mean, "spread": spread}


def _map(tree, function):
    if isinstance(tree, dict):
        return {k: _map(v, function) for k, v in tree.items()}
    return function(tree) if tree is not None else None


# ------------------------------------------------------------------ using it
class MotionSpace:
    """The fitted space on the CPU, NumPy in and out."""

    def __init__(self, saved):
        config = saved["config"]
        if (config["past"], config["future"], config["latent"], config.get("modes")) \
                != (PAST, FUTURE, LATENT, PRIOR_MODES):
            raise RuntimeError("motion_space.pt was fitted with other sizes - fit it again")
        self.net = Net(np.array(config["low"]), np.array(config["high"]))
        self.net.load_state_dict(saved["state"])
        self.net.eval()
        self.report = saved["report"]

    @torch.no_grad()
    def encode(self, past, future):
        mean, _ = self.net.encode(_t(past), _t(future))
        return mean.numpy()

    @torch.no_grad()
    def decode(self, past, z):
        return self.net.decode(_t(past), _t(z)).numpy()

    @torch.no_grad()
    def sample(self, past, seed=0):
        torch.manual_seed(seed)
        return self.net.prior_of(_t(past)).sample().numpy()

    @torch.no_grad()
    def roll(self, past, count, seed=0):
        torch.manual_seed(seed)
        return _roll(self.net, _t(past)[None], count)[0].numpy()


def _t(x):
    return torch.as_tensor(np.asarray(x, np.float32))
