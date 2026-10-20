"""Training the student, and measuring whether it is worth having.

THE COMPARISON THAT MATTERS is not the training loss. It is the student against
the live solver and the teacher, on held-out blocks of a real hand, scored the
way the teacher is scored - against the OPERATOR: finger posture, how often the
robot touches when the operator does, the gap it leaves when it does, jerk and
worst step. `Teacher.measure` produces all three rows, so they are comparable.

TWO SETS, USED DIFFERENTLY:

  SYNTHETIC pairs are the bulk - invented hands, invented motions, clean input,
  teacher labels. They teach the correspondence, and they can grow overnight
  without anyone at a bench.

  REAL pairs are small and precious: the operator's own hand, its own noise,
  the same teacher. They are the fine-tune and the evaluation, and evaluation
  is held out by BLOCK rather than by frame. Splitting a 3,575-frame block down
  the middle leaves neighbouring frames on both sides, 8 ms apart, and a model
  that memorised the training half scores well on a test set it has effectively
  already seen.

NOISE IS ADDED TO THE INPUT AND NOT TO THE TARGET. That is the whole of Phase 4
and it costs nothing at inference: the target stays the clean hand's joint
vector, so the network learns to undo the sensor's error as a side effect of
learning the mapping. The synthetic keypoints are clean, so they are the ones
that can carry it; the real ones already have the operator's own noise baked in
and get none added. Both views get it - the distances the student reads live
are measured off a noisy sensor too.

THREE LABELS PER FRAME, all from the teacher: the pose it solved, which thumb
pairs it had in contact when it solved it (measured off its own kinematics,
not off the operator - it is the teacher's behaviour the student copies), and
the seconds until each pair next touches, which only a finished run can know.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..robot.model import FINGERS
from . import collision, geometry, student
from .kinematics import ChainFK
from . import objective
from .objective import COLLIDE_TOL_MM, MADE_MM, PENETRATION_MM
from .student import (BASIS, CHANNELS, DILATIONS, HORIZONS, TTC_MAX, WINDOW,
                      Student, pca_basis, windows)

# Loss weights. Joint error is in radians and tip error in metres, which differ
# by roughly the length of a finger - so these are not free parameters, they are
# the unit conversion plus an opinion about which matters.
W_JOINT = 1.0
W_TIP = 20.0            # metres -> comparable to radians on this hand
W_LIMIT = 10.0          # only bites outside the URDF's range
# Frame-to-frame change, matched against the LABEL's - so a still label asks
# for a still hand and a striking label asks for the strike. This was 0.5.
# Raised after the distance features arrived and the student rattled:
# measured at 3.0 against 0.5, same everything else, held out - jerk 2.27e4
# to 1.99e4 with contact and gap unchanged (66% / 7.3 mm to 68% / 6.5 mm),
# and with the cumulative norm beside it, 1.66e4. Unlike an acceleration
# penalty (see W_ACCEL) it never erased a strike, because the strike is in
# the label it matches.
W_VELOCITY = 3.0
# WHY W_ACCEL IS ZERO, measured twice against two different teachers.
#
# Against the OLD labels - a solver that never made contact, so a hand that
# never struck - this was 300, and it earned it: the student matched the
# solver's accuracy and was barely smoother (jerk 8.3e3 against 8.8e3, teacher
# 1.8e3), because matching the label's velocity says nothing about a still
# hand rattling. An absolute penalty on the student's own acceleration fixed
# that.
#
# Against the NEW labels it does the opposite. The teacher now closes a pinch,
# and a pinch is a strike: 23 degrees in seven frames. Held out, same network,
# same windows, same schedule:
#
#     W_ACCEL 300   contact 18%   gap at touch 22 mm   jerk 1.15e5
#     W_ACCEL  30   contact 15%   gap at touch 18 mm   jerk 6.1e4
#     W_ACCEL   0   contact 55%   gap at touch 10 mm   jerk 7.3e4
#
# The penalty erased the strike, arrived short, and bought no smoothness for
# it - jerk was WORSE at 300 than at 0, because a network asked to reproduce
# a fast event and punished for producing one compromises badly everywhere. A
# relative version, penalising only acceleration beyond the label's, was
# worse again (1% and 0% at 0.3 and 1.0): the student's strike lands a frame
# or two off the teacher's, so the penalty sees a fast event where the label
# has none, and suppresses it.
#
# So there is no acceleration term. The labels are already centred-smoothed;
# what roughness the student adds is not something this term removes, and
# arriving on time is worth more than not rattling. Being both smooth AND on
# time is what the forward heads are for.
W_ACCEL = 0.0

# The auxiliary heads. Small against the pose terms on purpose: they exist to
# shape the trunk, not to be the objective, and a cross-entropy near 0.7 at
# initialisation would otherwise outweigh a joint loss of 0.05 twenty to one.
W_CONTACT_HEAD = 0.2
W_TTC_HEAD = 0.2

# THUMB-PAIR DISTANCES, student against teacher, through the same kinematics
# the tip term uses. The tip term already places every fingertip; this one is
# about the DIFFERENCE between two of them, which is what a contact is, and
# it is weighted up on the frames where the teacher was touching. Metres
# squared like the tip term, so the same scale.
W_PAIR = 20.0
PAIR_TOUCHING = 4.0     # extra weight on a pair the teacher had closed

# SELF-COLLISION, the teacher's own term through the student's FK: every
# cross-digit pair's surface gap from `collision.Proxy`, squared past the
# proxy's tolerance, summed over pairs. The labels are clear of themselves
# now, so this mostly bites between them - where a student interpolating two
# clear poses can pass through one. Millimetres squared: a pair three
# millimetres too deep costs nine, times this.
W_COLLIDE = 8e-3

# TOUCH, the other side of the same coin. A penalty that only charges
# penetration is one-sided, and a network unsure of a fingertip to four
# millimetres answers a one-sided penalty by hedging away from it: trained
# with the collision term alone, the student stopped 3 to 6 mm short of the
# skin on every pinch the operator made - the labels touching at -0.1. So
# wherever the LABEL touches (its thumb-to-fingertip surface gap under
# MADE_MM), a surface gap above the label's costs the same as one below
# the tolerance does. Symmetric pressure around the label; the pinch lands
# on it. Millimetres squared, like W_COLLIDE - and lighter than it, because
# at equal weight the thumb took the short way to the fingertip, through
# the finger: 42% of held-out frames penetrating by the proxy. The collision
# term has to outweigh the pull it answers.
W_TOUCH = 1e-3
TOUCH_SLACK_MM = 1.0    # above the label's gap before it costs anything

# THE REPAIR IS NOT IN THE LOSS. Live, the network's answer goes through
# `collision.repair` before it is commanded, and the same repair can run
# inside the loss so the geometric terms score the repaired pose
# (`REPAIR_STEPS` straight-through steps). Measured, two steps of it cost
# three degrees of posture on the held-out blocks (12.5 against 9.4) and did
# not touch the hover it was meant to cure; the collision term reads the raw
# pose either way. Off by default, kept for the comparison.
REPAIR_STEPS = 0

# A RUN THAT NEVER TRAINED IS STOPPED, NOT SAVED.
#
# 192-channel seed 1 sat at 100% held-out self-penetration on every one of its
# 21 epochs - the collision loss flat at 292 from first to last while the joint
# loss trained normally - and finished and wrote a checkpoint like any other.
# Nothing read the number it printed each epoch. Loaded as `student.pt`, it
# would have driven a hand that overlaps itself on every frame.
#
# THE RULE IS ON THE LOWEST VALUE SO FAR, NOT THE CURRENT ONE, because healthy
# runs spike. Measured, per epoch, on the 16-block labels:
#
#                  epoch 1   epoch 2   worst later       final
#     96  seed 0     0.1%      0.2%        4.3%          0.3%
#     96  seed 1    54.7%      0.1%        1.2%          0.3%
#     192 seed 0    53.8%      0.1%   16.2% (1st real)   0.5%
#     192 seed 1   100.0%    100.0%      100.0%        100.0%
#
# A threshold on any single epoch would have killed a good run - two healthy
# ones open at 54%. What separates the failure is that it is NEVER low: every
# healthy run is at or under 0.1% by epoch 2, the collapsed one never below
# 100%. So after COLLAPSE_EPOCH, a run whose best epoch is still above
# COLLAPSE_PCT is stopped - a gap of 0.1 against 100 either side of 50.
COLLAPSE_EPOCH = 3
COLLAPSE_PCT = 50.0


def collapsed(penetrations, epoch=COLLAPSE_EPOCH, limit=COLLAPSE_PCT):
    """True if a run's held-out self-penetration never came down.

    `penetrations` are the per-epoch held-out percentages so far, pretraining
    only. Decided once, when `epoch` of them are in, on the LOWEST - see
    COLLAPSE_EPOCH for why the current value would kill healthy runs. NaN is an
    empty held-out set and says nothing either way, so it is ignored rather
    than allowed to read as "never low".
    """
    seen = [value for value in penetrations if not np.isnan(value)]
    return len(penetrations) == epoch and bool(seen) and min(seen) > limit

# THE OBJECTIVE ITSELF, scored on the student's own pose rather than on its
# distance from the teacher's.
#
# WHY THIS EXISTS. The teacher solves a whole trajectory and then smooths it
# with a centred filter, so `q_star` contains poses that depend on frames the
# student will never have seen - a causal network cannot represent them, and a
# student scored only against them inherits a floor it cannot cross. Measured
# on the shipped student, the distillation gap is 8.0 deg of joint error and
# 21 mm of fingertip against the teacher's own answer, and it is FLAT across
# contact and free frames, so it is not the contact trade either.
#
# Scored against `objective.per_frame`, the student is free to find the causal
# pose that satisfies the same terms the teacher was satisfying. The label
# stays as a trust region through `W_JOINT` rather than disappearing: without
# it the student would be solving a 600-step problem in one forward pass from
# a cold start.
#
# MEASURED, AND IT DOES NOT PAY - off, kept for the comparison. Its posture gain
# is circular (that score is `per_frame`'s own first term), distance from the
# teacher rises monotonically with the weight - 7.87, 8.19, 9.93, 11.36 deg at
# 0, 0.01, 0.05, 0.20 - and at 0.01 one seed of three self-penetrated on 23.2%
# of frames. See `objective.per_frame` for the numbers and what to fix first.
W_DIRECT = 0.0

# The self-error head's weight, applied when training with `belief=True`.
# Small: it shapes the trunk and gives the executor something to read, and it
# is not what the network is for. Tied to belief because "how wrong am I" and
# "how good was my input" are one claim and are measured together.
W_SPREAD = 0.05

# The grasp head's weight, applied when training with `authority=True`. It is
# trained to be
# the best THREE-dimensional account of the same label the pose head fits in
# sixteen, so it is deliberately not competing: it will always be worse, and
# what it is for is being worse in a useful way - a pose that stays a plausible
# grasp when the fingertips stop being worth believing.
W_INTENT = 1.0

# How many of the window's output frames are actually supervised, counting back
# from the newest.
#
# THIS IS NOT A TUNING KNOB, IT IS A BUG FIX. The network's receptive field is
# 61 frames and the window is 20, so no position sees a full history - and the
# EARLY positions see almost none, because the causal padding feeds them zeros.
# Averaging the loss over all twenty asks the network to predict frames from
# histories that do not exist, and those unlearnable positions dominate the
# gradient.
#
# Live, only the LAST frame of the window is ever used. Measured, the
# difference is stark: with all twenty supervised, adding a smoothness term
# made the student monotonically WORSE on every metric including the one it
# targeted - jerk 8.3e3, 1.11e4, 1.01e4 for weights 0, 300, 2000 - because the
# term was spending itself on the startup transient rather than on the frame
# that ships.
SUPERVISE = 8

# Held out by block, not by frame - see the module docstring. These three span
# a still hold, a fast transition and the worst tracking in the session.
HELD_OUT = ("fist_hold", "curl_fast", "occlusion")

# THE FINE-TUNE REPLAYS THE PRETRAINING. Fine-tuned on the real session
# alone, the student drifted 11 to 13 degrees from the synthetic labels it
# had pretrained to within 6 - on object-free frames and on the wraps
# alike - and the object prior went with it: 16 mm into a cylinder the
# labels hold at 1.4. The real session has no objects in it, so nothing in
# that stage asked for the prior back. The fine-tune now trains on the real
# windows plus an equal number of synthetic ones, at the same 0.1x rate, so
# what the operator's hand teaches is added to the pretraining rather than
# written over it. The synthetic subset is drawn ONCE for the whole stage,
# not per epoch - a fresh draw each epoch would show the fine-tune more of
# the pretraining set for the same cost, and is untried.
REPLAY = 1.0            # synthetic windows per real window in the fine-tune

# The thumb row's partner fingers, in the row's order, as indices into the
# student's kinematics - which walks FINGERS, thumb last.
THUMB = FINGERS.index("thumb")
PARTNERS = [FINGERS.index(f) for f in ("index", "middle", "ring")]


def load_pairs(path):
    with np.load(Path(path)) as handle:
        return {key: handle[key] for key in handle.files}


# ------------------------------------------------------------------ labels
@torch.no_grad()
def contact_labels(fk, proxy, q, batch=8192):
    """Which thumb pairs the TEACHER had closed at each frame: (n, 3) in {0, 1}.

    Measured off the teacher's own pose through the robot's kinematics, the
    way `Teacher.measure` scores it - a surface gap under MADE_MM. Not off
    the operator's distance: the label is what the teacher did, and on a
    frame where the operator pinched but the teacher did not close, the
    student should not be told it did.
    """
    out = []
    for at in range(0, len(q), batch):
        chunk = torch.as_tensor(np.asarray(q[at:at + batch]), dtype=torch.float32,
                                device=fk.xyz.device)
        _, origins, frames = fk.walk(chunk)
        out.append((proxy.tip_gaps(origins, frames) < MADE_MM).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def ttc_labels(time_to_contact):
    """Seconds to the next touch per thumb pair, clipped to TTC_MAX and put on
    0..1. Infinite - never again in this run - becomes 1, "not soon"."""
    ttc = np.asarray(time_to_contact, dtype=np.float32)[:, list(geometry.THUMB_ROW)]
    return (np.minimum(ttc, TTC_MAX) / TTC_MAX).astype(np.float32)


def positive_weight(labels, cap=20.0):
    """Per-pair weight on the positive class, negatives over positives.

    A middle-finger contact is a few percent of the frames; unweighted, the
    head learns to say "no" and is right 97% of the time. Capped so a pair
    with almost no positives does not get an absurd weight from them.
    """
    positives = np.asarray(labels).sum(axis=0)
    negatives = len(labels) - positives
    return np.clip(negatives / np.maximum(positives, 1.0), 1.0, cap).astype(np.float32)


class Windows(torch.utils.data.Dataset):
    """Windows into one pair file, with the noise model applied on the fly.

    Applied here rather than baked into the file so the same clean sample is
    seen with different noise every epoch - which is the point of augmenting
    rather than just collecting more data.

    Each item is five tensors: the aligned window, the raw window, the pose
    labels (T, horizons, joints), the contact labels (T, 3) and the
    time-to-contact labels (T, 3).
    """

    def __init__(self, keypoints, raw, q, contact, ttc, bounds, noise=None,
                 length=WINDOW, rows=None, horizons=HORIZONS):
        self.keypoints = keypoints.astype(np.float32)
        self.raw = raw.astype(np.float32)
        self.q = q.astype(np.float32)
        self.contact = contact.astype(np.float32)
        self.ttc = ttc.astype(np.float32)
        self.length = length
        self.noise = noise
        # Room for the furthest horizon, so every window has a real future to
        # be scored against rather than a padded one.
        self.horizons = tuple(horizons)
        self.reach = length + max(self.horizons)
        self.starts = (windows(bounds, self.reach) if rows is None else rows)

    def __len__(self):
        return len(self.starts)

    def _noisy(self, window):
        # Millimetres in, millimetres out: the noise model was fitted on the
        # sensor's own units and the pairs are stored in metres. Reset per
        # view, so the raw window gets its own draw: the two views are not
        # the same points, and a correlated draw would have to be transformed
        # through the alignment to be honest. Independent is merely
        # pessimistic.
        self.noise.reset()
        window = self.noise.sequence(window * 1000.0, 1.0 / self.noise.hz)
        return (window * 0.001).astype(np.float32)

    def __getitem__(self, index):
        at = int(self.starts[index])
        stop = at + self.length
        window, raw = self.keypoints[at:stop], self.raw[at:stop]
        if self.noise is not None:
            window, raw = self._noisy(window), self._noisy(raw)
        # Labels for every horizon: (T, horizons, joints). Sliced rather than
        # shifted in the loss, so a window near the end of a run cannot quietly
        # read across into the next one - `windows` already guaranteed the room.
        target = np.stack([self.q[at + h:at + h + self.length]
                           for h in self.horizons], axis=1)
        return tuple(torch.from_numpy(np.ascontiguousarray(x)) for x in
                     (window, raw, target, self.contact[at:stop],
                      self.ttc[at:stop]))


def losses(out, target, contact, ttc, chain, proxy, lower, upper,
           w_accel=W_ACCEL, w_pair=W_PAIR, pos_weight=None, w_collide=W_COLLIDE,
           aligned=None, raw=None, w_direct=W_DIRECT, w_spread=0.0,
           w_intent=0.0):
    """Ten terms, and each answers a way the others can be satisfied wrongly.

    JOINT is the obvious one and is not enough on its own: the distal joint has
    six times less authority over a fingertip than the spread joint, so equal
    joint error is very unequal tip error.

    TIP is what the mapping is actually for. Computed through the same
    kinematics the simulator uses, so the network's idea of where a tip is
    cannot drift from the model's.

    PAIR is the tip term for a contact: the thumb-to-finger distances the
    student's pose makes against the teacher's, weighted up where the teacher
    was touching. Two tips can each be a centimetre off and their distance
    exact, or each be close and their distance wrong - it is the second the
    pinch cares about.

    VELOCITY is matched against the LABEL's velocity rather than driven to
    zero. A plain smoothness penalty asks the hand to stop moving; this asks it
    to move the way the teacher did, which is the thing worth copying.

    LIMIT is zero inside the URDF's range and grows outside it. The basis makes
    implausible poses hard to represent; this makes impossible ones expensive.

    ACCEL is the student's own second difference, matched to nothing. See
    W_ACCEL for why it is off.

    COLLIDE is the teacher's self-collision term through the same proxy - see
    W_COLLIDE. TOUCH is its mirror: wherever the label touches, hovering
    above it costs what penetrating does - see W_TOUCH.

    CONTACT and TTC train the auxiliary heads - which pairs the teacher had
    closed, and how long until each closes next - and through them the trunk.
    """

    predicted = out["q"]
    # Only the tail of the window carries a real history - see SUPERVISE. The
    # differences are taken over the whole window first and sliced afterwards,
    # so the oldest supervised frame still has a real predecessor to differ
    # against rather than a boundary.
    tail = slice(-SUPERVISE, None)
    predicted_tail, target_tail = predicted[:, tail], target[:, tail]
    # The PRESENT horizon is index 0 by construction - see `student.HORIZONS`.
    present, present_target = predicted_tail[:, :, 0], target_tail[:, :, 0]
    flat = present.reshape(-1, predicted.shape[-1])
    flat_target = present_target.reshape(-1, target.shape[-1])
    contact_tail = contact[:, tail]

    # Joint error over EVERY horizon, so the future heads are trained; the
    # geometric terms only over the present one, because a fingertip position
    # is a statement about now and forward kinematics on a prediction would
    # score the network for a hand that has not happened.
    joint = nn.functional.mse_loss(predicted_tail, target_tail)
    # The collision and touch terms read the RAW pose; the geometry reads it
    # repaired when REPAIR_STEPS is on, which by default it is not.
    _, origins, frames = chain.walk(flat)
    gaps = proxy.gaps(origins, frames)
    collide = (torch.relu(-gaps - COLLIDE_TOL_MM) ** 2).sum(dim=1).mean()
    with torch.no_grad():
        _, label_origins, label_frames = chain.walk(flat_target)
        label_gap = proxy.tip_gaps(label_origins, label_frames)
        touching = (label_gap < MADE_MM).float()
    touch = ((torch.relu(gaps[:, proxy.tip_rows] - label_gap - TOUCH_SLACK_MM) ** 2)
             * touching).sum() / touching.sum().clamp_min(1.0)
    repaired = (collision.repair(chain, proxy, flat, tolerance=COLLIDE_TOL_MM,
                                 steps=REPAIR_STEPS) if REPAIR_STEPS else flat)
    tips, tips_target = chain(repaired)[:, :, -1], chain(flat_target)[:, :, -1]
    tip = nn.functional.mse_loss(tips, tips_target)
    pair = (tips[:, THUMB, None] - tips[:, PARTNERS]).norm(dim=-1)
    pair_target = (tips_target[:, THUMB, None] - tips_target[:, PARTNERS]).norm(dim=-1)
    weight = 1.0 + PAIR_TOUCHING * contact_tail.reshape(-1, pair.shape[-1])
    pair = (weight * (pair - pair_target) ** 2).mean()
    velocity = nn.functional.mse_loss(
        torch.diff(predicted[:, :, 0], dim=1)[:, tail],
        torch.diff(target[:, :, 0], dim=1)[:, tail])
    limit = (torch.relu(flat - upper) ** 2 + torch.relu(lower - flat) ** 2).mean()
    accel = (torch.diff(predicted[:, :, 0], n=2, dim=1)[:, tail] ** 2).mean()
    # HOW WRONG IT THINKS IT IS, trained through a Gaussian likelihood on the
    # joint term. The network can lower this loss two ways - be right, or admit
    # it is not - and only the first works on frames it CAN get right, because
    # claiming a large sigma everywhere pays the log term on every frame. What
    # it buys is a student that hedges where the input is bad instead of
    # uniformly. Off with BELIEF, since it is the same claim.
    spread = flat.new_zeros(())
    if Student.SPREAD in out and w_spread:
        log_sigma = out[Student.SPREAD][:, tail].reshape(flat.shape).clamp(-4.0, 2.0)
        residual = (present - present_target).reshape(flat.shape)
        spread = (0.5 * (residual / log_sigma.exp()) ** 2 + log_sigma).mean()

    # The three-number grasp, against the same label the pose head sees.
    grasp = flat.new_zeros(())
    if "q_intent" in out and w_intent:
        grasp = nn.functional.mse_loss(
            out["q_intent"][:, tail].reshape(flat.shape), flat_target)

    head = nn.functional.binary_cross_entropy_with_logits(
        out[Student.CONTACT][:, tail], contact_tail, pos_weight=pos_weight)
    soon = nn.functional.mse_loss(out[Student.TTC][:, tail], ttc[:, tail])
    # The teacher's own objective, on the student's pose. Same FK, same proxy,
    # same terms - see `objective.per_frame` and `W_DIRECT`.
    direct = flat.new_zeros(())
    if w_direct and aligned is not None:
        views = [v[:, tail].reshape(-1, *v.shape[2:]) for v in (aligned, raw)]
        intent = objective.intent_from_distance(
            torch.stack([(views[1][:, objective.TIP["thumb"]]
                          - views[1][:, objective.TIP[f]]).norm(dim=-1)
                         for f in objective.OTHERS], dim=1) * 1000.0)
        direct = objective.per_frame(chain, proxy, flat, views[0], views[1],
                                     intent, lower, upper)

    total = (W_JOINT * joint + W_TIP * tip + w_pair * pair
             + W_VELOCITY * velocity + W_LIMIT * limit + w_accel * accel
             + w_collide * collide + W_TOUCH * touch
             + W_CONTACT_HEAD * head + W_TTC_HEAD * soon
             + w_direct * direct + w_spread * spread
             + w_intent * grasp)
    return total, {"joint": joint.item(), "tip": tip.item(),
                   "pair": pair.item(), "velocity": velocity.item(),
                   "limit": limit.item(), "accel": accel.item(),
                   "collide": collide.item(), "touch": touch.item(),
                   "contact": head.item(), "ttc": soon.item(),
                   "direct": direct.item(), "spread": spread.item(),
                   "grasp": grasp.item()}


def device_for(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_real(real, meta_blocks, held_out=HELD_OUT):
    """Train and test windows over the real pairs, held out by BLOCK.

    Returns (train starts, test starts). A window is assigned by the block its
    LAST frame is in, which is the frame the network is asked to predict.
    """
    wanted = {meta_blocks.index(b) for b in held_out if b in meta_blocks}
    starts = windows(real["run_bounds"], WINDOW + max(HORIZONS))
    last = real["block"][starts + WINDOW - 1]
    test = np.isin(last, list(wanted))
    return starts[~test], starts[test]


def evaluate(model, dataset, chain, proxy, device, batch=512):
    """Joint and fingertip error against the teacher's labels, the contact
    head's balanced accuracy, and the share of frames the student puts the
    hand through itself (by the proxy), on the last frame of each window -
    which is the only one the live path ever uses."""
    model.eval()
    joint, tip, count, penetrating = 0.0, 0.0, 0, 0
    hits = np.zeros((2, 3)), np.zeros((2, 3))       # (correct, total) x (no, yes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch)
    with torch.no_grad():
        for window, raw, target, contact, _ in loader:
            out = model.predict(window.to(device), raw.to(device))
            predicted = out["q"][:, -1, 0]
            truth = target[:, -1, 0].to(device)
            joint += float(torch.abs(predicted - truth).sum())
            links, origins, frames = chain.walk(predicted)
            tip += float(torch.norm(links[:, :, -1] - chain(truth)[:, :, -1],
                                    dim=-1).sum())
            penetrating += int((proxy.gaps(origins, frames).min(dim=1).values
                                < -PENETRATION_MM).sum())
            count += len(window)
            said = (out[Student.CONTACT][:, -1] > 0).cpu().numpy()
            was = contact[:, -1].numpy() > 0.5
            for cls in (0, 1):
                mask = was == bool(cls)                      # (n, 3)
                hits[0][cls] += ((said == bool(cls)) & mask).sum(axis=0)
                hits[1][cls] += mask.sum(axis=0)
    model.train()
    rate = hits[0] / np.maximum(hits[1], 1)
    # Balanced over the two classes, then averaged over the pairs that have
    # positives to be balanced against; a pair the held-out blocks never
    # close cannot score.
    scored = hits[1][1] > 0
    balanced = rate.mean(axis=0)[scored].mean() if scored.any() else float("nan")
    if not count:
        # NOT zeros. Divided by `max(count, 1)` this used to report 0.00 deg,
        # 0.00 mm - a perfect score - for a held-out set with nothing in it,
        # which is what a session recorded with `--blocks` that skipped all of
        # `HELD_OUT` gives you. NaN prints as nan and cannot be mistaken.
        return {"joint_deg": float("nan"), "tip_mm": float("nan"),
                "contact_acc": float("nan"), "penetration_pct": float("nan")}
    return {"joint_deg": np.degrees(joint / (count * len(chain.joint_names))),
            "tip_mm": 1000.0 * tip / (count * len(FINGERS)),
            "contact_acc": float(balanced),
            "penetration_pct": 100.0 * penetrating / count}


def train(synthetic, real, meta_blocks, model_out, noise=None, epochs=12,
          batch=256, learning_rate=2e-3, basis=BASIS, channels=CHANNELS,
          device=None, workers=0, w_accel=W_ACCEL, w_pair=W_PAIR,
          robot_hand="left", seed=0, w_direct=W_DIRECT, belief=False,
          authority=False, contact_gate=False, warmup=0, progress=print):
    """Pretrain on synthetic, fine-tune on real, evaluate on held-out blocks.

    `warmup` ramps the learning rate linearly from zero over that many epochs
    at the start of pretraining. Off by default; see `COLLAPSE_EPOCH` for why a
    wider network may need it.

    `belief` adds the sensor's fitted error as 21 input channels and the head
    that reports the student's own sigma; `authority` adds the three-synergy
    grasp head. BOTH ARE ARGUMENTS, not module flags, because the flags are
    read at import: `run_train` used to set `student.BELIEF`, then patch
    `student.FEATURES`, then patch this module's imported-by-value copy of it,
    then patch `W_SPREAD` - four assignments across two modules to turn one
    thing on, and `FEATURES = FEATURES + 21` compounds if it happens twice in
    a process. The architecture is decided here and travels by argument.
    """
    # SEEDED, so a rerun is a rerun and two seeds are a measurement. It was
    # unseeded, which meant the weight init, the dropout masks, the shuffle
    # and the augmentation draw all differed run to run - and middle-finger
    # contact ranged 0% to 23% across retrains that were meant to differ by
    # one knob. Nobody could say how much of that was the knob.
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device_for(device)
    from ..robot.model import HandModel, ensure_urdf

    hand_model = HandModel.load(ensure_urdf(robot_hand, None))
    joint_names = list(str(n) for n in real["joint_names"])
    lower, upper = hand_model.limits(joint_names)
    # The labels have to have been solved in the same joint order this walks.
    # Nothing downstream would notice otherwise: the names match, the shapes
    # match, and the student quietly learns a permutation of the hand.
    stored = [str(n) for n in synthetic["joint_names"]]
    if stored != joint_names:
        raise SystemExit("the synthetic and real labels are in different joint "
                         f"orders - {stored[:2]}... against {joint_names[:2]}...; "
                         "re-run `learn.py synth`")
    # The whole chain: tips for the tip term, every body's origin and frame
    # for the collision term and the contact labels - one FK per batch.
    chain = ChainFK(hand_model, joint_names).to(device)
    # `objective.per_frame` reads this off the chain. The Teacher sets it too;
    # both go through `objective.reach_scale` so they cannot disagree.
    chain.reach_scale = objective.reach_scale(
        dict(zip((str(f) for f in real["fingers"]), real["digit_scale"])))
    proxy = collision.for_hand(chain, lower, upper, robot_hand)
    lower = torch.tensor(lower, dtype=torch.float32, device=device)
    upper = torch.tensor(upper, dtype=torch.float32, device=device)

    # The basis is fitted on BOTH sets. Fitting it on synthetic alone would
    # give the student a manifold that does not quite contain the real hand's
    # solutions, and it could then never reach them however well it trained.
    mean, components = pca_basis(np.concatenate([synthetic["q_star"],
                                                 real["q_star"]]), basis)
    features = student.feature_width(belief)
    w_spread = W_SPREAD if belief else 0.0
    w_intent = W_INTENT if authority else 0.0
    model = Student(mean, components, channels=channels,
                    belief=belief, authority=authority).to(device)
    if belief:
        # The same fit the augmentation draws from, read as an INPUT rather
        # than as a corruption - see `belief.py`. Gated on the ARGUMENT: when
        # this read `student.BELIEF` it silently stopped firing the moment the
        # flag stopped being a global, and the run built a 102-channel input
        # layer then fed it 81 channels.
        from ..paths import NOISE_FIT
        from .belief import Belief
        model.belief = Belief.load(NOISE_FIT).to(device)
        model.belief.gate = bool(contact_gate)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)

    labels = {}
    for name, pairs in (("synthetic", synthetic), ("real", real)):
        labels[name] = (contact_labels(chain, proxy, pairs["q_star"]),
                        ttc_labels(pairs["time_to_contact"]))
    train_rows, test_rows = split_real(real, meta_blocks)
    sets = {
        "synthetic": Windows(synthetic["keypoints"], synthetic["keypoints_raw"],
                             synthetic["q_star"], *labels["synthetic"],
                             synthetic["run_bounds"], noise=noise),
        "real": Windows(real["keypoints"], real["keypoints_raw"], real["q_star"],
                        *labels["real"], real["run_bounds"], rows=train_rows),
        "test": Windows(real["keypoints"], real["keypoints_raw"], real["q_star"],
                        *labels["real"], real["run_bounds"], rows=test_rows),
    }
    touching = {name: 100.0 * labels[name][0].mean(axis=0) for name in labels}
    progress(f"  device {device}   basis {basis}   channels {channels}   "
             f"{sum(p.numel() for p in model.parameters()):,} parameters   "
             f"{features} features")
    progress(f"  synthetic {len(sets['synthetic']):,} windows   "
             f"real train {len(sets['real']):,}   held out {len(sets['test']):,}"
             f"  ({', '.join(HELD_OUT)})")
    progress("  teacher touching, % of frames (index / middle / ring):  "
             + "   ".join(f"{name} {a:.1f} / {b:.1f} / {c:.1f}"
                          for name, (a, b, c) in touching.items()))

    history = []
    # WHAT THIS SCHEDULE COSTS. The fine-tune is the reason the student is
    # good on the operator's hand and mediocre on anything else: measured, 2.5
    # degrees on real frames it trained on, 4.0 on held-out real blocks, and
    # 8.95 on the SYNTHETIC set it pretrained on - worse than the 8.57 it
    # scores on gestures it never saw. There is no generalisation gap between
    # those last two; both are the pretraining being forgotten. The 0.1x rate
    # below slows that, it does not stop it.
    #
    # Kept because the live path only ever sees a real tracked hand, and 4
    # degrees on a novel real shape is the number that matters. Anyone reading
    # the gesture probe as a coverage failure should read this first - widening
    # the sampler was tried against it and did nothing. See DIVERSITY.
    schedule = [("synthetic", epochs), ("real", max(2, epochs // 3))]
    generator = torch.Generator().manual_seed(seed)
    for stage, rounds in schedule:
        dataset = sets[stage]
        if stage == "real" and REPLAY > 0:
            count = min(int(len(sets["real"]) * REPLAY), len(sets["synthetic"]))
            draw = torch.randperm(len(sets["synthetic"]), generator=generator)[:count]
            dataset = torch.utils.data.ConcatDataset(
                [sets["real"], torch.utils.data.Subset(sets["synthetic"], draw.tolist())])
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch, shuffle=True, num_workers=workers,
            drop_last=True)
        pos_weight = torch.tensor(positive_weight(
            np.concatenate([labels["real"][0], labels["synthetic"][0]])
            if stage == "real" and REPLAY > 0 else labels[stage][0]), device=device)
        # The fine-tune runs an order of magnitude slower. The real set is a
        # fiftieth the size and the pretrained weights are already close; at
        # the pretraining rate it forgets the synthetic set in one epoch.
        for group in optimiser.param_groups:
            group["lr"] = learning_rate * (1.0 if stage == "synthetic" else 0.1)
        # WARM-UP, pretraining only. Every width trains at the same rate, and
        # a wider network takes a larger effective step in output space - so
        # at 192 channels the first epoch opened 5 to 40 times deeper in
        # self-collision than at 96, and one seed of two never climbed out. A
        # ramp from zero lets the first steps be small. The fine-tune starts
        # from pretrained weights and does not need one.
        warm_steps = warmup * len(loader) if stage == "synthetic" else 0
        step, penetrations = 0, []
        for epoch in range(rounds):
            began, running, parts = time.perf_counter(), 0.0, {}
            for window, raw, target, contact, ttc in loader:
                if warm_steps:
                    scale = min(1.0, (step + 1) / warm_steps)
                    for group in optimiser.param_groups:
                        group["lr"] = learning_rate * scale
                step += 1
                out = model.predict(window.to(device), raw.to(device))
                loss, terms = losses(out, target.to(device), contact.to(device),
                                     ttc.to(device), chain, proxy, lower, upper,
                                     w_accel, w_pair, pos_weight,
                                     aligned=window.to(device), raw=raw.to(device),
                                     w_direct=w_direct, w_spread=w_spread,
                                     w_intent=w_intent)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                running += loss.item()
                for key, value in terms.items():
                    parts[key] = parts.get(key, 0.0) + value
            batches = max(len(loader), 1)
            scores = evaluate(model, sets["test"], chain, proxy, device)
            row = {"stage": stage, "epoch": epoch + 1,
                   "loss": running / batches,
                   **{k: v / batches for k, v in parts.items()}, **scores,
                   "seconds": round(time.perf_counter() - began, 1)}
            history.append(row)
            progress(f"    {stage:<9} {epoch + 1:2d}/{rounds}  "
                     f"loss {row['loss']:.5f}  joint {row['joint']:.5f}  "
                     f"pair {row['pair']:.2e}  coll {row['collide']:.2e}  "
                     f"touch {row['touch']:.2e}  "
                     f"head {row['contact']:.3f}  |  "
                     f"HELD OUT  {row['joint_deg']:5.2f} deg  "
                     f"{row['tip_mm']:5.2f} mm  contact {row['contact_acc']:.2f}  "
                     f"pen {row['penetration_pct']:4.1f}%"
                     f"   {row['seconds']:5.1f}s")
            # Stop a run that never trained - see `collapsed`.
            if stage == "synthetic":
                penetrations.append(row["penetration_pct"])
                if collapsed(penetrations):
                    raise SystemExit(
                        f"training collapsed: held-out self-penetration never "
                        f"fell below {min(penetrations):.1f}% in "
                        f"{COLLAPSE_EPOCH} epochs (healthy runs are under 1% by "
                        f"epoch 2). Nothing was saved. A wider network may need "
                        f"--warmup; a new seed may also clear it.")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean,
                # The TEACHER's own average pose at training time, kept so a
                # replay can tell whether this student is still about the solve
                # on disk. Not the PCA mean, which is fitted over the synthetic
                # set as well and so differs from any one session by several
                # degrees even when nothing is stale.
                "teacher_mean": real["q_star"].mean(axis=0),
                "basis": components, "channels": channels,
                "dilations": DILATIONS, "window": WINDOW,
                "horizons": HORIZONS,
                # What the input is, so a loader can refuse a checkpoint from
                # before the distance features rather than fail inside torch.
                "seed": seed,
                "belief": belief, "authority": authority,
                "contact_gate": bool(contact_gate), "warmup": int(warmup),
                "features": features,
                "ttc_max": TTC_MAX,
                "joint_names": joint_names}, model_out)
    return model, history, sets["test"], chain, device


def sliding(keypoints, raw, start, end, length):
    """Both views of one run as (n, T, 21, 3) windows, newest frame last."""
    out = []
    for view in (keypoints, raw):
        window = np.lib.stride_tricks.sliding_window_view(
            view[start:end], length, axis=0)                # (n, 21, 3, T)
        out.append(np.ascontiguousarray(window.transpose(0, 3, 1, 2)))
    return out


def run_windows(model, aligned, raw, device, head, batch=512):
    """The student's chosen head over a stack of windows, last frame only."""
    out = []
    with torch.no_grad():
        for at in range(0, len(aligned), batch):
            a = torch.from_numpy(aligned[at:at + batch]).float().to(device)
            r = torch.from_numpy(raw[at:at + batch]).float().to(device)
            out.append(model(a, r)[:, -1, head].cpu().numpy())
    return np.concatenate(out)


def rollout(model, keypoints, raw, bounds, blocks, meta_blocks, device,
            held_out=HELD_OUT, length=WINDOW, head=0):
    """Run the student causally over whole held-out runs.

    Whole runs, not shuffled windows, because the metrics that decide this are
    temporal: jerk and worst-step are differences between consecutive commands,
    and a shuffled batch has no consecutive anything. Each frame is predicted
    from the `length` frames ending at it and nothing later, which is exactly
    what the live path can supply.
    """
    wanted = {meta_blocks.index(b) for b in held_out if b in meta_blocks}
    pieces = []
    model.eval()
    for start, end in bounds:
        if int(blocks[start]) not in wanted or end - start < length:
            continue
        aligned, raw_windows = sliding(keypoints, raw, start, end, length)
        # The first `length - 1` frames have no full history. They are dropped
        # rather than padded: padding invents a hand that was not there, and on
        # the bench those frames simply do not exist yet either.
        pieces.append((start + length - 1, end,
                       run_windows(model, aligned, raw_windows, device, head)))
    model.train()
    return pieces


def compare(model, real, meta_blocks, device, hand_model, joint_names,
            robot_hand="left"):
    """Student against the live solver and the teacher, on held-out blocks.

    Same frames, same metrics, same code path as Phase 1's table - so the three
    rows are comparable, which is the only reason any of them mean anything.
    """
    from .teacher import Teacher

    pieces = rollout(model, real["keypoints"], real["keypoints_raw"],
                     real["run_bounds"], real["block"], meta_blocks, device)
    if not pieces:
        return None
    rows = np.concatenate([np.arange(a, b) for a, b, _ in pieces])
    student = np.concatenate([q for _, _, q in pieces])
    bounds, at = [], 0
    for a, b, _ in pieces:
        bounds.append((at, at + (b - a)))
        at += b - a
    aligned, raw = real["keypoints"][rows], real["keypoints_raw"][rows]
    dt = float(np.median(np.diff(real["t"])))
    scale = dict(zip((str(f) for f in real["fingers"]), real["digit_scale"]))
    teacher = Teacher(hand_model, joint_names, scale, device=str(device), hand=robot_hand)
    return [teacher.measure(name, aligned, raw, q, dt, bounds)
            for name, q in (("live-equivalent", real["q_live"][rows]),
                            ("teacher", real["q_star"][rows]),
                            ("student", student))]


def lookahead_skill(model, real, meta_blocks, device, held_out=HELD_OUT):
    """Is the forward head better than simply holding the present pose?

    THE BASELINE MATTERS MORE THAN THE NUMBER. A network that predicts 80 ms
    ahead and does no better than repeating what it sees has learned nothing
    about where a hand is going, and commanding it would buy lag back with
    error. So each horizon is scored against persistence on the same frames.

    Persistence here is the TEACHER's label at the present frame, held - not
    the student's own last answer. That makes it an oracle baseline and so
    the harder of the two to beat: it is asking whether predicting ahead
    beats knowing exactly where the hand is now and assuming it stays.
    """
    wanted = {meta_blocks.index(b) for b in held_out if b in meta_blocks}
    rows = []
    for start, end in real["run_bounds"]:
        if int(real["block"][start]) in wanted and end - start > WINDOW + 40:
            rows.append((start, end))
    if not rows:
        return []

    model.eval()
    out = []
    for index, horizon in enumerate(model.horizons):
        errors, naive = [], []
        for start, end in rows:
            aligned, raw = sliding(real["keypoints"], real["keypoints_raw"],
                                   start, end, WINDOW)
            at = start + WINDOW - 1
            future = at + horizon
            if future >= end:
                continue
            keep = end - future
            predicted = run_windows(model, aligned[:keep], raw[:keep], device,
                                    index)
            truth = real["q_star"][future:future + keep]
            held = real["q_star"][at:at + keep]
            errors.append(np.degrees(np.abs(predicted - truth)).mean())
            naive.append(np.degrees(np.abs(held - truth)).mean())
        if errors:
            out.append({"horizon": horizon,
                        "ms": round(1000 * horizon * float(np.median(np.diff(real["t"])))),
                        "student_deg": round(float(np.mean(errors)), 2),
                        "hold_deg": round(float(np.mean(naive)), 2)})
    model.train()
    return out
