"""The student: the network that runs live in place of the solver.

WHAT IT IS BUYING. Not accuracy in the present frame - measured, the per-frame
solve of the same input is as accurate as anything here, and an optimizer can
run live. What only a network can do, because each is statistical rather than
geometric:

  DENOISE. A solver retargets the frame it is given, noise included. A network
  trained on noisy windows against clean teacher answers learns the expected
  pose under the sensor's own error - it has seen what Leap noise looks like.

  PREDICT. The last 27 mm of a tap takes 56 ms, and a thumb that arrives when
  the operator's does moves 23 degrees in that time. A solver answers the
  present; the forward heads start the strike early, and the time-to-contact
  head says how early.

  FILL IN. In a fist the fingers are hallucinated by the tracker, and a solver
  solves the hallucination. A network that has seen every fist in the data
  learns what fingers do when the palm and thumb do this.

  CONTINUITY AND SPEED. A per-frame optimiser is a discontinuous function of
  its input; a network is continuous by construction, and one forward pass.

WHAT IT READS - TWO VIEWS OF THE SAME HAND, see `features`:

  The ALIGNED keypoints carry pose: bone directions, which finger is where.
  A pinch does not survive them - the per-digit scaling sends two touching
  tips 136 mm apart - so on their own they cannot say WHICH finger the thumb
  is closing on, and a student fed only these learned every pinch as an index
  pinch (middle-finger contact 0.4% against the teacher's 56%).

  The DISTANCE TABLE, from the RAW view, carries relations: every tip pair and
  every curl, in millimetres the sensor actually measured. A cell that is
  21 mm is a pinch on THAT finger; twenty frames of a cell closing is a
  strike arriving, and the causal stack takes whatever difference of them it
  needs. (Handing it the derivatives ready-made was tried and removed - see
  "measured, and off" below.) The same table the teacher's objective reads, indexed the
  same way - `geometry`.

THREE PIECES, and each is a deliberate constraint rather than a layer count:

  A CAUSAL TCN over the feature window. Causal because this has to run live:
  a centred window would need frames that have not happened. Dilations 1-2-4-8
  give a 61-frame receptive field, which covers the 20-frame window three
  times over, at four layers.

  A POSE BASIS on the output - PCA over the teacher's own solutions. Measured,
  the basis is the most expensive constraint in this file, and it now ships at
  the full sixteen; see BASIS.

  A ZERO-INITIALISED HEAD, so an untrained network emits the mean pose rather
  than noise, and training starts from something a hand could hold.

TWO AUXILIARY HEADS off the same trunk, trained but never commanded: which
fingers the teacher had in contact at this frame, and how long until each
thumb pair next touches. They are there to make the trunk carry the answer to
"which finger, and when" - the two things a joint-space loss is indifferent
to until a pinch lands on the wrong finger. Downstream they are what a
contact-aware executor would read.

The FK is `kinematics.py`, shared with the teacher's objective, walked off
the same URDF data `model.py` holds - so the network's idea of where a
fingertip is cannot drift from the simulator's or the teacher's.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import geometry
from .objective import TOUCH_MM

# The window the network reads. 20 frames at the sensor's ~120 Hz is ~170 ms of
# history - enough to see a snap begin and end, and short enough that the oldest
# frame still describes the gesture in progress. (The rate is read from the
# recording's timestamps wherever it matters; it is 119 Hz on this sensor, not
# the 125 the project once assumed.)
WINDOW = 20

# Dilations for the causal stack. Each block holds TWO kernel-3 convolutions,
# so the receptive field is 1 + 4*sum(dilations) = 61 frames, not the
# 1 + 2*sum(dilations) = 31 this comment claimed for a while - measured by
# bumping one input frame at a time and watching where the last output stops
# moving. Either number covers WINDOW, which is all the claim needed; anything
# less and the earliest frames in the window are invisible to the last output,
# which is the one that matters.
DILATIONS = (1, 2, 4, 8)
CHANNELS = 96

# PCA components on the output. Ten carried 98.3% of the old teacher's variance
# and reconstructed its labels to 2.3 degrees; sixteen is the identity.
#
# SIXTEEN, MEASURED. Same architecture, same data, 3000 steps, mean joint error
# on what it just trained on:
#
#     basis 10  2.95 deg        basis 16  1.56 deg
#
# For scale, every other knob tested against it: doubling the channels to 192
# buys 0.18, quadrupling the window to 80 frames buys 0.12, letting the network
# see 10 frames of FUTURE buys 0.15, and adding two more dilation blocks costs
# 0.12. The bottleneck was never capacity or history; it was that the answer
# had to be written in ten numbers. The shipped student is trained at sixteen.
#
# What the constraint gave up is a plausibility prior: with the basis at
# identity, the LIMIT term in `losses` is the only thing keeping a pose inside
# the URDF's range. `--basis 10` is still there for the comparison.
BASIS = 16

# How far ahead the network predicts, in frames: now, and roughly 40, 80 and
# 120 ms at the sensor's rate.
#
# WHY PREDICT AT ALL. The teacher is smooth because it smooths with a CENTRED
# window - it uses frames that have not happened. A causal student cannot, and
# measured, that is where it loses: teacher jerk 1.7e3, student and live solver
# both around 8e3. Smoothing causally would close it and pay in lag, which is
# the trade the live solver already makes and the reason it feels wrong.
#
# A head that predicts FORWARD buys the lag back. Command the 80 ms head and
# the hand is acting on where the operator will be, not where they were, so the
# smoothing costs nothing in felt latency - and it hides the pipeline's own
# delay at the same time.
#
# The zero horizon stays first and stays trained: it is the honest one, and
# having it in the same network keeps the others anchored to a mapping that is
# still correct at the present instant.
HORIZONS = (0, 5, 10, 15)

# ---------------------------------------------------------------- features
# Every constant here was measured on the recorded session, not chosen.
#
# The distance channels' scale: the 99th percentile of the thumb row in
# motion, so the channel sits near unit range on a moving hand.
DISTANCE_MM = 100.0

# Intent per thumb pair: 1 at touching, 0 beyond `INTENT_FAR_MM`, the same
# ramp `objective.intent_from_distance` uses for a whole hand, here per cell
# so the network is told which finger as well as how much.
INTENT_FAR_MM = 120.0

# ------------------------------------------------------ measured, and off
# Three input and head variants were built, measured and rejected as defaults.
# None is a module flag any more: a flag nothing ever set was a switch that
# looked live and was not - flipping one changed some paths and not others,
# and one of them raised on any caller that did not also pass a noise fit.
# What remains of each is what the measurement said, and the code that is
# still worth having is reached by argument.
#
# DISTANCE DERIVATIVES - removed. Velocity, acceleration and jerk of the
# distance table were read heavily and badly: on a STILL fist the output
# rattled at 5e3 jerk against the keypoint-only student's 626, and a Jacobian
# attribution put 99% of the output's frame-to-frame motion on the three
# derivative groups. With them zeroed, the prediction they were meant to buy
# came out BETTER - the 80 ms head beat holding still by 0.35 deg against 0.25,
# the 120 ms head by 1.21 against 0.90. A causal stack over twenty frames of
# distances computes whatever difference it needs, trained against smooth
# labels rather than fixed against noisy input. The implementation, and the
# stencil and rate it ran at, are in git history for a sensor that changes this.
#
# BELIEF CHANNELS - kept, `belief=True`. See `belief.py` for the result: 0.96
# deg closer to the teacher over three seeds, non-overlapping, at the cost of
# every contact number.
#
# GRASP HEAD - kept, `authority=True`, and it does not pay. One seed against
# the gated-belief student it would sit on: 6.81 deg from the teacher against
# 6.68, inside the 0.50 deg seed spread. THE REPRESENTATION WAS NOT THE
# PROBLEM: projecting the teacher's own poses onto the top three PCA
# directions - an oracle, using the answer - lands at 4.15 deg on fist_hold,
# 3.84 on curl_fast and 5.39 on occlusion, all well ahead of the pose head.
# Three numbers can express these hands; what the trained head cannot do is
# predict them from keypoints. A retry starts there, not at SYNERGIES.

# HOW MANY NUMBERS A GRASP IS. Three, and it is not a guess: the earlier work
# here found the same, and it is the standard result for human grasping - a
# hand closing on something moves along very few directions at once.
SYNERGIES = 3

# WHEN TO STOP BELIEVING THE KEYPOINTS. Authority hands over from the pose the
# keypoints imply to the pose the INTENT implies, and it is gated on two
# measured things rather than on a mode switch:
#
#   sigma, from `belief.py` - how far the reported points might be from the
#   real ones. Measured, a closed hand's fingers are the tracker's model, not
#   the operator's: a still fist reports 0.03 degrees of motion per frame,
#   which is a fit holding still and not a hand.
#
#   grab_strength, which the SDK does report usefully, unlike confidence. It
#   is over 0.5 on 23.7% of the recorded frames.
#
# alpha 0 is the pose the fingers say; alpha 1 is the pose the grasp says.
AUTHORITY_SIGMA_MM = 1.2     # sigma at which the keypoints are half-believed
AUTHORITY_GRAB = 0.5         # grab_strength at which the same is true

def feature_width(belief=False):
    """One frame's input width: aligned keypoints, the distance table, the
    thumb row's intent, and the per-keypoint sigma when `belief` is set.

    A FUNCTION, NOT A CONSTANT, because a checkpoint outlives the setting that
    made it - a loader passes what the checkpoint recorded, see
    `replay.student_rollout`.
    """
    return (21 * 3 + geometry.CELLS + len(geometry.THUMB_ROW)
            + (21 if belief else 0))


FEATURES = feature_width()


def features(aligned, raw, belief=None):
    """(b, T, 21, 3) aligned, (b, T, 21, 3) raw -> (b, T, FEATURES).

    The two views are the teacher's two views: the aligned one is what the
    solver is sent and carries directions; the raw one is rotation-only and
    carries distances. Everything derived here is causal and window-local, so
    a live caller with a rolling buffer of both views gets identical numbers.

    `belief` is a `belief.Belief` for a student trained with it, and its
    channels are computed HERE rather than handed in, so training and the live
    path cannot end up with two versions of the same estimate. Passing one
    IS what turns the channels on.
    """
    batch, frames = aligned.shape[0], aligned.shape[1]
    pose = aligned.reshape(batch, frames, -1)
    distance = geometry.pairwise(raw)                       # (b, T, CELLS) mm
    thumb = distance[..., list(geometry.THUMB_ROW)]
    intent = ((INTENT_FAR_MM - thumb) / (INTENT_FAR_MM - TOUCH_MM)).clamp(0.0, 1.0)
    parts = [pose, distance / DISTANCE_MM]
    # Passing a fit IS what turns the 21 sigma channels on, so a loaded belief
    # student and a baseline one each get exactly the width they were trained
    # with - the caller that owns the architecture passes the fit or does not.
    if belief is not None:
        parts.append(belief.channels(raw))
    return torch.cat(parts + [intent], dim=-1)


# ------------------------------------------------------------------- basis
def pca_basis(q, components=BASIS):
    """Mean pose and principal directions of the teacher's own solutions.

    Fitted on the LABELS, not on a hand-authored set of plausible poses. The
    question the basis has to answer is "what does the teacher produce", and
    the teacher's output is the only honest answer to it - a bank of poses
    someone thought looked reasonable would constrain the student to a
    different manifold than the one it is being trained to copy.
    """
    q = np.asarray(q, dtype=np.float64)
    mean = q.mean(axis=0)
    _, _, right = np.linalg.svd(q - mean, full_matrices=False)
    return mean.astype(np.float32), right[:components].astype(np.float32)


class CausalNorm(nn.Module):
    """GroupNorm whose statistics at frame t are taken over frames <= t.

    THE NORM IS PART OF THE SMOOTHING. This began as GroupNorm(8, C), which
    pools its statistics over TIME as well as channels - so the output at
    frame t was normalised by the mean of frames after it. Live, the newest
    frame is last and nothing follows it, so the frame that ships was honest;
    but every earlier supervised frame trained under statistics it could not
    have live, and `selftest` could not prove the network causal.

    The obvious repair, a per-frame LayerNorm, is causal and was measured to
    rattle: a norm that sees only one frame gives each frame its own scale,
    and the pooled statistics had been holding consecutive outputs together.
    Same data, same schedule, held out:

        per-frame norm            jerk 1.99e4   worst step 20.6 deg
        this, cumulative          jerk 1.66e4   worst step 16.6 deg

    So the statistics run CUMULATIVELY: at the newest frame they are exactly
    GroupNorm's over the whole window, which is what the old student shipped
    and the live path sees, and at every earlier frame they cover only what
    had happened. Causal at every position, and the bump test in `selftest`
    holds for it.
    """

    def __init__(self, channels, groups=8, eps=1e-5):
        super().__init__()
        self.groups, self.eps = groups, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        batch, channels, frames = x.shape
        grouped = x.reshape(batch, self.groups, channels // self.groups, frames)
        count = (channels // self.groups) * torch.arange(
            1, frames + 1, device=x.device, dtype=x.dtype)
        mean = grouped.sum(dim=2).cumsum(dim=-1) / count
        var = ((grouped * grouped).sum(dim=2).cumsum(dim=-1) / count
               - mean ** 2).clamp_min(0.0)
        normed = (grouped - mean[:, :, None]) / torch.sqrt(var[:, :, None] + self.eps)
        return (normed.reshape(batch, channels, frames) * self.weight[None, :, None]
                + self.bias[None, :, None])


class CausalBlock(nn.Module):
    """One dilated causal convolution, with a residual path.

    Left-padded and then trimmed, rather than padded symmetrically: symmetric
    padding lets the output at frame t see frame t+1, which trains a network
    that cannot exist live. It is an easy mistake to make and it does not show
    up as an error - only as a model that is mysteriously better offline than
    on the bench. `selftest` bumps the last frame and checks nothing earlier
    moves.
    """

    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm1 = CausalNorm(channels)
        self.norm2 = CausalNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv1(nn.functional.pad(x, (self.pad, 0)))
        y = self.drop(torch.relu(self.norm1(y)))
        y = self.conv2(nn.functional.pad(y, (self.pad, 0)))
        y = self.drop(torch.relu(self.norm2(y)))
        return x + y


class Student(nn.Module):
    """Two keypoint windows in, joint vector out - plus what it knows about
    contact.

    Input is the same 21 MANO keypoints the solver is given - metres, robot
    axes, wrist at the origin - and the raw view of the same frames, which the
    live path builds from the same hand with the alignment left off. See
    `features` for what is made of them.
    """

    # The auxiliary heads, one output per thumb pair in `geometry.THUMB_ROW`.
    INTENT = "intent"        # 3-D grasp coordinate, with authority=True
    SPREAD = "spread"        # log sigma of its OWN joint error, with belief=True
    CONTACT = "contact"      # logit: was the teacher touching on this pair
    TTC = "ttc"              # seconds to its next touch, clipped - see TTC_MAX

    def __init__(self, mean, basis, channels=CHANNELS, dilations=DILATIONS,
                 dropout=0.1, horizons=HORIZONS, belief=False, authority=False):
        super().__init__()
        # WHAT THIS STUDENT IS, recorded on the instance. `belief` and
        # `authority` decide an ARCHITECTURE - an input width and two heads -
        # so a loader rebuilds exactly what a checkpoint says it was trained
        # as, and nothing about the process it loads into can change that.
        self.belief_on = bool(belief)
        self.authority_on = bool(authority)
        self.horizons = tuple(horizons)
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("basis", torch.tensor(basis))
        self.input = nn.Conv1d(feature_width(self.belief_on), channels, 1)
        self.blocks = nn.Sequential(*[CausalBlock(channels, d, dropout)
                                      for d in dilations])
        # One head per horizon, from a SHARED trunk. Separate networks would
        # each have to learn the mapping again from the same data; sharing it
        # means the future heads only have to learn how a hand carries on,
        # which is the part that differs.
        self.head = nn.Conv1d(channels, len(self.horizons) * basis.shape[0], 1)
        # Zero-initialised, so an untrained network emits exactly the mean pose
        # rather than noise. Training then starts from a pose a hand can hold,
        # and the first epochs are spent improving it rather than escaping a
        # tangle of joints at their limits.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        pairs = len(geometry.THUMB_ROW)
        self.contact = nn.Conv1d(channels, pairs, 1)
        self.ttc = nn.Conv1d(channels, pairs, 1)
        # HOW WRONG IT THINKS IT IS, per joint, as log sigma. Zero-initialised
        # so an untrained network claims sigma = 1 rad everywhere rather than
        # something confident and arbitrary. Trained through a Gaussian
        # likelihood on the joint term, which is what lets it say "this frame
        # was guesswork" instead of hedging uniformly - see `train.losses`.
        self.spread = None
        if self.belief_on:
            self.spread = nn.Conv1d(channels, self.mean.shape[0], 1)
            nn.init.zeros_(self.spread.weight)
            nn.init.zeros_(self.spread.bias)
        # WHAT THE HAND IS TRYING TO DO, in three numbers, and a decoder back
        # to a pose. Not a bottleneck on the pose head - that was measured and
        # it costs 4.4 degrees on moving frames. This is a SECOND opinion,
        # blended in by `authority` only where the keypoints stop being worth
        # believing. Seeded from the top three directions of the same basis,
        # so it starts as the best three-dimensional summary of what the
        # teacher does rather than from noise.
        self.intent = self.synergy = None
        if self.authority_on:
            self.intent = nn.Conv1d(channels, SYNERGIES, 1)
            self.synergy = nn.Linear(SYNERGIES, self.mean.shape[0], bias=False)
            with torch.no_grad():
                self.synergy.weight.copy_(self.basis[:SYNERGIES].T)
        # The sensor's error model, with belief=True. Not a parameter and
        # not saved in the state dict - it is a fit, and it belongs to the
        # recording rather than to the network. `train` and `Mimic` both set
        # it from the same `noise.json`.
        self.belief = None

    def trunk(self, aligned, raw):
        """(b, T, 21, 3) x2 -> (b, channels, T), causal in T."""
        x = features(aligned, raw, self.belief).transpose(1, 2)
        return self.blocks(self.input(x))

    def predict(self, aligned, raw):
        """Everything the network says about a window, in one pass:

            q        (b, T, horizons, joints)   the pose, per head
            contact  (b, T, pairs)              logits, thumb row
            ttc      (b, T, pairs)              0..1, fraction of TTC_MAX
        """
        batch, frames = aligned.shape[0], aligned.shape[1]
        x = self.trunk(aligned, raw)
        coefficients = self.head(x).transpose(1, 2)        # (b, T, h*k)
        coefficients = coefficients.reshape(batch, frames, len(self.horizons),
                                            self.basis.shape[0])
        out = {"q": self.mean + coefficients @ self.basis,
               self.CONTACT: self.contact(x).transpose(1, 2),
               self.TTC: self.ttc(x).transpose(1, 2)}
        if self.spread is not None:
            out[self.SPREAD] = self.spread(x).transpose(1, 2)
        if self.intent is not None:
            intent = self.intent(x).transpose(1, 2)
            out[self.INTENT] = intent
            out["q_intent"] = self.mean + self.synergy(intent)
        return out

    def forward(self, aligned, raw):
        """(b, T, 21, 3) aligned, (b, T, 21, 3) raw -> (b, T, horizons, joints)."""
        return self.predict(aligned, raw)["q"]


# Where the time-to-contact head's range ends, in seconds. Beyond this the
# label is "not soon" and the head says 1. Half a second is three times the
# longest approach in the recorded taps, and long enough that an executor
# reading it has a whole strike of warning.
TTC_MAX = 0.5


def authority(sigma_mm, grab=None,
              half_sigma=AUTHORITY_SIGMA_MM, half_grab=AUTHORITY_GRAB):
    """How much to believe the GRASP over the fingertips: alpha in [0, 1].

    Two independent reasons to stop trusting a keypoint, combined so that
    either one is enough. Both are measured quantities rather than modes:

      sigma is `belief.Belief.sigma_for` - the sensor's fitted error for this
      keypoint, on this frame, in millimetres. High when the digit is moving
      fast or folded away behind the hand.

      grab is the SDK's `grab_strength`, which unlike `confidence` carries
      real signal. A hand closing on an object is a hand whose fingers the
      sensor is about to lose.

    Returns alpha per frame: 0 keeps the pose the keypoints imply, 1 takes the
    pose the three-dimensional grasp coordinate implies, and the live path
    blends between them. Nothing here is a threshold - both terms are smooth,
    because a hand that flips between two mappings at a boundary is worse than
    either of them.
    """
    from_sigma = (sigma_mm.mean(dim=-1) / half_sigma).clamp(0.0, 4.0)
    alpha = 1.0 - torch.exp(-from_sigma)
    if grab is not None:
        alpha = 1.0 - (1.0 - alpha) * (1.0 - (grab / half_grab).clamp(0.0, 1.0))
    return alpha.clamp(0.0, 1.0)


def blend(q_pose, q_intent, alpha):
    """The commanded pose: keypoints where they are worth believing, grasp
    where they are not. `alpha` broadcasts over the joint axis."""
    return q_pose + (q_intent - q_pose) * alpha[..., None]


# ------------------------------------------------------------------ windows
def windows(bounds, length=WINDOW):
    """Start indices whose whole window lies inside one run.

    A window that straddles a run boundary is a window over a discontinuity -
    the operator's hand teleporting from the end of one block to the start of
    the next - and training on those teaches the network to smooth across
    something that never happens live.
    """
    out = []
    for start, end in bounds:
        if end - start >= length:
            out.append(np.arange(start, end - length + 1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
