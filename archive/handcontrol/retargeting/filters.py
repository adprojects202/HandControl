"""Conditioning the sensor before anything is inferred from it.

Nothing in this file knows about robots, URDFs or this project. It takes
arrays in and gives arrays out, which is the point: the noise here is the
camera's, and the fix belongs at the sensor boundary rather than smeared
through the mapping that reads it.

WHY FILTER POSITIONS AND NOT ANGLES. An angle is a ratio of differences of
positions, so a millimetre of jitter on a bone joint becomes degrees of jitter
on the joint angle it defines, and worse for a short bone than a long one.
Filtering at the joint positions fixes the angles, the fingertip targets and
the drawn skeleton in one place; filtering the angles afterwards would leave
the other two noisy and would be fighting noise that has already been
amplified.

WHAT THE LEAP DOES NOT TELL YOU is which of those positions it measured and
which it inferred. A finger behind another finger is occluded, and the SDK
reports the hidden joints from a model prior - with `LEAP_HAND.confidence`
documented as "Not currently used (always 1.0)" and no per-joint quality field
anywhere in LeapC.h. The least trustworthy channel reports itself as perfect,
so the filtering cannot be selective and has to be uniform.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

# One Euro defaults, for joint positions in millimetres at ~100 Hz.
#
# MIN_CUTOFF sets how still a still hand looks: lower is steadier and lags
# more. BETA is how fast the cutoff opens up as the hand moves, and it is the
# whole trick - a fixed low-pass has to choose once between jitter at rest and
# lag in motion, and this one re-chooses every frame.
MIN_CUTOFF = 1.2         # Hz
BETA = 0.012             # cutoff Hz added per mm/s of speed
D_CUTOFF = 1.0           # Hz, for the speed estimate itself

# Hampel: a spike gate, not a smoother.
# MAD from a short window underestimates sigma, so a nominally 3-sigma gate
# fires far more often than 3 sigma should. These are set from the measured
# false-positive rate on gaussian noise, not from the textbook values.
WINDOW = 15
N_SIGMAS = 4.0
MAD_TO_SIGMA = 1.4826    # makes MAD a consistent estimator of sigma for gaussians


def _alpha(cutoff, dt):
    """The exponential-smoothing constant for a given corner frequency."""
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuro:
    """The 1-euro filter (Casiez, Roussel & Vogel), one instance for N channels.

    An exponential filter whose cutoff rises with the measured speed of the
    signal. At rest the cutoff is low and jitter is crushed; in motion it opens
    and the lag collapses. That is the property a fixed alpha cannot have, and
    lag is what makes teleoperation feel wrong rather than merely noisy.

    Vectorised over the whole array - one filter for all 75 joint coordinates
    of a hand, each channel with its own speed estimate.
    """

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self.value = None            # last filtered value
        self.speed = None            # last filtered derivative

    def __call__(self, sample, dt):
        sample = np.asarray(sample, dtype=float)
        if self.value is None or self.value.shape != sample.shape:
            # First sample, or the shape changed under us - start from it
            # rather than easing toward it from a stale pose.
            self.value = sample.copy()
            self.speed = np.zeros_like(sample)
            return self.value.copy()

        dt = max(float(dt), 1e-6)
        speed = (sample - self.value) / dt
        self.speed += _alpha(self.d_cutoff, dt) * (speed - self.speed)
        cutoff = self.min_cutoff + self.beta * np.abs(self.speed)
        # alpha is per-channel here, because the cutoff is: each
        # coordinate carries its own speed estimate.
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        alpha = 1.0 / (1.0 + tau / dt)
        self.value += alpha * (sample - self.value)
        return self.value.copy()


class Hampel:
    """A causal spike gate: median plus median-absolute-deviation.

    Leap tracking throws single-frame outliers - a finger re-acquired in the
    wrong place, a hand briefly confused with its own reflection. Those are not
    noise to be averaged; averaging spreads them over the window. They are
    wrong samples, and the honest response is to not use them.

    Causal on purpose. A centred window would judge each sample against its
    neighbours on both sides and be strictly better at it, at the cost of half
    a window of latency on every frame - which is exactly the currency this
    whole file is trying not to spend.
    """

    def __init__(self, window=WINDOW, n_sigmas=N_SIGMAS):
        self.window = int(window)
        self.n_sigmas = float(n_sigmas)
        self.reset()

    def reset(self):
        self.history = []
        self.rejected = 0

    def __call__(self, sample):
        sample = np.asarray(sample, dtype=float)
        if len(self.history) < self.window:
            self.history.append(sample.copy())
            return sample.copy()

        past = np.stack(self.history)
        median = np.median(past, axis=0)
        deviation = MAD_TO_SIGMA * np.median(np.abs(past - median), axis=0)
        # A perfectly still signal has zero deviation, and every real sample
        # would then look like a spike. The floor is what stops the gate from
        # locking a stationary hand to its own median.
        threshold = self.n_sigmas * np.maximum(deviation, 1e-4)
        spike = np.abs(sample - median) > threshold
        cleaned = np.where(spike, median, sample)
        self.rejected += int(spike.sum())

        # The RAW sample goes into the history, not the cleaned one. Feeding
        # back the substituted median makes the window collapse toward itself:
        # the deviation estimate shrinks, the threshold shrinks with it, and
        # the gate starts rejecting ordinary samples - measured at 75% of them
        # on plain gaussian noise before this line was fixed.
        self.history.append(sample.copy())
        self.history.pop(0)
        return cleaned


class HandFilter:
    """Hampel then One Euro, over every joint position of a tracked hand.

    Returns a new hand rather than mutating: the raw one stays available, which
    is what makes it possible to draw both and see what the filter is actually
    doing.

    Resets itself when tracking is lost or the hand id changes. Carrying a
    filter state across a dropout would ease the new hand out of where the old
    one was last seen, which is a smooth, confident lie.
    """

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, window=WINDOW,
                 n_sigmas=N_SIGMAS, enabled=True):
        self.euro = OneEuro(min_cutoff, beta)
        self.hampel = Hampel(window, n_sigmas)
        self.enabled = bool(enabled)
        self.hand_id = None

    @property
    def rejected(self):
        return self.hampel.rejected

    def configure(self, min_cutoff=None, beta=None, enabled=None):
        if min_cutoff is not None:
            self.euro.min_cutoff = float(min_cutoff)
        if beta is not None:
            self.euro.beta = float(beta)
        if enabled is not None:
            self.enabled = bool(enabled)

    def reset(self):
        self.euro.reset()
        self.hampel.reset()
        self.hand_id = None

    def __call__(self, hand, dt):
        if hand is None:
            self.reset()
            return None
        if not self.enabled:
            return hand
        if hand.id != self.hand_id:
            self.reset()
            self.hand_id = hand.id

        # The palm rides in the same array as the bones. It is a reported point
        # like any other and things downstream measure against it, so leaving
        # it unfiltered would put the camera's jitter straight back into a
        # hand whose joints were cleaned. (`hand_frame` is built from
        # `direction` and `normal`, which are NOT filtered - see the note at
        # the top of this module.)
        digits = sorted(hand.bones)
        stacked = np.concatenate([hand.bones[d] for d in digits]
                                 + [hand.palm[None, :]])            # (26, 3)
        cleaned = self.euro(self.hampel(stacked), dt)

        bones, at = {}, 0
        for digit in digits:
            count = len(hand.bones[digit])
            bones[digit] = cleaned[at:at + count]
            at += count
        return replace(hand, bones=bones, palm=cleaned[at])
