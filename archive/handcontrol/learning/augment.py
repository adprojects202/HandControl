"""Putting the measured sensor error back onto a clean hand.

`noise.py` measures what the Leap does. This reproduces it. The two are
deliberately separate files: one reads recordings and writes numbers, the other
reads those numbers and writes keypoints, and neither needs the other's
machinery.

WHY NOT ONE GAUSSIAN. Every property measured on the real sensor says the
obvious implementation would train for noise nobody has:

  It is not a constant. Sigma runs 0.042 mm on a still hand to 7.98 mm on a
  fast one - a factor of 190 - so the amplitude is drawn from the hand's own
  speed, per keypoint, per frame.

  It is not independent. After dividing speed out, keypoints on DIFFERENT
  fingers still correlate at 0.34-0.82: the tracker gets the whole hand
  slightly wrong at once. Sampled independently, error averages away along a
  chain and the real error does not - which is the difference between a finger
  that looks noisy and one that bends the wrong way. So the variance is split
  three ways and only the last part is per keypoint.

  It is not white. Most of its power sits below 5 Hz. White noise is trivially
  removable by any temporal model, and a student trained against it would learn
  that the noise it meets on the bench is removable too. Each term is coloured
  by an AR(1) with the measured coefficient.

  It is not uniform over the hand. A hidden finger is 3 to 5.6 times worse than
  a visible one, so occlusion is a multiplier applied per digit rather than a
  flat increase.

  And it is not all noise. Snaps are single-frame relocations up to 98 mm -
  wrong samples, not error to be averaged - and they are injected as events at
  their measured rate rather than folded into a sigma.

USED AS A TRAINING AUGMENTATION, not a simulation: the target stays the clean
hand's joint vector, so the network learns to undo this as a side effect of
learning the mapping, at no inference cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..leap import DIGITS

# MANO slot -> which digit it belongs to. Slot 0 is the wrist and belongs to
# none of them; `spread_for` falls back to DIGITS[0] - the thumb's per-finger
# term - rather than to the global one alone, because a wrist that jittered
# independently of every digit would break the rigid-bone invariant the
# rest of this module keeps.
SLOT_DIGIT = [None] + [DIGITS[i // 4] for i in range(20)]


class NoiseModel:
    """The measured sensor error, as something you can add to a hand.

    Stateful across a sequence, because the noise is: `step` carries the AR(1)
    state forward, and calling it on shuffled frames would produce white noise
    with the right variance and the wrong character. Call `reset` between
    sequences.
    """

    def __init__(self, fit, rng=None):
        augmentation = fit["augmentation"] if "augmentation" in fit else fit
        curve = augmentation["sigma_vs_speed_mm"]
        self.speed_knots = np.asarray(curve["speed_mm_s"], dtype=float)
        self.sigma_knots = np.asarray(curve["sigma_mm"], dtype=float)
        split = augmentation["variance_split"]
        # Stored as standard deviations, which is what gets multiplied - the
        # measurement is a variance share, so this is where the square root
        # belongs and nowhere else.
        self.share = {name: float(np.sqrt(max(split[name], 0.0)))
                      for name in ("global", "per_finger", "per_keypoint")}
        self.ar1 = float(augmentation["ar1"])
        self.occlusion = dict(augmentation["occlusion_multiplier"])
        self.snap_rate = augmentation["snaps"]
        self.hz = float(augmentation["hz"])
        self.rng = rng or np.random.default_rng()
        self.reset()

    @classmethod
    def load(cls, path, rng=None):
        return cls(json.loads(Path(path).read_text()), rng)

    def reset(self):
        self.state = None

    def sigma_for(self, speed):
        """Interpolate the measured sigma(speed) curve. Flat outside its range:
        extrapolating a curve fitted over 0-500 mm/s to a hand moving at 2 m/s
        would invent an amplitude nothing measured."""
        return np.interp(speed, self.speed_knots, self.sigma_knots)

    def _white(self, shape):
        """One frame of the three-term structure, unit variance overall."""
        common = self.rng.standard_normal((1, 3)) * self.share["global"]
        finger = self.rng.standard_normal((5, 3)) * self.share["per_finger"]
        point = self.rng.standard_normal(shape) * self.share["per_keypoint"]
        spread = np.empty(shape)
        for slot in range(shape[0]):
            digit = SLOT_DIGIT[slot] if slot < len(SLOT_DIGIT) else None
            index = DIGITS.index(digit) if digit else 0
            spread[slot] = common[0] + finger[index]
        return spread + point

    def step(self, points, speed, occluded=(), snap=True):
        """One frame: clean (21,3) keypoints in millimetres, noisy ones out.

        `speed` is per keypoint in mm/s, from the CLEAN track - the amplitude
        is a function of how fast the hand is really moving, not of how fast
        the noisy version appears to.
        """
        points = np.asarray(points, dtype=float)
        white = self._white(points.shape)
        if self.state is None or self.state.shape != white.shape:
            self.state = white
        else:
            # AR(1). The (1 - a^2) factor keeps the stationary variance at one,
            # so colouring the noise changes its character and not its size.
            self.state = (self.ar1 * self.state
                          + np.sqrt(1.0 - self.ar1 ** 2) * white)

        sigma = np.asarray(self.sigma_for(speed), dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(len(points), float(sigma))
        scale = np.ones(len(points))
        for slot in range(len(points)):
            digit = SLOT_DIGIT[slot] if slot < len(SLOT_DIGIT) else None
            if digit in occluded:
                scale[slot] = self.occlusion.get(digit, 1.0)

        noisy = points + self.state * (sigma * scale)[:, None]
        if snap:
            noisy = self.snap(noisy)
        return noisy

    def snap(self, points):
        """Maybe relocate one digit, this frame.

        Applied to a whole digit rather than one keypoint: the tracker changing
        its mind about a finger moves the finger, not a point on it. Whether
        this frame gets one is drawn at the measured per-second rate, converted
        to per frame - so a slower recording does not get more snaps.
        """
        rate = float(self.snap_rate.get("per_second_moving", 0.0)) / self.hz
        if self.rng.random() > rate:
            return points
        digit = DIGITS[self.rng.integers(len(DIGITS))]
        magnitude = self.snap_rate["magnitude_mm"]
        # Lognormal through the measured quantiles rather than uniform to the
        # worst case: the p99 is 2.6 mm and the worst 98, and drawing uniformly
        # between them would make a once-a-minute event the common case.
        size = float(np.clip(self.rng.lognormal(
            np.log(max(magnitude["p90"], 1e-3)), 1.4),
            0.0, magnitude["worst"]))
        direction = self.rng.standard_normal(3)
        direction /= max(np.linalg.norm(direction), 1e-9)
        out = points.copy()
        for slot, owner in enumerate(SLOT_DIGIT):
            if owner == digit:
                out[slot] += direction * size
        return out

    def sequence(self, points, dt, occluded=(), snaps=True):
        """A whole clean trajectory, noised. (n,21,3) mm in, same shape out.

        Speed is measured from the clean track here rather than passed in, so
        the caller cannot accidentally drive the amplitude with the noise it is
        about to add.
        """
        points = np.asarray(points, dtype=float)
        speed = np.linalg.norm(np.gradient(points, axis=0) / dt, axis=-1)
        self.reset()
        return np.stack([self.step(points[i], speed[i], occluded, snaps)
                         for i in range(len(points))])
