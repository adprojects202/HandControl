"""How much to trust each keypoint, from the sensor's own measured error.

WHY THIS EXISTS. Every mapping in this project has treated the tracked
keypoints as truth. They are not: the Leap's error spans two orders of
magnitude across the session - `noise.json` fits sigma against speed over a
190x range - and it is far worse on a digit the sensor cannot see. The SDK
reports a per-hand `confidence` field, and it is useless: the recording carries
one value for every tracked frame.

So the uncertainty is not read from the sensor, it is COMPUTED from the sensor's
fitted error model. Two terms, both measured, both live-computable from the
keypoints alone:

  SPEED. `augmentation.sigma_vs_speed_mm` is a 60-knot curve of per-axis sigma
  against smoothed keypoint speed, fitted over the whole recording. A still
  fingertip sits near 0.04 mm; one moving at 500 mm/s is near 4 mm.

  SELF-OCCLUSION. `augmentation.occlusion_multiplier` is per digit, and it is
  measured as fist_hold over flat_hold - two still hands in front of the same
  sensor, where the only difference is that the fist puts the ring and middle
  chains behind the fingers in front. So the multiplier is what CURL costs,
  and the live proxy for it is how curled that digit is, which needs nothing
  but the hand.

The product is an estimate in millimetres per keypoint, and `student.features`
carries it as 21 channels. It is a BELIEF, not a measurement: it says how far
the reported point might be from the real one, which is exactly what a
downstream authority blend needs in order to stop trusting a finger the sensor
is inventing.

WHAT IT BOUGHT, over three seeds against three baseline seeds. Distance from
the teacher fell from 7.87 to 6.91 degrees with the two ranges not overlapping
(6.71-7.21 against 7.58-8.08), and the seed spread on the posture score fell
from 1.38 degrees to 0.32 - so it also made training markedly more repeatable.

WHAT IT COST, and this is why it is not on by default. Every contact number
got worse, all of them non-overlapping with the baseline: the pinch gap went
from 0.5-0.6 mm to 1.2-1.6, self-penetration from 0.3-0.7% of frames to
0.8-1.6, and posture on contact frames from 13.3-15.1 degrees to 15.5-16.0.

The mechanism explains the shape of that. `curl` is the live proxy, so the
channels say "distrust this digit" exactly when a digit is folded - and a
folded digit is usually one that is about to touch something. It backs off
where precision matters most. The fix to try is gating sigma on the thumb
row's intent, so an imminent touch suppresses the distrust rather than
inviting it; that is untested.

One more thing the per-block numbers say: the gain is where CURL is, not
where hiding is. Against baseline seed 0, fist_hold improves 2.26 degrees and
`curl_fast` 0.73, while the block actually named `occlusion` - the operator
deliberately hiding the hand - moves 0.01. That is consistent rather than
disappointing: `occlusion_multiplier` was fitted as fist_hold over flat_hold,
so a fist is the case it was measured on, and curl cannot see a hand hidden
by being turned away.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..leap import DIGITS

# Which digit each MANO-21 slot belongs to. Slot 0 is the wrist, which no digit
# owns and which the sensor sees whenever it sees the hand at all.
SLOT_DIGIT = [None] + [DIGITS[i // 4] for i in range(20)]

# Frames between the two samples a speed is taken over. One frame is the
# obvious choice and the wrong one: at 8 ms a still hand's tracking jitter
# differentiates into 116 mm/s against a real strike's 1,050 at the 99th
# percentile, so the fitted sigma curve would be read at the wrong end of
# itself. Three frames (24 ms) brings the jitter down to 67 mm/s against
# 1,020, a 15:1 separation; wider starts to blur a strike, which lasts 56 ms.
STENCIL = 3

# A digit counts as fully self-occluded at this much total bend, in radians
# over its three bones. Measured: a flat hand sits near 0.2 and a fist near
# 3.4, so the ramp spends its range where the curl actually happens.
CURL_FULL = 2.6

# What a keypoint's sigma is divided by before it becomes a feature. The 99th
# percentile of the fitted curve, so the channel sits near unit range.
SIGMA_MM = 4.0

# Raw thumb-to-nearest-fingertip distances the gate ramps between, in mm.
# MEASURED, and the first values tried were wrong in a way worth recording.
# Borrowing the intent channel's ramp (120 mm down to 33) looked principled -
# same schedule the network already sees - and it inverted the gate: this
# operator's flat hand rests at 64 mm and their spread hand at 104, so a hand
# doing nothing was already halfway down the ramp, while `thumb_tap`, whose
# median over the block includes every retraction, came out LESS gated than
# `flat_hold`.
#
# Per block, raw thumb tip to nearest fingertip:
#
#     flat_hold      64 mm, near-constant      spread_hold   104 mm
#     fist_hold      34 mm, near-constant      thumb_tap     16 mm at p1, 20 at p10
#
# A FIST IS AS CLOSE AS A PINCH by this measure - 34 mm against a pinch's
# 16-20 - and a fist is where the belief channels bought the most (2.26
# degrees on `fist_hold`). So the ramp has to separate those two rather than
# just find "close", which means it lives well below 33 mm and not near it.
GATE_FAR_MM = 40.0
GATE_NEAR_MM = 18.0

# How much of sigma survives at full contact. Not zero: a touching digit is
# still measured by the same sensor with the same error, and claiming
# certainty there would be a different lie from the one being fixed.
GATE_FLOOR = 0.25


class Belief:
    """The sensor's fitted error model, as a live per-keypoint estimate.

    Built once from `noise.json` and then called per window. Holds no state
    between calls: the estimate for a frame depends on that frame and the two
    before it, which is what makes it identical offline and live.
    """

    def __init__(self, speed_mm_s, sigma_mm, occlusion, hz=125.0):
        self.speed = np.asarray(speed_mm_s, dtype=np.float32)
        self.sigma = np.asarray(sigma_mm, dtype=np.float32)
        # Per SLOT, not per digit, so the lookup is one gather rather than a
        # dict read inside the loop. The wrist takes 1.0: it is the one point
        # no amount of curl hides.
        self.multiplier = np.array(
            [1.0] + [occlusion.get(SLOT_DIGIT[i], 1.0) for i in range(1, 21)],
            dtype=np.float32)
        self.hz = float(hz)
        # WHETHER AN IMMINENT PINCH SUPPRESSES THE DISTRUST. An attribute, not
        # a global read at call time: it changes what the 21 channels MEAN, so
        # a checkpoint has to record it and a loader has to restore it, the
        # same way `belief` and `authority` are carried.
        self.gate = False

    @classmethod
    def load(cls, path):
        """From a `learn.py noise` fit. Raises if the fit predates the fields."""
        fit = json.loads(path.read_text()) if hasattr(path, "read_text") else json.load(open(path))
        aug = fit["augmentation"]
        curve = aug["sigma_vs_speed_mm"]
        return cls(curve["speed_mm_s"], curve["sigma_mm"],
                   aug.get("occlusion_multiplier", {}), aug.get("hz", 125.0))

    def to(self, device):
        self._t = {k: torch.as_tensor(v, device=device)
                   for k, v in (("speed", self.speed), ("sigma", self.sigma),
                                ("multiplier", self.multiplier))}
        return self

    def tensors(self, device):
        cached = getattr(self, "_t", None)
        if cached is None or cached["speed"].device != device:
            self.to(device)
        return self._t

    def sigma_for(self, raw, hz=None):
        """(b, T, 21, 3) metres -> (b, T, 21) millimetres of expected error.

        The RAW view, deliberately: sigma was fitted on what the sensor
        reported, and the aligned view has been through a per-digit scaling
        that would stretch the speeds it is read against.
        """
        t = self.tensors(raw.device)
        rate = self.hz if hz is None else hz
        points = raw * 1000.0                                   # mm

        # Speed over `STENCIL` frames, with the first few frames of a window
        # holding the first sample rather than differencing against something
        # invented, so their speed is zero rather than a jump from nothing.
        lead = points[:, :1].expand(-1, STENCIL, -1, -1)
        padded = torch.cat([lead, points], dim=1)
        step = points - padded[:, :points.shape[1]]
        speed = step.norm(dim=-1) * (rate / STENCIL)            # (b, T, 21) mm/s

        # The fitted curve, read by interpolation. `torch.searchsorted` wants
        # the query contiguous and the table 1-D, so this is done flat.
        flat = speed.reshape(-1).contiguous()
        index = torch.searchsorted(t["speed"], flat).clamp(1, len(self.speed) - 1)
        lo, hi = t["speed"][index - 1], t["speed"][index]
        span = (hi - lo).clamp_min(1e-6)
        frac = ((flat - lo) / span).clamp(0.0, 1.0)
        sigma = torch.lerp(t["sigma"][index - 1], t["sigma"][index], frac)
        sigma = sigma.reshape(speed.shape)

        return sigma * (1.0 + (t["multiplier"] - 1.0) * curl(raw))

    def channels(self, raw, hz=None, gate=None):
        """`sigma_for`, scaled to sit near unit range as a network input.

        `gate` defaults to this fit's own `gate`, which training and the
        loaders set from the checkpoint. See `contact_gate` for what it is for.
        """
        sigma = self.sigma_for(raw, hz)
        if self.gate if gate is None else gate:
            sigma = sigma * contact_gate(raw)
        return sigma / SIGMA_MM


def curl(raw):
    """(b, T, 21, 3) -> (b, T, 21) in [0, 1]: how curled each slot's digit is.

    THE LIVE STAND-IN FOR SELF-OCCLUSION, and it is the same quantity the
    multiplier was fitted against. `occlusion_penalty` divides the fist's
    per-digit sigma by the flat hand's - two still hands, same sensor, and the
    only difference between them is that the fist folds each chain behind the
    one in front. So what the multiplier measures is the cost of CURL, and a
    curl is three bone angles the hand itself carries.

    The wrist is 0 always: no amount of folding hides it.
    """
    points = raw
    out = points.new_zeros(points.shape[:-1])                   # (b, T, 21)
    for digit in range(5):
        base = 1 + 4 * digit
        bones = [points[..., base + k + 1, :] - points[..., base + k, :]
                 for k in range(3)]
        total = points.new_zeros(points.shape[:2])
        for a, b in zip(bones[:-1], bones[1:]):
            cosine = (torch.nn.functional.normalize(a, dim=-1)
                      * torch.nn.functional.normalize(b, dim=-1)).sum(-1)
            total = total + torch.arccos(cosine.clamp(-1.0, 1.0))
        amount = (total / CURL_FULL).clamp(0.0, 1.0)
        out[..., base:base + 4] = amount[..., None]
    return out


def contact_gate(raw):
    """(b, T, 21, 3) -> (b, T, 21) in [GATE_FLOOR, 1]: keep sigma, or suppress it.

    WHY THIS EXISTS. The belief channels bought 0.96 degrees against the
    teacher over three seeds, with the two groups' ranges not overlapping -
    and made every contact number worse by the same standard: the pinch gap
    went 0.5-0.6 mm to 1.2-1.6, self-penetration 0.3-0.7% of frames to
    0.8-1.6, posture on contact frames 13.3-15.1 degrees to 15.5-16.0.

    The cause is in `curl`. It is the live proxy for self-occlusion, so the
    channels say "distrust this digit" precisely when a digit folds - and a
    folding digit is usually one on its way to touching something. The layer
    backs off where precision matters most.

    So the sigma a digit carries is scaled down as its tip approaches the
    thumb's, on the same ramp the intent channel already uses. The thumb takes
    the nearest fingertip, since it is the other half of every pinch. Nothing
    else moves: a digit far from a pinch keeps the full estimate, which is
    where the gain came from.

    NOT ZERO AT CONTACT - see `GATE_FLOOR`. The sensor is no better on a
    touching finger than a free one; what changes is only that the network
    should stop being told to hedge there.
    """
    tips = [1 + 4 * d + 3 for d in range(5)]                 # 4, 8, 12, 16, 20
    thumb, fingers = tips[0], tips[1:]
    points = raw * 1000.0                                   # mm

    to_thumb = torch.stack(                                 # (b, T, 4)
        [(points[..., f, :] - points[..., thumb, :]).norm(dim=-1) for f in fingers],
        dim=-1)
    span = max(GATE_FAR_MM - GATE_NEAR_MM, 1e-6)
    near = ((GATE_FAR_MM - to_thumb) / span).clamp(0.0, 1.0)  # 1 at contact

    out = points.new_ones(points.shape[:-1])                # (b, T, 21)
    # The thumb answers to whichever finger is closest to it.
    out[..., 1:5] = 1.0 - (1.0 - GATE_FLOOR) * near.max(dim=-1).values[..., None]
    for slot, digit in enumerate(range(1, 5)):              # index, middle, ring
        base = 1 + 4 * digit
        out[..., base:base + 4] = (
            1.0 - (1.0 - GATE_FLOOR) * near[..., slot])[..., None]
    return out
