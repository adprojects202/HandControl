"""What a camera's tracker makes of a hand, as it moves: a true hand (in its
palm frame, handmotion.in_palm) -> the hand WiLoR would give, for the
reader to learn from (reader.py). Nothing here is made up from scratch -
the errors are WiLoR's own, on FreiHAND's photos (intent_pairs.py):

    noise = TrackerNoise()
    near = noise.neighbours(person)                       # (T, NEIGHBOURS), once per clip
    seen, found = noise.corrupt(person, near, hidden, rng)   # (B, T, 21, 3), (B, T)

    errors      a true pose's error is one WiLoR made on a FreiHAND hand
                posed like it (one of its NEIGHBOURS nearest training
                hands: a pinch is misread as pinches are, a fist as
                fists). The training photos' errors are smaller than the
                held-out ones' (WiLoR was trained on them): each point's
                are scaled to the held-out level, measured on the even
                evaluation photos (the odd ones are for testing).
    in time     a misread stays a while - the same view, the same mistake:
                one error, from a hand posed as this one is when it begins,
                faded into the next - and a little is fresh every frame
                (jitter). How long a misread lasts, the jitter and the
                error's size are drawn per clip (SLOW_S, JITTER, SCALE) - a
                camera, a light, a person. Two of WiLoR's errors are never
                mixed in full: they share a bias, which would add
    hidden      points the object hides (grab.views): the tracker still
                gives them, guessed - drawn towards another hand's
                (GUESS); with most of the hand hidden it loses it
    lost        no reading at all: the hand lost (bursts, LOST_S) or the
                tracker slower than RATE (a frame skipped, SKIPPED)

Points are given as the tracker's would be: in their palm frame, made
again from the hand given (wrist, knuckles and all wrong together).
"""

from __future__ import annotations

import numpy as np
import torch

from . import intent_pairs
from .handmotion import RATE, in_palm

NEIGHBOURS = 32
SCALE = (0.5, 2.0)            # the held-out error times this, per clip
SLOW_S = (0.2, 2.0)           # how long a misread lasts
FADE = 5                      # frames from one misread to the next
JITTER = (0.0, 0.3)           # a fresh error each frame, times this
GUESS = (0.5, 1.0)            # a hidden point's way to another hand's
LOST_HIDDEN = 0.5             # share of the 21 points hidden: the hand lost
LOST_S = (0.05, 1.0)          # a burst of no hand
LOST_EVERY_S = 5.0            # bursts come this often on average
SKIPPED = (0.0, 0.5)          # share of frames with no new reading


class TrackerNoise:
    def __init__(self, poses=None, errors=None):
        """WiLoR's errors on FreiHAND, or the true hands `poses` (P, 21, 3)
        and errors on them `errors` given (palm frames)."""
        if poses is not None:
            self.poses, self.errors = np.asarray(poses, np.float32), np.asarray(errors, np.float32)
            self.scale = np.ones(self.poses.shape[1])
            return
        training = intent_pairs.load("training")
        evaluation = intent_pairs.load("evaluation")
        true, error = _errors(training)
        _, held = _errors(evaluation, slice(0, None, 2))
        scale = np.sqrt((held ** 2).sum(-1).mean(0) / np.maximum((error ** 2).sum(-1).mean(0), 1e-12))
        self.poses = true.astype(np.float32)                                  # (P, 21, 3)
        self.errors = (error * scale[:, None]).astype(np.float32)
        self.scale = scale

    def neighbours(self, person, chunk=4096):
        """Each pose's NEIGHBOURS nearest FreiHAND training hands (T, NEIGHBOURS)."""
        bank = torch.as_tensor(self.poses.reshape(len(self.poses), -1))
        out = []
        for part in np.array_split(np.asarray(person, np.float32), max(1, len(person) // chunk)):
            d = torch.cdist(torch.as_tensor(part.reshape(len(part), -1)), bank)
            out.append(d.topk(NEIGHBOURS, largest=False).indices.numpy().astype(np.int32))
        return np.concatenate(out)

    def corrupt(self, person, near, hidden, rng, scale=None):
        """True hands (B, T, 21, 3) in palm frames, their neighbours (B, T,
        NEIGHBOURS), points hidden (B, T, 21) or None -> the tracker's
        hands (B, T, 21, 3), zero where `found` (B, T) is False."""
        person = np.asarray(person, np.float32)
        count, length = person.shape[:2]
        size = rng.uniform(*SCALE, count) if scale is None else np.full(count, float(scale))
        # A misread: one error WiLoR made on a hand posed as this one is when
        # it begins, kept for its span (the same view, the same mistake),
        # faded into the next over FADE frames; and a little fresh every
        # frame (jitter).
        span = rng.uniform(*SLOW_S, count)[:, None] * RATE                    # frames
        change = rng.uniform(size=(count, length)) < 1 / span
        change[:, 0] = True
        begins = np.maximum.accumulate(np.where(change, np.arange(length), 0), 1)
        error = _faded(self.errors[np.take_along_axis(self._pick(near, rng), begins, 1)], FADE)
        jitter = rng.uniform(*JITTER, count)[:, None, None, None] * self.errors[self._pick(near, rng)]
        seen = person + size[:, None, None, None] * (error + jitter)
        found = np.ones((count, length), bool)
        if hidden is not None:
            hidden = np.asarray(hidden, bool)
            guess = self.poses[rng.integers(len(self.poses), size=count)][:, None]
            way = rng.uniform(*GUESS, count)[:, None, None, None]
            seen = np.where(hidden[..., None], seen + way * (guess - seen), seen)
            found &= hidden.mean(-1) <= LOST_HIDDEN
        found &= rng.uniform(size=(count, length)) >= rng.uniform(*SKIPPED, count)[:, None]
        for b in range(count):
            for _ in range(rng.poisson(length / (LOST_EVERY_S * RATE))):
                start = rng.integers(length)
                found[b, start:start + max(1, round(rng.uniform(*LOST_S) * RATE))] = False
        seen = in_palm(seen.astype(float), "Left").astype(np.float32)
        return np.where(found[..., None, None], seen, 0), found

    def _pick(self, near, rng):
        """One of each frame's neighbours (B, T, NEIGHBOURS) -> (B, T)."""
        choice = rng.integers(near.shape[-1], size=(*near.shape[:-1], 1))
        return np.take_along_axis(near, choice, -1)[..., 0]


def _faded(error, frames):
    """A moving mean over `frames` along time (axis 1), the ends held."""
    pad = frames // 2
    padded = np.concatenate([np.repeat(error[:, :1], pad, 1), error, np.repeat(error[:, -1:], pad, 1)], 1)
    total = np.concatenate([np.zeros_like(padded[:, :1]), np.cumsum(padded, 1)], 1)
    return (total[:, frames:] - total[:, :-frames]) / frames


def _errors(pairs, rows=slice(None)):
    """True hands in palm frames (N, 21, 3) and the tracker's errors there,
    where it found the hand."""
    found = pairs["found"].astype(bool)
    found[np.setdiff1d(np.arange(len(found)), np.arange(len(found))[rows])] = False
    true = in_palm(pairs["person"][found].astype(float), intent_pairs.SIDE)
    seen = in_palm(pairs["seen"][found].astype(float), intent_pairs.SIDE)
    return true, seen - true
