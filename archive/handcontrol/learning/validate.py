"""Checking a synthetic sample against the recording it was fitted from.

TWO CHECKS, AND THE SECOND EXISTS BECAUSE THE FIRST CANNOT SEE WHAT IT MISSES.
`compare` measures shape - distances between fingertips - and a sample can pass
it while being rotated, mirrored or translated bodily off the hand, because a
rigid motion preserves every distance. That happened: 150,000 pairs were solved
for a hand held sideways, with bone lengths right, shape coverage at 98.5% and
the solver reporting ordinary residuals. It showed up on screen and nowhere
else.

`check_frame` is the answer - absolute placement against the recording, which
is the one thing a rigid motion does change.
"""

from __future__ import annotations

import numpy as np


PROBES = (("thumb-index gap", 4, 8), ("index-middle gap", 8, 12),
          ("index-ring spread", 8, 16), ("wrist-middle reach", 0, 12),
          ("thumb-ring gap", 4, 16))
# How far a synthetic fingertip's mean may sit from the recorded one before the
# two are not in the same frame. Generous - the sampler is deliberately flatter
# than the recording, so the means genuinely differ - but a wrong frame here is
# a rotation of the whole hand, which is centimetres, not millimetres.
FRAME_TOLERANCE = 0.05


def check_frame(real, synthetic):
    """Refuse to write pairs whose keypoints are not in the robot's axes.

    THE BUG THIS EXISTS FOR was invisible in every check that came before it.
    Bone lengths were right, the shape distributions covered the real ones, the
    solver converged and reported ordinary residuals - because a rotation
    changes none of those. It only showed up on screen, as a hand floating
    sideways, after 150,000 frames had been solved against it.

    So the test is on absolute placement, which is the one thing a rotation
    does change and the one thing the shape checks are all blind to by
    construction. On the KNUCKLES, not the tips: a set that closes on
    objects a fifth of the time moves its mean fingertip by design and its
    knuckles by nothing, while a rotation moves both alike.
    """
    problems = []
    for name, slot in (("thumb", 1), ("index", 5), ("middle", 9),
                       ("ring", 13)):
        gap = float(np.linalg.norm(real[:, slot].mean(0)
                                   - synthetic[:, slot].mean(0)))
        if gap > FRAME_TOLERANCE:
            problems.append(f"{name} knuckle {gap * 1000:.0f} mm")
    if problems:
        raise SystemExit(
            "  synthetic keypoints are not in the robot's axes - mean "
            "knuckle is far from the recorded one:\n    "
            + ", ".join(problems)
            + "\n  a rotation passes every shape check; see check_frame.")


def compare(real, synthetic):
    """Does the synthetic set COVER the real one? Not: does it match it.

    Coverage is the requirement, and it is the weaker one on purpose. The
    synthetic set is deliberately flatter than the real distribution - most of
    a recorded session is a hand at rest, and a student trained on that
    marginal would be best at the pose it needs least help with. What would be
    fatal is a synthetic set that never reaches a shape the sensor will show
    it.

    SO THE TEST IS ON THE SUPPORT, not quantile against quantile. Comparing the
    synthetic 95th percentile against the real 95th demands the synthetic be
    MORE spread than the real at every quantile, which is a different and much
    stronger claim than the one that matters - and it fails a sampler that
    reaches every real shape while simply spending less of its time at the
    extremes. What is reported instead is the fraction of real frames whose
    shape falls inside the synthetic range, which is the question a student
    actually cares about: will it ever be shown something it never trained on.
    """
    gap = lambda p, a, b: np.linalg.norm(p[:, a] - p[:, b], axis=1)  # noqa: E731
    print(f"\n  {'metric':<22}{'real p5':>9}{'p50':>8}{'p95':>8}  "
          f"|{'syn p5':>9}{'p50':>8}{'p95':>8}  | inside")
    inside = []
    for name, a, b in PROBES:
        r, s = gap(real, a, b), gap(synthetic, a, b)
        covered = float(np.mean((r >= s.min()) & (r <= s.max())))
        inside.append(covered)
        rq, sq = np.percentile(r, [5, 50, 95]), np.percentile(s, [5, 50, 95])
        print(f"  {name:<22}" + "".join(f"{x:9.1f}" for x in rq) + "  |"
              + "".join(f"{x:9.1f}" for x in sq)
              + f"  |{100 * covered:6.1f}%")
    worst = min(inside)
    print(f"\n  worst coverage {100 * worst:.1f}% of real frames. "
          + ("Good." if worst > 0.98 else
             "Under 98% - widen MARGIN or the sampler's ranges."))
    return worst


