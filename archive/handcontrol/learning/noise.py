"""What the Leap's error looks like, and how to reproduce it.

Nothing here knows about robots. Arrays in, parameters out.

The output that matters is `augmentation()` - not a report to read once, but
the parameters `augment.NoiseModel` injects into clean synthetic hands during
training (Phase 3). Every
number measured here ends up as an augmentation, so a number that is wrong here
trains a model to be robust to noise nobody has.

FOUR ERROR MODES, MEASURED SEPARATELY, because a single fitted gaussian
produces none of them:

  JITTER, speed-dependent. Measured at 0.042 mm on a still hand and 7.98 mm on
  a fast one - a factor of 190. A constant is wrong by two orders of magnitude
  whichever end it is fitted at.

  CORRELATED ERROR. After dividing out speed, keypoints on DIFFERENT fingers
  still correlate at 0.34-0.82: the tracker gets the whole hand slightly wrong
  at once. Independent per-keypoint gaussians average away down a chain and the
  measured error does not.

  SNAPS. Single-frame relocations when a finger is re-acquired - up to 98 mm
  here. Wrong samples rather than noise, and the class no filter fixes.

  FREEZES. The service repeating its last answer, reported as perfect tracking.
  Measured at zero in this session, which is one mode not to model.

WHY BONE LENGTH IS THE MEASURING STICK. Every other estimate of the error needs
to know what the hand was really doing, and nothing does. A bone length does
not: it is rigid, the operator cannot change it, so every millimetre it moves is
the tracker being wrong. That makes it the one estimate valid during MOTION,
which is where the error lives and where a hold-still capture says nothing.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, welch

from ..retargeting.filters import MAD_TO_SIGMA
from .sessions import tracked_runs

# 21 frames at ~119 Hz is 175 ms. Long enough that hand motion is smooth
# across it and short enough that it does not eat a real transition. Order 3
# so a curl's acceleration is followed rather than clipped.
RESIDUAL_WINDOW = 21
RESIDUAL_ORDER = 3

# A snap is judged against the frame-to-frame step distribution of the SAME
# recording, not an absolute millimetre figure: what counts as a jump depends
# on how fast the hand was going, and a threshold in millimetres would flag
# every fast block and no slow one.
SNAP_SIGMAS = 8.0


# Speed bins for the jitter fit, mm/s. Open at the top.
SPEED_BINS = (0.0, 5.0, 20.0, 50.0, 120.0, 300.0, 800.0, np.inf)

HOLD_BLOCKS = ("flat_hold", "fist_hold", "spread_hold")


def smooth(x, axis=0):
    """The signal, with everything above the hand's own bandwidth taken out.

    What is left over when this is subtracted is the part the operator cannot
    have produced. A hand cannot move meaningfully at 30 Hz; the sensor
    reports at 119."""
    if x.shape[axis] < RESIDUAL_WINDOW:
        return x.copy()
    return savgol_filter(x, RESIDUAL_WINDOW, RESIDUAL_ORDER, axis=axis)


def residuals(bones, dt):
    """Per keypoint: the high-frequency residual, and the speed it was moving.

    Returns (resid (n,5,5,3), speed (n,5,5)) - the speed is taken from the
    SMOOTHED track, so it is the hand's speed rather than the noise's."""
    flat = bones.reshape(len(bones), -1)
    base = smooth(flat, axis=0)
    resid = (flat - base).reshape(bones.shape)
    velocity = np.gradient(base, axis=0) / dt
    speed = np.linalg.norm(velocity.reshape(bones.shape), axis=-1)
    return resid, speed


def jitter_vs_speed(resid, speed):
    """sigma(speed). The headline result: noise is not a constant.

    Binned rather than regressed because the relationship is not a line - it is
    flat and tiny up to a walking pace and then climbs - and a line fitted
    through both regimes describes neither."""
    out = []
    magnitude = np.abs(resid)
    for low, high in zip(SPEED_BINS, SPEED_BINS[1:]):
        pick = (speed >= low) & (speed < high)
        if pick.sum() < 200:
            continue
        # sigma per axis from the pooled residuals of every keypoint in the bin
        sigma = float(np.sqrt(np.mean(magnitude[pick] ** 2)))
        out.append({"speed_lo": low, "speed_hi": None if np.isinf(high) else high,
                    "sigma_mm": round(sigma, 4),
                    "samples": int(pick.sum())})
    return out


def rigidity(bones, digits):
    """Bone-length error, per digit. Model-free, and valid during motion.

    The median over the block is taken as the true length - the operator has
    one - and the spread around it is the tracker. Reported as a robust sigma
    so a handful of snaps do not become the answer."""
    lengths = np.linalg.norm(np.diff(bones, axis=2), axis=3)      # (n,5,4)
    truth = np.median(lengths, axis=0)
    error = lengths - truth
    out = {}
    for slot, digit in enumerate(digits):
        # bone 0 of the thumb is zero-length padding in the SDK's layout
        bones_here = error[:, slot, 1:] if digit == "thumb" else error[:, slot]
        mad = np.median(np.abs(bones_here - np.median(bones_here)))
        out[digit] = {"sigma_mm": round(float(np.std(bones_here)), 4),
                      "robust_sigma_mm": round(float(MAD_TO_SIGMA * mad), 4),
                      "worst_mm": round(float(np.abs(bones_here).max()), 3),
                      "length_mm": [round(float(v), 2) for v in truth[slot]]}
    return out


def correlation(resid, speed):
    """Is a finger's error independent per keypoint, or does the chain move
    together?

    The answer decides the augmentation. Independent per-keypoint gaussians and
    a single per-finger offset look the same in a variance table and completely
    different to a model: one averages away down the chain, the other does
    not, and the one that does not is what bends a finger the wrong way.

    SPEED IS DIVIDED OUT FIRST, and that correction is the whole reason this
    function is not two lines. Residual magnitude rises with how fast the hand
    is moving, and the whole hand moves at once - so correlating raw magnitudes
    finds every keypoint correlated with every other, at 0.9, whatever the
    spatial structure is. Normalising each keypoint by the sigma its OWN speed
    predicts removes that common driver and leaves the coupling actually being
    asked about."""
    series = np.linalg.norm(resid, axis=-1)                       # (n,5,5)
    # sigma(speed) from this same data, as a smooth curve rather than the
    # reported bins - a step function would leave the bin edges in the result.
    order = np.argsort(speed, axis=None)
    flat_speed = speed.reshape(-1)[order]
    flat_series = series.reshape(-1)[order]
    window = max(64, len(flat_speed) // 200)
    edges = np.arange(0, len(flat_speed), window)
    knots = np.array([flat_speed[a:a + window].mean() for a in edges])
    sigmas = np.array([flat_series[a:a + window].std() + 1e-6 for a in edges])
    predicted = np.interp(speed, knots, sigmas)
    series = series / predicted

    within, across = [], []
    series = series - series.mean(axis=0, keepdims=True)
    unit = series / (np.linalg.norm(series, axis=0) + 1e-12)
    for a_digit in range(5):
        for a_joint in range(1, 5):
            for b_digit in range(5):
                for b_joint in range(1, 5):
                    if (a_digit, a_joint) >= (b_digit, b_joint):
                        continue
                    value = float(unit[:, a_digit, a_joint]
                                  @ unit[:, b_digit, b_joint])
                    (within if a_digit == b_digit else across).append(value)
    return {"within_finger": round(float(np.mean(within)), 4),
            "across_fingers": round(float(np.mean(across)), 4),
            "ratio": round(float(np.mean(within) / max(np.mean(across), 1e-6)), 2)}


def snaps(bones, dt, digits):
    """Single-frame relocations: how often, how big, how long.

    Counted on fingertips only. A snap is the tracker changing its mind about
    where a whole finger is, and the tip is where that shows up largest and
    where it matters - it is the point every keyvector is drawn from."""
    tips = bones[:, :, 4, :]                                      # (n,5,3)
    step = np.linalg.norm(np.diff(tips, axis=0), axis=-1)         # (n-1,5)
    seconds = len(step) * dt
    out = {}
    for slot, digit in enumerate(digits):
        series = step[:, slot]
        median = np.median(series)
        mad = MAD_TO_SIGMA * np.median(np.abs(series - median)) + 1e-9
        hit = series > median + SNAP_SIGMAS * mad
        # consecutive flagged frames are one event, not several
        events = int(np.sum(hit & ~np.concatenate([[False], hit[:-1]])))
        out[digit] = {
            "events": events,
            "per_second": round(events / max(seconds, 1e-6), 3),
            "median_step_mm": round(float(median), 4),
            "threshold_mm": round(float(median + SNAP_SIGMAS * mad), 3),
            "mean_snap_mm": round(float(series[hit].mean()), 3) if hit.any() else 0.0,
            "worst_mm": round(float(series.max()), 3),
        }
    return out


def freezes(bones, dt):
    """Consecutive frames the sensor reported identically.

    The service repeats its last answer when it has nothing new, and it does
    not say so: `LEAP_HAND.confidence` is documented as always 1.0. A frozen
    hand is indistinguishable from a still one in the data, and the only tell
    is bit-exact repetition."""
    tips = bones[:, :, 4, :].reshape(len(bones), -1)
    same = np.all(np.diff(tips, axis=0) == 0.0, axis=1)
    runs, run = [], 0
    for flag in same:
        if flag:
            run += 1
        elif run:
            runs.append(run + 1)
            run = 0
    if run:
        runs.append(run + 1)
    return {"frames_repeated": int(same.sum()),
            "fraction": round(float(same.mean()), 5),
            "events": len(runs),
            "longest_frames": int(max(runs)) if runs else 0,
            "longest_ms": round(max(runs) * dt * 1000, 1) if runs else 0.0}


def spectrum(resid, dt):
    """Where the residual's power sits. If it is white, a memoryless
    augmentation is honest; if it is low-frequency, the augmentation needs
    memory or the model learns to average noise that does not average."""
    series = np.linalg.norm(resid[:, 1, 4, :], axis=-1)      # index fingertip
    if len(series) < 256:
        return {}
    freq, power = welch(series - series.mean(), fs=1.0 / dt, nperseg=256)
    total = power.sum() + 1e-12
    return {"below_5hz": round(float(power[freq < 5].sum() / total), 3),
            "5_to_20hz": round(float(power[(freq >= 5) & (freq < 20)].sum() / total), 3),
            "above_20hz": round(float(power[freq >= 20].sum() / total), 3),
            "peak_hz": round(float(freq[np.argmax(power)]), 2)}


def occlusion_penalty(per_block, digits):
    """How much worse a hidden finger is than a visible one.

    Both are still hands held in front of the same sensor, so the only
    difference between the two numbers is that the fist puts the ring and
    middle chains behind the fingers in front. Whatever that ratio is, it is
    what the augmentation has to be able to reproduce - uniformly noising every
    keypoint trains for a hand the sensor does not see."""
    flat = per_block.get("flat_hold", {}).get("rigidity")
    fist = per_block.get("fist_hold", {}).get("rigidity")
    if not flat or not fist:
        return {}
    return {digit: round(fist[digit]["robust_sigma_mm"]
                         / max(flat[digit]["robust_sigma_mm"], 1e-6), 2)
            for digit in digits}


def augmentation(resid, speed, per_block, digits, dt):
    """The diagnostics above, turned into something a training loop can call.

    Everything above answers "what does the sensor do". This answers "what
    should be added to a clean synthetic hand", which is the only reason any of
    it was measured. Four terms, and each one is here because the measurement
    said the obvious single-gaussian version would be wrong:

    SIGMA(SPEED) as knots to interpolate, not one number. Measured across a
    factor of 190 between a still hand and a fast one, so a constant is wrong
    by two orders of magnitude whichever end it is fitted at.

    A VARIANCE SPLIT into global, per-finger and per-keypoint. Recovered from
    the two correlations: the across-finger figure is the share of variance
    that every keypoint holds in common, the within-finger figure adds the
    share a chain holds together, and what is left is independent. Sampling all
    of it independently would produce noise that averages away down a chain,
    and the measured noise does not.

    AN AR(1) COEFFICIENT, because the residual is not white - most of its power
    sits below 5 Hz. White augmentation is trivially removable by any temporal
    model, which would teach it that the noise it meets on the bench is
    removable too.

    SNAPS as a rate and a magnitude distribution, kept separate from the
    gaussian part. They are wrong samples rather than noise, and a model that
    has never seen one will follow it.
    """
    magnitude = np.linalg.norm(resid, axis=-1)

    # sigma(speed): the fitted curve, as knots rather than the report's bins
    order = np.argsort(speed, axis=None)
    ordered_speed = speed.reshape(-1)[order]
    ordered_mag = magnitude.reshape(-1)[order]
    window = max(256, len(ordered_speed) // 60)
    starts = np.arange(0, len(ordered_speed) - window, window)
    knots = [[round(float(ordered_speed[a:a + window].mean()), 2),
              round(float(ordered_mag[a:a + window].std()), 5)] for a in starts]

    # variance split, pooled over the blocks that have both correlations
    within = float(np.mean([b["correlation"]["within_finger"]
                            for b in per_block.values()]))
    across = float(np.mean([b["correlation"]["across_fingers"]
                            for b in per_block.values()]))
    across = min(max(across, 0.0), 0.99)
    within = min(max(within, across), 0.99)
    split = {"global": round(across, 4),
             "per_finger": round(within - across, 4),
             "per_keypoint": round(1.0 - within, 4)}

    # AR(1) on the speed-normalised residual, per component
    flat = resid.reshape(len(resid), -1)
    flat = flat - flat.mean(axis=0, keepdims=True)
    lag = np.mean([float(np.corrcoef(flat[:-1, k], flat[1:, k])[0, 1])
                   for k in range(0, flat.shape[1], 7)])
    lag = float(np.clip(lag, 0.0, 0.999))

    # snaps, split by what the block was for: a still hand and a moving one
    # have rates an order of magnitude apart and the augmentation should not
    # average them into one.
    def rate(phase):
        picked = [b for name, b in per_block.items()
                  if (name in HOLD_BLOCKS) == (phase == "hold")]
        if not picked:
            return 0.0
        return round(float(np.mean([b["snaps"][d]["per_second"]
                                    for b in picked for d in digits])), 3)

    worst = max((b["snaps"][d]["worst_mm"]
                 for b in per_block.values() for d in digits), default=0.0)
    steps = np.linalg.norm(np.diff(resid[:, :, 4, :], axis=0), axis=-1)
    return {
        "sigma_vs_speed_mm": {"speed_mm_s": [k[0] for k in knots],
                              "sigma_mm": [k[1] for k in knots]},
        "variance_split": split,
        "ar1": round(lag, 4),
        "hz": round(1.0 / dt, 1),
        "occlusion_multiplier": occlusion_penalty(per_block, digits),
        "snaps": {"per_second_still": rate("hold"),
                  "per_second_moving": rate("motion"),
                  "magnitude_mm": {
                      "p50": round(float(np.percentile(steps, 50)), 3),
                      "p90": round(float(np.percentile(steps, 90)), 3),
                      "p99": round(float(np.percentile(steps, 99)), 3),
                      "worst": round(float(worst), 2)}},
        "note": "sigma is per axis, millimetres, device frame. Interpolate "
                "sigma_vs_speed_mm on the smoothed speed of each keypoint, "
                "split the variance by variance_split, and colour each term "
                "with the ar1 coefficient at the stated rate.",
    }


def analyse(data, meta):
    digits = meta["digits"]
    names = meta["blocks"]
    block_id = data["block"]
    bones = data["bones"]
    time = data["t"]
    tracked = data["tracked"]

    per_block, pooled_resid, pooled_speed = {}, [], []
    for index, name in enumerate(names):
        pick = block_id == index
        if not pick.any():
            continue
        runs = tracked_runs(pick, tracked, data["session"])
        if not runs:
            continue
        dt = float(np.median(np.diff(time[pick])))
        resid = np.concatenate([residuals(bones[run], dt)[0] for run in runs])
        speed = np.concatenate([residuals(bones[run], dt)[1] for run in runs])
        rows = bones[np.concatenate(runs)]
        pooled_resid.append(resid)
        pooled_speed.append(speed)

        # snaps and freezes are per-run: both are defined by what the PREVIOUS
        # frame was, and across a gap there is no previous frame.
        snap_runs = [snaps(bones[run], dt, digits) for run in runs]
        freeze_runs = [freezes(bones[run], dt) for run in runs]
        seconds = sum(len(run) for run in runs) * dt
        snap = {d: {"events": sum(s[d]["events"] for s in snap_runs),
                    "per_second": round(sum(s[d]["events"] for s in snap_runs)
                                        / max(seconds, 1e-6), 3),
                    "median_step_mm": round(float(np.mean(
                        [s[d]["median_step_mm"] for s in snap_runs])), 4),
                    "worst_mm": round(max(s[d]["worst_mm"] for s in snap_runs), 3)}
                for d in digits}
        freeze = {"frames_repeated": sum(f["frames_repeated"] for f in freeze_runs),
                  "fraction": round(sum(f["frames_repeated"] for f in freeze_runs)
                                    / max(sum(len(r) for r in runs), 1), 5),
                  "events": sum(f["events"] for f in freeze_runs),
                  "longest_frames": max(f["longest_frames"] for f in freeze_runs),
                  "longest_ms": round(max(f["longest_ms"] for f in freeze_runs), 1)}

        per_block[name] = {
            "frames": int(pick.sum()),
            "tracked": int(sum(len(run) for run in runs)),
            "lost_pct": round(100.0 * (1 - sum(len(r) for r in runs) / pick.sum()), 2),
            "runs": len(runs),
            "hz": round(1.0 / dt, 1),
            "median_speed_mm_s": round(float(np.median(speed)), 2),
            "rigidity": rigidity(rows, digits),
            "snaps": snap,
            "freezes": freeze,
            "correlation": correlation(resid, speed),
            "spectrum": spectrum(resid, dt),
        }

    dt = float(np.median(np.diff(time)))
    resid = np.concatenate(pooled_resid)
    speed = np.concatenate(pooled_speed)

    still = {k: v for k, v in per_block.items() if k in HOLD_BLOCKS}
    rest_sigma = float(np.mean([
        v["rigidity"][d]["robust_sigma_mm"]
        for v in still.values() for d in digits])) if still else float("nan")

    return {
        "session": meta.get("started"),
        "operator": meta.get("operator"),
        "hand": meta.get("hand"),
        "frames": int(len(bones)),
        "hz": round(1.0 / dt, 1),
        "jitter_vs_speed": jitter_vs_speed(resid, speed),
        "rest_rigidity_sigma_mm": round(rest_sigma, 4),
        "occlusion_penalty": occlusion_penalty(per_block, digits),
        "augmentation": augmentation(resid, speed, per_block, digits, dt),
        "blocks": per_block,
    }
