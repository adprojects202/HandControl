"""The offline teacher: recorded sessions in, clean joint targets out.

WHY THERE IS A TEACHER. The learned retargeter has to answer in about two
milliseconds inside a control loop. This does not: it runs over a file, sees
the whole trajectory at once, and can be scored against the operator rather
than against itself. Everything the live solver gives up for latency is
available here.

WHAT IT IS NOW. A batched trajectory solve of the objective in `objective.py`,
warm-started from one pass of the live keyvector solver, then centred
smoothing. The old teacher - forward pass, backward pass, average, smooth,
re-solve, all through the live solver's own objective - is gone, and the
reason is one number: on 55k real frames it could not make a pinch. Not once.
Its keypoints were per-digit scaled and rest-rotated, and a pinch does not
survive either transform (21 mm raw, 136 mm as the solver saw it); it matched
the 136 faithfully. Fixing that needed a second view of the hand and a
different objective for the thumb, and `objective.py` says why each term is
the way it is.

WHAT IT MEASURED when this objective first replaced the old one - whole
recording, every block, against the live solver:

    fingers        6.90 -> 3.16 deg      better on every block
    flat hand      3.21 -> 0.08 deg
    contact made   0%   -> 82%           index 93, middle 58, ring 64
    gap at touch   135  -> 3.9 mm        the touch floor
    jerk           8.1e3 -> 3.5e3
    worst step     25.6 -> 13.8 deg

Those are the numbers that justified the change and they are kept as such.
The collision, clearance and object terms have moved them since; what
`learn.py teacher` prints TODAY is 2.69 deg and 95% contact, and the shipped
table lives in `learning/README.md` rather than in three docstrings that
each drifted to a different snapshot.

The live solver still runs here for two reasons. It is the warm start - the
batched solve converges from it in 600 steps, and from a neutral pose it does
not - and it is the `live-equivalent` row of the comparison, which is what
the student is ultimately judged against.
"""

from __future__ import annotations

import time
from argparse import Namespace

import numpy as np
from scipy.signal import savgol_filter

from ..robot.model import DISTAL_LINKS, TIP_LINKS
from . import collision, geometry, objective
from .kinematics import ChainFK
from .sessions import (fit_bone_offsets, fit_proportions, hands_from,
                       keypoints_for, raw_keypoints_for, runs_for)

SOLVER_ORDER = ("thumb", "index", "middle", "ring")

# Centred smoothing on the solved joints. 21 frames is 176 ms at the sensor's
# rate, and centred, so it costs no lag - the whole reason for doing this
# offline. The width was swept on a thumb-tap recording: from none to this,
# jerk fell elevenfold and the worst single-frame step from 15.9 to 5.3 deg,
# while the pose itself did not move.
SMOOTH_WINDOW = 21
SMOOTH_ORDER = 2

# What the live bridge runs at, for the comparison. Not a choice - it is what
# `KeyvectorBridge.start` passes, and changing it here would make the headroom
# figure describe a bridge nobody runs.
LIVE_ALPHA = 0.5


def solver(urdf, joint_names, alpha, maxeval=None):
    """The live keyvector solver, configured the way `Mimic.start_bridge`
    configures it. Used for the warm start and for the live-equivalent row."""
    # Imported here, not at module scope: `build` pulls in dex_retargeting,
    # Pinocchio and torch, and nothing else in this package needs them. A
    # session that only wants the noise model should not pay for the solver.
    from keyvector_bridge import build

    retargeting = build(Namespace(
        urdf=str(urdf), joints=",".join(joint_names),
        tips=",".join(TIP_LINKS[f] for f in SOLVER_ORDER),
        distals=",".join(DISTAL_LINKS[f] for f in SOLVER_ORDER),
        wrist="palm_base_link", objective="bones", posture=2,
        scaling=1.0, alpha=alpha, orientation=True, open_init=True))
    if maxeval:
        options = getattr(retargeting.optimizer, "opt", None)
        if options is not None:
            options.set_maxeval(int(maxeval))
            options.set_ftol_abs(1e-8)
    return retargeting


def pass_over(retargeting, points):
    """Solve a whole run with the live solver, warm-started frame to frame."""
    from keyvector_bridge import solve_frame

    out = np.empty((len(points), len(retargeting.optimizer.target_joint_names)))
    for index in range(len(points)):
        if index == 0:
            retargeting.last_qpos = np.zeros(out.shape[1], dtype=np.float32)
        out[index] = solve_frame(retargeting, points[index])
    return out


def smooth(q):
    """Centred smoothing, narrowed to fit a short run.

    Savitzky-Golay handles the ends by fitting a polynomial to them, so a
    21-frame window over a 43-frame run is mostly extrapolation and can leave
    the ends further out than it found them. `curl_fast` fragments into runs of
    forty frames when tracking drops, and those are exactly the frames where a
    wild end would matter most."""
    window = min(SMOOTH_WINDOW, len(q) if len(q) % 2 else len(q) - 1)
    if window <= SMOOTH_ORDER + 1:
        return q.copy()
    return savgol_filter(q, window, SMOOTH_ORDER, axis=0)


class Teacher:
    """Everything the objective needs about this robot, built once.

    The FK, the joint limits, the collision proxy and the finger-length
    ratio are properties of the hand and the operator, not of a run - so they
    are fitted here and every run is solved against the same ones.
    """

    def __init__(self, model, joint_names, digit_scale, device=None, hand="left"):
        import torch

        self.model = model
        self.joint_names = list(joint_names)
        self.hand = hand
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.fk = ChainFK(model, self.joint_names).to(self.device)
        self.lower, self.upper = model.limits(self.joint_names)
        # The hand's own links as capsules and boxes, offsets measured off
        # the meshes once and cached beside the model - see `collision`.
        self.proxy = collision.for_hand(self.fk, self.lower, self.upper, hand)
        self.fk.lateral = objective.lateral_axis(self.fk)
        self._census = None
        self.fk.reach_scale = objective.reach_scale(digit_scale)

    def teach(self, seed, aligned, raw, intent, steps=None, objects=None, present=None):
        """One run: batched solve from the warm start, then smoothed.
        `steps` overrides `objective.STEPS` - the synthetic set solves in
        fewer, see `cli.SYNTH_STEPS`. `objects` with a per-frame `present`
        mask puts something in the hand - see `objective.held`."""
        if objects is not None:
            objects = objects.to(self.device)
        q = smooth(objective.solve(self.fk, self.proxy, seed, aligned, raw,
                                   intent, self.lower, self.upper,
                                   steps=steps or objective.STEPS,
                                   objects=objects, present=present))
        if objects is not None and objective.POLISH_STEPS:
            # The smoothing pass can draw a jumping digit through the
            # object; this pulls it back out - see `objective.polish`.
            q = objective.polish(self.fk, self.proxy, q, aligned, raw, intent,
                                 self.lower, self.upper, objects, present)
        return q

    def census(self):
        """MuJoCo's contacts, for the penetration column of `measure`."""
        if self._census is None:
            self._census = collision.Census(self.joint_names, self.hand)
        return self._census

    def measure(self, name, aligned, raw, q, dt, bounds, samples=4000,
                objects=None, present=None):
        """Posture, contact, the gap at touch, jerk and worst step, on a stride.

        No aperture error over all frames: on an open hand the robot's
        thumb-index distance is genuinely 150 mm more than the operator's raw
        one - bigger hand, thumb splayed - and that is not something the
        objective should close. What a pinch is judged on is the gap left when
        the operator touches, and how often it closes at all."""
        stride = max(1, len(aligned) // samples)
        fingers, thumb, made, gap, _, posture = objective.score(
            self.fk, self.proxy, q[::stride], aligned[::stride], raw[::stride])
        # Penetration from MuJoCo's contacts, not from the proxy the solve
        # optimised - the number that cannot flatter the term.
        truth = self.census().survey(q[::stride], objective.PENETRATION_MM)
        against = {}
        if objects is not None:
            against = collision.Census(self.joint_names, self.hand, objects=objects).objects_survey(
                q[::stride], None if present is None else np.asarray(present)[::stride].any(axis=1))
        return {"name": name, **against,
                "gap_at_touch_mm": rounded(gap, 2),
                "posture_deg": round(fingers, 2),
                # The same number split by whether the direction target was
                # being asked for - see `objective.score`. A mapping that
                # closes contacts must score worse on the pooled one.
                "posture_free_deg": rounded(posture["free"], 2),
                "posture_contact_deg": rounded(posture["contact"], 2),
                "posture_untouched_deg": rounded(posture["untouched"], 2),
                "thumb_deg": round(thumb, 2),
                # Split the same way, and read instead of the pooled one: the
                # thumb is the digit the objective releases COMPLETELY, so
                # `thumb_deg` averages a targeted regime with an untargeted
                # one and moves with the share of touching frames.
                "thumb_free_deg": rounded(posture["thumb_free"], 2),
                "thumb_contact_deg": rounded(posture["thumb_contact"], 2),
                "contact_pct": rounded(100.0 * made["all"], 1),
                "contact_by_finger": {f: rounded(100.0 * v, 1)
                                      for f, v in made.items() if f != "all"},
                "penetration_pct": truth["penetration_pct"],
                "worst_penetration_mm": truth["worst_penetration_mm"],
                "penetration_pairs": truth["pairs"],
                "jerk": round(jerk(q, dt, bounds), 1),
                "worst_step_deg": round(worst_step(q, bounds), 2)}


def rounded(value, digits):
    """`value` rounded for a JSON report, or None where it is NaN.

    NaN is how an empty split reports itself - no contact frames, no free
    frames - and it is not valid JSON. Named rather than written `x == x` at
    each use: that test is how NaN is detected, and it reads to everyone else
    as a condition that is always true.
    """
    return None if np.isnan(value) else round(float(value), digits)


def jerk(q, dt, bounds):
    """Third derivative of the commanded angles, RMS over joints, rad/s^3.

    PER RUN, POOLED. Runs are concatenated in the array and minutes apart on
    the bench, so a difference taken across a seam is the hand teleporting from
    the end of one block to the start of the next. Measured that way the
    teacher scored WORSE than the mapping it improves on, entirely on seams it
    never solved across."""
    parts, weights = [], []
    for start, end in bounds:
        run = q[start:end]
        if len(run) < 4:
            continue
        parts.append(float(np.mean((np.diff(run, n=3, axis=0) / dt ** 3) ** 2)))
        weights.append(len(run))
    if not parts:
        return float("nan")
    return float(np.sqrt(np.average(parts, weights=weights)))


def worst_step(q, bounds):
    """Largest single-frame move of any joint, in degrees. Per run, same
    reason as `jerk`."""
    steps = [np.abs(np.diff(q[start:end], axis=0)).max()
             for start, end in bounds if end - start > 1]
    return float(np.degrees(max(steps))) if steps else float("nan")


def solve_session(data, meta, model, retarget, fingers, blocks=None, limit=0,
                  maxeval=2000, progress=None):
    """Solve one pooled session, run by run. Returns everything to save.

    RUN BY RUN, not session by session. The warm start, the acceleration
    term and the centred window are all temporal, and two blocks laid end to
    end are adjacent in the array and minutes apart on the bench.
    """
    joint_names = retarget.joint_names
    names = meta["blocks"]
    keep = ({names.index(k) for k in blocks} if blocks else None)
    runs = runs_for(data, minimum=SMOOTH_WINDOW * 2, blocks=keep)
    if limit:
        runs = [run[:limit] for run in runs]
    if not runs:
        return None
    tracked = np.concatenate(runs)
    dt = float(np.median(np.diff(data["t"])))

    # Proportions off the open-hand blocks, which is what they describe. A
    # fist's digit lengths are the same numbers measured through the tracker's
    # worst occlusion.
    open_blocks = [names.index(b) for b in ("flat_hold", "spread_hold") if b in names]
    open_rows = [i for i in tracked if data["block"][i] in open_blocks]
    sample = open_rows[::10] or list(tracked[::200])
    digit_scale = fit_proportions(retarget, [hands_from(data, i) for i in sample], fingers)

    # The rest fit uses the flat block only. Digit LENGTHS do not change when
    # a hand spreads, so fitting them across both open blocks is free accuracy;
    # a rest ORIENTATION does change, and averaging the fan into the definition
    # of straight put -18.6 degrees of adduction on the index.
    rest_blocks = [names.index(b) for b in ("flat_hold",) if b in names]
    rest_rows = [i for i in tracked if data["block"][i] in rest_blocks]
    bone_offsets = fit_bone_offsets(
        retarget, model, [hands_from(data, i) for i in (rest_rows[::10] or sample)], fingers)

    # Two views of every frame. Aligned carries the directions; raw carries
    # the distances. See `objective.py` for the measurement that split them.
    aligned = np.stack([keypoints_for(retarget, data, i, digit_scale, bone_offsets)
                        for i in tracked])
    raw = np.stack([raw_keypoints_for(retarget, data, i) for i in tracked])
    intent = data["pinch_strength"][tracked].astype(np.float32)

    live = solver(model.path, joint_names, alpha=LIVE_ALPHA)
    warm = solver(model.path, joint_names, alpha=1.0, maxeval=maxeval)
    teacher = Teacher(model, joint_names, digit_scale)

    q_live = np.empty((len(aligned), len(joint_names)))
    q_star = np.empty_like(q_live)
    bounds, at = [], 0
    live_seconds = teach_seconds = 0.0
    for number, run in enumerate(runs, start=1):
        span = slice(at, at + len(run))
        if progress:
            progress(number, len(runs), names[int(data["block"][run[0]])],
                     len(run), at, len(aligned))
        mark = time.perf_counter()
        q_live[span] = pass_over(live, aligned[span])
        live_seconds += time.perf_counter() - mark
        mark = time.perf_counter()
        seed = pass_over(warm, aligned[span])
        q_star[span] = teacher.teach(seed, aligned[span], raw[span], intent[span])
        teach_seconds += time.perf_counter() - mark
        bounds.append((at, at + len(run)))
        at += len(run)

    rows = [teacher.measure("live-equivalent", aligned, raw, q_live, dt, bounds),
            teacher.measure("teacher", aligned, raw, q_star, dt, bounds)]
    rows[0]["ms_per_frame"] = round(1000 * live_seconds / len(aligned), 2)
    rows[1]["ms_per_frame"] = round(1000 * teach_seconds / len(aligned), 2)

    distances = geometry.pairwise(raw)
    return {"points": aligned, "raw": raw, "intent": intent,
            "distances": distances,
            "time_to_contact": geometry.time_to_contact(
                distances, data["t"][tracked], bounds, objective.TOUCH_MM),
            "q_star": q_star, "q_live": q_live,
            "rows": tracked, "bounds": np.array(bounds, dtype=np.int64),
            "digit_scale": digit_scale, "bone_offsets": bone_offsets,
            "metrics": rows, "dt": dt,
            "joint_names": joint_names, "runs": len(runs)}
