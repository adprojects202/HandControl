#!/usr/bin/env python3
"""Every trained student on one table, with the seed spread beside the deltas.

    .venv/bin/python compare_runs.py

WHY THIS EXISTS. Training was unseeded until recently, so every row of the
comparison tables in the READMEs was one draw. Two seeds of one setting land
2.66 degrees apart on held-out fingers, which is larger than most of the
differences those tables are being read for. This prints the spread WITH the
numbers so a difference smaller than it cannot be mistaken for a result.

Posture is split by contact state - see `objective.score`. The pooled number
penalises any mapping that closes contacts, because the objective releases the
direction target during a touch; `free` is the number that is actually being
asked for, and `vs teacher` is the one with no metric argument in it at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from handcontrol.learning import objective, train as T          # noqa: E402
from handcontrol.learning.replay import student_rollout          # noqa: E402
from handcontrol.learning.sessions import sessions_under         # noqa: E402
from handcontrol.learning.teacher import Teacher                 # noqa: E402
from handcontrol.robot.model import HandModel, ensure_urdf       # noqa: E402

# label -> checkpoint, grouped so the spread within a group is visible
RUNS = [
    ("baseline", [("seed 0", "student_seed0.pt"), ("seed 1", "student_seed1.pt"),
                  ("seed 2", "student_seed2.pt")]),
    ("no objects in labels", [("seed 0", "noobj_seed0.pt"), ("seed 1", "noobj_seed1.pt"),
                              ("seed 2", "noobj_seed2.pt")]),
    ("direct objective", [("w 0.01", "direct_0.01.pt"), ("w 0.05", "direct_0.05.pt"),
                          ("w 0.20", "direct_0.20.pt")]),
    ("belief channels", [("seed 0", "belief_seed0.pt"), ("seed 1", "belief_seed1.pt"),
                         ("seed 2", "belief_seed2.pt")]),
    ("belief + contact gate", [("seed 0", "gated_seed0.pt"),
                               ("seed 1", "gated_seed1.pt"),
                               ("seed 2", "gated_seed2.pt")]),
    ("direct 0.01, 3 seeds", [("seed 0", "direct_0.01.pt"),
                              ("seed 1", "direct001_seed1.pt"),
                              ("seed 2", "direct001_seed2.pt")]),
]


def held_out_runs(labels, blocks):
    """The runs whose last frame is in a held-out block, long enough to score."""
    wanted = {blocks.index(b) for b in T.HELD_OUT if b in blocks}
    return [(a, b) for a, b in labels["run_bounds"]
            if int(labels["block"][a]) in wanted and b - a > T.WINDOW + 40]


def score_one(path, labels, rows, teacher):
    """(posture dict, joint error vs the teacher, contact %) for one checkpoint."""
    q, valid, stored, _ = student_rollout(str(path), labels["keypoints"],
                                          labels["keypoints_raw"], np.array(rows),
                                          device=torch.device("cpu"))
    if stored != list(teacher.joint_names):
        return None
    keep = np.flatnonzero(valid)
    _, _, made, _, _, posture = objective.score(
        teacher.fk, teacher.proxy, q[keep],
        labels["keypoints"][keep], labels["keypoints_raw"][keep])
    against = float(np.degrees(np.abs(q[keep] - labels["q_star"][keep])).mean())
    return posture, against, 100.0 * made["all"]


def main():
    session = sorted(sessions_under())[-1]
    labels = dict(np.load(session / "teacher.npz"))
    blocks = json.loads((session / "meta.json").read_text())["blocks"]
    names = [str(n) for n in labels["joint_names"]]
    scale = dict(zip((str(f) for f in labels["fingers"]), labels["digit_scale"]))
    teacher = Teacher(HandModel.load(ensure_urdf("left", None)), names, scale, device="cpu")
    rows = held_out_runs(labels, blocks)

    print(f"\n  held out: {', '.join(T.HELD_OUT)} — {len(rows)} runs\n")
    print(f"  {'run':28s}{'pooled':>9}{'free':>8}{'contact':>9}"
          f"{'untouched':>11}{'vs teacher':>12}{'contact':>9}")

    models = Path("data/models")
    for group, members in RUNS:
        found = []
        for label, name in members:
            path = models / name
            if not path.exists():
                continue
            out = score_one(path, labels, rows, teacher)
            if out is None:
                print(f"  {group + ' ' + label:28s}  (different joint order — skipped)")
                continue
            posture, against, contact = out
            found.append((label, posture, against, contact))
            print(f"  {group + ' ' + label:28s}{posture['all']:8.2f}°{posture['free']:7.2f}°"
                  f"{posture['contact']:8.2f}°{posture['untouched']:10.2f}°"
                  f"{against:11.2f}°{contact:8.1f}%")
        if len(found) > 1:
            pooled = np.array([f[1]["all"] for f in found])
            teach = np.array([f[2] for f in found])
            print(f"  {'  └ mean ± sd':28s}{pooled.mean():8.2f}°"
                  f"{'':26s}{teach.mean():11.2f}°")
            print(f"  {'  └ spread (max − min)':28s}{np.ptp(pooled):8.2f}°"
                  f"{'':26s}{np.ptp(teach):11.2f}°"
                  f"   ± {pooled.std(ddof=1):.2f} / ± {teach.std(ddof=1):.2f}")
        if found:
            print()

    print("  A difference smaller than a group's own spread is not a result.")
    print("  `vs teacher` scatters ~3x less than `pooled`, so it is the column")
    print("  with the power to separate two settings - and the one with no")
    print("  argument about what the metric means.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
