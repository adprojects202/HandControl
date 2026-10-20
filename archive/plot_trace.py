#!/usr/bin/env python
"""Plot a trace CSV: how well the robot hand followed the human one.

    python plot_trace.py logs/trace-20260901-143022.csv
    python plot_trace.py logs/trace-*.csv --save comparison.png

Four panels, chosen because between them they answer the only questions worth
asking of a retargeting:

  PINCH APERTURE - thumb-to-fingertip distance for all three fingers, both
  hands, over time. The most useful panel in the file, and the one that judges
  a retargeting: a mapping that tracks has each pair rise and fall together,
  and touches zero when you touch. A mapping that saturates has the robot's
  curve flatten while yours keeps going.

  Computed from the hum_ and rob_ columns rather than read from pinch_human /
  pinch_robot, which only carry the index pair - so touching the thumb to each
  finger in turn is readable without re-recording.

  FINGERTIP ERROR - per finger, the distance between where your fingertip was
  mapped to and where the robot's actually got. Separated by finger because
  the thumb is a different problem from the fingers and averaging them hides
  exactly that.

  Read this one with care under keyvector. That mapping does not try to put
  tips at absolute points - it matches the vectors between them - so a large
  residual here is not necessarily a failure, and the pinch panel above is
  the one that judges it. What this panel is good for is spotting a mapping
  that has gone somewhere silly, and for watching the thumb, whose error
  falling is the clearest sign the workspace offset is being absorbed rather
  than fought.

  ACCELERATION - the second difference of the commanded joint angles, which
  is what the axis plots. This is what makes a hand look nervous rather than
  inaccurate, and it is the thing the filtering is for. Not the same question
  as accuracy, and worth its own axis.

  SOLVE TIME - only under keyvector. A solver quietly hitting its iteration
  cap looks exactly like a bad mapping until you plot this.

Where the file spans a change of mapping, the background is banded by mode -
that is what makes one recording an A/B rather than two recordings you have to
trust were comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring")
BAND = {"raw": "#7C8492", "angles": "#5B9DFF", "tip": "#FFB86B",
        "keyvector": "#7EE787", "posebank": "#FF7B9C"}


def load(path):
    import csv
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def column(rows, name, default=np.nan):
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        value = row.get(name, "")
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            out[i] = default
    return out


def vector(rows, prefix):
    return np.stack([column(rows, f"{prefix}_{a}") for a in "xyz"], axis=1)


def mode_spans(rows):
    """(start_index, end_index, mode) for each contiguous run of one mapping."""
    spans, start = [], 0
    modes = [row.get("mode", "") for row in rows]
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            spans.append((start, i - 1, modes[start]))
            start = i
    return spans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="\n".join(__doc__.splitlines()[2:6]))
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--save", type=Path, help="write a PNG instead of showing")
    parser.add_argument("--joint", default="index_proximal_joint",
                        help="which joint's acceleration to plot")
    args = parser.parse_args(argv)

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, len(args.csv), figsize=(7.5 * len(args.csv), 11),
                                squeeze=False, sharex="col")
    for c, path in enumerate(args.csv):
        rows = load(path)
        t = column(rows, "t")
        tracked = column(rows, "tracked") > 0.5

        # every panel gets the mode bands, so a switch is visible everywhere
        for axis in axes[:, c]:
            for start, end, mode in mode_spans(rows):
                if mode in BAND and end > start:
                    axis.axvspan(t[start], t[end], color=BAND[mode], alpha=.10,
                                 lw=0)

        # thumb-to-each-finger, both hands. Dashed is you, solid is the robot,
        # one colour per finger - so a pair that tracks is two lines of the
        # same colour lying on top of each other.
        pairs, ax = {}, axes[0, c]
        h_thumb, r_thumb = vector(rows, "hum_thumb"), vector(rows, "rob_thumb")
        for finger, colour in zip(("index", "middle", "ring"),
                                  ("#1F5FD0", "#1E7F42", "#B26A00")):
            h = np.linalg.norm(h_thumb - vector(rows, f"hum_{finger}"), axis=1) * 1000
            r = np.linalg.norm(r_thumb - vector(rows, f"rob_{finger}"), axis=1) * 1000
            h = np.where(tracked, h, np.nan)
            pairs[finger] = (h, r)
            ax.plot(t, h, lw=1.5, ls="--", color=colour, alpha=.75)
            ax.plot(t, r, lw=1.4, color=colour, label=f"thumb-{finger}")
        ax.axhline(0, color="#999", lw=.6)
        ax.set_ylabel("thumb to fingertip (mm)")
        ax.set_title(f"{path.name}\n"
                     + "  ".join(f"{m}: {t[e]-t[s]:.0f}s"
                                 for s, e, m in mode_spans(rows) if m in BAND))
        ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False,
                  title="dashed = your hand", title_fontsize=7)

        ax = axes[1, c]
        for finger in FINGERS:
            want, got = vector(rows, f"hum_{finger}"), vector(rows, f"rob_{finger}")
            error = np.linalg.norm(want - got, axis=1) * 1000
            ax.plot(t, np.where(tracked, error, np.nan), lw=1.1, label=finger)
        ax.set_ylabel("fingertip error (mm)")
        ax.legend(loc="upper right", fontsize=8, ncol=4, frameon=False)

        ax = axes[2, c]
        q = column(rows, f"cmd_{args.joint}")
        accel = np.abs(np.gradient(np.gradient(q, t), t))
        ax.plot(t, np.degrees(accel), lw=.9, color="#1F5FD0")
        ax.set_ylabel(f"|accel| deg/s²\n{args.joint}")
        ax.set_yscale("symlog")

        ax = axes[3, c]
        ms = column(rows, "solve_ms")
        if np.isfinite(ms).any():
            ax.plot(t, ms, lw=.9, color="#1E7F42")
            ax.set_ylabel("solver (ms)")
        else:
            ax.text(.5, .5, "no keyvector frames in this recording",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888")
            ax.set_yticks([])
        ax.set_xlabel("seconds")

        # the numbers worth quoting, per mapping
        print(f"\n{path.name}")
        for start, end, mode in mode_spans(rows):
            if mode not in BAND or end - start < 10:
                continue
            piece = slice(start, end + 1)
            live = tracked[piece]
            if not live.any():
                continue
            errs = {f: np.nanmedian(np.linalg.norm(
                        vector(rows, f"hum_{f}")[piece][live]
                        - vector(rows, f"rob_{f}")[piece][live], axis=1)) * 1000
                    for f in FINGERS}
            gaps = {f: np.nanmedian(np.abs(h[piece][live] - r[piece][live]))
                    for f, (h, r) in pairs.items()}
            closest = {f: (np.nanmin(h[piece][live]), np.nanmin(r[piece][live]))
                       for f, (h, r) in pairs.items()}
            print(f"  {mode:10s} {t[end]-t[start]:5.1f}s")
            print("    pinch error   " + "  ".join(
                  f"thumb-{f} {g:5.1f} mm" for f, g in gaps.items()))
            print("    closest touch " + "  ".join(
                  f"{f} you {a:5.1f} robot {b:5.1f}"
                  for f, (a, b) in closest.items()))
            print("    tip error     " + "  ".join(
                  f"{f[:3]} {e:5.1f} mm" for f, e in errs.items()))

    figure.tight_layout()
    if args.save:
        figure.savefig(args.save, dpi=130)
        print(f"\nwrote {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
