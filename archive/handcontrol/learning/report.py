"""Printing a noise fit, and plotting it.

Presentation only - separated from `noise.py` because that file is the
measurement and this one is how it is looked at, and a plot should never be
able to change a number.
"""

from __future__ import annotations

import numpy as np

from .noise import HOLD_BLOCKS

def report(fit, digits):
    print("\n" + "=" * 72)
    print(f"  LEAP NOISE  -  {fit['frames']} frames @ {fit['hz']} Hz  "
          f"({fit['operator'] or 'unnamed'}, {fit['hand']})")
    print("=" * 72)

    print("\n  JITTER vs SPEED       the headline: noise is not a constant")
    print(f"  {'hand speed mm/s':<20}{'sigma mm':>10}{'samples':>12}")
    for row in fit["jitter_vs_speed"]:
        span = (f"{row['speed_lo']:.0f}-{row['speed_hi']:.0f}"
                if row["speed_hi"] else f"{row['speed_lo']:.0f}+")
        print(f"  {span:<20}{row['sigma_mm']:>10.3f}{row['samples']:>12,}")

    print("\n  RIGIDITY ERROR        bone lengths cannot change; this is all tracker")
    print(f"  {'block':<16}" + "".join(f"{d:>9}" for d in digits)
          + f"{'lost%':>8}{'runs':>6}")
    for name, block in fit["blocks"].items():
        row = "".join(f"{block['rigidity'][d]['robust_sigma_mm']:9.3f}"
                      for d in digits)
        print(f"  {name:<16}{row}{block['lost_pct']:8.2f}{block['runs']:6d}")

    if fit["occlusion_penalty"]:
        print("\n  OCCLUSION PENALTY     fist vs flat, same hand, same stillness")
        print("  " + "  ".join(f"{d} x{v}" for d, v in
                               fit["occlusion_penalty"].items()))

    print("\n  SNAPS                 fingertip relocations per second")
    print(f"  {'block':<16}" + "".join(f"{d:>9}" for d in digits) + "   worst mm")
    for name, block in fit["blocks"].items():
        row = "".join(f"{block['snaps'][d]['per_second']:9.2f}" for d in digits)
        worst = max(block["snaps"][d]["worst_mm"] for d in digits)
        print(f"  {name:<16}{row}{worst:11.1f}")

    print("\n  CORRELATION           speed divided out; does the error move as one?")
    print(f"  {'block':<16}{'within':>9}{'across':>9}{'ratio':>8}")
    for name, block in fit["blocks"].items():
        c = block["correlation"]
        print(f"  {name:<16}{c['within_finger']:9.3f}{c['across_fingers']:9.3f}"
              f"{c['ratio']:8.2f}")

    print("\n  SPECTRUM              index fingertip residual, power fraction")
    print(f"  {'block':<16}{'<5Hz':>8}{'5-20Hz':>9}{'>20Hz':>8}{'peak':>8}")
    for name, block in fit["blocks"].items():
        s = block["spectrum"]
        if s:
            print(f"  {name:<16}{s['below_5hz']:8.2f}{s['5_to_20hz']:9.2f}"
                  f"{s['above_20hz']:8.2f}{s['peak_hz']:7.1f}Hz")

    frozen = [(n, b["freezes"]) for n, b in fit["blocks"].items()
              if b["freezes"]["events"]]
    print("\n  FREEZES               repeated frames the SDK reports as tracked")
    if not frozen:
        print("    none in any block")
    for name, f in frozen:
        print(f"    {name:<16}{f['events']:5d} events, "
              f"{100 * f['fraction']:5.2f}% of frames, "
              f"longest {f['longest_ms']:.0f} ms")

    a = fit["augmentation"]
    print("\n  AUGMENTATION          what `augment.NoiseModel` injects")
    print(f"    sigma(speed)        {a['sigma_vs_speed_mm']['sigma_mm'][0]:.3f} mm "
          f"at rest -> {a['sigma_vs_speed_mm']['sigma_mm'][-1]:.2f} mm at "
          f"{a['sigma_vs_speed_mm']['speed_mm_s'][-1]:.0f} mm/s")
    split = a["variance_split"]
    print(f"    variance split      global {split['global']:.3f}  "
          f"per-finger {split['per_finger']:.3f}  "
          f"per-keypoint {split['per_keypoint']:.3f}")
    print(f"    colour              AR(1) {a['ar1']:.3f} at {a['hz']} Hz")
    print(f"    snaps               {a['snaps']['per_second_still']}/s still, "
          f"{a['snaps']['per_second_moving']}/s moving, "
          f"worst {a['snaps']['magnitude_mm']['worst']} mm")
    print()


def plot(fit, data, meta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    digits = meta["digits"]
    names = list(fit["blocks"])
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0][0]
    rows = fit["jitter_vs_speed"]
    centres = [r["speed_lo"] if r["speed_hi"] is None
               else 0.5 * (r["speed_lo"] + r["speed_hi"]) for r in rows]
    ax.plot(centres, [r["sigma_mm"] for r in rows], "o-", color="crimson")
    ax.set_xscale("symlog"); ax.set_yscale("log")
    ax.set_xlabel("hand speed (mm/s)"); ax.set_ylabel("residual sigma (mm)")
    ax.set_title("jitter is a function of speed, not a constant")
    ax.grid(alpha=0.3)

    width = 0.8 / len(digits)
    for ax, key, label, title in (
            (axes[0][1], "rigidity", "bone-length sigma (mm)",
             "rigidity error - all of it is the tracker"),
            (axes[1][0], "snaps", "snaps / second", "fingertip relocations")):
        field = "robust_sigma_mm" if key == "rigidity" else "per_second"
        for k, digit in enumerate(digits):
            ax.bar(np.arange(len(names)) + k * width,
                   [fit["blocks"][n][key][digit][field] for n in names],
                   width, label=digit)
        ax.set_xticks(np.arange(len(names)) + 0.4)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    for name in HOLD_BLOCKS:
        if name not in fit["blocks"]:
            continue
        pick = (data["block"] == meta["blocks"].index(name)) & data["tracked"]
        tips = data["bones"][pick][:, 1, 4, :]
        ax.plot(np.linalg.norm(tips - tips.mean(axis=0), axis=-1), lw=0.6,
                label=name)
    ax.set_xlabel("frame"); ax.set_ylabel("index tip deviation (mm)")
    ax.set_title("a 'still' hand, as the sensor sees it")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    print(f"  plot written to {path}")


