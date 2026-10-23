"""Train the general tracker on this Mac, then test it.

    python scripts/train_tracker.py

Needs data/learning/library.npz (Learning panel > Build library). Trains by
`learning.method` for `learning.iterations` rounds over `learning.envs`
copies of the task sim, writing data/learning/runs/<stamp>/: run.json,
checkpoints, progress.jsonl, a TensorBoard log, and evaluation.json - each
motion's result. The recorded motions were never trained on: they are the
one-shot test. The Learning panel's "Train" runs this; its progress goes to
data/learning/status.json either way.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.learning import background, train  # noqa: E402


def main():
    background.begin("train")
    try:
        folder = train.train()
        background.report(text="testing on every motion")
        results = train.evaluate(folder)
        (folder / "evaluation.json").write_text(json.dumps(results, indent=2))
    except Exception as failure:
        background.finish(state="failed", error=str(failure) or type(failure).__name__)
        raise
    report(results)
    print(f"\n{folder}")
    background.finish(state="done", text="trained and tested")


def report(results):
    for recorded, title in ((True, "recorded motions - never trained on (one-shot)"),
                            (False, "generated motions - trained on")):
        rows = [r for r in results if r["recorded"] == recorded]
        if not rows:
            continue
        print(f"\n{title}: {len(rows)}")
        print("  dropped   policy mm / deg    planner mm / deg    held still deg")
        for r in rows if recorded else []:
            print(f"  {str(r['dropped']):7s}   {r['position_mm']:5.1f} / {r['turn_deg']:4.1f}"
                  f"      {r['planner_position_mm']:5.1f} / {r['planner_turn_deg']:4.1f}"
                  f"        {r['still_turn_deg']:4.1f}   {Path(r['source']).parent.name}")
        held = [r for r in rows if not r["dropped"]]
        print(f"  held {len(held)}/{len(rows)}; median policy "
              f"{np.median([r['position_mm'] for r in rows]):.1f} mm / "
              f"{np.median([r['turn_deg'] for r in rows]):.1f} deg, planner "
              f"{np.median([r['planner_position_mm'] for r in rows]):.1f} mm / "
              f"{np.median([r['planner_turn_deg'] for r in rows]):.1f} deg, held still "
              f"{np.median([r['still_turn_deg'] for r in rows]):.1f} deg")


if __name__ == "__main__":
    main()
