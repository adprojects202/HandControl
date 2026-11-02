"""The reader (src/learning/reader.py): the tracker's hands -> points in the
motion space - SEEDS times, each measured on clips it never saw, against
the labelling solver given the same hands. Needs the motion space, GRAB
(scripts/build_grab.py, scripts/build_grab_views.py) and FreiHAND's intent
pairs (scripts/build_intent_pairs.py):

    python scripts/fit_reader.py

Writes data/learning/reader.pt. About 20 min a seed on this Mac (GPU), and
a few minutes for the solver.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import reader  # noqa: E402


def main():
    started = time.monotonic()

    def progress(done, total, text=""):
        print(f"[{time.monotonic() - started:6.0f} s] {done}/{total} {text}", flush=True)

    report = reader.fit(progress)
    print(f"wrote {reader.MODEL}")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
