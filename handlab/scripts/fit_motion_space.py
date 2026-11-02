"""Fit the motion space - the robot hand's motions as points in a learnt
space, the intent space (src/learning/motion_space.py) - SEEDS times, each
measured on held-out clips, and print the measures.

    python scripts/fit_motion_space.py

Writes data/learning/motion_space.pt. About 25 min a seed on this Mac (GPU, contacts on the CPU).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import motion_space  # noqa: E402


def main():
    started = time.monotonic()

    def progress(done, total, text=""):
        print(f"[{time.monotonic() - started:6.0f} s] {done}/{total} {text}", flush=True)

    report = motion_space.fit(progress)
    print(f"wrote {motion_space.MODEL}")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
