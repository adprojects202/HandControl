"""GRAB's hands put onto the robot hand (src/learning/grab.py), for the
motion space. Needs GRAB's hands first (in .venv-wilor):

    ../.venv-wilor/bin/python wilor/grab.py
    python scripts/build_grab.py

Writes data/learning/intent/grab_robot.npz. About 20 min on this Mac.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import grab  # noqa: E402


def main():
    started = time.monotonic()

    def progress(done, total, text=""):
        if done % 50 == 0 or done == total:
            print(f"[{time.monotonic() - started:6.0f} s] {done}/{total} {text}", flush=True)

    print(grab.label(progress))


if __name__ == "__main__":
    main()
