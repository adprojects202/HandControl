"""Build the hand-motion library: the robot hand's motions - random, the
thumb touching fingertips, hand signs, fingers in turn - and every recorded
hand, as shapes to follow.

    python scripts/build_hand_library.py

Writes data/learning/hand_library.npz (src/learning/hand_library.py). About
4 min. The Learning panel's "Build library" (learning hand motion) runs
this; its progress goes to data/learning/status.json either way.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import background, hand_library  # noqa: E402


def main():
    started = time.monotonic()
    background.begin("hand_library")

    def progress(done, total, text=""):
        print(f"[{time.monotonic() - started:6.0f} s] {done}/{total} {text}", flush=True)
        background.report(done=done, total=total, text=text)

    try:
        summary = hand_library.build(progress)
    except Exception as failure:
        background.finish(state="failed", error=str(failure) or type(failure).__name__)
        raise
    print(f"wrote {hand_library.LIBRARY}: {summary}")
    background.finish(state="done", text="hand library built")


if __name__ == "__main__":
    main()
