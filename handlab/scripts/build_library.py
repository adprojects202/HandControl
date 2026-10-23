"""Build the tracker's trajectory library from the recordings.

    python scripts/build_library.py

Writes data/learning/library.npz (src/learning/trajectories.py): starts the
hand can grip, generated and recorded motions, each tried by the planner.
About 30 minutes on all cores. The Learning panel's "Build library" runs
this; its progress goes to data/learning/status.json either way.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import background, trajectories  # noqa: E402


def main():
    started = time.monotonic()
    background.begin("library")

    def progress(done, total, text=""):
        print(f"[{time.monotonic() - started:6.0f} s] {done}/{total} {text}", flush=True)
        background.report(done=done, total=total, text=text)

    try:
        summary = trajectories.build(progress)
    except Exception as failure:
        background.finish(state="failed", error=str(failure) or type(failure).__name__)
        raise
    print(f"wrote {trajectories.LIBRARY}: {summary}")
    background.finish(state="done", text="library built")


if __name__ == "__main__":
    main()
