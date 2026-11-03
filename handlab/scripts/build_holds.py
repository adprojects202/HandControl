"""GRAB's one-hand holds put onto the robot hand with their objects - what
the physics policy learns to keep hold of (src/learning/holds.py).

    python scripts/build_holds.py

Needs GRAB's robot motion and views (scripts/build_grab.py,
build_grab_views.py) and the reader (scripts/fit_reader.py). Writes
data/learning/intent/holds.npz. About 4 min. The Learning panel's "Build
holds" (learning holding objects) runs this; its progress goes to
data/learning/status.json either way.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import background, holds  # noqa: E402


def main():
    started = time.monotonic()
    background.begin("holds")

    def progress(done, total, text=""):
        print(f"[{time.monotonic() - started:6.0f} s] {done}/{total} {text}", flush=True)
        background.report(done=done, total=total, text=text)

    try:
        summary = holds.build(progress)
    except Exception as failure:
        background.finish(state="failed", error=str(failure) or type(failure).__name__)
        raise
    print(f"wrote {holds.HOLDS}:\n{summary}")
    background.finish(state="done", text="holds built")


if __name__ == "__main__":
    main()
