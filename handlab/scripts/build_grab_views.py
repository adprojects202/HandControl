"""Where a camera could see GRAB's hands from, and which of their points the
object or the table hides there (src/learning/grab.py views()) - what the
reader learns a tracker loses. Needs GRAB's hands and meshes first (in
.venv-wilor):

    ../.venv-wilor/bin/python wilor/grab.py
    python scripts/build_grab_views.py

Writes data/learning/intent/grab_views.npz. Under a minute on this Mac.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import grab  # noqa: E402


def main():
    started = time.monotonic()

    def progress(done, total, text=""):
        if done % 200 == 0 or done == total:
            print(f"[{time.monotonic() - started:6.0f} s] {done}/{total} {text}", flush=True)

    print(grab.views(progress))


if __name__ == "__main__":
    main()
