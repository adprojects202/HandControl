"""Label FreiHAND's true hands with the robot poses they mean - the robot
side of the intent pairs (src/learning/intent_pairs.py); the camera side is
wilor/freihand.py's. Then, where that has run, how a tracker's own gaps read
the touches.

    python scripts/build_intent_pairs.py

Writes data/learning/intent/freihand_labels_<set>.npz. About 5 min.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import intent_pairs  # noqa: E402


def main():
    started = time.monotonic()
    shown = set()

    def progress(done, total, text=""):
        if done == total or done // 5000 not in shown:
            shown.add(done // 5000)
            print(f"[{time.monotonic() - started:6.0f} s] {text} {done}/{total}", flush=True)

    for name, summary in intent_pairs.label(progress).items():
        print(f"{name}: {summary}")
    for name in intent_pairs.SOURCES:
        try:
            pairs = intent_pairs.load(name)
        except (FileNotFoundError, RuntimeError) as missing:
            print(f"{name}: no tracker hands yet ({missing})")
            continue
        print(f"{name}: {intent_pairs.report(pairs['person'], pairs['touching'], pairs['seen'])}")


if __name__ == "__main__":
    main()
