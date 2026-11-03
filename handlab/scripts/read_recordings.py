"""The reader on your recordings: every recording with WiLoR's hands
(../.venv-wilor/bin/python wilor/recording.py) read as live by the reader and
by the solver - the numbers printed, and a video of each into renders/:
the camera, the robot hand as the reader moves it, as the solver does
(src/learning/read_recordings.py).

    python scripts/read_recordings.py

Under a minute.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import read_recordings  # noqa: E402
from src.paths import ROOT  # noqa: E402


def main():
    out = ROOT / "renders"
    out.mkdir(exist_ok=True)
    for hands in sorted((ROOT / "data" / "tasks").glob(f"*/*/{read_recordings.HANDS}")):
        folder = hands.parent
        name = f"{folder.parent.name}-{folder.name}"
        results = read_recordings.read(folder)
        read_recordings.render(folder, out / f"{name}-reader.mp4")
        print(name, json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
