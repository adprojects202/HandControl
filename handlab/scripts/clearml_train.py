"""Train the general tracker on the ClearML agent (the Windows PC with the
RTX 3090), as a ClearML pipeline, and fetch the run.

    python scripts/clearml_train.py

src/learning/remote.py has the pipeline. Needs this code committed and
pushed (the agent clones it from GitHub) and ~/clearml.conf. The finished
run lands in data/learning/runs/ like one trained here. The Learning
panel's "Train on ClearML" runs this; its progress goes to
data/learning/status.json either way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning import background, remote  # noqa: E402


def main():
    background.begin("clearml")
    try:
        folder = remote.run(lambda done, total, text="": background.report(
            done=done, total=total, text=text))
    except Exception as failure:
        background.finish(state="failed", error=str(failure) or type(failure).__name__)
        raise
    print(f"fetched {folder}")
    background.finish(state="done", text=f"trained on ClearML: {folder.name}")


if __name__ == "__main__":
    main()
