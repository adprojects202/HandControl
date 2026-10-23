"""One step of the ClearML training pipeline, run by the agent (see
src/learning/remote.py). Its parameters ("handlab" section) say which:

    library   build the trajectory library from the recordings dataset;
              uploads it as the "library" artifact
    train     train by the learning.* parameters on a library (the
              library step's artifact, or a library dataset), test it, and
              upload the run folder as the "run" artifact

Not for running by hand: the pipeline sets the parameters.
"""

import json
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clearml import Dataset, Task  # noqa: E402

from src import settings  # noqa: E402
from src.learning import background, train, trajectories  # noqa: E402

PARAMETERS = {"step": "train", "recordings_dataset": "", "library_dataset": "",
              "library_task": "", "method": "ppo", "envs": 2048, "iterations": 1500,
              "teacher_rounds": 4}


def main():
    task = Task.init(project_name="HandRL", task_name="handlab step", reuse_last_task_id=False)
    params = task.connect(dict(PARAMETERS), name="handlab")
    _mirror_progress(task)
    try:
        if params["step"] == "library":
            library_step(task, params)
        else:
            train_step(task, params)
    except Exception as failure:
        background.finish(state="failed", error=str(failure) or type(failure).__name__)
        raise


def library_step(task, params):
    background.begin("library")
    recordings = Path(Dataset.get(dataset_id=params["recordings_dataset"]).get_local_copy())
    settings.set("recording.folder", str(recordings))           # this run only, not saved
    summary = trajectories.build(lambda done, total, text="": background.report(
        done=done, total=total, text=text))
    task.upload_artifact("library", artifact_object=str(trajectories.LIBRARY),
                         metadata=summary, wait_on_upload=True)
    background.finish(state="done", text="library built")


def train_step(task, params):
    background.begin("train")
    if params["library_task"]:
        path = Task.get_task(task_id=params["library_task"]).artifacts["library"].get_local_copy()
    else:
        folder = Path(Dataset.get(dataset_id=params["library_dataset"]).get_local_copy())
        path = folder / trajectories.LIBRARY.name
    library = trajectories.load(Path(path))
    for key in ("method", "envs", "iterations", "teacher_rounds"):
        settings.set(f"learning.{key}", params[key])                # this run only
    folder = train.train(library=library)
    background.report(text="testing on every motion")
    results = train.evaluate(folder, library=library)
    (folder / "evaluation.json").write_text(json.dumps(results, indent=2))
    archive = shutil.make_archive(str(folder), "zip", root_dir=folder.parent, base_dir=folder.name)
    task.upload_artifact("run", artifact_object=archive, wait_on_upload=True)
    background.finish(state="done", text="trained and tested")


def _mirror_progress(task):
    """The status file's progress, as the ClearML task's progress (percent)."""
    def mirror():
        while True:
            status = background.read() or {}
            if status.get("total"):
                task.set_progress(int(100 * status.get("done", 0) / status["total"]))
            time.sleep(10)
    threading.Thread(target=mirror, daemon=True).start()


if __name__ == "__main__":
    main()
