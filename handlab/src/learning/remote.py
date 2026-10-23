"""Training on the ClearML agent - the Windows PC with the RTX 3090 - as a
ClearML pipeline, started from this Mac:

    folder = run(progress)          # -> data/learning/runs/<stamp>/, fetched

    library  (agent)  the trajectory library, built from the recordings
                      (only with `learning.clearml_library` "build"; with
                      "mine", this Mac's library is uploaded instead)
    train    (agent)  training by learning.*, then the test; the run folder
                      comes back as an artifact

Each step is scripts/clearml_step.py from the GitHub repository at this
very commit, so the work must be committed and pushed first (`blocked`
says so). settings.json is not: it changes as the window is used, so this
Mac's current one goes along with each step as a ClearML configuration
("settings") - the agent trains with exactly these hand, object and motor
settings. The agent installs requirements-train.txt - MuJoCo, PyTorch,
rsl_rl, no camera stack. Data goes up as ClearML datasets: this Mac's
library, or each recording's meta.json, frames.npz, pose3d.npz and sim.npz.

The pipeline's controller runs here, in this process: a single agent runs
the steps, and a controller queued on it too would wait for itself. While
it runs, the running step's progress and its last error numbers go into
the status file, so the Learning panel shows them; the finished run is
unpacked into data/learning/runs, where the Runs list and Recordings step 7
find it like a run trained here.

What the agent's GPU gives: the policy's network trains on it, but the
physics - nearly all the time - runs on the agent's CPU (MuJoCo), as here.
A real speed-up needs the simulation on the GPU (MJX / MuJoCo Warp).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path

from .. import settings
from ..paths import ROOT
from ..processing.files import POSE3D, SIM
from . import trajectories
from .train import RUNS

PROJECT = "HandRL"
STEP_SCRIPT = "scripts/clearml_step.py"
REQUIREMENTS = "requirements-train.txt"
RECORDING_FILES = ("meta.json", "frames.npz", POSE3D, SIM)
POLL_S = 15


def blocked(where=ROOT):
    """Why the pipeline cannot run the code at `where`, or "": the agent
    clones the repository, so what it runs must be committed and pushed."""
    git = _git("status", "--porcelain", "--", ".", ":!settings.json", cwd=where)
    if git is None:
        return "not a git repository"
    if git.strip():
        return "uncommitted changes - commit and push first"
    if not _git("remote", "get-url", "origin", cwd=where):
        return "no GitHub remote (origin)"
    _git("fetch", "--quiet", "origin", cwd=where)
    ahead = _git("rev-list", "--count", "@{u}..HEAD", cwd=where)
    if ahead is None:
        return "the branch is not on GitHub - push it first"
    if ahead.strip() != "0":
        return "commits not pushed to GitHub - push first"
    return ""


def run(progress=lambda done, total, text="": None):
    """Run the pipeline to the end; returns the fetched run folder."""
    from clearml import PipelineController, Task      # here: only this path needs ClearML

    reason = blocked()
    if reason:
        raise RuntimeError(reason)
    config = settings.section("learning")
    queue = config["clearml_queue"]
    repo, commit = _git("remote", "get-url", "origin").strip(), _git("rev-parse", "HEAD").strip()
    root = _git("rev-parse", "--show-toplevel").strip()
    here = str(ROOT.relative_to(root))                 # handlab/, inside the repository
    progress(0, 1, "uploading data")
    build = config["clearml_library"] == "build"
    data = _upload_recordings() if build else _upload_library()

    def step_task(name):
        return Task.create(project_name=PROJECT, task_name=f"handlab {name}", repo=repo,
                           commit=commit, script=f"{here}/{STEP_SCRIPT}",
                           working_directory=here,
                           requirements_file=str(ROOT / REQUIREMENTS)).id

    training = {f"handlab/{key}": config[key]
                for key in ("method", "envs", "iterations", "teacher_rounds")}
    here_settings = {"settings": settings.current().path.read_text()}
    pipe = PipelineController(name="handlab tracker", project=PROJECT, add_run_number=True)
    if build:
        pipe.add_step("library", base_task_id=step_task("library"), execution_queue=queue,
                      configuration_overrides=here_settings,
                      parameter_override={"handlab/step": "library",
                                          "handlab/recordings_dataset": data})
        training["handlab/library_task"] = "${library.id}"
    else:
        training["handlab/library_dataset"] = data
    pipe.add_step("train", base_task_id=step_task("train"), execution_queue=queue,
                  parents=["library"] if build else None,
                  configuration_overrides=here_settings,
                  parameter_override={"handlab/step": "train", **training})
    finished = threading.Event()
    failure = []

    def control():
        try:
            pipe.start_locally(run_pipeline_steps_locally=False)
        except Exception as error:                    # reported below
            failure.append(error)
        finally:
            finished.set()

    threading.Thread(target=control, daemon=True).start()
    while not finished.wait(POLL_S):
        _mirror(pipe, progress)
    if failure or not pipe.is_successful():
        raise RuntimeError(f"the pipeline failed{': ' + str(failure[0]) if failure else ''}"
                           " - see it in ClearML (project HandRL)")
    return _fetch(Task.get_task(task_id=pipe.get_pipeline_dag()["train"].executed))


def _upload_library():
    """This Mac's library as a ClearML dataset; its id."""
    from clearml import Dataset

    if not trajectories.LIBRARY.exists():
        raise RuntimeError("no library - build it first, or set learning.clearml_library "
                           "to build")
    dataset = Dataset.create(dataset_project=PROJECT, dataset_name="handlab library")
    dataset.add_files(str(trajectories.LIBRARY))
    dataset.upload()
    dataset.finalize()
    return dataset.id


def _upload_recordings():
    """Every recording sent to sim, the files the library needs, as a
    ClearML dataset laid out as data/tasks; its id."""
    from clearml import Dataset

    from ..recording import episodes

    dataset = Dataset.create(dataset_project=PROJECT, dataset_name="handlab recordings")
    found = 0
    for folder in episodes():
        if not (folder / SIM).exists():
            continue
        for name in RECORDING_FILES:
            if (folder / name).exists():
                dataset.add_files(str(folder / name),
                                  dataset_path=f"{folder.parent.name}/{folder.name}")
        found += 1
    if not found:
        raise RuntimeError("no recording has been sent to sim")
    dataset.upload()
    dataset.finalize()
    return dataset.id


def _mirror(pipe, progress):
    """The running step's progress and last numbers, into `progress`."""
    from clearml import Task

    for name, node in pipe.get_pipeline_dag().items():
        job = getattr(node, "job", None)
        if node.executed or job is None:
            continue
        task = Task.get_task(task_id=job.task_id())
        status = task.get_status()
        if status == "queued":
            progress(0, 100, f"{name}: queued for the agent")
            continue
        scalars = task.get_last_scalar_metrics()
        turn = scalars.get("/track/turn_deg", {}).get("/track/turn_deg", {}).get("last")
        note = f", {turn:.1f} deg off while training" if turn is not None else ""
        progress(task.get_progress() or 0, 100, f"{name} on the agent: {status}{note}")


def _fetch(task):
    """The train step's run folder, unpacked into data/learning/runs."""
    target = unpack(Path(task.artifacts["run"].get_local_copy(extract_archive=False)))
    (target / "clearml.txt").write_text(f"trained on ClearML: task {task.id}\n"
                                        f"{task.get_output_log_web_page()}\n")
    return target


def unpack(archive, runs=RUNS):
    """A zipped run folder, unpacked into `runs`; the folder."""
    runs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as packed:
        packed.extractall(tmp)
        (run,) = [p for p in Path(tmp).iterdir() if p.is_dir()]
        target = runs / run.name
        shutil.move(str(run), str(target))
    return target


def _git(*args, cwd=ROOT):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

