"""Training on the ClearML agent's GPU - the Windows PC with the RTX 3090 - as
a ClearML pipeline, started from this Mac:

    folder = run(progress)          # -> data/learning/runs/<stamp>/, fetched

    train_gpu  (agent)  PPO over `learning.gpu_envs` task sims on the GPU
                        (src/learning/gpu.py: MuJoCo Warp + PyTorch CUDA),
                        `learning.iterations` rounds, then the test; the run
                        folder comes back as an artifact

What it trains on goes up as a ClearML dataset - the bundle
(`export_bundle`): this Mac's library, the task sim as MJCF (task.xml, the
model this Mac compiles, with this Mac's settings) and the hand mount. The
step is scripts/clearml_gpu_step.py from the GitHub repository at this very
commit, so the code must be committed and pushed first (`blocked` says so;
settings.json need not be - it is in the bundle). The agent installs
requirements-gpu.txt: MuJoCo Warp, CUDA PyTorch, rsl_rl.

The pipeline's controller runs here, in this process (one agent runs the
step; a controller queued on it would wait for itself). While it runs, the
step's progress and last error numbers go into the status file, so the
Learning panel shows them; the finished run is unpacked into
data/learning/runs, where Runs and Recordings step 7 take it like a run
trained here. Cancel (SIGTERM) stops the pipeline on ClearML too.
"""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import tempfile
import threading
import zipfile
from datetime import datetime
from pathlib import Path

import mujoco

from .. import settings
from ..paths import ROOT
from . import trajectories
from .train import RUNS

PROJECT = "HandRL"
STEP_SCRIPT = "scripts/clearml_gpu_step.py"
REQUIREMENTS = "requirements-gpu.txt"
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


def export_bundle(folder):
    """What the GPU trains on, into `folder`: library.npz, task.xml (the
    task sim as MJCF) and bundle.json (the hand mount, the fingertips)."""
    from ..sim.task import Task

    library = trajectories.load()
    if not library:
        raise RuntimeError("no library - build it first")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(trajectories.LIBRARY, folder / trajectories.LIBRARY.name)
    task = Task.build((library["start_position"][0], library["start_quat"][0]))
    # Attaching the robot leaves an empty default class in the export, which
    # newer MuJoCo (3.17, the agent's) refuses; it holds nothing.
    xml = task.spec.to_xml().replace("  <default>\n    <default/>\n  </default>\n", "", 1)
    (folder / "task.xml").write_text(xml)
    (folder / "bundle.json").write_text(json.dumps({
        "mount_position": task.mount[0].tolist(), "mount_quat": task.mount[1].tolist(),
        "tips": len(task.tips), "mujoco": mujoco.__version__,
        "library_made": str(library["made"]),
        "made": datetime.now().isoformat(timespec="seconds")}, indent=2))
    return folder


def run(progress=lambda done, total, text="": None):
    """Run the pipeline to the end; returns the fetched run folder."""
    from clearml import Dataset, PipelineController, Task   # here: only this needs ClearML

    reason = blocked()
    if reason:
        raise RuntimeError(reason)
    config = settings.section("learning")
    repo, commit = _git("remote", "get-url", "origin").strip(), _git("rev-parse", "HEAD").strip()
    here = str(ROOT.relative_to(_git("rev-parse", "--show-toplevel").strip()))
    progress(0, 100, "uploading the bundle")
    with tempfile.TemporaryDirectory() as tmp:
        dataset = Dataset.create(dataset_project=PROJECT, dataset_name="handlab gpu bundle")
        dataset.add_files(str(export_bundle(Path(tmp) / "bundle")))
        dataset.upload()
        dataset.finalize()
    step = Task.create(project_name=PROJECT, task_name="handlab train_gpu", repo=repo,
                       commit=commit, script=f"{here}/{STEP_SCRIPT}", working_directory=here,
                       requirements_file=str(ROOT / REQUIREMENTS))
    pipe = PipelineController(name="handlab tracker gpu", project=PROJECT, add_run_number=True)
    pipe.add_step("train_gpu", base_task_id=step.id, execution_queue=config["clearml_queue"],
                  parameter_override={"handlab/bundle": dataset.id,
                                      "handlab/envs": config["gpu_envs"],
                                      "handlab/iterations": config["iterations"]})
    finished, failure = threading.Event(), []

    def control():
        try:
            pipe.start_locally(run_pipeline_steps_locally=False)
        except Exception as error:                    # reported below
            failure.append(error)
        finally:
            finished.set()

    def cancel(signum, frame):
        _stop(pipe)
        raise SystemExit("cancelled")

    signal.signal(signal.SIGTERM, cancel)
    threading.Thread(target=control, daemon=True).start()
    while not finished.wait(POLL_S):
        _mirror(pipe, progress)
    if failure or not pipe.is_successful():
        reason = str(failure[0]) if failure else _step_error(pipe)
        raise RuntimeError(f"the pipeline failed: {reason or 'see it in ClearML (project HandRL)'}")
    return _fetch(Task.get_task(task_id=pipe.get_pipeline_dag()["train_gpu"].executed))


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


def _step_error(pipe):
    """The failed step's last error line, from its console log."""
    from clearml import Task

    for name, node in pipe.get_pipeline_dag().items():
        job = getattr(node, "job", None)
        if job is None:
            continue
        task = Task.get_task(task_id=job.task_id())
        if task.get_status() != "failed":
            continue
        lines = [line.strip() for report in task.get_reported_console_output(number_of_reports=50)
                 for line in report.splitlines() if line.strip()]
        errors = [line for line in lines if "Error" in line or "error" in line]
        return f"{name} on the agent - {(errors or lines or ['no log'])[-1][:300]}"
    return ""


def _stop(pipe):
    """Stop the pipeline and its running step on ClearML."""
    from clearml import Task

    for node in pipe.get_pipeline_dag().values():
        job = getattr(node, "job", None)
        if job is not None and not node.executed:
            Task.get_task(task_id=job.task_id()).mark_stopped(force=True)
    pipe.stop(mark_failed=False, mark_aborted=True)


def _fetch(task):
    """The step's run folder, unpacked into data/learning/runs."""
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
