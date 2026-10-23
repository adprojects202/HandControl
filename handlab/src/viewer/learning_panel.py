"""The Learning panel: build the tracker's library, train it, and look at
its runs. Building and training each run as a process of their own
(learning/background.py) - the window stays responsive, and a run goes on
when the window closes; Cancel stops it.

    Library   what data/learning/library.npz holds; Build library
    Train     method, sim copies and rounds (learning.*); Train here, or
              Train on ClearML (the agent with the RTX 3090; the run comes
              back into Runs); progress, and the error while it learns, live
    Runs      each run: the recordings (never trained on) and the motions
              it trained on, next to the planner and to holding still; which
              run Recordings step 7 uses; Test for a run not yet tested
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.implot import implot

from .. import settings
from ..learning import remote, train, trajectories
from .assistant_panel import AMBER
from .form import GREEN, RED, SettingsForm, row, table

POLL_S = 1.0                     # how often the status and progress files are read
CURVES = (("/track/turn_deg", "turn off (deg)", 1.0),
          ("/track/position_mm", "position off (mm)", 1.0),
          ("Policy/mean_std", "exploration (x100)", 100.0))


class TaskStatus:
    """The background task's status, read at most every POLL_S."""

    def __init__(self, task):
        self.task = task
        self._read_at, self._status = 0.0, None

    def get(self):
        if time.monotonic() - self._read_at > POLL_S:
            self._read_at, self._status = time.monotonic(), self.task.status()
        return self._status

    def refresh(self):
        self._read_at = 0.0

    def busy(self):
        status = self.get()
        return bool(status) and status.get("state") in ("starting", "running")


def _progress(status, which, poll):
    """The background task's bar and Cancel while `which` runs; how it ended
    after."""
    if not status or status.get("task") != which:
        return
    state = status.get("state")
    if state in ("starting", "running"):
        total = status.get("total") or 0
        fraction = status.get("done", 0) / total if total else 0.0
        started = datetime.fromisoformat(status["started"])
        elapsed = (datetime.now() - started).total_seconds()
        left = (f", ~{(elapsed / fraction - elapsed) / 60:.0f} min left"
                if 0.02 < fraction < 1 else "")
        imgui.ProgressBar(fraction, imgui.Vec2(-90, 0),
                          f"{elapsed / 60:.0f} min{left}")
        imgui.SameLine()
        if imgui.Button(f"Cancel##{which}"):
            poll.task.cancel()
            poll.refresh()
        imgui.PushTextWrapPos(0.0)
        imgui.TextDisabled(status.get("text", ""))
        imgui.PopTextWrapPos()
    elif state == "done":
        imgui.TextColored(GREEN, f"{status.get('text', 'done')} - {status.get('ended', '')}")
    elif state == "failed":
        imgui.PushTextWrapPos(0.0)
        imgui.TextColored(RED, f"failed: {status.get('error', '?')}")
        imgui.PopTextWrapPos()
        if status.get("log"):
            imgui.TextDisabled(f"log: {status['log']}")
    elif state == "cancelled":
        imgui.TextColored(AMBER, f"cancelled {status.get('ended', '')}")
    elif state == "stopped":
        imgui.TextColored(AMBER, "the process ended without finishing"
                          + (f" - log: {status['log']}" if status.get("log") else ""))


def _curve(plot_id, rows, height=170):
    """The error and exploration over the rounds of a run."""
    if not rows or not implot.BeginPlot(plot_id, imgui.Vec2(-1, height)):
        return
    fit = int(implot.AxisFlags.AutoFit)
    implot.SetupAxes("round", "deg / mm", fit, fit)
    rounds = [float(r["iteration"]) for r in rows]
    for key, label, scale in CURVES:
        values = [float(r.get(key, np.nan)) * scale for r in rows]
        implot.PlotLine(label, rounds, values)
    implot.EndPlot()


class LibrarySection:
    title = "Library"

    def __init__(self, poll):
        self.poll = poll
        self._made, self._lines, self._exists = None, [], False

    def draw(self):
        self._describe()
        for line in self._lines:
            imgui.TextDisabled(line)
        busy = self.poll.busy()
        imgui.BeginDisabled(busy)
        if imgui.Button("Build library again" if self._exists else "Build library"):
            self.poll.task.start("library")
            self.poll.refresh()
        imgui.EndDisabled()
        imgui.SetItemTooltip(
            "starts the hand can grip around where each recording's motion begins, "
            "generated motions from each, and the recordings' own - each tried by the "
            "planner. Needs a current Send to sim (Recordings, step 5). About 30 min, "
            "on every core" + (" - something else is running" if busy else ""))
        _progress(self.poll.get(), "library", self.poll)

    def _describe(self):
        """The library's summary, re-read when the file changes."""
        path = trajectories.LIBRARY
        made = path.stat().st_mtime if path.exists() else None
        if made == self._made:
            return
        self._made = made
        library = trajectories.load()
        self._exists = bool(library)
        if not library:
            self._lines = ["no library yet"]
            return
        source = library["motion_source"]
        feasible = library["motion_feasible"]
        kinds = ", ".join(f"{int(((source == k) & feasible).sum())}/{int((source == k).sum())} {k}"
                          for k in ("moves", "wobble") if (source == k).any())
        recorded = library["motion_recorded"]
        names = sorted({str(s).rstrip("/").split("/")[-2] for s in source[recorded]})
        self._lines = [
            f"made {str(library['made']).replace('T', ' ')}",
            f"{len(library['start_slip'])} starts the hand grips",
            f"feasible generated motions: {kinds}",
            f"recordings (the test, never trained on): {', '.join(names) or 'none'}"]


class TrainSection:
    title = "Train"

    def __init__(self, poll):
        self.poll = poll
        self.form = SettingsForm(lambda: None)
        self._rows, self._rows_at, self._run = [], 0.0, None
        self._blocked = ""

    def draw(self):
        method = settings.get("learning.method")
        if table("training"):
            row("method", "how the policy learns:\nppo - rsl_rl's PPO, the policy finding its "
                          "own way by trial and error\nplanner_teacher - the planner labels "
                          "every step and the policy copies (DAgger)")
            self.form.choice("learning.method")
            if method == "ppo":
                low, high = settings.limits("learning.envs")
                row("sim copies", f"copies of the task sim stepped together\n{low} - {high}")
                self.form.field("learning.envs")
                low, high = settings.limits("learning.iterations")
                row("rounds", f"PPO rounds; about 5 s each with 2048 copies\n{low} - {high}")
                self.form.field("learning.iterations")
            else:
                low, high = settings.limits("learning.teacher_rounds")
                row("rounds", "each runs every training motion once, the planner labelling "
                              "every step, then fits the policy to all labels so far; about "
                              f"20 min each\n{low} - {high}")
                self.form.field("learning.teacher_rounds")
            imgui.EndTable()
        if imgui.TreeNode("on ClearML"):
            if table("clearml"):
                row("queue", "the ClearML queue the agent (the Windows PC with the RTX 3090) "
                             "takes work from")
                self.form.text("learning.clearml_queue")
                low, high = settings.limits("learning.gpu_envs")
                row("GPU sim copies", "task sims the GPU steps together (MuJoCo Warp)\n"
                                      f"{low} - {high}")
                self.form.field("learning.gpu_envs")
                imgui.EndTable()
            imgui.TreePop()
        self.form.draw_message()
        busy = self.poll.busy()
        no_library = not trajectories.LIBRARY.exists()
        imgui.BeginDisabled(busy or no_library)
        if imgui.Button("Train"):
            self.poll.task.start("train")
            self.poll.refresh()
        imgui.EndDisabled()
        imgui.SetItemTooltip("build the library first" if no_library else
                             "trains a new run here, then tests it on every motion - "
                             "the recordings being the one-shot test"
                             + (" - something else is running" if busy else ""))
        imgui.SameLine()
        imgui.BeginDisabled(busy)
        if imgui.Button("Train on ClearML"):
            self._blocked = remote.blocked()          # git: asked on the click, not every frame
            if not self._blocked:
                self.poll.task.start("clearml")
                self.poll.refresh()
        imgui.EndDisabled()
        imgui.SetItemTooltip("PPO on the agent's GPU (MuJoCo Warp), as a ClearML pipeline: "
                             "this Mac's library and task sim go up, the run comes back into "
                             "Runs. Needs this code committed and pushed: the agent clones it "
                             "from GitHub")
        if self._blocked:
            imgui.TextColored(RED, f"cannot train on ClearML: {self._blocked}")
        status = self.poll.get()
        _progress(status, "train", self.poll)
        _progress(status, "clearml", self.poll)
        if status and status.get("task") == "train" and status.get("run"):
            _curve("##training", self._progress_of(status["run"]))
            imgui.TextDisabled("measured while it trains (exploring, or mixed with the "
                               "planner), so not its test")

    def _progress_of(self, run):
        if run != self._run or time.monotonic() - self._rows_at > POLL_S:
            self._rows, self._rows_at, self._run = train.progress(Path(run)), time.monotonic(), run
        return self._rows


class RunsSection:
    title = "Runs"

    def __init__(self, poll, job):
        self.poll = poll
        self.job = job                     # processing.Job: testing a run
        self.form = SettingsForm(lambda: None)
        self._runs, self._runs_at = [], 0.0
        self._cache = {}                   # run -> (made, run.json, evaluation, progress)

    def draw(self):
        if time.monotonic() - self._runs_at > POLL_S:
            self._runs, self._runs_at = train.runs(), time.monotonic()
        if not self._runs:
            imgui.TextDisabled("no runs yet")
            return
        chosen = train.chosen_run()
        imgui.TextDisabled("Recordings step 7 uses the run picked here")
        for folder in self._runs:
            config, evaluation, rows = self._read(folder)
            trained = any(folder.glob("model_*.pt"))
            if imgui.RadioButton(f"##use{folder.name}", folder == chosen) and trained:
                self.form.save("learning.run", folder.name)
            imgui.SetItemTooltip("use this run for Track (Recordings, step 7)"
                                 if trained else "no checkpoint yet")
            imgui.SameLine()
            # Runs from before progress.jsonl: count by their last checkpoint.
            checkpoints = [int(p.stem.split("_")[1]) + 1 for p in folder.glob("model_*.pt")]
            rounds = len(rows) or max(checkpoints, default=0)
            label = (f"{folder.name} - {config.get('method', 'ppo')}, "
                     f"{rounds}/{config.get('iterations', '?')} rounds")
            if imgui.TreeNode(f"{label}###{folder.name}"):
                self._results(folder, evaluation, trained)
                _curve(f"##{folder.name}", rows, 140)
                imgui.TreePop()

    def _results(self, folder, evaluation, trained):
        if evaluation is None:
            imgui.TextDisabled("not tested")
            imgui.BeginDisabled(self.job.busy or self.poll.busy() or not trained)
            if imgui.Button(f"Test##{folder.name}"):
                self.job.start(f"test {folder.name}", lambda progress: self._test(folder))
            imgui.EndDisabled()
            imgui.SetItemTooltip("runs its last checkpoint on every motion, without "
                                 "noise - about 20 s")
            if self.job.label == f"test {folder.name}":
                if self.job.busy:
                    imgui.TextDisabled("testing...")
                elif self.job.error:
                    imgui.TextColored(RED, self.job.error)
            return
        if not imgui.BeginTable(f"results{folder.name}", 4,
                                int(imgui.TableFlags.SizingStretchProp)
                                | int(imgui.TableFlags.RowBg)):
            return
        for header in ("motion", "tracker mm / deg", "planner mm / deg", "held still deg"):
            imgui.TableSetupColumn(header)
        imgui.TableHeadersRow()
        recorded = [r for r in evaluation if r["recorded"]]
        for result in recorded:
            name = result["source"].rstrip("/").split("/")[-2]
            self._result_row(name + (" - dropped" if result["dropped"] else ""),
                             result["position_mm"], result["turn_deg"], result,
                             result["dropped"])
        generated = [r for r in evaluation if not r["recorded"]]
        if generated:
            held = sum(not r["dropped"] for r in generated)
            median = {key: float(np.median([r[key] for r in generated])) for key in
                      ("position_mm", "turn_deg", "planner_position_mm", "planner_turn_deg",
                       "still_turn_deg")}
            self._result_row(f"trained on ({held}/{len(generated)} held)",
                             median["position_mm"], median["turn_deg"], median, False)
        imgui.EndTable()

    @staticmethod
    def _result_row(name, position, turn, result, dropped):
        imgui.TableNextRow()
        imgui.TableNextColumn()
        if dropped:
            imgui.TextColored(RED, name)
        else:
            imgui.Text(name)
        imgui.TableNextColumn()
        imgui.Text(f"{position:.1f} / {turn:.1f}")
        imgui.TableNextColumn()
        imgui.Text(f"{result['planner_position_mm']:.1f} / {result['planner_turn_deg']:.1f}")
        imgui.TableNextColumn()
        imgui.Text(f"{result['still_turn_deg']:.1f}")

    def _test(self, folder):
        results = train.evaluate(folder)
        (folder / "evaluation.json").write_text(json.dumps(results, indent=2))
        self._cache.pop(folder, None)

    def _read(self, folder):
        """(run.json, evaluation.json or None, progress rows), re-read when
        the run's files change."""
        made = tuple(path.stat().st_mtime if path.exists() else None for path in
                     (folder / "run.json", folder / "evaluation.json",
                      folder / train.PROGRESS))
        cached = self._cache.get(folder)
        if cached is None or cached[0] != made:
            config = json.loads((folder / "run.json").read_text())
            evaluation = (json.loads((folder / "evaluation.json").read_text())
                          if (folder / "evaluation.json").exists() else None)
            cached = (made, config, evaluation, train.progress(folder))
            self._cache[folder] = cached
        return cached[1:]
