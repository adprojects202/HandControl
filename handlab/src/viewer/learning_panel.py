"""The Learning panel: what to learn - object tracking (the general
tracker) or hand motion (the hand alone taking a person's hand shapes) -
its library, training, and its runs. Building and training each run as a
process of their own (learning/background.py) - the window stays
responsive, and a run goes on when the window closes; Cancel stops it.

    Library   learn (learning.learn); what its library holds; Build library
    Train     method, sim copies and rounds (learning.*); Train here (object
              tracking), or Train on ClearML (the agent with the RTX 3090;
              the run comes back into Runs); progress, and the error while
              it learns, live
    Runs      the runs learning it: object tracking - the recordings (never
              trained on) and the motions it trained on, next to the planner
              and to holding still, which run Recordings step 8 uses, Test
              for a run not yet tested; hand motion - the random motions and
              the recorded hands, next to holding still, which run
              Recordings step 6 uses
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
from ..learning import handmotion, remote, train, trajectories
from .assistant_panel import AMBER
from .form import GREEN, RED, SettingsForm, row, table

POLL_S = 1.0                     # how often the status and progress files are read
CURVES = {"object tracking": (("/track/turn_deg", "turn off (deg)", 1.0),
                               ("/track/position_mm", "position off (mm)", 1.0),
                               ("Policy/mean_std", "exploration (x100)", 100.0)),
          "hand motion": (("/hand/angle_deg", "shape off (deg)", 1.0),
                          ("/hand/tip_mm", "tips off (mm)", 1.0),
                          ("Policy/mean_std", "exploration (x100)", 100.0))}
HAND = "hand motion"


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


def _curve(plot_id, rows, learn, height=170):
    """The error and exploration over the rounds of a run."""
    if not rows or not implot.BeginPlot(plot_id, imgui.Vec2(-1, height)):
        return
    fit = int(implot.AxisFlags.AutoFit)
    implot.SetupAxes("round", "deg / mm", fit, fit)
    rounds = [float(r["iteration"]) for r in rows]
    for key, label, scale in CURVES[learn]:
        values = [float(r.get(key, np.nan)) * scale for r in rows]
        implot.PlotLine(label, rounds, values)
    implot.EndPlot()


class LibrarySection:
    title = "Library"

    def __init__(self, poll):
        self.poll = poll
        self.form = SettingsForm(lambda: None)
        self._made, self._lines, self._exists = None, [], False

    def draw(self):
        learn = settings.get("learning.learn")
        if table("learn"):
            row("learn", "what the policy learns:\nobject tracking - moving an object as a "
                         "recording did (the general tracker)\nhand motion - the hand alone "
                         "taking a person's hand shapes as they move (on the ClearML GPU)")
            self.form.choice("learning.learn")
            imgui.EndTable()
        self.form.draw_message()
        hand = learn == HAND
        self._describe(hand)
        for line in self._lines:
            imgui.TextDisabled(line)
        busy = self.poll.busy()
        imgui.BeginDisabled(busy)
        if imgui.Button("Build library again" if self._exists else "Build library"):
            self.poll.task.start("hand_library" if hand else "library")
            self.poll.refresh()
        imgui.EndDisabled()
        imgui.SetItemTooltip(
            ("random motions of the robot hand between poses it can take, and every "
             "recorded hand - as hand shapes to follow. About 10 s" if hand else
             "starts the hand can grip around where each recording's motion begins, "
             "generated motions from each, and the recordings' own - each tried by the "
             "planner. Needs a current Send to sim (Recordings, step 5). About 30 min, "
             "on every core") + (" - something else is running" if busy else ""))
        _progress(self.poll.get(), "hand_library" if hand else "library", self.poll)

    def _describe(self, hand):
        """The library's summary, re-read when the file or what is learnt
        changes."""
        path = handmotion.LIBRARY if hand else trajectories.LIBRARY
        made = (hand, path.stat().st_mtime if path.exists() else None)
        if made == self._made:
            return
        self._made = made
        if hand:
            self._describe_hand()
            return
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

    def _describe_hand(self):
        library = handmotion.load()
        self._exists = bool(library)
        if not library:
            self._lines = ["no hand library yet"]
            return
        source = library["source"]
        recorded = source != "random"
        names = sorted({str(s).rstrip("/").split("/")[-1] for s in source[recorded]})
        seconds = library["length"].sum() / int(library["rate"])
        self._lines = [
            f"made {str(library['made']).replace('T', ' ')}",
            f"{int((~recorded).sum())} random motions, {int(recorded.sum())} recorded "
            f"hands - {seconds / 60:.0f} min in all",
            f"recordings: {', '.join(names) or 'none'}"]


class TrainSection:
    title = "Train"

    def __init__(self, poll):
        self.poll = poll
        self.form = SettingsForm(lambda: None)
        self._rows, self._rows_at, self._run = [], 0.0, None
        self._blocked = ""

    def draw(self):
        method = settings.get("learning.method")
        hand = settings.get("learning.learn") == HAND
        if not hand and table("training"):
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
        if hand:
            imgui.SetNextItemOpen(True, int(imgui.Cond.Once))
        if imgui.TreeNode("on ClearML"):
            if table("clearml"):
                row("queue", "the ClearML queue the agent (the Windows PC with the RTX 3090) "
                             "takes work from")
                self.form.text("learning.clearml_queue")
                low, high = settings.limits("learning.gpu_envs")
                row("GPU sim copies", "task sims the GPU steps together (MuJoCo Warp)\n"
                                      f"{low} - {high}")
                self.form.field("learning.gpu_envs")
                if hand:
                    low, high = settings.limits("learning.iterations")
                    row("rounds", f"PPO rounds\n{low} - {high}")
                    self.form.field("learning.iterations")
                imgui.EndTable()
            imgui.TreePop()
        self.form.draw_message()
        busy = self.poll.busy()
        no_library = not (handmotion.LIBRARY if hand else trajectories.LIBRARY).exists()
        imgui.BeginDisabled(busy or no_library or hand)
        if imgui.Button("Train"):
            self.poll.task.start("train")
            self.poll.refresh()
        imgui.EndDisabled()
        imgui.SetItemTooltip("hand motion trains on the ClearML GPU only" if hand else
                             "build the library first" if no_library else
                             "trains a new run here, then tests it on every motion - "
                             "the recordings being the one-shot test"
                             + (" - something else is running" if busy else ""))
        imgui.SameLine()
        imgui.BeginDisabled(busy or no_library)
        if imgui.Button("Train on ClearML"):
            self._blocked = remote.blocked()          # git: asked on the click, not every frame
            if not self._blocked:
                self.poll.task.start("clearml")
                self.poll.refresh()
        imgui.EndDisabled()
        imgui.SetItemTooltip("build the library first" if no_library else
                             "PPO on the agent's GPU (MuJoCo Warp), as a ClearML pipeline: "
                             "this Mac's library and sim go up, the run comes back into "
                             "Runs. Needs this code committed and pushed: the agent clones it "
                             "from GitHub")
        if self._blocked:
            imgui.TextColored(RED, f"cannot train on ClearML: {self._blocked}")
        status = self.poll.get()
        _progress(status, "train", self.poll)
        _progress(status, "clearml", self.poll)
        if status and status.get("task") == "train" and status.get("run"):
            _curve("##training", self._progress_of(status["run"]), "object tracking")
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
        self._runs, self._runs_at, self._learn = [], 0.0, None
        self._cache = {}                   # run -> (made, run.json, evaluation, progress)

    def draw(self):
        learn = settings.get("learning.learn")
        if time.monotonic() - self._runs_at > POLL_S or learn != self._learn:
            self._runs, self._runs_at, self._learn = train.runs(learn), time.monotonic(), learn
        if not self._runs:
            imgui.TextDisabled(f"no {learn} runs yet")
            return
        chosen = train.chosen_run(learn)
        imgui.TextDisabled("Recordings step 6 (follow the hand) uses the run picked here"
                           if learn == HAND else "Recordings step 8 uses the run picked here")
        for folder in self._runs:
            config, evaluation, rows = self._read(folder)
            trained = any(folder.glob("model_*.pt"))
            if imgui.RadioButton(f"##use{folder.name}", folder == chosen) and trained:
                self.form.save(train.LEARN[learn], folder.name)
            imgui.SetItemTooltip(("use this run for Recordings step 6" if learn == HAND else
                                  "use this run for Recordings step 8")
                                 if trained else "no checkpoint yet")
            imgui.SameLine()
            # Runs from before progress.jsonl: count by their last checkpoint.
            checkpoints = [int(p.stem.split("_")[1]) + 1 for p in folder.glob("model_*.pt")]
            rounds = len(rows) or max(checkpoints, default=0)
            label = (f"{folder.name} - {config.get('method', 'ppo')}, "
                     f"{rounds}/{config.get('iterations', '?')} rounds")
            if imgui.TreeNode(f"{label}###{folder.name}"):
                if learn == HAND:
                    self._hand_results(folder, evaluation)
                else:
                    self._results(folder, evaluation, trained)
                _curve(f"##{folder.name}", rows, learn, 140)
                imgui.TreePop()

    @staticmethod
    def _hand_results(folder, evaluation):
        """Medians over the random motions and over each recording's hands:
        the policy's shape and fingertip errors next to holding the start."""
        if evaluation is None:
            imgui.TextDisabled("not tested (the GPU tests it after training)")
            return
        groups = {"random motions": [r for r in evaluation if not r["recorded"]]}
        for result in evaluation:
            if result["recorded"]:
                name = result["source"].rstrip("/").split("/")[-1]
                groups.setdefault(name, []).append(result)
        if not imgui.BeginTable(f"results{folder.name}", 3,
                                int(imgui.TableFlags.SizingStretchProp)
                                | int(imgui.TableFlags.RowBg)):
            return
        for header in ("clips", "policy deg / tips mm", "held still deg / tips mm"):
            imgui.TableSetupColumn(header)
        imgui.TableHeadersRow()
        for name, results in groups.items():
            if not results:
                continue
            median = {key: float(np.median([r[key] for r in results]))
                      for key in ("angle_deg", "tip_mm", "still_angle_deg", "still_tip_mm")}
            imgui.TableNextRow()
            imgui.TableNextColumn()
            imgui.Text(f"{name} ({len(results)})")
            imgui.TableNextColumn()
            imgui.Text(f"{median['angle_deg']:.1f} / {median['tip_mm']:.1f}")
            imgui.TableNextColumn()
            imgui.Text(f"{median['still_angle_deg']:.1f} / {median['still_tip_mm']:.1f}")
        imgui.EndTable()

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
        for header in ("motion", "policy mm / deg", "planner mm / deg", "held still deg"):
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
