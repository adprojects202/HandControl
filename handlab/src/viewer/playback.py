"""The recording on show, shared by the Recordings and Real to Sim panels:
which recording, the frame, whether it plays - so both always show the same
moment.

    playback = Playback(on_open)
    playback.choose()                  # the recording dropdown
    playback.timeline()                # Play / Pause and the frame slider
    playback.episode, playback.index, playback.version
    playback.step(job, name, button, run, blocked, tip, done)   # a step's button

A step that finishes on its job's thread marks the recording changed; the
next `refresh()` (on the GUI thread) reloads it from disk and bumps
`version`, so a panel knows to redraw what it cached.
"""

from __future__ import annotations

import time

import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.implot import implot

from ..recording import Episode, episodes
from .form import RED


class Playback:
    def __init__(self, on_open=lambda episode: None):
        """`on_open(episode or None)` runs when another recording is chosen."""
        self.on_open = on_open
        self.episode = None
        self.index = 0                    # the frame on show
        self.playing = False
        self.version = 0                  # counts reloads from disk
        self.message = ""
        self._play_from = (0.0, 0.0)      # (wall clock, recording time) at Play
        self._changed = False
        self._folders = []
        self._list_open = False

    # ------------------------------------------------------------- choosing
    def choose(self):
        """The recording dropdown (rescanned as it opens), and why one did not open."""
        imgui.SetNextItemWidth(-1)
        shown = (f"{self.episode.folder.parent.name} / {self.episode.name}"
                 if self.episode else "choose a recording")
        opened = imgui.BeginCombo("##recording", shown)
        if opened:
            if not self._list_open:
                self._folders = episodes()
            if not self._folders:
                imgui.TextDisabled("none yet - record one in Camera")
            for folder in self._folders:
                chosen = self.episode is not None and folder == self.episode.folder
                if imgui.Selectable(f"{folder.parent.name} / {folder.name}", chosen) \
                        and not chosen:
                    self.open(folder)
            imgui.EndCombo()
        self._list_open = opened
        if self.message:
            imgui.TextColored(RED, self.message)

    def open(self, folder):
        if self.episode is not None:
            self.episode.close()
        self.episode, self.message = None, ""
        try:
            self.episode = Episode(folder)
        except Exception as failure:              # a damaged folder
            self.message = f"cannot open {folder.name}: {failure}"
        self.index, self.playing = 0, False
        self.version += 1
        self.on_open(self.episode)

    def mark_changed(self, episode):
        """A step finished on `episode` (any thread)."""
        if episode is self.episode:
            self._changed = True

    def refresh(self):
        """Reload the recording if a step changed it; advance while playing."""
        episode = self.episode
        if episode is None:
            return
        if self._changed:
            self._changed = False
            episode.reload()
            self.version += 1
        if self.playing:
            wall, start = self._play_from
            last = episode.frames - 1
            self.index = int(np.searchsorted(episode.times, start + time.monotonic() - wall,
                                             side="right")) - 1
            if self.index >= last:
                self.index, self.playing = last, False

    # ------------------------------------------------------------- the frame
    def timeline(self, name="frame"):
        episode = self.episode
        last = episode.frames - 1
        if imgui.Button(f"{'Pause' if self.playing else 'Play'}##{name}"):
            self.playing = not self.playing
            if self.playing:
                if self.index >= last:
                    self.index = 0
                self._play_from = (time.monotonic(), float(episode.times[self.index]))
        imgui.SameLine()
        imgui.SetNextItemWidth(-1)
        changed, index = imgui.SliderInt(f"##{name}", self.index, 0, last,
                                         f"frame %d   {episode.times[self.index]:.2f} s")
        if changed:
            self.seek(index)

    def seek(self, index):
        self.index, self.playing = int(np.clip(index, 0, self.episode.frames - 1)), False

    def time(self):
        return float(self.episode.times[self.index])

    def plot_cursor(self, line):
        """The frame on show as a line on the open plot (x in recording
        seconds); a click there moves it."""
        implot.DragLineX(line, self.time(), imgui.Vec4(1, 1, 1, 0.8), 1,
                         int(implot.DragToolFlags.NoInputs))
        if implot.IsPlotHovered() and imgui.IsMouseClicked(0):
            self.seek(np.searchsorted(self.episode.times, implot.GetPlotMousePos().x))

    # ------------------------------------------------------------- steps
    def step(self, job, name, button, run, blocked, tip, done=None):
        """A step's button on the recording on show, its progress while it
        runs, and its error. `run(episode, progress)` runs on `job`'s thread;
        `blocked` says why it cannot run yet ("" when it can)."""
        episode = self.episode
        label = f"{name} {episode.folder}"
        imgui.BeginDisabled(job.busy or bool(blocked))
        if imgui.Button(f"{button} again##{name}" if done else f"{button}##{name}"):
            job.start(label, lambda progress: run(episode, progress),
                      lambda _: self.mark_changed(episode))
        imgui.EndDisabled()
        imgui.SetItemTooltip(blocked or tip)
        progress(job, label, name)

    def working(self, job, *names):
        """Whether `job` runs one of the steps `names` on the recording on show."""
        return job.busy and job.label in {f"{name} {self.episode.folder}" for name in names}


def progress(job, label, name):
    """`job`'s bar and Cancel while it runs `label`; its error after."""
    if job.label != label:
        return
    if job.busy:
        fraction = job.done / job.total if job.total else 0.0
        imgui.ProgressBar(fraction, imgui.Vec2(-90, 0),
                          f"{job.text}  {time.monotonic() - job.started:.0f} s")
        imgui.SameLine()
        if imgui.Button(f"Cancel##{name}"):
            job.cancel()
    elif job.error:
        imgui.PushTextWrapPos(0.0)
        imgui.TextColored(RED, job.error)
        imgui.PopTextWrapPos()
