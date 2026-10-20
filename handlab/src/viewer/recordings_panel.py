"""Recordings: play back a recording, with what each step found drawn on,
and run the steps.

Pick a recording (the list is rescanned when it opens). It plays at the
speed it was recorded, mirrored like the live preview when `vision.mirror`
is on (the recording itself never is), with - each switchable - the hands'
skeletons, the objects' masks (segment step) and the boxes the describe step
gave (on the frames they were given for).

The steps, in order, each run off the GUI thread and save into the
recording's folder:

    Describe   sends `assistant.clip_frames` frames, spread evenly, to the
               vision-language model -> context.json (task, objects, ...)
    Segment    masks of each object the context names, every frame, by
               `segment.model` (SAM 2.1 or SAM 3) -> masks.npz

"Sent frames" shows the frames Describe sent, numbered as the model saw
them, with the same overlays; click one to jump the player there.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings
from ..assistant import context, jpeg
from ..processing.segment import segment
from ..recording import Episode, episodes
from ..vision.overlay import draw, object_colour
from .assistant_panel import AMBER, draw_context, status
from .form import RED, SettingsForm
from .image import GuiImage

SHEET_COLUMNS = 4
TILE = (320, 240)                       # px, one sent frame in the sheet
MODELS = {"facebook/sam2.1-hiera-tiny": "SAM 2.1 tiny (box)",
          "facebook/sam2.1-hiera-small": "SAM 2.1 small (box)",
          "facebook/sam2.1-hiera-base-plus": "SAM 2.1 base+ (box)",
          "facebook/sam2.1-hiera-large": "SAM 2.1 large (box)",
          "facebook/sam3": "SAM 3 (name; HF_TOKEN)"}
MODEL_TIP = ("SAM 2.1 (open) starts from each object's box in the context; "
             "SAM 3 (gated: needs HF_TOKEN in handlab/.env) finds each object "
             "by its name. Both track through the whole recording")


class RecordingsSection:
    title = "Recordings"

    def __init__(self, assistant, job):
        self.assistant = assistant
        self.job = job                   # processing.Job: one step at a time
        self.form = SettingsForm(lambda: None)
        self.viewer = None
        self.player = GuiImage()
        self.sheet = GuiImage()
        self.episode = None
        self.message = ""
        self.show = {"skeleton": True, "masks": True, "boxes": True}
        self._folders = []
        self._list_open = False
        self._index = 0                  # the frame on show
        self._rendered = None            # what `_rgb` was drawn from
        self._rgb = None
        self._sheet_key = None
        self._sheet_rgb = None
        self._playing = False
        self._play_from = (0.0, 0.0)     # (wall clock, recording time) at Play
        self._context = None             # context.json of the episode
        self._changed = False            # a step finished: reload from disk

    def attach(self, viewer):
        self.viewer = viewer

    def draw(self):
        self._choice()
        if self.message:
            imgui.TextColored(RED, self.message)
        episode = self.episode
        if episode is None:
            return
        if self._changed:
            self._changed = False
            self._context = episode.context
            episode.reload_masks()
            self._rendered = self._sheet_key = None
        meta = episode.meta
        imgui.TextDisabled(f"{episode.frames} frames, {meta.get('duration_s', 0):.1f} s, "
                           f"{meta.get('camera', {}).get('name', '?')}")
        if not episode.complete:
            imgui.TextColored(AMBER, "incomplete - the recording did not finish")
        if not episode.frames:
            return
        self._player(episode)
        self._toggles()
        imgui.SeparatorText("1  describe")
        self._describe(episode)
        imgui.SeparatorText("2  segment")
        self._segment(episode)
        if self._context is not None:
            self._sent_frames(episode)
            if imgui.TreeNodeEx("task context", int(imgui.TreeNodeFlags.DefaultOpen)):
                imgui.TextDisabled(f"described {self._context.get('asked', '?')} by "
                                   f"{self._context.get('model', '?')}")
                draw_context(self._context.get("context") or {})
                imgui.TreePop()

    # ------------------------------------------------------------- choice
    def _choice(self):
        imgui.SetNextItemWidth(-1)
        shown = self.episode.name if self.episode else "choose a recording"
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
                    self._open(folder)
            imgui.EndCombo()
        self._list_open = opened

    def _open(self, folder):
        if self.episode is not None:
            self.episode.close()
        self.episode, self.message = None, ""
        try:
            self.episode = Episode(folder)
        except Exception as failure:              # a damaged folder
            self.message = f"cannot open {folder.name}: {failure}"
            return
        self._index, self._playing = 0, False
        self._rendered = self._rgb = self._sheet_key = None
        self._context = self.episode.context

    # ------------------------------------------------------------- player
    def _player(self, episode):
        last = episode.frames - 1
        if self._playing:
            wall, start = self._play_from
            now = start + time.monotonic() - wall
            self._index = int(np.searchsorted(episode.times, now, side="right")) - 1
            if self._index >= last:
                self._index, self._playing = last, False
        key = (self._index, settings.get("vision.mirror"), tuple(self.show.values()))
        if self._rendered != key:
            rgb = episode.image(self._index)
            if rgb is not None:
                self._rgb = self._overlay(episode, self._index, rgb)
            self._rendered = key
        if self._rgb is not None and self.viewer is not None:
            self.player.draw(self.viewer, self._rgb)

        if imgui.Button("Pause" if self._playing else "Play"):
            self._playing = not self._playing
            if self._playing:
                if self._index >= last:
                    self._index = 0
                self._play_from = (time.monotonic(), float(episode.times[self._index]))
        imgui.SameLine()
        imgui.SetNextItemWidth(-1)
        changed, index = imgui.SliderInt(
            "##frame", self._index, 0, last,
            f"frame %d   {episode.times[self._index]:.2f} s")
        if changed:
            self._index, self._playing = index, False

    def _toggles(self):
        for name in self.show:
            _, self.show[name] = imgui.Checkbox(name, self.show[name])
            imgui.SameLine()
        imgui.NewLine()

    def _overlay(self, episode, index, rgb, mirror=None):
        """`rgb` (frame `index`) with what the steps found drawn on."""
        masks, boxes = [], []
        found = episode.masks
        if self.show["masks"] and found:
            masks = [(episode.mask(track, index), object_colour(track), str(name))
                     for track, name in enumerate(found["names"])]
        record = self._context or {}
        if self.show["boxes"] and record.get("context"):
            sent = record.get("frames") or []
            for track, item in enumerate(record["context"].get("objects", [])):
                image = item.get("image", -1)
                if 0 <= image < len(sent) and sent[image]["index"] == index:
                    box = item["box"]
                    boxes.append(((box["left"], box["top"], box["right"], box["bottom"]),
                                  object_colour(track), f"{item['name']} (box)"))
        hands = episode.hands_at(index) if self.show["skeleton"] else []
        if mirror is None:
            mirror = settings.get("vision.mirror")
        return draw(rgb, hands, mirror=mirror, masks=masks, boxes=boxes)

    # ------------------------------------------------------------- steps
    def _describe(self, episode):
        mine = self.assistant.label == str(episode.folder)
        imgui.BeginDisabled(self.assistant.busy)
        if imgui.Button("Describe again" if self._context else "Describe"):
            self._ask(episode)
        imgui.EndDisabled()
        imgui.SetItemTooltip(f"send {settings.get('assistant.clip_frames')} frames, "
                             "spread over the recording, and save the task context "
                             "with it")
        if mine:
            status(self.assistant)
            answer = self.assistant.answer
            if answer is not None and answer.data is None:
                imgui.TextColored(AMBER, answer.note or "no task context came back")

    def _ask(self, episode):
        frames = episode.sample(settings.get("assistant.clip_frames"))
        times = [float(episode.times[i]) for i in frames]
        max_side = settings.get("assistant.image_max_side")

        def load():                                   # on the assistant's thread
            images = episode.read_frames(frames)
            if len(images) != len(frames):
                raise RuntimeError(f"read {len(images)} of {len(frames)} frames "
                                   "from the video")
            return [jpeg(rgb, max_side) for rgb in images]

        def save(answer, question, config):           # on the assistant's thread
            sent = [{"index": i, "time": round(t, 3)} for i, t in zip(frames, times)]
            episode.save_context(context.record(answer, question, config, sent))
            if self.episode is episode:
                self._changed = True

        self.assistant.describe(load, times, on_done=save, label=str(episode.folder))

    def _segment(self, episode):
        job, label = self.job, f"segment {episode.folder}"
        objects = ((self._context or {}).get("context") or {}).get("objects")
        imgui.SetNextItemWidth(imgui.GetContentRegionAvail().x * 0.55)
        self.form.choice("segment.model", show=lambda name: MODELS.get(name, name))
        imgui.SetItemTooltip(MODEL_TIP)
        imgui.SameLine()
        imgui.BeginDisabled(job.busy or not objects)
        if imgui.Button("Segment again" if episode.masks else "Segment"):
            def done(_):
                if self.episode is episode:
                    self._changed = True
            job.start(label, lambda progress: segment(episode, progress), done)
        imgui.EndDisabled()
        imgui.SetItemTooltip("masks of each object the context names, in every frame"
                             if objects else "describe the recording first: the "
                             "context names the objects")
        if job.label == label:
            if job.busy:
                fraction = job.done / job.total if job.total else 0.0
                imgui.ProgressBar(fraction, imgui.Vec2(-90, 0),
                                  f"{job.text}  {time.monotonic() - job.started:.0f} s")
                imgui.SameLine()
                if imgui.Button("Cancel"):
                    job.cancel()
            elif job.error:
                imgui.PushTextWrapPos(0.0)
                imgui.TextColored(RED, job.error)
                imgui.PopTextWrapPos()
        found = episode.masks
        if found:
            imgui.TextDisabled(f"masks by {MODELS.get(str(found['model']), found['model'])}, "
                               f"{found['made']}")
            for track, name in enumerate(found["names"]):
                area = found["area"][track]
                seen = float((area > 0).mean()) * 100
                imgui.TextColored(imgui.Vec4(*(c / 255 for c in object_colour(track)), 1.0),
                                  f"  {name}: in {seen:.0f}% of frames")
                note = str(found["notes"][track])
                if note:
                    imgui.SameLine()
                    imgui.TextColored(AMBER, note)

    # ------------------------------------------------------------- sent frames
    def _sent_frames(self, episode):
        sent = [frame["index"] for frame in self._context.get("frames") or []]
        if not sent or not imgui.TreeNodeEx(f"sent frames ({len(sent)})",
                                            int(imgui.TreeNodeFlags.DefaultOpen)):
            return
        key = (tuple(sent), settings.get("vision.mirror"), tuple(self.show.values()),
               id(episode.masks))
        if self._sheet_key != key:
            self._sheet_rgb = self._contact_sheet(episode, sent)
            self._sheet_key = key
        if self._sheet_rgb is not None and self.viewer is not None:
            self.sheet.draw(self.viewer, self._sheet_rgb)
            if imgui.IsItemClicked():
                self._jump(sent)
            imgui.SetItemTooltip("click a frame to show it in the player")
        imgui.TreePop()

    def _contact_sheet(self, episode, sent):
        """The sent frames in a grid, each numbered as the model saw it."""
        tiles = []
        for number, (index, rgb) in enumerate(zip(sent, episode.read_frames(sent))):
            tile = cv2.resize(self._overlay(episode, index, rgb), TILE,
                              interpolation=cv2.INTER_AREA)
            text = f"image {number}  {episode.times[index]:.1f} s"
            cv2.putText(tile, text, (6, TILE[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(tile, text, (6, TILE[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        if not tiles:
            return None
        while len(tiles) % SHEET_COLUMNS:
            tiles.append(np.zeros_like(tiles[0]))
        return np.vstack([np.hstack(tiles[row:row + SHEET_COLUMNS])
                          for row in range(0, len(tiles), SHEET_COLUMNS)])

    def _jump(self, sent):
        """Show the clicked tile's frame in the player."""
        mouse, corner = imgui.GetMousePos(), imgui.GetItemRectMin()
        size = imgui.GetItemRectSize()
        rows = (len(sent) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
        column = int((mouse.x - corner.x) / size.x * SHEET_COLUMNS)
        row = int((mouse.y - corner.y) / size.y * rows)
        number = row * SHEET_COLUMNS + column
        if 0 <= column < SHEET_COLUMNS and 0 <= number < len(sent):
            self._index, self._playing = sent[number], False
