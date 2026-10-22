"""Recordings: play back a recording, with what each step found drawn on,
and run the steps.

Pick a recording (the list is rescanned when it opens). It plays at the
speed it was recorded, mirrored like the live preview when `vision.mirror`
is on (the recording itself never is), with - each switchable - the hands'
skeletons, the objects' masks (segment step) and the boxes the describe step
gave (on the frames they were given for).

The steps, in order, each run off the GUI thread and save into the
recording's folder:

    1 Describe  sends `assistant.clip_frames` frames, spread evenly, to the
                vision-language model -> context.json (task, objects, ...)
    2 Segment   masks of each object the context names, every frame, by
                `segment.model` (SAM 2.1 or SAM 3) -> masks.npz
    3 Hands     each hand followed as one hand, labels by vote, phantoms
                dropped, gaps filled, smoothed -> hands_clean.npz
    4 Object    each object's motion in the image, and relative to the hand
                holding it -> object.npz; plotted below, the plot's cursor
                is the player's frame (click the plot to go there)
    5 Transfer to sim
                Fit 3D: the hands and the object in metres, from the
                camera's calibration and the object's known size, and the
                object in the palm -> pose3d.npz; plotted like step 4.
                Send to sim: that motion in the robot hand's frame -> sim.npz;
                with "show in sim", a ghost box in the world follows the
                player's frame

"Sent frames" shows the frames Describe sent, numbered as the model saw
them, with the same overlays; click one to jump the player there.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.implot import implot

from .. import settings
from ..assistant import context, jpeg
from ..processing.hands import clean_hands
from ..processing.objects import measure_objects
from ..processing.pose3d import POSE3D, edges as box_edges, fit_3d, load as load_pose, project
from ..processing.transfer import SIM, transfer
from ..processing.segment import segment
from ..recording import Episode, episodes
from ..vision.overlay import draw, object_colour
from ..vision.tracker import SIDES
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

    def __init__(self, assistant, job, show_demo=lambda pose: None):
        """`show_demo(pose or None)` puts the sim's demo ghost where a
        transferred recording's object is at the player's frame."""
        self.assistant = assistant
        self.job = job                   # processing.Job: one step at a time
        self.show_demo = show_demo
        self.in_sim = True
        self.form = SettingsForm(lambda: None)
        self.viewer = None
        self.player = GuiImage()
        self.sheet = GuiImage()
        self.episode = None
        self.message = ""
        self.show = {"skeleton": True, "masks": True, "boxes": True, "axis": True,
                     "3D": True}
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
            episode.reload()
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
        imgui.SeparatorText("3  hands")
        self._step(episode, "hands", "Clean hands", clean_hands,
                   "" if episode.hands else "the recording has no hands",
                   "follow each hand as one hand through the recording: labels "
                   "by vote, phantom hands dropped, short gaps filled, smoothed")
        self._hands_summary(episode)
        imgui.SeparatorText("4  object motion")
        self._step(episode, "object", "Measure", measure_objects,
                   "" if episode.masks else "segment the recording first",
                   "each object's centre, size and angle in every frame, and "
                   "relative to the hand holding it")
        self._motion(episode)
        imgui.SeparatorText("5  transfer to sim")
        self._step(episode, "pose3d", "Fit 3D", fit_3d, self._blocked_3d(episode),
                   "the hand and the object in metres: the hand from MediaPipe's "
                   "shape, the object as a box of the World section's object size",
                   done=bool(episode.arrays(POSE3D)))
        self._pose3d(episode)
        self._step(episode, "transfer", "Send to sim", transfer,
                   "" if episode.arrays(POSE3D) else "fit 3D first",
                   "the object's motion in the robot hand's frame, anchored where "
                   "the World section places the calibration object",
                   done=bool(episode.arrays(SIM)))
        self._sim(episode)
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
        self.show_demo(None)
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
        axes = []
        motion = episode.arrays("object.npz")
        if self.show["axis"] and motion:
            for track in range(len(motion["names"])):
                if motion["visible"][track, index]:
                    length = 0.6 * np.sqrt(motion["area"][track, index]
                                           * rgb.shape[0] * rgb.shape[1])
                    axes.append((motion["centroid"][track, index],
                                 float(motion["heading"][track, index]), length,
                                 object_colour(track)))
        lines = []
        pose = load_pose(episode)
        calibration = episode.calibration
        if self.show["3D"] and pose and calibration is not None:
            lines = _pose_lines(pose, index, calibration)
        hands = episode.hands_at(index) if self.show["skeleton"] else []
        if mirror is None:
            mirror = settings.get("vision.mirror")
        return draw(rgb, hands, mirror=mirror, masks=masks, boxes=boxes, axes=axes,
                    lines=lines)

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
        objects = ((self._context or {}).get("context") or {}).get("objects")
        imgui.SetNextItemWidth(imgui.GetContentRegionAvail().x * 0.55)
        self.form.choice("segment.model", show=lambda name: MODELS.get(name, name))
        imgui.SetItemTooltip(MODEL_TIP)
        imgui.SameLine()
        self._step(episode, "segment", "Segment", segment,
                   "" if objects else "describe the recording first: the context "
                   "names the objects",
                   "masks of each object the context names, in every frame",
                   done=bool(episode.masks))
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

    def _step(self, episode, name, button, run, blocked, tip, done=None):
        """A step's button, its progress while it runs, and its error.
        `run(episode, progress)` runs on the processing thread; `blocked`
        says why it cannot run yet ("" when it can)."""
        job, label = self.job, f"{name} {episode.folder}"
        imgui.BeginDisabled(job.busy or bool(blocked))
        if imgui.Button(f"{button} again" if done else button):
            def finished(_):
                if self.episode is episode:
                    self._changed = True
            job.start(label, lambda progress: run(episode, progress), finished)
        imgui.EndDisabled()
        imgui.SetItemTooltip(blocked or tip)
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

    def _sim(self, episode):
        """The transferred motion's summary, and the ghost in the sim."""
        sim = episode.arrays(SIM)
        if not sim:
            self.show_demo(None)
            return
        imgui.SameLine()
        _, self.in_sim = imgui.Checkbox("show in sim", self.in_sim)
        imgui.SetItemTooltip("a ghost box in the world follows this recording's "
                             "object at the player's frame")
        note = ("" if str(sim["hand"]) == "Left" else
                " - recorded with the right hand: not yet mirrored for the robot's "
                "left hand")
        imgui.TextDisabled(f"{int(sim['valid'].sum())} frames in the robot hand's frame"
                           f"{note}")
        index = self._index
        if self.in_sim and sim["valid"][index]:
            self.show_demo((sim["position"][index].astype(float),
                            sim["quat"][index].astype(float)))
        else:
            self.show_demo(None)

    @staticmethod
    def _blocked_3d(episode):
        if episode.calibration is None:
            return ("calibrate the camera first (Camera section > Calibrate...): "
                    f"{episode.meta.get('camera', {}).get('name', '?')}")
        if not episode.clean_hands or not episode.arrays("object.npz"):
            return "run steps 3 and 4 first"
        return ""

    def _pose3d(self, episode):
        pose = load_pose(episode)
        if not pose:
            return
        size = " x ".join(f"{v:.0f}" for v in pose["box_mm"])
        for track, name in enumerate(pose["names"]):
            fit = np.nanmedian(pose["object_fit"][track])
            scale = pose["hand_scale"][track]
            palm = "one still palm" if pose["still_hand"][track] else "the palm each frame"
            imgui.TextColored(imgui.Vec4(*(c / 255 for c in object_colour(track)), 1.0),
                              f"{name} as a {size} mm box: outline fit {fit:.2f}; "
                              f"hand {scale:.2f}x MediaPipe's size; in {palm}")
        marked = pose["marker_id"] >= 0
        if marked.any():
            compared = np.isfinite(pose["outline_error_mm"])
            versus = (f"; outline fit vs marker: {np.median(pose['outline_error_mm'][compared]):.0f}"
                      f" mm, {np.median(pose['outline_error_deg'][compared]):.0f} deg (median)"
                      if compared.any() else "")
            imgui.TextColored(imgui.Vec4(1.0, 0.24, 0.78, 1.0),
                              f"ArUco marker in {marked.mean():.0%} of frames - its pose used "
                              f"there{versus}")
        elif settings.get("vision.aruco"):
            imgui.TextDisabled("no ArUco marker seen")
        times = [float(t) for t in episode.times]
        if not implot.BeginPlot("##pose3d", imgui.Vec2(-1, 170)):
            return
        implot.SetupAxes("s", "deg / mm")
        for track, name in enumerate(pose["names"]):
            implot.PlotLine(f"{name} - turn in palm (deg)##{track}", times,
                            [float(v) for v in pose["in_palm_turn"][track]])
            implot.PlotLine(f"{name} - out of palm (mm)##{track}", times,
                            [float(v) for v in 1000 * pose["in_palm_position"][track, :, 2]])
        if (pose["marker_id"] >= 0).any():
            implot.PlotLine("outline vs marker (mm)", times,
                            [float(v) for v in pose["outline_error_mm"]])
            implot.PlotLine("outline vs marker (deg)", times,
                            [float(v) for v in pose["outline_error_deg"]])
        implot.DragLineX(1, times[self._index], imgui.Vec4(1, 1, 1, 0.8), 1,
                         int(implot.DragToolFlags.NoInputs))
        if implot.IsPlotHovered() and imgui.IsMouseClicked(0):
            self._index = int(np.clip(np.searchsorted(episode.times, implot.GetPlotMousePos().x),
                                      0, episode.frames - 1))
            self._playing = False
        implot.EndPlot()
        imgui.TextDisabled("how far the object has turned in the palm since first "
                           "seen (least over the box's symmetries), and how far its "
                           "centre is out from the palm")

    def _hands_summary(self, episode):
        clean = episode.clean_hands
        if not clean:
            return
        hands = ", ".join(f"{side.lower()} ({frames / episode.frames:.0%} of frames)"
                          for side, frames in zip(clean["track_side"], clean["track_frames"]))
        imgui.TextDisabled(f"{int(clean['tracks'])} hand(s): {hands or 'none'}")
        imgui.TextDisabled(f"{int(clean['relabelled'])} labels corrected, "
                           f"{int(clean['phantoms'])} phantom detections dropped, "
                           f"{int(clean['filled_frames'])} frames filled")

    def _motion(self, episode):
        """Per object: who holds it, and its angle and size through time."""
        motion = episode.arrays("object.npz")
        if not motion:
            return
        times = [float(t) for t in episode.times]
        for track, name in enumerate(motion["names"]):
            slot = int(motion["hand"][track])
            held = (f"held by the {SIDES[slot].lower()} hand in "
                    f"{motion['contact'][track, :, slot].mean():.0%} of frames"
                    if slot >= 0 else "not held")
            imgui.TextColored(imgui.Vec4(*(c / 255 for c in object_colour(track)), 1.0),
                              f"{name}: {held}")
        if not implot.BeginPlot("##motion", imgui.Vec2(-1, 170)):
            return
        implot.SetupAxes("s", "deg")
        for track, name in enumerate(motion["names"]):
            implot.PlotLine(f"{name} - turn in image##{track}", times,
                            [float(v) for v in np.degrees(motion["turn"][track])])
            implot.PlotLine(f"{name} - turn in hand##{track}", times,
                            [float(v) for v in np.degrees(motion["in_hand_turn"][track])])
        implot.DragLineX(0, times[self._index], imgui.Vec4(1, 1, 1, 0.8), 1,
                         int(implot.DragToolFlags.NoInputs))
        if implot.IsPlotHovered() and imgui.IsMouseClicked(0):
            seconds = implot.GetPlotMousePos().x
            self._index = int(np.clip(np.searchsorted(episode.times, seconds),
                                      0, episode.frames - 1))
            self._playing = False
        implot.EndPlot()
        imgui.TextDisabled("how far each object has turned since first seen "
                           "(counter-clockwise +), in the image and in the "
                           "holding hand; gaps: not seen")

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


PALM_AXES = ((255, 70, 70), (70, 220, 70), (80, 140, 255))   # x, y, z: red, green, blue
PALM_AXIS_M = 0.04


def _pose_lines(pose, index, calibration):
    """The 3D step's fit at frame `index`, as lines to draw: each object's box
    (magenta where posed from an ArUco marker, yellow from its outline) and
    each hand's palm axes."""
    lines = []
    for track in range(len(pose["names"])):
        rotation = pose["object_rotation"][track, index]
        if np.isfinite(rotation).all():
            corners = project(box_edges(pose["box_mm"] / 1000), rotation,
                              pose["object_position"][track, index], calibration)
            from_marker = pose["object_source"][track, index] == 2
            colour, width = ((255, 60, 200), 2) if from_marker else ((255, 255, 0), 1)
            lines += [(a, b, colour, width) for a, b in corners]
    for slot in range(pose["palm_position"].shape[1]):
        centre, axes = pose["palm_position"][index, slot], pose["palm_rotation"][index, slot]
        if np.isfinite(centre).all():
            for axis, colour in enumerate(PALM_AXES):
                (a, b), = project([(np.zeros(3), PALM_AXIS_M * np.eye(3)[axis])],
                                  axes, centre, calibration)
                lines.append((a, b, colour, 2))
    return lines
