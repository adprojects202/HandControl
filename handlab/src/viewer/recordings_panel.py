"""Recordings: play back a recording, with what each step found drawn on,
and run the steps that read it - what the person did, from the video.

Pick a recording (the list is rescanned when it opens; the Real to Sim panel
shows the same one, at the same frame). It plays at the speed it was
recorded, mirrored like the live preview when `vision.mirror` is on (the
recording itself never is), with - each switchable - the hands' skeletons,
the objects' masks (segment step), the boxes the describe step gave (on the
frames they were given for), each object's axis (step 4) and the 3D fit
(Real to Sim).

The steps, in order, each run off the GUI thread and save into the
recording's folder:

    1 Describe  sends `assistant.clip_frames` frames, spread evenly, to the
                vision-language model -> context.json (task, objects, ...);
                below it the frames it sent (click one to jump there), the
                task context, and the assistant's own settings - provider,
                model, key, prompt - with "Describe frame" to try it on the
                live camera
    2 Segment   masks of each object the context names, every frame, by
                `segment.model` (SAM 2.1 or SAM 3) -> masks.npz
    3 Hands     each hand followed as one hand, labels by vote, phantoms
                dropped, gaps filled, smoothed -> hands_clean.npz
    4 Object    each object's motion in the image, and relative to the hand
                holding it -> object.npz; plotted below, the plot's cursor
                is the player's frame (click the plot to go there)

Bringing it into the sim - the hand, the object in 3D, the robot doing it -
is the Real to Sim panel's.
"""

from __future__ import annotations

import cv2
import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.implot import implot

from .. import settings
from ..assistant import context, jpeg
from ..processing.hands import clean_hands
from ..processing.objects import measure_objects
from ..processing.pose3d import edges as box_edges, load as load_pose, project
from ..processing.segment import segment
from ..vision.overlay import draw, object_colour
from ..vision.tracker import SIDES
from .assistant_panel import AMBER, draw_context, status
from .form import SettingsForm
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

    def __init__(self, assistant, assistant_settings, job, playback):
        """`assistant_settings`: the assistant's settings (AssistantSection),
        drawn inside step 1; `playback`: the recording on show (shared)."""
        self.assistant = assistant
        self.assistant_settings = assistant_settings
        self.job = job                   # processing.Job: one step at a time
        self.playback = playback
        self.form = SettingsForm(lambda: None)
        self.viewer = None
        self.player = GuiImage()
        self.sheet = GuiImage()
        self.show = {"skeleton": True, "masks": True, "boxes": True, "axis": True,
                     "3D": True}
        self._rendered = None            # what `_rgb` was drawn from
        self._rgb = None
        self._sheet_key = None
        self._sheet_rgb = None
        self._context = None             # context.json of the episode
        self._version = None             # playback.version the caches are of

    def attach(self, viewer):
        self.viewer = viewer

    def draw(self):
        playback = self.playback
        playback.choose()
        playback.refresh()
        episode = playback.episode
        if episode is None:
            return
        if self._version != playback.version:
            self._version = playback.version
            self._context = episode.context
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
        playback.step(self.job, "hands", "Clean hands", clean_hands,
                      "" if episode.hands else "the recording has no hands",
                      "follow each hand as one hand through the recording: labels "
                      "by vote, phantom hands dropped, short gaps filled, smoothed",
                      done=bool(episode.clean_hands))
        self._hands_summary(episode)
        imgui.SeparatorText("4  object motion")
        playback.step(self.job, "object", "Measure", measure_objects,
                      "" if episode.masks else "segment the recording first",
                      "each object's centre, size and angle in every frame, and "
                      "relative to the hand holding it",
                      done=bool(episode.arrays("object.npz")))
        self._motion(episode)

    # ------------------------------------------------------------- player
    def _player(self, episode):
        index = self.playback.index
        key = (index, self.playback.version, settings.get("vision.mirror"),
               tuple(self.show.values()))
        if self._rendered != key:
            rgb = episode.image(index)
            if rgb is not None:
                self._rgb = self._overlay(episode, index, rgb)
                back = episode.back_image(index)
                if back is not None:
                    self._rgb = _beside(self._rgb, self._back_overlay(episode, index, back))
            self._rendered = key
        if self._rgb is not None and self.viewer is not None:
            self.player.draw(self.viewer, self._rgb)
        self.playback.timeline("recordings")

    def _back_overlay(self, episode, index, rgb):
        """The back camera's frame with its tracked hands, mirrored like the
        front's preview."""
        hands = episode.back_hands_at(index) if self.show["skeleton"] else []
        return draw(rgb, hands, None, mirror=settings.get("vision.mirror"))

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

    # ------------------------------------------------------------- 1 describe
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
        if self._context is not None:
            self._sent_frames(episode)
            if imgui.TreeNode("task context"):
                imgui.TextDisabled(f"described {self._context.get('asked', '?')} by "
                                   f"{self._context.get('model', '?')}")
                draw_context(self._context.get("context") or {})
                imgui.TreePop()
        if imgui.TreeNode("assistant: model, key, prompt"):
            self.assistant_settings.draw()
            imgui.TreePop()

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
            self.playback.mark_changed(episode)

        self.assistant.describe(load, times, on_done=save, label=str(episode.folder))

    def _sent_frames(self, episode):
        sent = [frame["index"] for frame in self._context.get("frames") or []]
        if not sent or not imgui.TreeNode(f"sent frames ({len(sent)})"):
            return
        key = (tuple(sent), settings.get("vision.mirror"), tuple(self.show.values()),
               self.playback.version)
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
            self.playback.seek(sent[number])

    # ------------------------------------------------------------- 2 - 4
    def _segment(self, episode):
        objects = ((self._context or {}).get("context") or {}).get("objects")
        imgui.SetNextItemWidth(imgui.GetContentRegionAvail().x * 0.55)
        self.form.choice("segment.model", show=lambda name: MODELS.get(name, name))
        imgui.SetItemTooltip(MODEL_TIP)
        imgui.SameLine()
        self.playback.step(self.job, "segment", "Segment", segment,
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
        self.playback.plot_cursor(0)
        implot.EndPlot()
        imgui.TextDisabled("how far each object has turned since first seen "
                           "(counter-clockwise +), in the image and in the "
                           "holding hand; gaps: not seen")


def _beside(front, back):
    """Two frames side by side, the back one scaled to the front's height."""
    height = front.shape[0]
    width = round(back.shape[1] * height / back.shape[0])
    return np.hstack([front, cv2.resize(back, (width, height), interpolation=cv2.INTER_AREA)])


PALM_AXES = ((255, 70, 70), (70, 220, 70), (80, 140, 255))   # x, y, z: red, green, blue
PALM_AXIS_M = 0.04


def _pose_lines(pose, index, calibration):
    """The 3D fit at frame `index`, as lines to draw: each object's box
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
