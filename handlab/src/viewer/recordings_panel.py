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
                Send to sim: the person's hand laid on the robot's palm ->
                hand_sim.npz, and once fitted in 3D the object's motion in
                the robot hand's frame -> sim.npz; with "show in sim", the
                hand as a skeleton on the plan stage and a ghost box at the
                hand follow the player's frame
    6 Follow the hand
                the trained hand-motion policy takes the shape of the
                person's hand in the hand sim -> follow.npz; with "show in
                sim", the plan stage's hand replays it under the skeleton
    7 Plan in sim
                the robot hand, with its servos, grips the object and
                follows that motion in the task sim -> plan.npz; with "show
                in sim", the plan stage beside the hand replays it at the
                player's frame
    8 Run the trained policy
                the trained policy (src/learning) follows that motion - no
                retraining: the one-shot test -> track.npz; shown like step 7

"Sent frames" shows the frames Describe sent, numbered as the model saw
them, with the same overlays; click one to jump the player there.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.implot import implot

from .. import settings
from ..assistant import context, jpeg
from ..processing.hands import clean_hands
from ..processing.objects import measure_objects
from ..processing.files import FOLLOW, HAND_SIM, PLAN, TRACK, stale
from ..processing.follow import follow_hand, summary as follow_summary
from ..processing.plan import plan, summary as plan_summary
from ..learning.train import chosen_run
from ..processing.track import track as track_with_tracker
from ..processing.pose3d import POSE3D, edges as box_edges, fit_3d, load as load_pose, project
from ..processing.transfer import SIM, send_to_sim
from ..processing.segment import segment
from ..recording import Episode, episodes
from ..vision.overlay import draw, object_colour
from ..vision.tracker import SIDES
from .assistant_panel import AMBER, draw_context, status
from .form import GREEN, RED, SettingsForm
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

    def __init__(self, assistant, job, show_demo=lambda pose: None,
                 show_plan=lambda frame: None, show_skeleton=lambda points: None):
        """`show_demo(pose or None)` puts the sim's demo ghost where a
        transferred recording's object is at the player's frame;
        `show_plan(frame or None)` shows the plan stage at it (World.set_plan);
        `show_skeleton(points or None)` the person's hand on the stage
        (World.set_skeleton)."""
        self.assistant = assistant
        self.job = job                   # processing.Job: one step at a time
        self.show_demo = show_demo
        self.show_plan = show_plan
        self.show_skeleton = show_skeleton
        self.in_sim = True
        self.stage_shows = TRACK             # on the plan stage, when both exist
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
        self._step(episode, "transfer", "Send to sim", send_to_sim,
                   "" if (episode.clean_hands or episode.hands) else "the recording has no hands",
                   "the person's hand laid on the robot's palm (a skeleton on the plan "
                   "stage); and, once fitted in 3D, the object's motion in the robot "
                   "hand's frame, placed as the World section's training setting says",
                   done=bool(episode.arrays(HAND_SIM) or episode.arrays(SIM)))
        self._sim(episode)
        imgui.SeparatorText("6  follow the hand")
        self._step(episode, "follow", "Follow", follow_hand, self._blocked_follow(episode),
                   "the trained hand-motion policy (Learning, learn: hand motion) takes the "
                   "shape of the person's hand through the recording - in the hand sim, no "
                   "object, no retraining; the plan stage replays it under the skeleton",
                   done=bool(episode.arrays(FOLLOW)))
        frames = {FOLLOW: self._follow_result(episode)}
        imgui.SeparatorText("7  plan in sim")
        self._step(episode, "plan", "Plan", plan, self._blocked_plan(episode),
                   "the robot hand, with its servos and the World section's mount, "
                   "searches for a grip on the object, then follows the motion sent "
                   "to sim - no training; about 20 s",
                   done=bool(episode.arrays(PLAN)))
        frames[PLAN] = self._sim_result(episode, PLAN, "planned")
        imgui.SeparatorText("8  run the trained policy")
        self._step(episode, "track", "Run", track_with_tracker, self._blocked_track(episode),
                   "the trained policy (Learning > Runs) follows this motion - no retraining, "
                   "the one-shot test; the grip is step 7's when current, else searched "
                   "(about 10 s)",
                   done=bool(episode.arrays(TRACK)))
        frames[TRACK] = self._sim_result(episode, TRACK, "run")
        self._stage(episode, frames)
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
        self.show_plan(None)
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
                back = episode.back_image(self._index)
                if back is not None:
                    self._rgb = _beside(self._rgb, self._back_overlay(episode, back))
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

    def _back_overlay(self, episode, rgb):
        """The back camera's frame with its tracked hands, mirrored like the
        front's preview."""
        hands = episode.back_hands_at(self._index) if self.show["skeleton"] else []
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
        """What was sent to sim, and the skeleton and ghost box in the sim."""
        hand, sim = episode.arrays(HAND_SIM), episode.arrays(SIM)
        if not hand and not sim:
            self.show_demo(None)
            self.show_skeleton(None)
            return
        imgui.SameLine()
        _, self.in_sim = imgui.Checkbox("show in sim", self.in_sim)
        imgui.SetItemTooltip("the person's hand stands on the plan stage beside the hand as "
                             "a skeleton, a ghost box at the hand follows this recording's "
                             "object, and the plan stage replays step 6, 7 or 8 - all at "
                             "the player's frame")
        index = self._index
        if hand:
            mirrored = " (mirrored onto the robot's left)" if str(hand["side"]) == "Right" else ""
            imgui.TextDisabled(f"hand: {int(hand['present'].sum())} frames of the "
                               f"{str(hand['side']).lower()} hand{mirrored}")
            if HAND_SIM in stale(episode):
                imgui.TextColored(AMBER, f"out of date - {stale(episode)[HAND_SIM]}")
            shown = self.in_sim and hand["present"][index]
            self.show_skeleton(hand["skeleton"][index].astype(float) if shown else None)
        else:
            self.show_skeleton(None)
        if not sim:
            imgui.TextDisabled("object: fit 3D, then Send to sim again")
            self.show_demo(None)
            return
        note = ("" if str(sim["hand"]) == "Left" else
                " - recorded with the right hand: not yet mirrored for the robot's "
                "left hand")
        imgui.TextDisabled(f"object: {int(sim['valid'].sum())} frames in the robot hand's "
                           f"frame{note}")
        if SIM in stale(episode):
            imgui.TextColored(AMBER, f"out of date - {stale(episode)[SIM]}")
        if self.in_sim and sim["valid"][index]:
            self.show_demo((sim["position"][index].astype(float),
                            sim["quat"][index].astype(float)))
        else:
            self.show_demo(None)

    @staticmethod
    def _blocked_follow(episode):
        """Follow sends the hand to sim afresh itself; it needs the hands and
        a hand-motion run."""
        if not (episode.clean_hands or episode.hands):
            return "the recording has no hands"
        if chosen_run("hand motion") is None:
            return "no hand-motion run yet (Learning panel, learn: hand motion)"
        return ""

    def _follow_result(self, episode):
        """Step 6's summary and errors; returns the plan stage's frame at the
        player's frame - the hand alone - or None."""
        result = episode.arrays(FOLLOW)
        if not result:
            return None
        if FOLLOW in stale(episode):
            imgui.TextColored(AMBER, f"out of date - {stale(episode)[FOLLOW]}")
        imgui.TextDisabled(f"hand-motion policy: run {Path(str(result['run'])).name}, "
                           f"{result['checkpoint']}")
        numbers = follow_summary(result)
        imgui.TextDisabled(f"following: finger segments {numbers['angle_deg']:.0f} deg, "
                           f"fingertips {numbers['tip_mm']:.0f} mm off (median) - at rest it "
                           f"would be {numbers['still_angle_deg']:.0f} deg, "
                           f"{numbers['still_tip_mm']:.0f} mm")
        times = [float(t) for t in result["time"]]
        if implot.BeginPlot("##follow", imgui.Vec2(-1, 170)):
            implot.SetupAxes("s", "deg / mm")
            for key, label, scale in (("angle_error", "segments off (deg)", np.degrees(1)),
                                      ("tip_error", "tips off (mm)", 1000),
                                      ("still_angle_error", "at rest: segments off (deg)",
                                       np.degrees(1)),
                                      ("still_tip_error", "at rest: tips off (mm)", 1000)):
                implot.PlotLine(label, times, [float(v) for v in result[key] * scale])
            self._plot_cursor(episode, 3)
            implot.EndPlot()
        moment = float(episode.times[self._index])
        row = int(np.clip(np.searchsorted(result["time"], moment), 0, len(result["time"]) - 1))
        return result["qpos"][row].astype(float), None, None

    @staticmethod
    def _blocked_plan(episode):
        """Plan and Run send to sim afresh themselves when it is missing or
        out of date; they need the 3D fit, and a left hand."""
        if not episode.arrays(POSE3D):
            return "fit 3D first"
        sim = episode.arrays(SIM)
        if sim and str(sim["hand"]) != "Left":
            return "recorded with the right hand: not yet mirrored for the robot's left hand"
        return ""

    @classmethod
    def _blocked_track(cls, episode):
        if cls._blocked_plan(episode):
            return cls._blocked_plan(episode)
        if chosen_run() is None:
            return "no trained policy yet (Learning panel > Train)"
        return ""

    def _stage(self, episode, frames):
        """Which result the plan stage replays - step 6's or step 7's - and
        show it, or hide the stage: a result out of date is not shown (it
        was made for where the object was), nor anything while Plan or Run
        works on this recording."""
        if self.job.busy and self.job.label in (f"plan {episode.folder}",
                                                f"track {episode.folder}",
                                                f"follow {episode.folder}"):
            self.show_plan(None)
            return
        out_of_date = stale(episode)              # made for another placement or motion
        shown = {name: frame for name, frame in frames.items()
                 if frame is not None and name not in out_of_date}
        if len(shown) > 1:
            imgui.TextDisabled("plan stage shows:")
            for name, label in ((FOLLOW, "hand follow"), (PLAN, "plan"),
                                (TRACK, "trained policy")):
                if name not in shown:
                    continue
                imgui.SameLine()
                if imgui.RadioButton(label, self.stage_shows == name):
                    self.stage_shows = name
        name = self.stage_shows if self.stage_shows in shown else next(iter(shown), None)
        self.show_plan(shown[name] if self.in_sim and name else None)

    def _sim_result(self, episode, name, verb):
        """Step 6's or 7's summary and errors; returns the plan stage's frame
        at the player's frame, or None."""
        result = episode.arrays(name)
        if not result:
            return None
        if name in stale(episode):
            imgui.TextColored(AMBER, f"out of date - {stale(episode)[name]}")
        mount = settings.get("world.hand_orientation")
        if str(result["orientation"]) != mount:
            imgui.TextColored(AMBER, f"{verb} with the hand {result['orientation']}; "
                                     f"the World section has it {mount} now")
        if "run" in result:
            imgui.TextDisabled(f"trained policy: run {Path(str(result['run'])).name}, "
                               f"{result['checkpoint']}")
        numbers = plan_summary(result)
        imgui.TextDisabled(f"grip: the object slips {numbers['grip_slip_mm']:.0f} mm and "
                           f"turns {numbers['grip_turn_deg']:.0f} deg when let go")
        if numbers["dropped_at"] is None:
            imgui.TextColored(GREEN, "held it through the whole motion")
        else:
            imgui.TextColored(RED, f"lost it {numbers['dropped_at']:.1f} s into the motion")
        (position, most), (turn, most_turn) = numbers["position_mm"], numbers["turn_deg"]
        (still, _), (still_turn, _) = numbers["still_position_mm"], numbers["still_turn_deg"]
        imgui.TextDisabled(f"following: {position:.0f} mm, {turn:.0f} deg off (median; "
                           f"worst {most:.0f} mm, {most_turn:.0f} deg) - held still it "
                           f"would be {still:.0f} mm, {still_turn:.0f} deg")
        valid = episode.arrays(SIM)["valid"]
        start = float(episode.times[int(np.flatnonzero(valid)[0])]) if valid.any() else 0.0
        hold = float(result["hold"])
        following = result["time"] >= hold
        times = [float(t) for t in result["time"][following] - hold + start]
        if implot.BeginPlot(f"##{name}", imgui.Vec2(-1, 170)):
            implot.SetupAxes("s", "mm / deg")
            for key, label, scale in (("position_error", "off (mm)", 1000),
                                      ("turn_error", "off (deg)", np.degrees(1)),
                                      ("still_position_error", "held still: off (mm)", 1000),
                                      ("still_turn_error", "held still: off (deg)",
                                       np.degrees(1))):
                implot.PlotLine(label, times,
                                [float(v) for v in result[key][following] * scale])
            self._plot_cursor(episode, 2)
            implot.EndPlot()
        imgui.TextDisabled("how far the simulated object is from where the recording "
                           "has it, and how far it would be had it stayed put")
        moment = float(episode.times[self._index]) - start + hold
        row = int(np.clip(np.searchsorted(result["time"], moment), 0, len(result["time"]) - 1))
        return (result["qpos"][row].astype(float),
                (result["position"][row].astype(float), result["quat"][row].astype(float)),
                (result["goal_position"][row].astype(float), result["goal_quat"][row].astype(float)))

    def _plot_cursor(self, episode, line):
        """The player's frame as a line on the open plot; a click there
        moves the player."""
        implot.DragLineX(line, float(episode.times[self._index]), imgui.Vec4(1, 1, 1, 0.8), 1,
                         int(implot.DragToolFlags.NoInputs))
        if implot.IsPlotHovered() and imgui.IsMouseClicked(0):
            self._index = int(np.clip(np.searchsorted(episode.times, implot.GetPlotMousePos().x),
                                      0, episode.frames - 1))
            self._playing = False

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
            if not pose["boxed"][track]:
                imgui.TextDisabled(f"{name}: not fitted - not the object being moved")
                continue
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
            if not pose["boxed"][track]:
                continue
            implot.PlotLine(f"{name} - turn in palm (deg)##{track}", times,
                            [float(v) for v in pose["in_palm_turn"][track]])
            implot.PlotLine(f"{name} - out of palm (mm)##{track}", times,
                            [float(v) for v in 1000 * pose["in_palm_position"][track, :, 2]])
        if (pose["marker_id"] >= 0).any():
            implot.PlotLine("outline vs marker (mm)", times,
                            [float(v) for v in pose["outline_error_mm"]])
            implot.PlotLine("outline vs marker (deg)", times,
                            [float(v) for v in pose["outline_error_deg"]])
        self._plot_cursor(episode, 1)
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
        self._plot_cursor(episode, 0)
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


def _beside(front, back):
    """Two frames side by side, the back one scaled to the front's height."""
    height = front.shape[0]
    width = round(back.shape[1] * height / back.shape[0])
    return np.hstack([front, cv2.resize(back, (width, height), interpolation=cv2.INTER_AREA)])


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
