"""Real to Sim: a recording brought into the sim, on the robot hand - what
the person's hand did, and what happened to the object.

The recording and its frame are the Recordings panel's (shared); the
timeline here moves them too. What is in the sim follows that frame:

    person's hand   the hand laid on the robot's palm, as a skeleton on the
                    plan stage beside the robot (World.set_skeleton)
    object          the recording's object, as a ghost box at the robot hand
                    (World.set_demo)
    robot           the plan stage's hand doing one of the results below -
                    following the hand, or moving the object by plan or by
                    trained policy (World.set_plan) - chosen under the timeline

HAND. Needs only the hands (Recordings, step 3). *Follow*: the trained
hand-motion policy (the run picked beside it; Learning, learn: hand motion)
takes the shape of the person's hand through the recording, in the hand sim
(processing/follow.py) - its error against the person's hand shape, next to
staying at rest.

OBJECT. *Fit 3D* (processing/pose3d.py) poses the object being moved and the
hand holding it in metres, from the camera's calibration; *placement* says
where on the robot hand the object goes (object.training: a fixed pose at
the World section's object position, or where the person held it, matched
to the robot's palm). Then the robot hand moves it in the task sim: *Plan*
(the planner, no training; processing/plan.py) or *Run policy* (the trained
tracker, the one-shot test; processing/track.py).

What is sent to sim - the hand (hand_sim.npz) and the object (sim.npz,
processing/transfer.py) - is kept current by itself: whenever either is
missing or out of date (the hands cleaned again, a new 3D fit, another
placement), it is sent again in the background. A result made from an
out-of-date one says so, and is not replayed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.implot import implot

from .. import settings
from ..learning import train
from ..processing.files import FOLLOW, HAND_SIM, PLAN, SIM, TRACK, stale
from ..processing.follow import follow_hand, summary as follow_summary
from ..processing.plan import plan, summary as plan_summary
from ..processing.pose3d import fit_3d, load as load_pose
from ..processing.track import track
from ..processing.transfer import transfer, transfer_hand
from ..vision.overlay import object_colour
from .assistant_panel import AMBER
from .form import GREEN, RED, SettingsForm, row, table
from .learning_panel import run_choice
from .playback import progress

HAND, TRACKER = "hand motion", "object tracking"
RESULTS = {FOLLOW: "following the hand", PLAN: "plan", TRACK: "trained policy"}
PLACEMENT_TIP = ("where on the robot hand the recording's object goes:\n"
                 "fixed pose - its motion, centred on the World section's object "
                 "position\nfrom the hand - where the person held it relative to their "
                 "palm, matched to the robot's palm (along the palm scaled to the robot's "
                 "size, out of the palm palm-surface to palm-surface)")


class RealToSimSection:
    title = "Real to Sim"

    def __init__(self, job, playback, rest, show_demo, show_plan, show_skeleton):
        """`rest`: the robot's joints at rest; `show_demo`, `show_plan`,
        `show_skeleton`: put the recording's object, the plan stage and the
        person's hand in the sim (None hides)."""
        self.job = job                   # processing.Job, shared with Recordings
        self.rest = np.asarray(rest, float)
        self.playback = playback
        self.show_demo, self.show_plan, self.show_skeleton = show_demo, show_plan, show_skeleton
        self.form = SettingsForm(lambda: None)
        self.show = {"person's hand": True, "object": True}
        self.stage_shows = FOLLOW
        self._sent = None                # what was last sent by itself (see _keep_sent)

    def draw(self):
        playback = self.playback
        playback.choose()
        playback.refresh()
        episode = playback.episode
        if episode is None or not episode.frames:
            self._hide()
            return
        playback.timeline("real to sim")
        self._keep_sent(episode)
        frames = {FOLLOW: self._follow_frame(episode), PLAN: self._result_frame(episode, PLAN),
                  TRACK: self._result_frame(episode, TRACK)}
        self._view(episode, frames)
        imgui.SeparatorText("hand")
        self._hand(episode)
        imgui.SeparatorText("object")
        self._object(episode)

    # ------------------------------------------------------------- the sim
    def _view(self, episode, frames):
        """What the sim shows, and which result the robot on the stage plays."""
        for name in self.show:
            _, self.show[name] = imgui.Checkbox(name, self.show[name])
            imgui.SameLine()
        imgui.NewLine()
        busy = self.playback.working(self.job, "follow", "plan", "track")
        out_of_date = stale(episode)
        shown = {name: frame for name, frame in frames.items()
                 if frame is not None and name not in out_of_date and not busy}
        imgui.AlignTextToFramePadding()
        imgui.TextDisabled("robot:")
        if not shown:
            imgui.SameLine()
            imgui.TextDisabled("nothing to replay yet - Follow, Plan or Run policy below")
        for name in shown:
            imgui.SameLine()
            if imgui.RadioButton(RESULTS[name], self.stage_shows == name):
                self.stage_shows = name
        if self.stage_shows not in shown:
            self.stage_shows = next(iter(shown), self.stage_shows)
        frame = shown.get(self.stage_shows)
        if frame is not None and not self.show["object"]:
            frame = (frame[0], None, None)
        index = self.playback.index
        hand = episode.arrays(HAND_SIM)
        self.show_skeleton(hand["skeleton"][index].astype(float)
                           if self.show["person's hand"] and hand and hand["present"][index]
                           else None)
        sim = episode.arrays(SIM)
        self.show_demo((sim["position"][index].astype(float), sim["quat"][index].astype(float))
                       if self.show["object"] and sim and sim["valid"][index] else None)
        if frame is None and hand and self.show["person's hand"]:
            frame = (self.rest, None, None)       # the stage stands, the robot at rest
        self.show_plan(frame)

    def _hide(self):
        self.show_skeleton(None)
        self.show_demo(None)
        self.show_plan(None)

    def _keep_sent(self, episode):
        """Send the hand and the object to sim when either is missing or out
        of date - on the job's thread, when it is free. A send that failed is
        not tried again until something it reads changes."""
        wanted = []
        if (episode.clean_hands or episode.hands) and (
                not episode.arrays(HAND_SIM) or HAND_SIM in stale(episode)):
            wanted.append("hand")
        if load_pose(episode) and (not episode.arrays(SIM) or SIM in stale(episode)):
            wanted.append("object")
        label = f"send {episode.folder}"
        progress(self.job, label, "send")
        if not wanted or self.job.busy:
            return
        key = (label, tuple(wanted), self.playback.version, settings.get("object.training"),
               tuple(settings.get("object.position_mm")))
        if key == self._sent:
            return
        self._sent = key

        def send(progress_):
            if "hand" in wanted:
                transfer_hand(episode, progress_)
            if "object" in wanted:
                transfer(episode, progress_)

        self.job.start(label, send, lambda _: self.playback.mark_changed(episode))

    # ------------------------------------------------------------- hand
    def _hand(self, episode):
        hand = episode.arrays(HAND_SIM)
        if not (episode.clean_hands or episode.hands):
            imgui.TextDisabled("the recording has no hands")
            return
        if hand:
            mirrored = " (mirrored onto the robot's left)" if str(hand["side"]) == "Right" else ""
            imgui.TextDisabled(f"the {str(hand['side']).lower()} hand{mirrored}, "
                               f"{int(hand['present'].sum())} frames, on the robot's palm")
        if not episode.clean_hands:
            imgui.TextColored(AMBER, "raw hands: clean them first for a steadier hand "
                                     "(Recordings, step 3)")
        run_choice(self.form, HAND)
        imgui.SameLine()
        self.playback.step(self.job, "follow", "Follow", follow_hand,
                           "" if train.chosen_run(HAND) else
                           "no hand-motion run yet (Learning, learn: hand motion)",
                           "the trained hand-motion policy takes the shape of the person's "
                           "hand through the recording, in the hand sim - no object, no "
                           "retraining; the robot on the stage replays it",
                           done=bool(episode.arrays(FOLLOW)))
        self._follow_result(episode)

    def _follow_frame(self, episode):
        result = episode.arrays(FOLLOW)
        if not result:
            return None
        row_ = int(np.clip(np.searchsorted(result["time"], self.playback.time()), 0,
                           len(result["time"]) - 1))
        return result["qpos"][row_].astype(float), None, None

    def _follow_result(self, episode):
        result = episode.arrays(FOLLOW)
        if not result:
            return
        if FOLLOW in stale(episode):
            imgui.TextColored(AMBER, f"out of date - {stale(episode)[FOLLOW]}")
        numbers = follow_summary(result)
        imgui.TextDisabled(f"run {Path(str(result['run'])).name}: finger segments "
                           f"{numbers['angle_deg']:.0f} deg, fingertips "
                           f"{numbers['tip_mm']:.0f} mm off (median) - at rest "
                           f"{numbers['still_angle_deg']:.0f} deg, "
                           f"{numbers['still_tip_mm']:.0f} mm")
        times = [float(t) for t in result["time"]]
        if implot.BeginPlot("##follow", imgui.Vec2(-1, 150)):
            implot.SetupAxes("s", "deg / mm")
            for key, label, scale in (("angle_error", "segments off (deg)", np.degrees(1)),
                                      ("tip_error", "tips off (mm)", 1000),
                                      ("still_angle_error", "at rest: segments (deg)",
                                       np.degrees(1)),
                                      ("still_tip_error", "at rest: tips (mm)", 1000)):
                implot.PlotLine(label, times, [float(v) for v in result[key] * scale])
            self.playback.plot_cursor(3)
            implot.EndPlot()

    # ------------------------------------------------------------- object
    def _object(self, episode):
        if table("placement"):
            row("placement", PLACEMENT_TIP)
            self.form.choice("object.training")
            imgui.EndTable()
        if settings.get("object.training") == "fixed pose":
            x, y, z = settings.get("object.position_mm")
            imgui.TextDisabled(f"centred at {x:.0f}, {y:.0f}, {z:.0f} mm (World section)")
        self.form.draw_message()
        self.playback.step(self.job, "pose3d", "Fit 3D", fit_3d, _blocked_3d(episode),
                           "the hand and the object in metres: the hand from MediaPipe's "
                           "shape, the object as a box of the World section's object size",
                           done=bool(load_pose(episode)))
        self._pose3d(episode)
        sim = episode.arrays(SIM)
        if sim:
            note = ("" if str(sim["hand"]) == "Left" else
                    " - recorded with the right hand: not yet mirrored for the robot's left")
            imgui.TextDisabled(f"{int(sim['valid'].sum())} frames on the robot hand{note}")
        imgui.Spacing()
        imgui.TextDisabled("the robot hand moves it in the task sim:")
        self.playback.step(self.job, "plan", "Plan", plan, _blocked_plan(episode),
                           "the planner: searches a grip on the object, then follows the "
                           "motion - no training; about 20 s",
                           done=bool(episode.arrays(PLAN)))
        imgui.SameLine()
        imgui.TextDisabled("planner, no training")
        self._result(episode, PLAN, "planned")
        run_choice(self.form, TRACKER)
        imgui.SameLine()
        self.playback.step(self.job, "track", "Run policy", track,
                           _blocked_plan(episode) or ("" if train.chosen_run() else
                                                      "no trained tracker yet (Learning)"),
                           "the trained tracker follows this motion - no retraining, the "
                           "one-shot test; the grip is the plan's when current, else "
                           "searched (about 10 s)",
                           done=bool(episode.arrays(TRACK)))
        self._result(episode, TRACK, "run")

    def _pose3d(self, episode):
        pose = load_pose(episode)
        if not pose:
            return
        size = " x ".join(f"{v:.0f}" for v in pose["box_mm"])
        for track_, name in enumerate(pose["names"]):
            if not pose["boxed"][track_]:
                continue
            fit = np.nanmedian(pose["object_fit"][track_])
            imgui.TextColored(imgui.Vec4(*(c / 255 for c in object_colour(track_)), 1.0),
                              f"{name} as a {size} mm box: outline fit {fit:.2f}; hand "
                              f"{pose['hand_scale'][track_]:.2f}x MediaPipe's size")
        marked = pose["marker_id"] >= 0
        if marked.any():
            compared = np.isfinite(pose["outline_error_mm"])
            versus = (f"; outline vs marker {np.median(pose['outline_error_mm'][compared]):.0f}"
                      f" mm, {np.median(pose['outline_error_deg'][compared]):.0f} deg"
                      if compared.any() else "")
            imgui.TextColored(imgui.Vec4(1.0, 0.24, 0.78, 1.0),
                              f"ArUco marker in {marked.mean():.0%} of frames{versus}")
        times = [float(t) for t in episode.times]
        if not implot.BeginPlot("##pose3d", imgui.Vec2(-1, 150)):
            return
        implot.SetupAxes("s", "deg / mm")
        for track_, name in enumerate(pose["names"]):
            if not pose["boxed"][track_]:
                continue
            implot.PlotLine(f"{name} - turn in palm (deg)##{track_}", times,
                            [float(v) for v in pose["in_palm_turn"][track_]])
            implot.PlotLine(f"{name} - out of palm (mm)##{track_}", times,
                            [float(v) for v in 1000 * pose["in_palm_position"][track_, :, 2]])
        self.playback.plot_cursor(1)
        implot.EndPlot()

    def _result_frame(self, episode, name):
        """The plan stage's frame for result `name` (plan or run) at the
        frame on show: (joints, object pose, goal pose), or None."""
        result, sim = episode.arrays(name), episode.arrays(SIM)
        if not result or not sim:
            return None
        valid = sim["valid"]
        start = float(episode.times[int(np.flatnonzero(valid)[0])]) if valid.any() else 0.0
        moment = self.playback.time() - start + float(result["hold"])
        row_ = int(np.clip(np.searchsorted(result["time"], moment), 0, len(result["time"]) - 1))
        return (result["qpos"][row_].astype(float),
                (result["position"][row_].astype(float), result["quat"][row_].astype(float)),
                (result["goal_position"][row_].astype(float),
                 result["goal_quat"][row_].astype(float)))

    def _result(self, episode, name, verb):
        """A plan's or a run's summary and errors."""
        result = episode.arrays(name)
        if not result:
            return
        if name in stale(episode):
            imgui.TextColored(AMBER, f"out of date - {stale(episode)[name]}")
        mount = settings.get("world.hand_orientation")
        if str(result["orientation"]) != mount:
            imgui.TextColored(AMBER, f"{verb} with the hand {result['orientation']}; "
                                     f"the World section has it {mount} now")
        if "run" in result:
            imgui.TextDisabled(f"run {Path(str(result['run'])).name}, {result['checkpoint']}")
        numbers = plan_summary(result)
        if numbers["dropped_at"] is None:
            imgui.TextColored(GREEN, "held it through the whole motion")
        else:
            imgui.TextColored(RED, f"dropped it {numbers['dropped_at']:.1f} s into the motion")
        imgui.SameLine()
        imgui.TextDisabled(f"(grip slips {numbers['grip_slip_mm']:.0f} mm, turns "
                           f"{numbers['grip_turn_deg']:.0f} deg when let go)")
        (position, most), (turn, most_turn) = numbers["position_mm"], numbers["turn_deg"]
        (still, _), (still_turn, _) = numbers["still_position_mm"], numbers["still_turn_deg"]
        imgui.TextDisabled(f"object {position:.0f} mm, {turn:.0f} deg off the recording "
                           f"(median; worst {most:.0f} mm, {most_turn:.0f} deg) - left "
                           f"where it started: {still:.0f} mm, {still_turn:.0f} deg")
        valid = episode.arrays(SIM)["valid"] if episode.arrays(SIM) else np.zeros(1, bool)
        start = float(episode.times[int(np.flatnonzero(valid)[0])]) if valid.any() else 0.0
        hold = float(result["hold"])
        following = result["time"] >= hold
        times = [float(t) for t in result["time"][following] - hold + start]
        if implot.BeginPlot(f"##{name}", imgui.Vec2(-1, 150)):
            implot.SetupAxes("s", "mm / deg")
            for key, label, scale in (("position_error", "off (mm)", 1000),
                                      ("turn_error", "off (deg)", np.degrees(1)),
                                      ("still_position_error", "left there: off (mm)", 1000),
                                      ("still_turn_error", "left there: off (deg)",
                                       np.degrees(1))):
                implot.PlotLine(label, times,
                                [float(v) for v in result[key][following] * scale])
            self.playback.plot_cursor(2)
            implot.EndPlot()


def _blocked_3d(episode):
    if episode.calibration is None:
        return ("calibrate the camera first (Camera section > Calibrate...): "
                f"{episode.meta.get('camera', {}).get('name', '?')}")
    if not episode.clean_hands or not episode.arrays("object.npz"):
        return "needs the hands and the object's motion (Recordings, steps 3 and 4)"
    return ""


def _blocked_plan(episode):
    if not load_pose(episode):
        return "fit 3D first"
    sim = episode.arrays(SIM)
    if sim and str(sim["hand"]) != "Left":
        return "recorded with the right hand: not yet mirrored for the robot's left hand"
    return ""
