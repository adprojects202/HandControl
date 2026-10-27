"""Mimic: copy the robot's poses with your hand, so the hand-motion policy
can learn what the camera makes of them (learning/mimic.py).

*Start mimic* turns the camera on and takes over the plan stage beside the
robot: the robot there shows a pose, your hand stands on it as a skeleton
(laid on the robot's palm), and the line under the button says what to do
- copy it, then hold it. When the robot's fingertips touch, touch yours.
Your left hand is best (the robot is a left hand). *Stop* keeps what was
recorded so far; *Discard* drops it. Sessions are saved in
data/learning/mimic/ and used the next time a hand-motion run trains.
"""

from __future__ import annotations

import time

from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings
from ..learning import mimic
from ..learning.handmotion import followed, in_palm, mount_palm, on_robot
from .form import GREEN, RED, SettingsForm, row, table

SAY = {mimic.MOVE: "watch the robot", mimic.READY: "copy it with your hand",
       mimic.HOLD: "hold it"}
KIND = {"touch": "fingertips touch", "sign": "a hand sign", "curl": "fingers curled"}


class MimicSection:
    title = "Mimic"

    def __init__(self, camera, apply_vision, show_prompt, release, camera_name):
        """`show_prompt(frame, skeleton)` puts the pose to copy and your hand
        on the plan stage, ahead of anything else; `release()` gives the
        stage back; `camera_name()` names the camera."""
        self.camera = camera
        self.show_prompt, self.release = show_prompt, release
        self.camera_name = camera_name
        self.form = SettingsForm(apply_vision)
        self.session = None
        self.started = 0.0
        self.message = ""                # how the last session ended
        self.error = ""                  # why one could not start
        self._palm = None
        self._camera_was_off = False
        self._saved = None              # (count, minutes) of saved sessions, cached

    def draw(self):
        if self.session is None:
            self._idle()
        else:
            self._running()
        imgui.PushTextWrapPos(0.0)
        if self.error:
            imgui.TextColored(RED, self.error)
        elif self.message:
            imgui.TextDisabled(self.message)
        imgui.PopTextWrapPos()
        self.form.draw_message()

    def _idle(self):
        if imgui.Button("Start mimic"):
            self._start()
        imgui.SetItemTooltip("the robot beside the hand shows poses; copy each with your "
                             "hand (left is best) and hold it - when its fingertips touch, "
                             "touch yours. Teaches the hand-motion policy what the camera "
                             "makes of your hand")
        imgui.SameLine()
        if table("mimic"):
            low, high = settings.limits("learning.mimic_prompts")
            row("poses", f"poses in a session, {mimic.MOVE_S + mimic.READY_S + mimic.HOLD_S:.1f}"
                         f" s each\n{low} - {high}")
            self.form.field("learning.mimic_prompts")
            imgui.EndTable()
        if self._saved is None:
            paths = mimic.sessions()
            minutes = sum(float(mimic.load(p)["time"][-1]) for p in paths) / 60
            self._saved = (len(paths), minutes)
        count, minutes = self._saved
        imgui.TextDisabled(f"{count} session(s) saved, {minutes:.0f} min" if count else
                           "no sessions yet")

    def _running(self):
        now = time.monotonic() - self.started
        shown = self.session.target(now)
        frame = self.camera.latest()
        if frame is not None:
            self.session.add(frame, frame.time - self.started)
        if shown is None:
            self._finish(keep=True)
            return
        q, phase, number = shown
        hand = followed(frame.hands) if frame is not None else None
        skeleton = (on_robot(in_palm(hand.world.astype(float), hand.side), self._palm)
                    if hand is not None else None)
        self.show_prompt((q, None, None), skeleton)
        total = len(self.session.joints)
        imgui.ProgressBar(now / self.session.length_s, imgui.Vec2(-1, 0),
                          f"pose {number + 1} of {total}")
        colour = GREEN if phase == mimic.HOLD else imgui.Vec4(0.95, 0.75, 0.30, 1.0)
        imgui.TextColored(colour, f"{SAY[phase]} - {self.session.left(now):.1f} s - "
                                  f"{KIND[str(self.session.kinds[number])]}")
        if hand is None:
            imgui.TextColored(RED, "no hand in view")
        if imgui.Button("Stop"):
            self._finish(keep=True)
        imgui.SetItemTooltip("stop and keep what was recorded")
        imgui.SameLine()
        if imgui.Button("Discard"):
            self._finish(keep=False)

    def _start(self):
        self.message = self.error = ""
        try:
            prompted = mimic.prompts(settings.get("learning.mimic_prompts"))
        except RuntimeError as failure:
            self.error = str(failure)
            return
        self._palm = self._palm or mount_palm()
        self._camera_was_off = not settings.get("vision.enabled")
        if self._camera_was_off:
            self.form.save("vision.enabled", True)
        self.session = mimic.MimicSession(prompted)
        self.started = time.monotonic()

    def _finish(self, keep):
        session, self.session = self.session, None
        self.release()
        if self._camera_was_off:
            self.form.save("vision.enabled", False)
        if not keep:
            self.message = "discarded"
            return
        path = session.save(self.camera_name())
        held = [row_ for row_ in session.rows if row_[5] == mimic.HOLD]
        seen = sum(row_[3] for row_ in held) / max(1, len(held))
        self.message = (f"saved {path.name}: {len(session.rows)} frames, your hand seen in "
                        f"{seen:.0%} of the holds" if path else "nothing recorded")
        self._saved = None
