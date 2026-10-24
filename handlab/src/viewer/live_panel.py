"""Live: the robot hand following your hand, as the camera sees it - the
trained hand-motion policy driving the joint targets (learning/live.py)
instead of the sliders.

*Start live* turns the camera on (and *Stop live* off again, if it was off)
and follows the run picked beside it (Learning, learn: hand motion). The
purple command hand shows what it asks for; the real hand only moves while
streaming (Hardware section), slew-limited as always. While live, the Joints
sliders are off.
"""

from __future__ import annotations

from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings
from .form import RED, SettingsForm
from .learning_panel import run_choice

HAND = "hand motion"
START_TIP = ("turn the camera on and have the trained hand-motion policy follow your "
             "hand: its joint targets drive the purple command hand. The real hand "
             "only moves while streaming (Hardware section)")


class LiveSection:
    title = "Live"

    def __init__(self, live, apply_vision):
        """`live`: learning.live.LiveHand; `apply_vision()` takes the saved
        vision settings into the camera."""
        self.live = live
        self.form = SettingsForm(apply_vision)
        self._camera_was_off = False

    def draw(self):
        live = self.live
        if live.running:
            if imgui.Button("Stop live"):
                self.stop()
        else:
            if imgui.Button("Start live"):
                self._start()
            imgui.SetItemTooltip(START_TIP)
        imgui.SameLine()
        imgui.BeginDisabled(live.running)
        run_choice(self.form, HAND, width=1.0)
        imgui.EndDisabled()
        if live.error:
            imgui.PushTextWrapPos(0.0)
            imgui.TextColored(RED, live.error)
            imgui.PopTextWrapPos()
        elif live.running:
            angle = f" - shape {live.angle_deg:.0f} deg off" if live.side else ""
            imgui.TextDisabled(f"{live.status}{angle}")
        self.form.draw_message()

    def _start(self):
        self._camera_was_off = not settings.get("vision.enabled")
        if self._camera_was_off:
            self.form.save("vision.enabled", True)
        if not self.live.start() and self._camera_was_off:
            self.form.save("vision.enabled", False)

    def stop(self):
        self.live.stop()
        if self._camera_was_off:
            self.form.save("vision.enabled", False)
            self._camera_was_off = False
