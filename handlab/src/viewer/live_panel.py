"""Live: the robot hand following your hand, as the camera sees it - the
trained hand-motion policy driving the joint targets (learning/live.py)
instead of the sliders. In the Vision panel, under Camera: live instead of
recording.

*Start live* turns the camera on (and *Stop live* off again, if it was off)
and follows the run picked beside it (Learning, learn: hand motion). *input*
says what the policy is given: your hand through the mapping (retarget.py,
fitted to your Mimic sessions - *Fit mapping*, with how it did on the part
it was not fitted to), or your hand's own shape. The
purple command hand shows what it asks for; the real hand only moves while
streaming (Hardware section), slew-limited as always. While live, the Joints
sliders are off.
"""

from __future__ import annotations

from mujoco.experimental.dear_imgui import dear_imgui as imgui

import threading

from .. import settings
from ..learning import mimic, retarget
from .form import RED, SettingsForm, row, table
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
        self._fitting = None             # the thread fitting the mapping, while it runs
        self._fit_error = ""
        self._fitted = None              # (made, report line) of the saved mapping

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
        imgui.BeginDisabled(live.running)
        if table("live"):
            row("input", "mapping - your hand through the mapping fitted to your Mimic "
                         "sessions: the robot pose you mean, pinches made\ncamera shape - "
                         "your hand's own shape (the robot's is built otherwise: pinches "
                         "stay open)")
            self.form.choice("learning.live_input")
            imgui.EndTable()
        imgui.EndDisabled()
        if settings.get("learning.live_input") == "mapping":
            self._mapping()
        if live.error:
            imgui.PushTextWrapPos(0.0)
            imgui.TextColored(RED, live.error)
            imgui.PopTextWrapPos()
        elif live.running:
            angle = f" - shape {live.angle_deg:.0f} deg off" if live.side else ""
            touching = (f" - pinching {', '.join(live.touching)}"
                        if live.mapping is not None and live.touching else "")
            imgui.TextDisabled(f"{live.status}{angle}{touching}")
        self.form.draw_message()

    def _mapping(self):
        """The mapping's state, and Fit mapping."""
        fitting = self._fitting is not None and self._fitting.is_alive()
        imgui.BeginDisabled(fitting or self.live.running or not mimic.sessions())
        if imgui.Button("Fitting the mapping ..." if fitting else "Fit mapping"):
            self._fit_error = ""
            self._fitting = threading.Thread(target=self._fit, daemon=True)
            self._fitting.start()
        imgui.EndDisabled()
        imgui.SetItemTooltip("fit the mapping to every Mimic session (about 30 s); "
                             "Live uses it from its next start")
        made = retarget.MODEL.stat().st_mtime if retarget.MODEL.exists() else None
        if self._fitted is None or self._fitted[0] != made:
            fitted = retarget.report(retarget.load())
            self._fitted = (made, (
                f"fitted to {fitted['pairs']} pairs; on held-out ones: joints "
                f"{fitted['joints_deg']:.0f} deg off, pinches made {fitted['touches_made_when_meant']:.0%}"
                f" (false {fitted['touches_made_when_not']:.0%})") if fitted else "not fitted yet")
        imgui.SameLine()
        imgui.TextDisabled(self._fitted[1])
        if self._fit_error:
            imgui.TextColored(RED, self._fit_error)

    def _fit(self):
        try:
            retarget.fit()
        except Exception as failure:         # shown under the button
            self._fit_error = str(failure) or type(failure).__name__

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
