"""Camera: what the camera sees, with the tracked hands drawn on - and
recording it.

The camera and the hand tracker run on their own thread (src/vision); this
only shows the latest frame (see image.py). Settings are saved to
settings.json and applied to the running camera at once (see
Camera.reconfigure). Record writes an episode of the task named beside it
(src/recording); it stops by itself if the camera does.
"""

from __future__ import annotations

from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings
from ..vision.devices import camera_names
from .form import RED, SettingsForm, row, table
from .image import GuiImage

MIRROR_TIP = ("show the preview flipped left-right, like a mirror. Preview "
              "only: tracking and recordings use the camera's true image")
TASK_TIP = ("what the recording is of; recordings are kept per task, in "
            "data/tasks/<task>/<date-time>/")
CAMERA_TIP = ("the cameras this computer lists, rescanned when the list opens;\n"
              "a USB camera plugged in now shows up here once the system has it")
# Typed in: (setting, label, tooltip).
FIELDS = (
    ("vision.capture_width", "width (px)", "asked of the camera; it picks the nearest it has"),
    ("vision.capture_height", "height (px)", "asked of the camera; it picks the nearest it has"),
    ("vision.max_hands", "hands", "most hands tracked at once"),
    ("vision.min_detection", "find", "confidence needed to find a new hand, 0-1"),
    ("vision.min_tracking", "follow", "confidence needed to keep following a hand, 0-1;\n"
                                      "below it, the hand is searched for again"),
)


class CameraSection:
    title = "Camera"

    def __init__(self, camera, recorder, apply_vision):
        """`apply_vision()` takes the saved vision settings into the camera."""
        self.camera = camera
        self.recorder = recorder
        self.apply_vision = apply_vision
        self.form = SettingsForm(apply_vision)
        self.viewer = None
        self.preview = GuiImage()
        self._cameras = None     # names by camera number, or None if unknowable
        self._list_open = False

    def attach(self, viewer):
        self.viewer = viewer

    def draw(self):
        changed, on = imgui.Checkbox("on", settings.get("vision.enabled"))
        imgui.SetItemTooltip("the camera, and hand tracking on every frame")
        if changed:
            self.form.save("vision.enabled", on)
        imgui.SameLine()
        changed, mirror = imgui.Checkbox("mirror", settings.get("vision.mirror"))
        imgui.SetItemTooltip(MIRROR_TIP)
        if changed:
            self.form.save("vision.mirror", mirror)
        imgui.SameLine()
        self._camera_choice()

        if self.camera.error:
            imgui.PushTextWrapPos(0.0)
            imgui.TextColored(RED, self.camera.error)
            imgui.PopTextWrapPos()
            if imgui.Button("Retry"):
                self.apply_vision()
        else:
            imgui.TextDisabled(self.camera.status)

        self._record()
        frame = self.camera.latest()
        if frame is not None and self.viewer is not None:
            self.preview.draw(self.viewer, frame.image)

        if imgui.TreeNode("camera settings"):
            if table("camera"):
                for key, label, tip in FIELDS:
                    low, high = settings.limits(key)
                    row(label, f"{tip}\n{low:g} - {high:g}")
                    self.form.field(key, fmt="%.2f")
                imgui.EndTable()
            imgui.TreePop()
        self.form.draw_message()

    def _camera_choice(self):
        """Pick the camera by name where the system gives names; by number
        elsewhere."""
        imgui.SetNextItemWidth(-1)
        index = settings.get("vision.camera_index")
        if self._cameras is None and not self._list_open:
            self._cameras = camera_names()
        if self._cameras is None:                     # no names on this system
            self.form.field("vision.camera_index")
            imgui.SetItemTooltip("camera number: 0 is the first the system lists")
            return
        shown = (self._cameras[index] if index < len(self._cameras)
                 else f"camera {index} (not connected)")
        opened = imgui.BeginCombo("##camera", shown)
        if opened:
            if not self._list_open:                   # rescan as the list opens
                self._cameras = camera_names()
            if not self._cameras:
                imgui.TextDisabled("no camera found")
            for number, name in enumerate(self._cameras):
                if imgui.Selectable(f"{name}##{number}", number == index) \
                        and number != index:
                    self.form.save("vision.camera_index", number)
            imgui.EndCombo()
        else:
            imgui.SetItemTooltip(CAMERA_TIP)
        self._list_open = opened

    def _record(self):
        recorder = self.recorder
        if recorder.active and not self.camera.running:
            recorder.stop()                       # the camera went off
        if recorder.active:
            if imgui.Button("Stop"):
                recorder.stop()
            imgui.SameLine()
            imgui.TextColored(RED, f"recording  {recorder.seconds:.1f} s, "
                                   f"{recorder.frames} frames")
        else:
            imgui.BeginDisabled(self.camera.latest() is None)
            if imgui.Button("Record"):
                index = settings.get("vision.camera_index")
                names = self._cameras or []
                name = names[index] if index < len(names) else f"camera {index}"
                recorder.start(settings.get("recording.task"), name,
                               settings.section("vision"))
            imgui.EndDisabled()
            imgui.SetItemTooltip("record the camera, the tracked hands and the "
                                 "robot as an episode of this task")
            imgui.SameLine()
            imgui.AlignTextToFramePadding()
            imgui.Text("task")
            imgui.SetItemTooltip(TASK_TIP)
            imgui.SameLine()
            imgui.SetNextItemWidth(-1)
            self.form.text("recording.task")
        if recorder.error:
            imgui.TextColored(RED, recorder.error)
        elif not recorder.active and recorder.folder is not None:
            imgui.TextDisabled(f"saved {recorder.folder.parent.name}/"
                               f"{recorder.folder.name}")
