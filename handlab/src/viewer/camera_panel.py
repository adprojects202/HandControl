"""Camera: what the camera sees, with the tracked hands drawn on - and
recording it.

The camera and the hand tracker run on their own thread (src/vision); this
only shows the latest frame (see image.py). Settings are saved to
settings.json and applied to the running camera at once (see
Camera.reconfigure). Record writes an episode of the task named beside it
(src/recording); it stops by itself if the camera does.

Under the camera's name: its calibration (focal length, from pictures of a
checkerboard; src/vision/calibration.py) and *Calibrate...*, which opens a
window to take those pictures. The calibration is kept in settings.json by
camera name, and goes into every recording made with that camera.
"""

from __future__ import annotations

import cv2
import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings
from ..vision import calibration
from ..vision.devices import camera_names
from .form import RED, SettingsForm, row, table
from .image import GuiImage

GREEN = imgui.Vec4(0.45, 0.85, 0.45, 1.0)
CALIBRATE_HOW = (
    "Show the checkerboard - printed, or on a phone or tablet screen - so all "
    "of it is in view. Take a picture each time it is found, moving it between "
    "pictures: near and far, tilted left, right, up and down, and into each "
    "corner of the view. Then Calibrate, and Save.")

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
        self._session = None     # calibration.CalibrationSession while its window is open
        self._calibration_image = GuiImage()
        self._shown = (None, None)                    # what the window's image was drawn from
        self._calibration_rgb = None
        self.calibration_message = ""

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
        self._calibration_line()

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
            self._aruco_settings()
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

    def _aruco_settings(self):
        changed, on = imgui.Checkbox("ArUco markers", settings.get("vision.aruco"))
        imgui.SetItemTooltip("look for ArUco markers: drawn on the preview; Fit 3D takes "
                             "the object's pose from one stuck in the middle of a face")
        if changed:
            self.form.save("vision.aruco", on)
        if not settings.get("vision.aruco") or not table("aruco"):
            return
        row("dictionary", "the marker family printed")
        self.form.choice("vision.aruco_dictionary")
        low, high = settings.limits("vision.aruco_mm")
        row("printed (mm)", "the marker's black square, side, measured on the print\n"
                            f"{low:g} - {high:g}")
        self.form.field("vision.aruco_mm", fmt="%.1f")
        imgui.EndTable()

    def camera_name(self):
        index = settings.get("vision.camera_index")
        names = self._cameras or []
        return names[index] if index < len(names) else f"camera {index}"

    # ------------------------------------------------------------- calibration
    def _calibration_line(self):
        frame = self.camera.latest()
        name = self.camera_name()
        if frame is not None:
            height, width = frame.raw.shape[:2]
            known = calibration.for_camera(name, width, height)
        else:
            entry = settings.get("camera_calibration").get(name)
            known = calibration.Calibration.from_entry(entry) if entry else None
        imgui.AlignTextToFramePadding()
        if known is not None:
            imgui.TextDisabled(f"calibrated: focal {known.fx:.0f} px, error "
                               f"{known.rms_px:.2f} px ({known.made[:10]})")
        else:
            imgui.TextColored(imgui.Vec4(0.95, 0.75, 0.30, 1.0), "not calibrated")
        imgui.SameLine()
        if imgui.SmallButton("Calibrate..."):
            if self._session is not None:
                self._session.close()
            self._session = calibration.CalibrationSession(
                settings.get("vision.checkerboard"), settings.get("vision.checkerboard_square_mm"))
            self.calibration_message = ""
        imgui.SetItemTooltip("measure this camera's focal length and lens distortion "
                             "from pictures of a checkerboard")
        if self._session is not None:
            self._calibration_window(name, frame)

    def _calibration_window(self, name, frame):
        session = self._session
        imgui.SetNextWindowSize(imgui.Vec2(560, 720), int(imgui.Cond.FirstUseEver))
        visible, keep_open = imgui.Begin(f"Calibrate camera###calibrate", True)
        if visible:
            imgui.Text(name)
            if frame is None:
                imgui.TextColored(RED, "turn the camera on")
            else:
                session.offer(frame.raw)
                height, width = frame.raw.shape[:2]
                imgui.SameLine()
                imgui.TextDisabled(f"{width} x {height}")
            imgui.PushTextWrapPos(0.0)
            imgui.TextDisabled(CALIBRATE_HOW)
            imgui.PopTextWrapPos()
            self._board_size()
            self._calibration_view(session)
            self._calibration_controls(name, session)
        imgui.End()
        if not keep_open:
            session.close()
            self._session = None

    def _board_size(self):
        imgui.AlignTextToFramePadding()
        imgui.Text("inner corners")
        imgui.SetItemTooltip("corners where four squares meet, across and down: a board "
                             "of 10 x 7 squares has 9 x 6")
        for index in (0, 1):
            imgui.SameLine()
            imgui.SetNextItemWidth(60)
            self.form.field("vision.checkerboard", index)
        imgui.SameLine()
        imgui.Text("  square (mm)")
        imgui.SetItemTooltip("one square's width and height, measured with a ruler on "
                             "the board as shown - on a screen it may not be square")
        for index in (0, 1):
            imgui.SameLine()
            imgui.SetNextItemWidth(60)
            self.form.field("vision.checkerboard_square_mm", index, fmt="%.1f")
        square = tuple(settings.get("vision.checkerboard_square_mm"))
        if square != self._session.square:            # same pictures, new shape
            self._session.square, self._session.result = square, None
        if tuple(settings.get("vision.checkerboard")) != self._session.pattern:
            self._session.close()                     # a new board: start again
            self._session = calibration.CalibrationSession(
                settings.get("vision.checkerboard"), square)

    def _calibration_view(self, session):
        frame, corners = session.latest
        if frame is None or self.viewer is None:
            imgui.TextDisabled("looking for the board ...")
            return
        mirror = settings.get("vision.mirror")
        key = (id(frame), len(session.views), mirror)
        if self._shown != key:
            shown = frame.copy()
            for view in session.views:                # where pictures were taken
                hull = cv2.convexHull(view.reshape(-1, 2)).astype(np.int32)
                cv2.polylines(shown, [hull], True, (120, 200, 255), 2, cv2.LINE_AA)
            if corners is not None:
                cv2.drawChessboardCorners(shown, session.pattern, corners, True)
            self._calibration_rgb = cv2.flip(shown, 1) if mirror else shown
            self._shown = key
        self._calibration_image.draw(self.viewer, self._calibration_rgb)
        if corners is not None:
            imgui.TextColored(GREEN, "board found")
        else:
            imgui.TextDisabled("no board in view")

    def _calibration_controls(self, name, session):
        imgui.BeginDisabled(session.latest[1] is None)
        if imgui.Button("Take picture"):
            self.calibration_message = session.take()
        imgui.EndDisabled()
        imgui.SameLine()
        if imgui.Button("Undo"):
            session.undo()
        imgui.SameLine()
        if imgui.Button("Clear"):
            session.clear()
        imgui.SameLine()
        imgui.Text(f"{len(session.views)} pictures ({calibration.MIN_VIEWS}+ needed)")
        if self.calibration_message:
            imgui.TextColored(RED, self.calibration_message)
        imgui.BeginDisabled(session.busy or len(session.views) < calibration.MIN_VIEWS)
        if imgui.Button("Calibrate"):
            session.calibrate()
        imgui.EndDisabled()
        if session.busy:
            imgui.SameLine()
            imgui.TextDisabled("calibrating ...")
        if session.error:
            imgui.TextColored(RED, session.error)
        result = session.result
        if result is None:
            return
        quality = ("good" if result.rms_px < 0.5 else "usable" if result.rms_px < 1.0
                   else "poor - take more pictures, more varied")
        imgui.Text(f"board squares {result.square_mm[0]:.1f} x {result.square_mm[1]:.1f} mm")
        imgui.Text(f"focal {result.fx:.1f} x {result.fy:.1f} px, centre "
                   f"({result.cx:.0f}, {result.cy:.0f}), error {result.rms_px:.2f} px: "
                   f"{quality}")
        imgui.TextDisabled("distortion " + ", ".join(f"{k:.3f}" for k in result.distortion))
        if imgui.Button(f"Save for {name}"):
            calibration.save(name, result)
            self.calibration_message = ""
        imgui.SetItemTooltip("keep it in settings.json for this camera; recordings "
                             "made with it carry it from now on")

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
                recorder.start(settings.get("recording.task"), self.camera_name(),
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
