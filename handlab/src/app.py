"""Wires the pieces together: world, controller, hardware bus, camera,
assistant, window.

The control loop, once per sim tick:

    sliders -> controller.targets
    controller.step(dt) -> command ---> purple command hand
                                   \\--> hardware bus (written when armed + sending)
    hardware bus encoders ----------> white encoder hand
    (no reading: nothing connected) -> white encoder hand at zero

Beside it, on threads of their own: the camera tracks the human hand
(src/vision) and can record it - frames, hands and the robot's state - as an
episode of a task (src/recording). Steps then run on a recording: the
assistant has a vision-language model say what task it shows, as structured
data (src/assistant); segmenting tracks each object it names
(src/processing). None of it moves the robot.
"""

from __future__ import annotations

import numpy as np

from . import settings
from .assistant import Assistant
from .control import JointController
from .hardware import HardwareBus
from .viewer import studio
from .viewer.assistant_panel import AssistantSection
from .viewer.camera_panel import CameraSection
from .viewer.recordings_panel import RecordingsSection
from .viewer.hardware_panel import HardwareSection
from .viewer.joints_panel import JointsSection
from .viewer.panel import SidePanel
from .viewer.world_panel import WorldSection
from .processing import Job
from .recording import Recorder
from .vision import Camera
from .world import World


class App:
    def __init__(self):
        self.world = World()
        self.robot = self.world.robot
        self.controller = JointController(self.robot)
        self.zero = np.zeros(len(self.robot.names))
        self.bus = HardwareBus(self.robot.names)
        self.camera = Camera()
        self.recorder = Recorder(self.robot.names, self.robot_state)
        self.camera.recorder = self.recorder
        self.assistant = Assistant()
        self.processing = Job()             # steps run on a saved recording
        # Asked for by the GUI, done by tick: the sim thread owns the model.
        self._apply_world = False
        self._reset_object = False
        self._demo = None           # the demo ghost's pose in the mount frame, or None

    def tick(self, dt):
        if self._apply_world:
            self._apply_world = False
            self.world.apply()
        if self._reset_object:
            self._reset_object = False
            self.world.reset_object()
        self.world.set_demo(self._demo)
        command = self.controller.step(dt)
        self.bus.send(command)
        self.world.set_pose("command", command)
        # Without a reading (nothing connected, or after a disconnect) the
        # encoder hand is shown at zero: an unposed hand is left to physics
        # and drifts, and a last reading would look live when it is not.
        encoders = self.bus.encoders
        self.world.set_pose("encoder", self.zero if encoders is None else encoders)

    def start_streaming(self):
        """Begin sending the command, starting from where the hand IS."""
        encoders = self.bus.encoders
        if encoders is None:
            self.bus.error = "no encoder reading yet - cannot start from the hand's pose"
            return
        self.controller.prime(encoders)
        self.bus.send(self.controller.command())
        self.bus.request("stream_on")

    def apply_world(self):
        """Take the saved world and object settings in, on the next tick."""
        self._apply_world = True

    def reset_object(self):
        """Put the object back where it is placed, on the next tick."""
        self._reset_object = True

    def show_demo(self, pose):
        """Show a recording's object in the sim as the demo ghost - (position,
        quaternion) in the hand mount's frame - or hide it (None)."""
        self._demo = pose

    def apply_settings(self):
        """Push saved settings into the running parts (after a GUI edit)."""
        self.controller.max_speed = settings.get("control.max_speed")
        self.bus.reconfigure(settings.section("hardware"))

    def robot_state(self):
        """(command, encoders or None) now - sampled with each recorded frame."""
        return self.controller.command(), self.bus.encoders

    def apply_vision(self):
        """Take the saved vision settings into the camera (starts or stops it)."""
        self.camera.reconfigure(settings.section("vision"))

    def plugins(self):
        """What runs inside the window: the Robot panel on the right, the
        Vision panel on the left - both in Studio's side panes, never over
        the 3D view."""
        return [
            SidePanel("Robot", [
                JointsSection(self.robot, self.controller, lambda: self.bus.encoders),
                WorldSection(self.apply_world, self.reset_object),
                HardwareSection(self.robot, self.bus, self.start_streaming,
                                self.apply_settings),
            ]),
            SidePanel("Vision", [
                CameraSection(self.camera, self.recorder, self.apply_vision),
                RecordingsSection(self.assistant, self.processing, self.show_demo),
                AssistantSection(self.assistant, self.camera),
            ], beside="Options"),
        ]

    def run(self):
        self.bus.start()
        self.apply_vision()
        try:
            studio.run(self.world, self.tick, self.plugins())
        finally:
            self.recorder.stop()
            self.camera.close()
            self.bus.close()
