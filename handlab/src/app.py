"""Wires the pieces together: world, controller, hardware bus, camera,
assistant, window.

The control loop, once per sim tick:

    sliders (or Live: the hand-motion policy following the camera's hand)
                                     -> controller.targets
    controller.step(dt) -> command ---> purple command hand
                                   \\--> hardware bus (written when armed + sending)
    hardware bus encoders ----------> white encoder hand
    (no reading: nothing connected) -> white encoder hand at zero

Beside it, on threads of their own: the camera tracks the human hand
(src/vision) and can record it - frames, hands and the robot's state - as an
episode of a task (src/recording). Steps then run on a recording: the
assistant has a vision-language model say what task it shows, as structured
data (src/assistant); segmenting tracks each object it names (Recordings,
steps 1-4). Real to Sim takes the hand and the object into the robot hand's
frame and has the robot do it in the sim (src/processing, src/sim). None of
it moves the robot; Live only sets the targets, which reach the hardware
only while streaming.

The panels: Robot on the right (Live, Joints, World, Hardware); Vision
(Camera, Recordings), Real to Sim and Learning on the left. Recordings and
Real to Sim show the same recording at the same frame (viewer/playback.py).
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
from .viewer.hardware_panel import HardwareSection
from .viewer.joints_panel import JointsSection
from .viewer.learning_panel import LibrarySection, RunsSection, TaskStatus, TrainSection
from .viewer.live_panel import LiveSection
from .viewer.panel import SidePanel
from .viewer.playback import Playback
from .viewer.real_to_sim_panel import RealToSimSection
from .viewer.recordings_panel import RecordingsSection
from .viewer.world_panel import WorldSection
from .learning.background import Background
from .learning.live import LiveHand
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
        self.back_camera = Camera(stream="back")   # the other side of the hand, if set
        self.recorder = Recorder(self.robot.names, self.robot_state)
        self.camera.recorder = self.back_camera.recorder = self.recorder
        self.assistant = Assistant()
        self.processing = Job()             # steps run on a saved recording
        self.learning = Background()        # library builds and training, as processes
        self.live = LiveHand(self.camera, self.controller.set_targets, self.robot.names)
        # Asked for by the GUI, done by tick: the sim thread owns the model.
        self._apply_world = False
        self._reset_object = False
        self._demo = None           # the demo ghost's pose in the mount frame, or None
        self._plan = None           # the plan stage's frame (World.set_plan), or None
        self._skeleton = None       # a person's hand on the stage (World.set_skeleton), or None
        self._stop_live = None      # Live's stop, once the panels are made

    def tick(self, dt):
        if self._apply_world:
            self._apply_world = False
            self.world.apply()
        if self._reset_object:
            self._reset_object = False
            self.world.reset_object()
        self.world.set_demo(self._demo)
        self.world.set_plan(self._plan)
        self.world.set_skeleton(self._skeleton)
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

    def show_plan(self, frame):
        """Show one moment of a planned task on the plan stage beside the hand
        (World.set_plan), or hide it (None)."""
        self._plan = frame

    def hide_recording(self):
        """Take a recording's object, plan stage and hand out of the sim."""
        self._demo = self._plan = self._skeleton = None

    def show_skeleton(self, points):
        """Show a person's hand on the plan stage - its landmarks (21, 3) m
        in the hand mount's frame - or hide it (None)."""
        self._skeleton = points

    def apply_settings(self):
        """Push saved settings into the running parts (after a GUI edit)."""
        self.controller.max_speed = settings.get("control.max_speed")
        self.bus.reconfigure(settings.section("hardware"))

    def robot_state(self):
        """(command, encoders or None) now - sampled with each recorded frame."""
        return self.controller.command(), self.bus.encoders

    def apply_vision(self):
        """Take the saved vision settings into the cameras (starts or stops
        them): the back camera is the same settings with its own number, on
        when one is set and it is not the front camera."""
        vision = settings.section("vision")
        self.camera.reconfigure(vision)
        back = vision["back_camera_index"]
        self.back_camera.reconfigure(dict(
            vision, camera_index=back,
            enabled=vision["enabled"] and back >= 0 and back != vision["camera_index"]))

    def plugins(self):
        """What runs inside the window: the Robot panel on the right, the
        Vision, Real to Sim and Learning panels on the left - all in Studio's
        side panes, never over the 3D view."""
        status = TaskStatus(self.learning)
        playback = Playback(on_open=lambda episode: self.hide_recording())
        live = LiveSection(self.live, self.apply_vision)
        self._stop_live = live.stop
        return [
            SidePanel("Robot", [
                live,
                JointsSection(self.robot, self.controller, lambda: self.bus.encoders,
                              live=lambda: self.live.running),
                WorldSection(self.apply_world, self.reset_object),
                HardwareSection(self.robot, self.bus, self.start_streaming,
                                self.apply_settings),
            ]),
            SidePanel("Vision", [
                CameraSection(self.camera, self.recorder, self.apply_vision,
                              self.back_camera),
                RecordingsSection(self.assistant, AssistantSection(self.assistant, self.camera),
                                  self.processing, playback),
            ], beside="Options"),
            SidePanel("Real to Sim", [
                RealToSimSection(self.processing, playback, self.zero, self.show_demo,
                                 self.show_plan, self.show_skeleton),
            ], beside="Vision"),
            SidePanel("Learning", [
                LibrarySection(status),
                TrainSection(status),
                RunsSection(status, Job()),
            ], beside="Real to Sim"),
        ]

    def run(self):
        self.bus.start()
        self.apply_vision()
        try:
            studio.run(self.world, self.tick, self.plugins())
        finally:
            if self._stop_live is not None:
                self._stop_live()
            self.recorder.stop()
            self.camera.close()
            self.back_camera.close()
            self.bus.close()
