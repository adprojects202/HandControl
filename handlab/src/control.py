"""Targets in, joint command out.

    targets   what was asked for (the sliders, later a policy or a retargeter)
    command   what actually goes out to the robot: the targets clamped to the
              URDF limits and slew-limited to `control.max_speed`

The purple hand in the world shows `command`, so what you see is exactly what
the hardware is sent. Thread-safe: the viewer writes targets while the sim
loop steps.
"""

from __future__ import annotations

import threading

import numpy as np

from . import settings

# A stalled loop must not buy one big step: dt is capped at this, seconds.
MAX_DT = 0.05


class JointController:
    def __init__(self, robot, max_speed=None):
        self.robot = robot
        self.max_speed = float(settings.get("control.max_speed")
                               if max_speed is None else max_speed)
        self._targets = np.zeros(len(robot))
        self._command = np.zeros(len(robot))
        self._lock = threading.Lock()

    def targets(self):
        with self._lock:
            return self._targets.copy()

    def command(self):
        with self._lock:
            return self._command.copy()

    def set_target(self, index, value):
        with self._lock:
            self._targets[index] = np.clip(value, self.robot.joints[index].lower,
                                           self.robot.joints[index].upper)

    def set_targets(self, q):
        with self._lock:
            self._targets = self.robot.clamp(q)

    def prime(self, q):
        """Jump targets AND command to `q` - where the robot already is - so
        starting to stream moves nothing until a target changes."""
        with self._lock:
            self._targets = self.robot.clamp(q)
            self._command = self._targets.copy()

    def step(self, dt):
        """Move the command toward the targets, at most max_speed * dt."""
        limit = self.max_speed * min(max(dt, 0.0), MAX_DT)
        with self._lock:
            self._command += np.clip(self._targets - self._command, -limit, limit)
            return self._command.copy()
