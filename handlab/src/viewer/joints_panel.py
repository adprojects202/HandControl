"""Joints: one slider per URDF joint, grouped by finger.

Built from the robot description alone - names, grouping and slider ranges
all come from the URDF, so a different hand gets its own sliders with no code
change. A slider sets that joint's TARGET; the purple hand shows the command
that results, and the "robot" column reads the real encoders. While Live
follows a hand (live_panel.py), it sets the targets and the sliders are off.
"""

from __future__ import annotations

import math

from mujoco.experimental.dear_imgui import dear_imgui as imgui


class JointsSection:
    title = "Joints"

    def __init__(self, robot, controller, encoders, live=lambda: False):
        """`encoders()` returns the robot's joint angles, or None without one;
        `live()` whether Live sets the targets now."""
        self.robot = robot
        self.controller = controller
        self.encoders = encoders
        self.live = live

    def draw(self):
        robot_q = self.encoders()
        live = self.live()
        if live:
            imgui.TextDisabled("Live sets the targets - stop it to use the sliders")
        imgui.BeginDisabled(live)
        self._sliders(robot_q)
        imgui.EndDisabled()

    def _sliders(self, robot_q):
        if imgui.Button("Zero all"):
            self.controller.set_targets([0.0] * len(self.robot))
        imgui.SameLine()
        imgui.BeginDisabled(robot_q is None)
        if imgui.Button("Match robot"):
            self.controller.set_targets(robot_q)
        imgui.SetItemTooltip("Set every target to where the real hand is now")
        imgui.EndDisabled()

        targets = self.controller.targets()
        if not imgui.BeginTable("joints", 3, int(imgui.TableFlags.SizingStretchProp)):
            return
        imgui.TableSetupColumn("joint", int(imgui.TableColumnFlags.WidthFixed))
        imgui.TableSetupColumn("target", int(imgui.TableColumnFlags.WidthStretch))
        imgui.TableSetupColumn("robot", int(imgui.TableColumnFlags.WidthFixed))
        imgui.TableHeadersRow()
        for group, indices in self.robot.groups().items():
            imgui.TableNextRow()
            imgui.TableNextColumn()
            imgui.SeparatorText(group)
            for i in indices:
                joint = self.robot.joints[i]
                imgui.TableNextRow()
                imgui.TableNextColumn()
                imgui.AlignTextToFramePadding()
                imgui.Text(joint.label)
                imgui.SetItemTooltip(joint.name)
                imgui.TableNextColumn()
                imgui.SetNextItemWidth(-1)
                changed, value = imgui.SliderAngle(
                    f"##{joint.name}", float(targets[i]),
                    math.degrees(joint.lower), math.degrees(joint.upper),
                    "%.1f deg")
                if changed:
                    self.controller.set_target(i, value)
                imgui.TableNextColumn()
                imgui.Text("  --  " if robot_q is None
                           else f"{math.degrees(robot_q[i]):6.1f}")
        imgui.EndTable()
