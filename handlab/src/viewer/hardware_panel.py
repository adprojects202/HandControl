"""Hardware: the device, its temperature, the two switches that make the
hand move, motion presets, config and the motor map.

Every edit goes through the settings form (`form.py`): checked, saved to
settings.json, committed on Enter or focus loss, then applied live, each the
safe way (see `hardware/bus.py`).

Torque and Send are separate on purpose. Torque on holds the hand where it
is. Send starts writing the joint command - and first snaps the targets to
where the hand already is, so flipping it moves nothing until a slider does.
"""

from __future__ import annotations

import time

from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings
from ..hardware import list_ports
from .form import RED, SettingsForm

AMBER = imgui.Vec4(0.98, 0.75, 0.25, 1.0)
PORT_REFRESH = 2.0       # s between serial-port scans while the list is open

# (setting, label, how it takes effect)
CONFIG = (
    ("hardware.baudrate", "baud", "reconnects"),
    ("hardware.command_hz", "write Hz", "goal writes per second - live"),
    ("hardware.read_hz", "read Hz", "encoder reads per second - live"),
    ("control.max_speed", "max rad/s", "command slew limit - live"),
    ("hardware.current_limit_ma", "current mA", "applies at the next Torque on"),
    ("hardware.profile_velocity", "profile vel", "0.229 rpm per count, 0 = unlimited "
                                                 "- written to the motors now"),
    ("hardware.profile_acceleration", "profile acc", "0 = unlimited - written to the motors now"),
    ("hardware.position_p", "P gain", "written to the motors now"),
    ("hardware.position_i", "I gain", "written to the motors now"),
    ("hardware.position_d", "D gain", "written to the motors now"),
    ("hardware.temp_derate_c", "derate C", "above this a joint's current cap falls "
                                           "linearly towards zero - live"),
    ("hardware.temp_cutoff_c", "cutoff C", "at this, torque off for the whole hand; "
                                           "re-arm once 5 C below - live"),
)
NEW_MOTOR = {"id": 0, "sign": 1, "zero": 2048}
LOOP = ("How a slider reaches the hand:\n"
        "  target -> clamp to URDF limits -> slew limit (max rad/s)\n"
        "  -> goal written at write Hz -> servo velocity profile\n"
        "  -> servo position PID (P/I/D) on its own encoder\n"
        "  -> current clipped at current mA.\n"
        "The software is open loop: encoders only draw the white hand.\n"
        "The closed loop is the PID inside each servo.")


class HardwareSection:
    title = "Hardware"

    def __init__(self, robot, bus, start_streaming, apply_settings):
        """`apply_settings()` pushes the saved settings into the running parts."""
        self.robot = robot
        self.bus = bus
        self.start_streaming = start_streaming
        self.form = SettingsForm(apply_settings)
        self._ports, self._scanned, self._list_open = [], 0.0, False

    # ------------------------------------------------------------------ draw
    def draw(self):
        state = self.bus.status()
        self._device(state)
        imgui.TextWrapped(state["note"])
        if state["error"]:
            imgui.PushTextWrapPos(0.0)
            imgui.TextColored(RED, state["error"])
            imgui.PopTextWrapPos()
        self.form.draw_message()
        if state["connected"]:
            self._temperature(state["thermal"])
        self._switches(state)
        imgui.Spacing()
        self._motion()
        if imgui.TreeNodeEx("Config", int(imgui.TreeNodeFlags.DefaultOpen)):
            self._config()
            imgui.TreePop()
        if imgui.TreeNodeEx("Motor map", 0):
            self._motor_map()
            imgui.TreePop()

    def _device(self, state):
        port = settings.get("hardware.port")
        options = [("", "none"), ("mock", "mock (echoes commands)")]
        options += [(device, device) for device, _ in self._ports]
        if port not in dict(options):
            options.append((port, f"{port} (not found)"))
        imgui.AlignTextToFramePadding()
        imgui.Text("device")
        imgui.SameLine()
        imgui.SetNextItemWidth(-1)
        imgui.BeginDisabled(state["streaming"])
        opened = imgui.BeginCombo("##device", dict(options)[port])
        if opened:
            for value, text in options:
                if imgui.Selectable(text, value == port) and value != port:
                    self.form.save("hardware.port", value)
            imgui.EndCombo()
        imgui.EndDisabled()
        if state["streaming"]:
            imgui.SetItemTooltip("stop sending to change device")
        self._scan_ports(opened)

    def _scan_ports(self, list_open):
        """Rescan when the list opens, and every PORT_REFRESH while it stays
        open - so a U2D2 plugged in now appears without a restart."""
        now = time.monotonic()
        if (list_open and not self._list_open) or not self._scanned \
                or (list_open and now - self._scanned > PORT_REFRESH):
            try:
                self._ports = list_ports()
            except Exception as failure:
                self.form.message = f"port scan failed: {failure}"
            self._scanned = now
        self._list_open = list_open

    def _switches(self, state):
        if not state["connected"]:
            imgui.BeginDisabled(not state["port"])
            if imgui.Button("Connect"):
                self.bus.request("connect")
            imgui.EndDisabled()
            return
        changed, armed = imgui.Checkbox("Torque on", state["armed"])
        if changed:
            self.bus.request("arm" if armed else "disarm")
        imgui.SameLine()
        imgui.BeginDisabled(not state["armed"])
        changed, streaming = imgui.Checkbox("Send", state["streaming"])
        if changed:
            if streaming:
                self.start_streaming()
            else:
                self.bus.request("stream_off")
        imgui.EndDisabled()
        imgui.SetItemTooltip("write the joint command to the motors")
        imgui.SameLine()
        if imgui.Button("Disconnect"):
            self.bus.request("disconnect")

    def _temperature(self, thermal):
        """One line: the hottest joint, coloured by what the guard is doing.
        Hover for every joint."""
        hot = thermal["hottest"]
        if hot is None:
            imgui.TextDisabled("temp --")
            return
        text = f"temp {hot[1]:.0f} C max ({hot[0].removesuffix('_joint')})"
        if thermal["level"] == "trip":
            imgui.TextColored(RED, text + " - CUT OFF")
        elif thermal["level"] == "derate":
            imgui.TextColored(AMBER, text + f" - derating {len(thermal['derated'])}")
        else:
            imgui.Text(text)
        imgui.SetItemTooltip("\n".join(
            f"{name.removesuffix('_joint'):22s} {'--' if c is None else f'{c:.0f} C'}"
            + ("  derated" if name in thermal["derated"] else "")
            for name, c in thermal["temperatures"].items()))

    def _motion(self):
        """Quick settings: one pick sets max speed and the servo profile."""
        presets = settings.get("motion_presets")
        active = settings.motion_preset()
        imgui.AlignTextToFramePadding()
        imgui.Text("motion")
        imgui.SetItemTooltip(LOOP)
        imgui.SameLine()
        imgui.SetNextItemWidth(-1)
        if imgui.BeginCombo("##motion", active or "custom"):
            for name, preset in presets.items():
                if imgui.Selectable(name, name == active) and name != active:
                    self._apply_preset(name)
                imgui.SetItemTooltip(
                    f"max {preset['max_speed']} rad/s, profile vel "
                    f"{preset['profile_velocity']}, acc {preset['profile_acceleration']}"
                    + ("  (0 = none)" if 0 in (preset['profile_velocity'],
                                                preset['profile_acceleration']) else ""))
            imgui.EndCombo()

    def _apply_preset(self, name):
        try:
            settings.apply_motion_preset(name)
        except (KeyError, ValueError) as failure:
            self.form.message = str(failure)
            return
        settings.save()
        self.form.message = ""
        self.form.on_saved()

    def _config(self):
        if not imgui.BeginTable("config", 2, int(imgui.TableFlags.SizingStretchProp)):
            return
        imgui.TableSetupColumn("name", int(imgui.TableColumnFlags.WidthFixed))
        imgui.TableSetupColumn("value", int(imgui.TableColumnFlags.WidthStretch))
        for key, label, effect in CONFIG:
            imgui.TableNextRow()
            imgui.TableNextColumn()
            imgui.AlignTextToFramePadding()
            imgui.Text(label)
            span = settings.limits(key)
            imgui.SetItemTooltip(effect + (f"\nrange {span[0]} .. {span[1]}" if span else ""))
            imgui.TableNextColumn()
            imgui.SetNextItemWidth(-1)
            if settings.choices(key):
                self.form.choice(key, show=lambda v: f"{v:,}")
            else:
                self.form.number(key, settings.get(key),
                                 lambda v, key=key: self.form.save(key, v))
        imgui.EndTable()

    def _motor_map(self):
        imgui.TextDisabled("per joint; reconnects to apply")
        motors = settings.get("hardware.motors")
        flags = int(imgui.TableFlags.SizingStretchProp) | int(imgui.TableFlags.RowBg)
        if not imgui.BeginTable("motors", 4, flags):
            return
        imgui.TableSetupColumn("joint", int(imgui.TableColumnFlags.WidthStretch))
        imgui.TableSetupColumn("id", int(imgui.TableColumnFlags.WidthFixed), 34)
        imgui.TableSetupColumn("rev", int(imgui.TableColumnFlags.WidthFixed))
        imgui.TableSetupColumn("zero", int(imgui.TableColumnFlags.WidthFixed), 46)
        imgui.TableHeadersRow()
        for joint in self.robot.joints:
            entry = motors.get(joint.name, NEW_MOTOR)

            def save(field, value, name=joint.name, entry=entry):
                self.form.save(f"hardware.motors.{name}", {**entry, field: value})

            imgui.TableNextRow()
            imgui.TableNextColumn()
            imgui.AlignTextToFramePadding()
            imgui.Text(f"{joint.group} {joint.label}")
            imgui.SetItemTooltip(joint.name + ("" if joint.name in motors
                                               else "\nnot in settings yet"))
            imgui.TableNextColumn()
            imgui.SetNextItemWidth(-1)
            self.form.number(f"{joint.name}.id", entry["id"],
                         lambda v, save=save: save("id", v))
            imgui.TableNextColumn()
            changed, reversed_ = imgui.Checkbox(f"##{joint.name}.sign",
                                                entry["sign"] == -1)
            imgui.SetItemTooltip("mounted reversed (sign -1): a positive joint "
                                 "angle needs a negative tick delta")
            if changed:
                save("sign", -1 if reversed_ else 1)
            imgui.TableNextColumn()
            imgui.SetNextItemWidth(-1)
            self.form.number(f"{joint.name}.zero", entry["zero"],
                         lambda v, save=save: save("zero", v))
        imgui.EndTable()
