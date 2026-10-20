"""The thread that owns the serial port. Nothing else touches the bus.

Three states, separate on purpose because they carry different risks:

    connected   the port is open and the motors answered
    armed       torque on: the motors hold position and will act on goals
    streaming   the joint command is being written to them

`armed` reports the motors' torque, and these motors power up energised, so
a hand can be armed the moment it connects. The arm REQUEST still matters: it
writes mode, limits, profile and PID from settings before re-energising.

Only `connected` happens on its own (at startup, when settings.json names a
port). Arming and streaming are always explicit requests, and every bus error
drops streaming and records why - it never crashes the simulator.

Why a thread: reading the encoders costs ~16 ms, a visible stutter inside a
real-time sim loop. The loop hands over the latest command with `send` and
reads the latest encoder frame from `encoders`; it never waits for the bus.
Requests (`request("arm")` ...) are queued and run here, in order, so a slow
arm (~160 register writes) does not freeze the simulator either.

Settings changed at runtime come in through `reconfigure`, which applies each
one the safe way: a new port, baud rate or motor map reconnects (sending
stops, torque is left as it is); new response registers are written to the
motors at once; rates and temperature thresholds take effect on the next
loop; the current limit on the next arm, since it can only be written with
torque off.

Every encoder read also brings back current, voltage and temperature (one
transaction, see `dynamixel.read_state`). Temperatures feed the ThermalGuard
(`thermal.py`): hot joints are derated, and at the cutoff torque goes off.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .. import settings
from .dynamixel import open_link
from .thermal import COOL_MARGIN, ThermalGuard

READ_FAILURES_TO_FAIL = 10     # consecutive failed encoder reads before giving up
ACTIONS = ("connect", "disconnect", "arm", "disarm", "stream_on", "stream_off",
           "registers")
NO_PORT = "no device selected"
# Settings that only take effect on a fresh connection.
RECONNECT = ("port", "baudrate", "motors")
# RAM registers on the XL330: safe to rewrite while the hand is holding.
REGISTERS = ("profile_velocity", "profile_acceleration",
             "position_p", "position_i", "position_d")


class HardwareBus:
    def __init__(self, joint_names, config=None):
        self.joint_names = list(joint_names)
        self.config = config or settings.section("hardware")
        self.link = None
        self.connected = False
        self.armed = False
        self.streaming = False
        self.encoders = None        # latest joint angles from the robot, rad
        self.telemetry = None       # latest full read: see dynamixel.read_state
        self.thermal = ThermalGuard(self.joint_names, self.config["temp_derate_c"],
                                    self.config["temp_cutoff_c"])
        self.note = NO_PORT
        self.error = ""
        self._command = None
        self._requests = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._read_failures = 0

    # ------------------------------------------------------- from outside
    def start(self):
        self._thread = threading.Thread(target=self._run, name="handlab-bus",
                                        daemon=True)
        self._thread.start()
        if self.config["port"]:
            self.request("connect")
        return self

    def request(self, action, payload=None):
        if action not in ACTIONS:
            raise ValueError(f"unknown bus action {action!r}")
        self._requests.put((action, payload))

    def reconfigure(self, config):
        """New hardware settings; see the module docstring for what each does."""
        old, self.config = self.config, dict(config)
        self.thermal.derate_c = config["temp_derate_c"]      # next read applies it
        self.thermal.cutoff_c = config["temp_cutoff_c"]
        if any(old[key] != config[key] for key in RECONNECT):
            self.request("connect" if config["port"] else "disconnect")
            return
        changed = [key for key in REGISTERS if old[key] != config[key]]
        if changed and self.connected:
            self.request("registers", changed)

    def send(self, command):
        """The latest joint command. Written only while armed and streaming."""
        self._command = np.asarray(command, dtype=float).copy()

    def status(self):
        return {"port": self.config["port"], "connected": self.connected,
                "armed": self.armed, "streaming": self.streaming,
                "note": self.note, "error": self.error,
                "thermal": self.thermal.status()}

    def close(self):
        """Stop the thread and close the port. Torque is left as it is: the
        hand keeps holding its last goal rather than dropping what it holds."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._disconnect()

    # ------------------------------------------------------------- thread
    def _run(self):
        next_write = next_read = time.monotonic()
        while not self._stop.is_set():
            self._handle_requests()
            now = time.monotonic()
            busy = False
            if self.streaming and self.armed and self._command is not None \
                    and now >= next_write:
                next_write = now + 1.0 / self.config["command_hz"]
                self._guard(lambda: self.link.write_positions(self._command))
                busy = True
            if self.connected and now >= next_read:
                next_read = now + 1.0 / self.config["read_hz"]
                self._read()
                busy = True
            if not busy:
                time.sleep(0.001)

    def _handle_requests(self):
        while True:
            try:
                action, payload = self._requests.get_nowait()
            except queue.Empty:
                return
            if action == "connect":
                self._connect()
            elif action == "disconnect":
                self._disconnect()
                self.note = "disconnected" if self.config["port"] else NO_PORT
            elif not self.connected:
                self.error = f"{action}: not connected"
            elif action == "registers":
                self.link.config = self.config
                wrong = self._guard(lambda: self.link.write_registers(payload))
                if wrong:
                    self.error = "did not take: " + ", ".join(
                        f"{joint}.{field}" for joint, field in wrong[:3])
            elif action == "arm":
                if self.thermal.tripped:
                    hot = self.thermal.hottest()
                    self.error = (f"too hot to arm: {hot[0]} at {hot[1]:.0f} C - wait "
                                  f"for every joint below "
                                  f"{self.thermal.cutoff_c - COOL_MARGIN:.0f} C")
                    continue
                self.link.config = self.config      # current limit, profile, PID
                self.streaming = False
                here = self._guard(self.link.arm)
                if here is not None:
                    self.encoders = here
                self.armed = bool(self.link.torque_on)
                # arm rewrote every goal current to the full limit
                self._derate([(i, s) for i, s in enumerate(self.thermal.scale) if s < 1.0])
            elif action == "disarm":
                self.streaming = False
                self._guard(self.link.disarm)
                self.armed = bool(self.link.torque_on)
            elif action == "stream_on":
                if not self.armed:
                    self.error = "arm (torque on) before streaming"
                else:
                    self.error = ""
                    self.streaming = True
            elif action == "stream_off":
                self.streaming = False

    def _connect(self):
        self._disconnect()
        self.error = ""
        try:
            self.link = open_link(self.joint_names, self.config)
            if self.link is None:
                self.note = NO_PORT
                return
            self.note = f"connecting to {self.config['port']} ..."
            self.link.connect()
        except Exception as failure:          # serial, SDK, OS: all of them
            self.link = None
            self.note = "not connected"
            self.error = str(failure)
            return
        missing = [n for n, model in self.link.found.items() if model is None]
        self.connected = True
        self.armed = bool(self.link.torque_on)
        self._read_failures = 0
        self.note = (f"{len(self.joint_names) - len(missing)}/{len(self.joint_names)} motors"
                     + (f", missing {', '.join(missing)}" if missing else ""))

    def _disconnect(self):
        self.streaming = False
        if self.link is not None:
            try:
                self.link.close()
            except Exception:
                pass
        self.link = None
        self.connected = self.armed = False
        self.encoders = None

    def _read(self):
        try:
            state = self.link.read_state()
            self.telemetry = state
            self.encoders = state["position"]
            self._read_failures = 0
        except Exception as failure:
            # One dropped frame is not a dead bus; a run of them is.
            self._read_failures += 1
            self.error = str(failure)
            if self._read_failures >= READ_FAILURES_TO_FAIL:
                self.streaming = False
            return
        changed, trip = self.thermal.update(state["temperature"])
        if trip:
            self._trip()
        elif changed:
            self._derate(changed)

    def _trip(self):
        """Cutoff reached: stop sending and cut torque on the whole hand."""
        name, celsius = self.thermal.hottest()
        self.streaming = False
        if self.armed:
            self._guard(self.link.disarm)
            self.armed = bool(self.link.torque_on)
        self.error = (f"{name} reached {celsius:.0f} C (cutoff "
                      f"{self.thermal.cutoff_c:.0f} C) - torque off")

    def _derate(self, changed):
        """Write the new per-joint current caps (goal current, RAM)."""
        limit = self.config["current_limit_ma"]
        for index, scale in changed:
            name = self.joint_names[index]
            self._guard(lambda: self.link.set_goal_current(name, scale * limit))

    def _guard(self, call):
        """Run one bus call; on any failure stop streaming and say why.

        Any, not just DynamixelError: an unplugged adapter raises
        SerialException or OSError from under the SDK.
        """
        try:
            return call()
        except Exception as failure:
            self.streaming = False
            self.error = str(failure)
            return None
