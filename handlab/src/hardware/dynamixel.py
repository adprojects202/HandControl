"""The hand's DYNAMIXEL XL330-M288 servos on one TTL bus, through a U2D2.

Measured on the bench hand, not taken from a datasheet:

  * The bus runs at 4 Mbaud. Single reads drop at that rate over an FTDI
    bridge (one motor answered a ping, then failed eleven reads in a row), so
    every transaction retries. Without it a healthy motor looks absent.
  * Operating mode 5, current-based position control: the current limit is
    what stops a finger that hit something from pulling until a part gives.
  * A sync write gets no reply, so `txPacket` returns while the frame is still
    on the wire. A read issued straight after collides with it on the
    half-duplex line and the write is lost - 20 of 20 at no gap, 0 at 0.5 ms.
    Hence WRITE_SETTLE after every goal write.
  * Reading 16 encoders costs ~16 ms (the macOS FTDI latency floor), so reads
    run slower than writes. See `bus.py`. The cost is the round trip, not the
    bytes, so one read brings back everything from present current to
    present temperature - see `read_state`.
  * The port is opened EXCLUSIVE (TIOCEXCL): once we have it, macOS refuses
    every later open. Two programs writing one half-duplex bus is two
    masters, and nothing on the wire would say so. It cannot catch a program
    that opened the port before us without the flag; close those first.

Joint angles are URDF radians throughout. `motors` in settings.json maps each
joint to its motor ID, mounting sign and zero tick.
"""

from __future__ import annotations

import fcntl
import math
import termios
import time

import numpy as np

PROTOCOL = 2.0
TICKS_PER_TURN = 4096
CURRENT_POSITION_MODE = 5
WRITE_SETTLE = 0.001     # s after a sync write before the bus is touched again

# XL330 control table: name -> (address, bytes)
CONTROL = {
    "model": (0, 2),
    "operating_mode": (11, 1),
    "current_limit": (38, 2),
    "torque_enable": (64, 1),
    "position_d": (80, 2),
    "position_i": (82, 2),
    "position_p": (84, 2),
    "goal_current": (102, 2),
    "profile_acceleration": (108, 4),
    "profile_velocity": (112, 4),
    "goal_position": (116, 4),
    "present_current": (126, 2),
    "present_velocity": (128, 4),
    "present_position": (132, 4),
    "present_voltage": (144, 2),
    "present_temperature": (146, 1),
}
# Two's complement on read-back: a motor holding still reports -3 mA.
SIGNED = {"present_current", "present_velocity", "present_position",
          "goal_position", "goal_current"}
RPM_PER_COUNT = 0.229                    # velocity registers
# One sync read covers this whole block, per motor.
STATE_START = CONTROL["present_current"][0]
STATE_END = CONTROL["present_temperature"][0] + CONTROL["present_temperature"][1]


class DynamixelError(RuntimeError):
    pass


def _signed(value, length):
    value = int(value)
    if value >= 1 << (8 * length - 1):
        value -= 1 << (8 * length)
    return value


def list_ports():
    """(device, description) for every USB serial port, likeliest first.

    macOS always offers a Bluetooth port and a debug console that open fine
    and answer nothing; those are left out, as are the tty. twins of cu. ports.
    """
    from serial.tools import list_ports as tools

    ports = []
    for port in tools.comports():
        device = port.device
        if "Bluetooth" in device or "debug-console" in device \
                or device.startswith("/dev/tty."):
            continue
        ports.append((device, port.description or device))
    return sorted(ports, key=lambda entry: "usb" not in entry[0].lower())


class DynamixelLink:
    """Sixteen servos, addressed by URDF joint name."""

    def __init__(self, joint_names, config):
        self.joint_names = list(joint_names)
        self.config = config
        self.port = config["port"]
        missing = [n for n in self.joint_names if n not in config["motors"]]
        if missing:
            raise DynamixelError("no motor in settings.json hardware.motors for: "
                                 + ", ".join(missing))
        self.motors = {n: config["motors"][n] for n in self.joint_names}
        for name, motor in self.motors.items():
            if motor["sign"] not in (1, -1):
                raise DynamixelError(f"{name}: sign must be 1 or -1")
        self.retries = 5
        self.torque_on = False
        self.connected = False
        self.found = {}
        self._port = self._packet = None

    # ---------------------------------------------------------- plumbing
    def connect(self):
        from dynamixel_sdk import PacketHandler, PortHandler

        self._port = PortHandler(self.port)
        try:
            # setBaudRate opens the port itself. The SDK examples call openPort
            # first, which opens it at the default baud and then again here.
            if not self._port.setBaudRate(self.config["baudrate"]):
                raise DynamixelError(f"cannot set {self.config['baudrate']} baud")
        except OSError as failure:              # SerialException is one
            if "busy" in str(failure).lower():
                raise DynamixelError(f"{self.port} is in use by another program "
                                     "- close it first") from failure
            raise DynamixelError(f"cannot open {self.port}: {failure}") from failure
        fcntl.ioctl(self._port.ser.fileno(), termios.TIOCEXCL)
        self._packet = PacketHandler(PROTOCOL)
        self.connected = True
        self.found = self.ping_all()
        # Ask whether the hand is energised rather than assume it is not: these
        # motors power up with torque on.
        self.torque_on = self._read_torque()
        return self

    def close(self):
        if self._port is not None:
            self._port.closePort()
        self.connected = False

    def read(self, motor_id, field):
        address, length = CONTROL[field]
        reader = {1: self._packet.read1ByteTxRx, 2: self._packet.read2ByteTxRx,
                  4: self._packet.read4ByteTxRx}[length]
        last = None
        for _ in range(self.retries):
            value, result, error = reader(self._port, motor_id, address)
            if result == 0 and error == 0:
                return _signed(value, length) if field in SIGNED else value
            last = (result, error)
        raise DynamixelError(f"no reply from id {motor_id} ({field}): {last}")

    def write(self, motor_id, field, value):
        address, length = CONTROL[field]
        writer = {1: self._packet.write1ByteTxRx, 2: self._packet.write2ByteTxRx,
                  4: self._packet.write4ByteTxRx}[length]
        last = None
        for _ in range(self.retries):
            result, error = writer(self._port, motor_id, address, int(value))
            if result == 0 and error == 0:
                return
            last = (result, error)
        raise DynamixelError(f"write {field}={value} to id {motor_id} failed: {last}")

    def ping_all(self):
        """joint -> model number, or None where nothing answered."""
        found = {}
        for name in self.joint_names:
            found[name] = None
            for _ in range(self.retries):
                model, result, _ = self._packet.ping(self._port, self.motors[name]["id"])
                if result == 0:
                    found[name] = model
                    break
        return found

    def _read_torque(self):
        for name in self.joint_names:
            if self.found.get(name) is None:
                continue
            try:
                return bool(self.read(self.motors[name]["id"], "torque_enable"))
            except DynamixelError:
                continue
        return False

    # ---------------------------------------------------------- torque
    def configure(self):
        """Mode, current limit, velocity profile and PID on every motor.
        Torque must be off to change the mode, so this leaves it off."""
        c = self.config
        for name in self.joint_names:
            motor_id = self.motors[name]["id"]
            self.write(motor_id, "torque_enable", 0)
            self.write(motor_id, "operating_mode", CURRENT_POSITION_MODE)
            self.write(motor_id, "current_limit", c["current_limit_ma"])
            self.write(motor_id, "goal_current", c["current_limit_ma"])
            self.write(motor_id, "profile_velocity", c["profile_velocity"])
            self.write(motor_id, "profile_acceleration", c["profile_acceleration"])
            self.write(motor_id, "position_p", c["position_p"])
            self.write(motor_id, "position_i", c["position_i"])
            self.write(motor_id, "position_d", c["position_d"])
        self.torque_on = False

    def write_registers(self, fields):
        """Rewrite RAM registers (profile, PID) from config on every motor,
        torque on or off, and read them back. Returns (joint, field) pairs
        that still disagree after one retry: on a bus that drops packets, a
        partial write is silent - one fingertip snapping while the rest ease."""
        wrong = []
        for name in self.joint_names:
            motor_id = self.motors[name]["id"]
            for field in fields:
                value = int(self.config[field])
                self.write(motor_id, field, value)
                if self.read(motor_id, field) != value:
                    self.write(motor_id, field, value)
                    if self.read(motor_id, field) != value:
                        wrong.append((name, field))
        return wrong

    def torque(self, on):
        for name in self.joint_names:
            self.write(self.motors[name]["id"], "torque_enable", 1 if on else 0)
        self.torque_on = bool(on)

    def arm(self):
        """Energise without jumping, and return where the hand is.

        Order matters: a goal register keeps whatever was last written, so
        enabling torque first snaps the hand to a goal from some earlier
        session. Goal := present position first, then torque on.

        Not quite motionless: `configure` has to turn torque off to set the
        mode, so for the ~160 writes it takes the hand is limp, and fingers
        resting against a wire or a neighbour settle - up to ~2.4 deg on the
        bench hand. They are held wherever they settled.
        """
        self.configure()
        here = self.read_positions()
        self.write_positions(here)
        self.torque(True)
        return here

    def disarm(self):
        self.torque(False)

    # ------------------------------------------------- joints <-> ticks
    def to_ticks(self, name, radians):
        motor = self.motors[name]
        return int(round(motor["zero"] + motor["sign"] * radians * TICKS_PER_TURN
                         / (2.0 * math.pi)))

    def to_radians(self, name, ticks):
        motor = self.motors[name]
        return motor["sign"] * (ticks - motor["zero"]) * 2.0 * math.pi / TICKS_PER_TURN

    def read_positions(self):
        return self.read_state()["position"]

    def read_state(self):
        """Everything worth knowing about every joint, in ONE sync read.

        Returns arrays in joint order: position in rad, velocity in rad/s,
        current in mA, voltage in V, temperature in deg C. A motor the sync
        read misses has its position read singly - that one has to be there -
        and NaN for the rest until the next frame.
        """
        from dynamixel_sdk import GroupSyncRead

        length = STATE_END - STATE_START
        group = GroupSyncRead(self._port, self._packet, STATE_START, length)
        for name in self.joint_names:
            group.addParam(self.motors[name]["id"])
        for _ in range(self.retries):
            if group.txRxPacket() == 0:
                break
        size = len(self.joint_names)
        state = {key: np.full(size, np.nan) for key in
                 ("position", "velocity", "current", "voltage", "temperature")}
        rad_s = RPM_PER_COUNT * 2.0 * math.pi / 60.0
        for index, name in enumerate(self.joint_names):
            motor_id = self.motors[name]["id"]
            sign = self.motors[name]["sign"]
            if not group.isAvailable(motor_id, STATE_START, length):
                state["position"][index] = self.to_radians(
                    name, self.read(motor_id, "present_position"))
                continue

            def field(key):
                address, size_ = CONTROL[key]
                value = group.getData(motor_id, address, size_)
                return _signed(value, size_) if key in SIGNED else value

            state["position"][index] = self.to_radians(name, field("present_position"))
            state["velocity"][index] = sign * field("present_velocity") * rad_s
            state["current"][index] = sign * field("present_current")
            state["voltage"][index] = field("present_voltage") / 10.0
            state["temperature"][index] = field("present_temperature")
        group.clearParam()
        return state

    def set_goal_current(self, name, milliamps):
        """Per-joint current cap, live: in mode 5 goal current is a RAM
        register bounding the position loop's output. Never above the limit."""
        milliamps = int(max(0, min(milliamps, self.config["current_limit_ma"])))
        self.write(self.motors[name]["id"], "goal_current", milliamps)

    def write_positions(self, q):
        """One sync write for the whole hand."""
        from dynamixel_sdk import GroupSyncWrite

        address, length = CONTROL["goal_position"]
        group = GroupSyncWrite(self._port, self._packet, address, length)
        for name, value in zip(self.joint_names, np.asarray(q, dtype=float)):
            ticks = max(0, min(TICKS_PER_TURN - 1, self.to_ticks(name, value)))
            group.addParam(self.motors[name]["id"], list(ticks.to_bytes(4, "little")))
        result = group.txPacket()
        group.clearParam()
        if result != 0:
            raise DynamixelError(f"sync write failed ({result})")
        time.sleep(WRITE_SETTLE)


class MockLink:
    """The same interface with no bus: encoders echo the last goal. Lets the
    whole command path run with no hand on the desk (port "mock")."""

    port = "mock"

    def __init__(self, joint_names, config=None):
        self.joint_names = list(joint_names)
        self.config = config
        self.goal = np.zeros(len(self.joint_names))
        self.temperature = np.full(len(self.joint_names), 30.0)   # tests heat it
        self.goal_current = {}
        self.torque_on = False
        self.connected = False
        self.found = {}

    def connect(self):
        self.connected = True
        self.found = {name: 1200 for name in self.joint_names}
        return self

    def close(self):
        self.connected = False

    def arm(self):
        self.torque_on = True
        return self.goal.copy()

    def disarm(self):
        self.torque_on = False

    def write_registers(self, fields):
        return []

    def read_positions(self):
        return self.goal.copy()

    def read_state(self):
        size = len(self.joint_names)
        return {"position": self.goal.copy(),
                "velocity": np.zeros(size), "current": np.zeros(size),
                "voltage": np.full(size, 5.0), "temperature": self.temperature.copy()}

    def set_goal_current(self, name, milliamps):
        self.goal_current[name] = int(milliamps)

    def write_positions(self, q):
        self.goal = np.asarray(q, dtype=float).copy()


def open_link(joint_names, config):
    """The link settings.json asks for, unconnected; None for no hardware."""
    if not config["port"]:
        return None
    if config["port"] == "mock":
        return MockLink(joint_names, config)
    return DynamixelLink(joint_names, config)
