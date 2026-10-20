"""The h4nd on real DYNAMIXEL XL330 servos, through a U2D2.

Everything here was measured against the hand rather than taken from a
datasheet:

  * The bus runs at 4 Mbaud. Nothing answers at 57600, 115200, 1M or 2M.
  * All sixteen motors are XL330-M288 (model 1200), IDs 1..16 with none
    missing - but at 4 Mbaud over an FTDI bridge, single reads drop. ID 13
    answered a ping and then failed eleven consecutive one-byte reads. Every
    transaction here retries, and that is not belt-and-braces: without it the
    bus looks like it has holes in it.
  * They arrive in operating mode 5, current-based position control, with a
    300 mA limit. That is the mode to keep: the current limit is what stops a
    finger that has hit something from pulling until a linkage gives.
  * Profile velocity reads back as 0, which means a goal change is a step, not
    a move. Something has to rate-limit; see `RateLimiter`, applied in the
    bus thread `_run`.

WHAT THIS FILE DOES NOT KNOW is which motor drives which joint, or what tick
each joint calls zero. Those are properties of one physical hand, they live in
config/hand_calibration.json, and until that file is written the defaults here
(IDs 1..16 in joint order, centre tick, positive sign) are a guess.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from .robot.model import joint_role

from .paths import CONFIG

CALIBRATION = CONFIG / "hand_calibration.json"

BAUDRATE = 4_000_000
PROTOCOL = 2.0

# XL330 control table: name -> (address, bytes)
CONTROL = {
    "model": (0, 2),
    "operating_mode": (11, 1),
    "current_limit": (38, 2),
    "torque_enable": (64, 1),
    "hardware_error": (70, 1),
    "position_d": (80, 2),
    "position_i": (82, 2),
    "position_p": (84, 2),
    "feedforward_2nd": (88, 2),
    "feedforward_1st": (90, 2),
    "goal_current": (102, 2),
    "profile_acceleration": (108, 4),
    "profile_velocity": (112, 4),
    "goal_position": (116, 4),
    "present_current": (126, 2),
    "present_position": (132, 4),
    "present_voltage": (144, 2),
    "present_temperature": (146, 1),
}

# Read back as two's complement. present_current is the one that bites: a
# motor holding still reports -3 mA, which unsigned is 65533.
SIGNED = {"present_current", "present_position", "goal_position",
          "goal_current"}

CURRENT_POSITION_MODE = 5
TICKS_PER_TURN = 4096
CENTRE_TICK = 2048
MODEL_XL330_M288 = 1200

# 0.229 rev/min per count. 200 is about 275 deg/s at the joint, so a 90 degree
# finger close takes 0.33 s. The old default of 40 was 55 deg/s and 1.64 s,
# which is the single biggest reason the hand felt slow - it is a register on
# the motor, not anything in this repo. Lower it if a finger is snatching.
PROFILE_VELOCITY = 200
# NONZERO. 0 means "infinite": the goal step is a jolt, and with
# PROFILE_VELOCITY uncapped beside it the first write after an arm is the
# fastest move the servo can make. That is what was on all sixteen when the
# hand was first read - velocity 0, acceleration 0 - and it is what makes a
# stale or distant goal dangerous rather than merely wrong.
#
# This is the DEFAULT, not the only place it can be set. It used to be:
# `configure` rewrites the profile registers from the link's own fields and
# `arm` calls `configure`, so a ramp set through `hw gains` was reverted at
# the moment it was needed - which is why this constant had to be changed to
# get off 0 at all. `write_gains` now folds what it wrote back into those
# fields, so the tuning path holds through an arm and
# `selftest.test_written_gains_survive_arming` keeps it that way.
#
# 20 is 0.64 s to reach the profile velocity - 20 x 214.577 rev/min^2 against
# 200 x 0.229 rev/min - which reads as firm rather than soft and is slow
# enough that a goal nobody meant does not arrive before it can be seen. That
# ramp is also what puts the command up to 28.5 deg ahead of the encoder on a
# large move, so it is the number `MAX_SPEED` below is reasoning about; the
# two comments disagreed on it once, at a quarter-second against 0.64.
PROFILE_ACCELERATION = 20
CURRENT_LIMIT_MA = 300

# Position PID, written on every arm so the hand feels the same after a power
# cycle. It did not used to be: `configure` set the mode, the current limit and
# the profile but never the gains, so behaviour came from whatever was last
# left in EEPROM - invisible from here, and different on every hand.
#
# These are what was found on the bench hand, not factory values (XL330-M288
# ships P=900, I=0, D=0), so writing them changes nothing today. They are here
# to be changed: `main.py hw gains` reads and sets them live.
#
# If a joint oscillates, note that the same argument applies as in the
# simulator - the current limit is a clip, so a joint saturates on any decent
# step and arrives with speed to shed. More P makes that worse, not better.
# D is what sheds it. But D is differentiated from a 0.088 deg/tick encoder,
# so too much of it turns quantisation into buzz. D=600 against P=250 is a
# long way from the factory ratio in that direction; try halving D before
# touching P, and give PROFILE_ACCELERATION a nonzero value so each goal
# change is ramped instead of stepped.
POSITION_P = 250
POSITION_I = 0
POSITION_D = 600
# rad/s software slew. NOT a free number: it has to sit near what the servo's
# own velocity profile can actually deliver, or the command runs away from the
# hand on every large move.
#
# MEASURED, on the seven-pose run. PROFILE_VELOCITY 200 is 4.80 rad/s at the
# joint and the encoder reached 4.60 at the 99th percentile, so the servo's
# ceiling is real; this limiter was set to 6.0, which is a quarter faster than
# the hand can move. Deriving it removes that much.
#
# IT DOES NOT REMOVE THE TRANSIENT, and the measurement says why. During any
# large move the command runs up to 28.5 deg ahead of the encoder, and the
# cause is the ACCELERATION ramp, not the top speed: PROFILE_ACCELERATION 20
# takes about 0.64 s to reach 4.8 rad/s, while the limiter steps to its full
# rate in one tick. That ramp is deliberate and is what keeps a goal nobody
# meant from arriving before it can be seen, so the lead it creates is a cost
# worth paying rather than a fault to tune out.
#
# That lead is also why the executor no longer looks at it. Free air produced
# a LARGER lead (28.5 deg) than a finger actually pushing on a cylinder
# (17.9), so a stall test built on it could only fire on the wrong one - see
# `executor.STALL_MA`, which is what detects a stall instead.
PROFILE_RAD_S_PER_COUNT = 0.229 * 2.0 * math.pi / 60.0      # 0.02398 rad/s


def profile_rad_s(counts=None):
    """What a `profile_velocity` register value is worth at the joint, rad/s."""
    return (PROFILE_VELOCITY if counts is None else counts) * PROFILE_RAD_S_PER_COUNT


MAX_SPEED = profile_rad_s()
COMMAND_HZ = 100.0     # goal writes per second down the bus
WRITE_SETTLE = 0.001   # s to let a sync write clear before the next transaction
READ_FAILURES_TO_FAIL = 10   # consecutive encoder-read failures before the output goes offline


class DynamixelError(RuntimeError):
    pass


def _signed(value, length):
    """Two's complement for a sync-read field, which comes back unsigned."""
    value = int(value)
    if value >= 1 << (8 * length - 1):
        value -= 1 << (8 * length)
    return value


def ticks_to_radians(ticks):
    return ticks * (2.0 * math.pi / TICKS_PER_TURN)


def radians_to_ticks(radians):
    return radians * (TICKS_PER_TURN / (2.0 * math.pi))


def list_serial_ports():
    """(device, label) for every port worth offering, likeliest first.

    macOS always advertises a Bluetooth serial port and a debug console. Both
    open happily and answer nothing, so a dropdown that lists them just invites
    a twenty-second wait for a timeout. Only real USB bridges are offered.
    """
    from serial.tools import list_ports

    ports = []
    for port in list_ports.comports():
        device = port.device
        if "Bluetooth" in device or "debug-console" in device:
            continue
        if device.startswith("/dev/tty."):
            continue          # on macOS the callout (cu.) side is the one to use
        label = port.description if port.description != "n/a" else device
        maker = port.manufacturer or ""
        ports.append((device, f"{label}" + (f" ({maker})" if maker else "")))
    return sorted(ports, key=lambda entry: "usb" not in entry[0].lower())


# Which way a motor turns for a positive joint angle is a property of how it
# was bolted on, not of the URDF, so it lives here rather than in the model.
#
# On this build every FLEXION actuator - the knuckle, middle and tip joints of
# all four digits, by role rather than by name - is mounted the opposite way
# round to the URDF's convention:
# the model has a positive angle curl the finger toward the palm, and the servo
# turns that way for a negative tick delta. The four adduction joints are
# mounted the way the model expects and are left alone.
#
# Twelve joints, one mechanical decision. Flipping them in the URDF instead
# would be the wrong place: the URDF describes the mechanism, which is fine,
# and doing it there would silently invalidate every pose, limit and IK result
# already expressed in joint terms. Here it changes only the tick conversion,
# which is exactly what is wrong.
#
# Measure before trusting it - `main.py hw watch`, bend one joint, see which
# way the number runs - and override per joint with `hw sign`.
def default_sign(joint_name):
    """-1 on every flexion actuator, +1 on the spreads.

    Keyed on what the joint MOVES, via model.joint_role, not on its name. The
    thumb is why, and getting this wrong is not hypothetical: an earlier
    version split the groups on whether "adduction" appeared in the name, which
    put `thumb_proximal` (a spread) in the flexion group and left
    `thumb_adduction` (a flexion) out of it. Both ended up inverted, and the
    thumb's two joints drove the hardware backwards while the simulator and the
    sliders looked perfectly normal - because nothing in the model was wrong.
    """
    return 1 if joint_role(joint_name) == "spread" else -1


def default_calibration(joint_names):
    """IDs 1..16 down the joint order, and the mounting signs above.

    The IDs are a starting point, not a discovered truth: nothing has confirmed
    that motor 5 is the middle finger's spread joint rather than some other
    one. The signs are a measured property of this build.
    """
    return {"baudrate": BAUDRATE,
            "motors": {name: {"id": i, "sign": default_sign(name),
                              "zero": CENTRE_TICK}
                       for i, name in enumerate(joint_names, start=1)}}


def load_calibration(joint_names, path=CALIBRATION):
    path = Path(path)
    data = default_calibration(joint_names)
    if path.exists():
        stored = json.loads(path.read_text())
        data["baudrate"] = stored.get("baudrate", data["baudrate"])
        data["calibrated"] = stored.get("calibrated", False)
        for name, entry in stored.get("motors", {}).items():
            if name in data["motors"]:
                data["motors"][name].update(entry)
    else:
        data["calibrated"] = False
    return data


SIGN_NOTE = ("sign is how the motor is mounted, not how the joint is modelled: "
             "-1 means a positive joint angle needs a negative tick delta. On "
             "this build every flexion actuator is mounted backwards and the "
             "four spread joints are not - by role, so the thumb's spread is "
             "thumb_proximal and its knuckle is thumb_adduction. Measure "
             "with `main.py hw watch` before trusting any of it.")


def save_calibration(data, path=CALIBRATION):
    """JSON has no comments, so the explanation ships as a field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**data, "_note": SIGN_NOTE}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


class MockHand:
    """The same interface with no bus behind it.

    Not a toy: it is how the panel toggle, the rate limiter and the command
    path get exercised without a hand on the desk to break.
    """

    name = "mock"

    def __init__(self, joint_names, **_):
        self.joint_names = list(joint_names)
        self.goal = np.zeros(len(self.joint_names))
        self.torque_on = False
        self.connected = False
        self.note = "mock link - no motors"
        # The mock carries the response registers too. Without them the tuning
        # path looked fine and wrote nothing: `set_servo` raised on the missing
        # method, the caller swallowed it into `error`, and three sliders were
        # silently inert on the one link that exists to test them.
        self.gains = {"position_p": POSITION_P, "position_i": POSITION_I,
                      "position_d": POSITION_D,
                      "profile_velocity": PROFILE_VELOCITY,
                      "profile_acceleration": PROFILE_ACCELERATION,
                      "current_limit": CURRENT_LIMIT_MA}

    def connect(self):
        self.connected = True
        return self

    def close(self):
        self.connected = False

    def ping_all(self):
        return {name: MODEL_XL330_M288 for name in self.joint_names}

    # A virtual object: closing joints cannot pass these angles, and the
    # current climbs with how far the goal is pushed past them. It is how
    # the grasp executor is exercised without a hand or a thing to hold.
    obstacle = None          # {joint index: angle}

    def read_positions(self):
        return self.read_state()[0]

    def read_state(self):
        positions = self.goal.copy()
        currents = np.zeros(len(self.joint_names))
        if self.obstacle:
            for k, limit in self.obstacle.items():
                if positions[k] > limit:
                    currents[k] = min(1500.0 * (positions[k] - limit), CURRENT_LIMIT_MA)
                    positions[k] = limit
        return positions, currents

    def write_positions(self, q):
        self.goal = np.asarray(q, dtype=float).copy()

    def torque(self, on):
        self.torque_on = bool(on)

    def sample_gains(self, name=None):
        return dict(self.gains)

    def read_gains(self):
        return {name: dict(self.gains) for name in self.joint_names}

    def write_gains(self, values, verify=True):
        # One dict, unlike the real link, because this mock's `configure` is a
        # no-op and has nothing to rewrite them from - see
        # `DynamixelHand._remember` for the bookkeeping that matters on metal.
        self.gains.update({k: int(v) for k, v in values.items()})
        return []

    def configure(self):
        self.torque_on = False
        return self

    def arm(self):
        self.torque_on = True
        return self.goal.copy()

    def disarm(self):
        self.torque_on = False

    def health(self):
        return [{"joint": n, "id": i, "position": CENTRE_TICK, "current_ma": 0,
                 "volts": 5.0, "celsius": 25, "error": 0}
                for i, n in enumerate(self.joint_names, start=1)]

    def status(self):
        return {"port": "mock", "motors": len(self.joint_names),
                "torque": self.torque_on, "note": self.note}


class DynamixelHand:
    """Sixteen XL330s on one TTL bus, addressed by joint name."""

    name = "dynamixel"

    def __init__(self, port, joint_names, calibration=None, baudrate=None,
                 current_ma=CURRENT_LIMIT_MA, profile_velocity=PROFILE_VELOCITY,
                 profile_acceleration=PROFILE_ACCELERATION, gains=None,
                 retries=5):
        self.port = port
        self.joint_names = list(joint_names)
        self.calibration = calibration or load_calibration(self.joint_names)
        self.baudrate = baudrate or self.calibration.get("baudrate", BAUDRATE)
        self.current_ma = current_ma
        self.profile_velocity = profile_velocity
        self.profile_acceleration = profile_acceleration
        self.gains = dict(gains) if gains else {"position_p": POSITION_P,
                                                "position_i": POSITION_I,
                                                "position_d": POSITION_D}
        self.retries = retries
        self.torque_on = False
        self.connected = False
        self.note = ""
        self._handler = None
        self._packet = None

    # ----------------------------------------------------------- plumbing
    @property
    def motors(self):
        return self.calibration["motors"]

    def ids(self):
        return [self.motors[name]["id"] for name in self.joint_names]

    def connect(self):
        from dynamixel_sdk import PacketHandler, PortHandler

        self._handler = PortHandler(self.port)
        if not self._handler.openPort():
            raise DynamixelError(f"cannot open {self.port}")
        if not self._handler.setBaudRate(self.baudrate):
            self._handler.closePort()
            raise DynamixelError(f"cannot set {self.baudrate} baud on {self.port}")
        self._packet = PacketHandler(PROTOCOL)
        self.connected = True

        found = self.ping_all()
        missing = [n for n, model in found.items() if model is None]
        self.note = (f"{len(found) - len(missing)}/{len(self.joint_names)} motors"
                     + (f", missing {', '.join(missing)}" if missing else ""))

        # Ask the hand whether it is energised rather than assuming it is not.
        # These motors arrive with torque already enabled, and a panel that
        # says "disarmed" over a hand that is holding position is worse than
        # no panel at all.
        self.torque_on = self._live_torque(found)
        return self

    def _live_torque(self, found=None):
        for name in self.joint_names:
            if found is not None and found.get(name) is None:
                continue
            try:
                return bool(self._read(self.motors[name]["id"], "torque_enable"))
            except DynamixelError:
                continue
        return False

    def close(self):
        if self._handler is not None:
            self._handler.closePort()
        self.connected = False

    def _retry(self, call, *args):
        """Every transaction, several times.

        At 4 Mbaud a single read fails often enough that one attempt makes a
        healthy motor look absent - which would be read as a wiring fault and
        sent someone looking for a broken cable.
        """
        last = None
        for _ in range(self.retries):
            value, result, error = call(self._handler, *args)
            if result == 0 and error == 0:
                return value
            last = (result, error)
        raise DynamixelError(f"no reply after {self.retries} tries: {args} {last}")

    def _read(self, motor_id, field):
        address, length = CONTROL[field]
        reader = {1: self._packet.read1ByteTxRx, 2: self._packet.read2ByteTxRx,
                  4: self._packet.read4ByteTxRx}[length]
        value = self._retry(reader, motor_id, address)
        if field in SIGNED and value >= 1 << (8 * length - 1):
            value -= 1 << (8 * length)
        return value

    def _write(self, motor_id, field, value):
        address, length = CONTROL[field]
        writer = {1: self._packet.write1ByteTxRx, 2: self._packet.write2ByteTxRx,
                  4: self._packet.write4ByteTxRx}[length]
        last = None
        for _ in range(self.retries):
            result, error = writer(self._handler, motor_id, address, int(value))
            if result == 0 and error == 0:
                return
            last = (result, error)
        raise DynamixelError(f"write {field}={value} to id {motor_id} failed: {last}")

    # -------------------------------------------------------------- setup
    def ping_all(self):
        """joint name -> model number, or None where nothing answered."""
        found = {}
        for name in self.joint_names:
            motor_id = self.motors[name]["id"]
            model = None
            for _ in range(self.retries):
                value, result, _ = self._packet.ping(self._handler, motor_id)
                if result == 0:
                    model = value
                    break
            found[name] = model
        return found

    def configure(self):
        """Current-limited position control, with a velocity profile.

        Torque has to be off to change the operating mode, so this turns it off
        and leaves it off - arming is a separate, deliberate act.
        """
        for name in self.joint_names:
            motor_id = self.motors[name]["id"]
            self._write(motor_id, "torque_enable", 0)
            self._write(motor_id, "operating_mode", CURRENT_POSITION_MODE)
            self._write(motor_id, "current_limit", self.current_ma)
            self._write(motor_id, "profile_velocity", self.profile_velocity)
            self._write(motor_id, "profile_acceleration", self.profile_acceleration)
            self._write(motor_id, "goal_current", self.current_ma)
            for field, value in self.gains.items():
                self._write(motor_id, field, value)
        self.torque_on = False
        return self

    def torque(self, on):
        for name in self.joint_names:
            self._write(self.motors[name]["id"], "torque_enable", 1 if on else 0)
        self.torque_on = bool(on)

    def arm(self):
        """Energise the hand without moving it, and return where it is.

        The order matters. A servo's goal position register keeps whatever was
        last written to it, so enabling torque first makes the hand snap to a
        goal from some earlier session before anything has commanded it. Set
        the goal to the present position, then enable.
        """
        self.configure()                    # needs torque off to set the mode
        here = self.read_positions()
        self.write_positions(here)
        self.torque(True)
        return here

    def disarm(self):
        self.torque(False)

    # ------------------------------------------------------- joints <-> ticks
    def to_ticks(self, name, radians):
        motor = self.motors[name]
        return int(round(motor["zero"] + motor["sign"] * radians_to_ticks(radians)))

    def to_radians(self, name, ticks):
        motor = self.motors[name]
        return motor["sign"] * ticks_to_radians(ticks - motor["zero"])

    def read_positions(self):
        return self.read_state()[0]

    def read_state(self):
        """Every joint's angle and present current in one bus transaction.

        Sixteen separate reads is sixteen chances to drop a packet, and at
        4 Mbaud that is not hypothetical. A sync read asks once, for the
        block from present_current through present_position - ten bytes
        per motor - and whatever it fails to bring back is retried the slow
        way rather than left as a hole. The current is what the grasp
        executor reads a stall from, so it rides along for free.
        """
        from dynamixel_sdk import GroupSyncRead

        start, _ = CONTROL["present_current"]
        p_address, p_length = CONTROL["present_position"]
        length = p_address + p_length - start
        group = GroupSyncRead(self._handler, self._packet, start, length)
        for name in self.joint_names:
            group.addParam(self.motors[name]["id"])
        for _ in range(self.retries):
            if group.txRxPacket() == 0:
                break

        angles = np.zeros(len(self.joint_names))
        currents = np.zeros(len(self.joint_names))
        for index, name in enumerate(self.joint_names):
            motor_id = self.motors[name]["id"]
            if group.isAvailable(motor_id, start, length):
                ticks = _signed(group.getData(motor_id, p_address, p_length), p_length)
                current = _signed(group.getData(motor_id, start, 2), 2)
            else:
                ticks = self._read(motor_id, "present_position")
                current = self._read(motor_id, "present_current")
            angles[index] = self.to_radians(name, ticks)
            currents[index] = float(current)
        group.clearParam()
        return angles, currents

    def write_positions(self, q):
        """One sync write for the whole hand - sixteen separate writes at this
        baud rate is sixteen chances to drop a packet mid-motion."""
        from dynamixel_sdk import GroupSyncWrite

        address, length = CONTROL["goal_position"]
        group = GroupSyncWrite(self._handler, self._packet, address, length)
        for name, value in zip(self.joint_names, np.asarray(q, dtype=float)):
            ticks = max(0, min(TICKS_PER_TURN - 1, self.to_ticks(name, value)))
            group.addParam(self.motors[name]["id"],
                           [ticks & 0xFF, (ticks >> 8) & 0xFF,
                            (ticks >> 16) & 0xFF, (ticks >> 24) & 0xFF])
        result = group.txPacket()
        group.clearParam()
        if result != 0:
            raise DynamixelError(f"sync write failed ({result})")
        # Let the bus drain before anything else touches it. A sync write gets
        # no status packets back, so `txPacket` returns while the frame is
        # still going out; a read issued immediately collides with it on the
        # half-duplex line and the write is simply lost. Measured on this hand:
        # with no gap, 20 of 20 goal writes never landed and the fingers did
        # not move; at 0.5 ms, none were lost. This is the loop's own bug, not
        # just a test's - the bus thread writes at 100 Hz and reads at 20 Hz,
        # so about a fifth of every command was being dropped in silence.
        time.sleep(WRITE_SETTLE)

    # ------------------------------------------------------------- status
    def status(self):
        return {"port": self.port, "baudrate": self.baudrate,
                "motors": len(self.joint_names), "torque": self.torque_on,
                "note": self.note}

    def read_gains(self):
        """(joint -> {field: value}) for everything that shapes the response."""
        fields = ("position_p", "position_i", "position_d",
                  "profile_velocity", "profile_acceleration", "current_limit")
        return {name: {f: self._read(self.motors[name]["id"], f) for f in fields}
                for name in self.joint_names}

    def sample_gains(self, name=None):
        """The response registers, read off ONE motor.

        All sixteen are written together and never diverge, and reading the
        lot costs 16 reads x 16 ms - a second and a half, on a bus the control
        loop is sharing. One motor answers the same question in 100 ms.
        """
        name = name or self.joint_names[0]
        motor_id = self.motors[name]["id"]
        fields = ("position_p", "position_i", "position_d",
                  "profile_velocity", "profile_acceleration", "current_limit")
        return {f: self._read(motor_id, f) for f in fields}

    def write_gains(self, values, verify=True):
        """Set any of the response fields on every motor, and check they took.

        Torque may stay on: these are RAM registers on the XL330, unlike the
        operating mode.

        The read-back is not paranoia. Sixteen writes are sixteen separate
        transactions on a bus that drops packets, and a partial success is
        silent and looks like nothing happened: found in the wild with fifteen
        motors on a 275 deg/s velocity profile and `middle_distal` still on
        zero, so one fingertip snapped while the rest eased. Returns the joints
        that disagreed after being rewritten, so a caller can say so.
        """
        for name in self.joint_names:
            for field, value in values.items():
                self._write(self.motors[name]["id"], field, int(value))
        # AND REMEMBER IT, or `configure` undoes it on the next arm. These
        # registers are also written by `configure` from the instance's own
        # fields, and `arm` calls `configure` - so a value set here and not
        # recorded here survives exactly until the hand is next armed, which
        # is the moment it was wanted. That is not hypothetical: it is why
        # `PROFILE_ACCELERATION` could not be taken off 0 with `hw gains` and
        # had to be changed at the module default instead.
        self._remember(values)
        if not verify:
            return []
        wrong = []
        for name in self.joint_names:
            for field, value in values.items():
                motor_id = self.motors[name]["id"]
                if self._read(motor_id, field) != int(value):
                    self._write(motor_id, field, int(value))
                    if self._read(motor_id, field) != int(value):
                        wrong.append((name, field))
        return wrong

    def _remember(self, values):
        """Fold written registers into the state `configure` rewrites.

        The three profile registers live in their own fields; everything else
        is a response gain and goes in `gains`, which `configure` walks.
        """
        for field, value in values.items():
            if field == "profile_velocity":
                self.profile_velocity = int(value)
            elif field == "profile_acceleration":
                self.profile_acceleration = int(value)
            elif field in ("current_limit", "goal_current"):
                self.current_ma = int(value)
            else:
                self.gains[field] = int(value)

    def health(self):
        rows = []
        for name in self.joint_names:
            motor_id = self.motors[name]["id"]
            try:
                rows.append({
                    "joint": name, "id": motor_id,
                    "position": self._read(motor_id, "present_position"),
                    "current_ma": self._read(motor_id, "present_current"),
                    "volts": self._read(motor_id, "present_voltage") / 10.0,
                    "celsius": self._read(motor_id, "present_temperature"),
                    "error": self._read(motor_id, "hardware_error"),
                })
            except DynamixelError as failure:
                rows.append({"joint": name, "id": motor_id, "error": str(failure)})
        return rows


class RateLimiter:
    """Software slew limit on top of the servo's own velocity profile.

    A slider drag can move a target a radian between frames. The profile
    velocity alone would take that as a normal move and run at it; this keeps
    the commanded angle within MAX_SPEED of where it was, so the hand follows
    the panel rather than snapping to it.
    """

    def __init__(self, size, max_speed=MAX_SPEED):
        self.value = np.zeros(size)
        self.max_speed = max_speed
        self.primed = False

    def prime(self, q):
        self.value = np.asarray(q, dtype=float).copy()
        self.primed = True

    def step(self, target, dt):
        target = np.asarray(target, dtype=float)
        if not self.primed:
            self.prime(target)
            return self.value
        limit = self.max_speed * dt
        delta = np.clip(target - self.value, -limit, limit)
        self.value = self.value + delta
        return self.value


class HardwareOutput:
    """The hardware side of the control loop, and its safety interlocks.

    Three separate states, on purpose, because they carry different risks:

      available - a link exists (a port was chosen)
      enabled   - commands are being sent to it
      armed     - the motors are energised and will act on them

    Nothing here is on by default. A bus error takes the output offline and
    records why; it never propagates, because a dropped packet should not end
    a simulation session.

    The bus runs on its own thread. It has to: reading all sixteen encoders
    costs 16 ms, which is the macOS FTDI latency floor rather than wire time -
    dropping the SDK's latency timer below it does return sooner, but only by
    truncating half the replies (13 clean reads out of 25 at 4 ms, 9 at 2 ms).
    Sixteen milliseconds inside a 100 Hz simulation loop is a visible stutter,
    so the loop hands over a target and reads back the last encoder frame, and
    never waits for the bus.
    """

    def __init__(self, link, size, max_speed=MAX_SPEED, hz=COMMAND_HZ,
                 read_hz=20.0, joint_names=None):
        self.size = size
        self.joint_names = list(joint_names) if joint_names else [str(k) for k in range(size)]
        self.limiter = RateLimiter(size, max_speed)
        self.interval = 1.0 / hz
        self.read_interval = 1.0 / read_hz
        self.enabled = False
        self.error = ""
        self.positions = None          # last encoder frame, radians
        self.currents = None           # last present current, mA, when the link has it
        # Closing to contact - see `executor.py`. Off until asked for.
        from .executor import GraspExecutor
        # No read rate any more: the executor latches on present current
        # alone, so it no longer cares how often the encoders are polled - see
        # `executor.STALL_MA` for the bench run that removed the position test.
        self.executor = GraspExecutor(self.joint_names)
        # What we believe is on the motors. Refreshed from the hand when a
        # link attaches and updated on every write, because the state message
        # goes out 20 times a second and the bus cannot be asked that often.
        self.servo = {"position_p": POSITION_P, "position_i": POSITION_I,
                      "position_d": POSITION_D,
                      "profile_velocity": PROFILE_VELOCITY,
                      "profile_acceleration": PROFILE_ACCELERATION,
                      "current_limit": CURRENT_LIMIT_MA}
        self.link = None
        self.port = ""
        self._target = np.zeros(size)
        self._stamp = None
        # Sensor frame arrival to the sync write that carried it, and the
        # write itself; exponential averages, plus the worst seen.
        self.latency_ms = 0.0
        self.write_ms = 0.0
        self._read_failures = 0
        self._pinching = None          # the student's contact head, when it is running
        self.worst_latency_ms = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if link is not None:
            self.attach(link)

    # ------------------------------------------------------------- state
    @property
    def available(self):
        return self.link is not None and getattr(self.link, "connected", False)

    @property
    def armed(self):
        return self.available and self.link.torque_on

    # -------------------------------------------------------------- link
    def attach(self, link):
        """Swap in a link chosen at runtime.

        Detaching first is the point: two open handles on one serial port is a
        bus with two masters.
        """
        self.detach()
        with self._lock:
            self.link = link
            self.port = getattr(link, "port", "mock" if link is not None else "")
            self.enabled = False
            self.error = ""
        if self.available:
            try:
                with self._lock:
                    self.servo.update(self.link.sample_gains())
            except Exception:
                pass          # keep the defaults; the panel is not worth a crash
            self._start()
        return self.available

    def detach(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        with self._lock:
            if self.link is not None:
                try:
                    self.link.close()
                except Exception:
                    pass
            self.link = None
            self.port = ""
            self.enabled = False
            self.positions = None
            self.currents = None      # a stale current would let `hot` fire on the next link's
        self.executor.reset()

    def _start(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop,),
                                        name="handcontrol-bus", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------- switches
    def enable(self, on):
        if on and not self.available:
            self.error = "no hand attached - pick a port"
            return False
        self.enabled = bool(on)
        if self.enabled:
            self.error = ""
            self._prime()
        return self.enabled

    def _prime(self):
        """Start following from where the hand actually is, not from zero."""
        try:
            with self._lock:
                here = self.link.read_positions()
            self.limiter.prime(here)
        except DynamixelError as failure:
            self._fail(failure)

    def arm(self, on):
        if not self.available:
            self.error = "no hand attached - pick a port"
            return False
        # Whichever way this goes, nothing stays held across it: disarming
        # lets the hand go limp and whatever it was gripping can move, so the
        # angle a finger stalled at stops meaning anything the moment torque
        # is cut. Re-arming onto a stale `hold_at` would drive it straight
        # back to a grip on empty air.
        self.executor.reset()
        try:
            with self._lock:
                if on:
                    here = self.link.arm()
                else:
                    self.link.disarm()
                    here = None
            if here is not None:
                self.limiter.prime(here)
            self.error = ""
        except Exception as failure:
            self._fail(failure)
        return self.armed

    def servo_tuning(self):
        return dict(self.servo)

    def enable_grip(self, on):
        """Closing to contact on or off. Off releases every held finger."""
        self.executor.enabled = bool(on)
        if not on:
            self.executor.reset()
        return self.executor.enabled

    def set_servo(self, values):
        """Write response registers and remember what was written."""
        if not self.available:
            return False
        with self._lock:
            wrong = self.link.write_gains(values)
        if wrong:
            self.error = ("did not take on: "
                          + ", ".join(f"{j}.{f}" for j, f in wrong[:3]))
        self.servo.update({k: int(v) for k, v in values.items()})
        return not wrong

    def command(self, targets, stamp=None, pinching=None):
        """What the hand should be doing. Non-blocking: the thread sends it.

        `stamp` is when the sensor frame behind it arrived, so the write that
        carries it can report the pipeline's own latency. `pinching` is the
        student's contact head when there is one, which is how the executor
        tells a pinch from a grasp - see `executor.pinched`.
        """
        self._target = np.asarray(targets, dtype=float).copy()
        self._pinching = None if pinching is None else np.asarray(pinching, dtype=float)
        self._stamp = stamp

    # ------------------------------------------------------------ thread
    def _run(self, stop):
        """The bus thread: writes targets at `hz`, reads encoders at `read_hz`.

        `stop` is passed in rather than read off `self`, because `detach`
        rebinds `self._stop` to a fresh Event and only waits a second for the
        join. A thread still inside a retrying read when that second expires
        would come back, see the NEW un-set Event, and carry on - two threads
        writing one half-duplex bus at 100 Hz each, which is the exact "two
        masters" failure `attach` says detaching first prevents. Holding the
        Event it was started with, a superseded thread always exits.
        """
        last_write = last_read = time.time()
        while not stop.is_set():
            now = time.time()
            did_something = False

            if self.enabled and self.armed and now - last_write >= self.interval:
                # A stalled thread must not buy a bigger step than a few
                # missed ticks - the ramp is the safety, not a formality.
                dt = min(now - last_write, 4.0 * self.interval)
                last_write = now
                # The executor first, the limiter after: a held finger's
                # target is pinned before it is slewed, so a stall is honoured
                # at the rate the hand can obey.
                wanted = self.executor.step(self._target, self.positions,
                                            self.currents,
                                            pinching=self._pinching)
                step = self.limiter.step(wanted, dt)
                stamp = self._stamp
                try:
                    began = time.perf_counter()
                    with self._lock:
                        if self.available:
                            self.link.write_positions(step)
                    took = (time.perf_counter() - began) * 1000.0
                    self.write_ms += 0.1 * (took - self.write_ms)
                    if stamp is not None:
                        latency = (time.time() - stamp) * 1000.0
                        self.latency_ms += 0.1 * (latency - self.latency_ms)
                        self.worst_latency_ms = max(self.worst_latency_ms, latency)
                except Exception as failure:
                    # NOT just DynamixelError. Every write goes through
                    # pyserial, and an unplugged adapter raises SerialException
                    # or OSError from under the SDK. Caught narrowly, those
                    # killed this thread outright: `enabled` stayed True and
                    # `link` stayed non-None, so `armed` kept answering yes and
                    # the panel kept drawing ARMED over a hand that had stopped
                    # receiving anything at all.
                    self._fail(failure)
                did_something = True

            if now - last_read >= self.read_interval:
                last_read = now
                try:
                    with self._lock:
                        if self.available:
                            if hasattr(self.link, "read_state"):
                                self.positions, self.currents = self.link.read_state()
                            else:
                                self.positions = self.link.read_positions()
                except Exception as failure:
                    # Reads fail on their own for a beat - a dropped packet is
                    # not a dead bus - so this records rather than disarms,
                    # until they stop coming back at all.
                    self.error = str(failure)
                    self._read_failures += 1
                    if self._read_failures >= READ_FAILURES_TO_FAIL:
                        self._fail(failure)
                else:
                    self._read_failures = 0
                did_something = True

            if not did_something:
                time.sleep(0.002)

    def _fail(self, failure):
        """Take the output offline. Whatever the executor was holding goes with
        it: `hold_at` is an angle read off encoders that are no longer being
        read, and re-arming onto a stale one drives the fingers back to a grip
        on something that has since moved."""
        self.error = str(failure)
        self.enabled = False
        self.executor.reset()

    # ------------------------------------------------------------ report
    def state(self):
        with self._lock:
            status = self.link.status() if self.available else {}
        positions = self.positions
        return {"available": self.available, "enabled": self.enabled,
                "armed": self.armed, "error": self.error,
                "grip": self.executor.state(),
                "latency_ms": round(self.latency_ms, 1),
                "worst_latency_ms": round(self.worst_latency_ms, 1),
                "write_ms": round(self.write_ms, 2),
                "port": status.get("port", self.port),
                "note": status.get("note", ""),
                "positions": None if positions is None
                             else [round(float(v), 5) for v in positions]}

    def close(self):
        self.detach()
