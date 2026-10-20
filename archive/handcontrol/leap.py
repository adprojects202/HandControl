"""The Leap Motion camera, bound straight to libLeapC with ctypes.

The Ultraleap SDK ships a Python binding, but it ships it as a compiled
`_leapc_cffi.cpython-312-darwin.so` - and this project runs 3.11, so that
file cannot be imported. Rather than rebuild it, the twenty-odd structs the
tracking path actually needs are declared here against the vendored
`LeapSDK/include/LeapC.h`. No compiler, no wheel, no version to match.

Only the tracking path is bound: create a connection, poll it, and read hands
out of `LEAP_TRACKING_EVENT`. Images, fiducials, config and the rest of the
header are deliberately absent - what is not declared cannot go stale.

Struct layout is the whole risk here, so `self_test()` checks the sizes that
the header states outright and the polling loop refuses events whose
`LEAP_CONNECTION_MESSAGE.size` disagrees with ours.

Units are the SDK's: millimetres, and the desktop frame with +X right,
+Y up away from the device, +Z toward the user.
"""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import (POINTER, Structure, byref, c_float, c_int32, c_int64,
                    c_uint32, c_uint64, c_void_p, sizeof)
from dataclasses import dataclass

import numpy as np

# Where the SDK puts itself on macOS. LEAPC_LIB overrides, for a build that
# lives somewhere else.
LIB_CANDIDATES = (
    os.environ.get("LEAPC_LIB"),
    "/Applications/Ultraleap Hand Tracking.app/Contents/LeapSDK/lib/libLeapC.dylib",
    "/Library/Application Support/Ultraleap/LeapSDK/lib/libLeapC.dylib",
    "/usr/local/lib/libLeapC.dylib",
    "libLeapC.dylib",
    "libLeapC.so.6",
)

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
BONES = ("metacarpal", "proximal", "intermediate", "distal")

# eLeapEventType. Tracking is the only one worth acting on; the others are
# named so the log line says something better than a number.
EVENT_NONE = 0
EVENT_CONNECTION = 1
EVENT_CONNECTION_LOST = 2
EVENT_DEVICE = 3
EVENT_DEVICE_FAILURE = 4
EVENT_POLICY = 5
EVENT_TRACKING = 0x100
EVENT_DEVICE_LOST = 0x104
EVENT_DEVICE_STATUS_CHANGE = 0x107
EVENT_TRACKING_MODE = 0x10B
EVENT_NAMES = {EVENT_NONE: "none", EVENT_CONNECTION: "connection",
               EVENT_CONNECTION_LOST: "connection-lost", EVENT_DEVICE: "device",
               EVENT_DEVICE_FAILURE: "device-failure", EVENT_POLICY: "policy",
               EVENT_TRACKING: "tracking", EVENT_DEVICE_LOST: "device-lost",
               EVENT_DEVICE_STATUS_CHANGE: "device-status",
               EVENT_TRACKING_MODE: "tracking-mode"}

# eLeapTrackingMode
MODE_DESKTOP, MODE_HMD, MODE_SCREENTOP = 0, 1, 2
MODES = {"desktop": MODE_DESKTOP, "hmd": MODE_HMD, "screentop": MODE_SCREENTOP}

RS_SUCCESS = 0
RS_TIMEOUT = 0xE2010004          # eLeapRS_Timeout: nothing arrived, not an error


class LeapError(RuntimeError):
    pass


# ------------------------------------------------------------------ structs
class LEAP_VECTOR(Structure):
    _pack_ = 1
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]

    def array(self):
        return np.array([self.x, self.y, self.z], dtype=float)


class LEAP_QUATERNION(Structure):
    _pack_ = 1
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float), ("w", c_float)]


class LEAP_BONE(Structure):
    _pack_ = 1
    _fields_ = [("prev_joint", LEAP_VECTOR), ("next_joint", LEAP_VECTOR),
                ("width", c_float), ("rotation", LEAP_QUATERNION)]


class LEAP_DIGIT(Structure):
    _pack_ = 1
    _fields_ = [("finger_id", c_int32), ("bones", LEAP_BONE * 4),
                ("is_extended", c_uint32)]


class LEAP_PALM(Structure):
    _pack_ = 1
    _fields_ = [("position", LEAP_VECTOR), ("stabilized_position", LEAP_VECTOR),
                ("velocity", LEAP_VECTOR), ("normal", LEAP_VECTOR),
                ("width", c_float), ("direction", LEAP_VECTOR),
                ("orientation", LEAP_QUATERNION)]


class LEAP_HAND(Structure):
    _pack_ = 1
    _fields_ = [("id", c_uint32), ("flags", c_uint32), ("type", c_uint32),
                ("confidence", c_float), ("visible_time", c_uint64),
                ("pinch_distance", c_float), ("grab_angle", c_float),
                ("pinch_strength", c_float), ("grab_strength", c_float),
                ("palm", LEAP_PALM), ("digits", LEAP_DIGIT * 5),
                ("arm", LEAP_BONE)]


class LEAP_FRAME_HEADER(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_void_p), ("frame_id", c_int64),
                ("timestamp", c_int64)]


class LEAP_TRACKING_EVENT(Structure):
    _pack_ = 1
    _fields_ = [("info", LEAP_FRAME_HEADER), ("tracking_frame_id", c_int64),
                ("nHands", c_uint32), ("pHands", POINTER(LEAP_HAND)),
                ("framerate", c_float)]


class LEAP_CONNECTION_MESSAGE(Structure):
    _pack_ = 1
    _fields_ = [("size", c_uint32), ("type", c_uint32), ("pointer", c_void_p),
                ("device_id", c_uint32)]


# LEAP_CONNECTION_CONFIG is deliberately not declared. The installed service
# rejects a filled-in one with eLeapRS_InvalidArgument whatever `size` says,
# and LeapCreateConnection documents NULL as "default configuration" - which
# is the default namespace and a device-centre tracking origin, i.e. exactly
# what this wants. One less struct to keep in step with the SDK.


def self_test():
    """Byte counts for the packed layout LeapC.h declares.

    The header wraps itself in `#pragma pack(1)`, so nothing is padded and
    LEAP_HAND is 1084 bytes rather than the 1088 natural alignment would give.
    Getting that wrong does not raise - it silently reads floats out of the
    middle of other floats - so the sizes are asserted before anything is read
    and the message-size check in `poll` confirms it against the live SDK."""
    expected = {LEAP_VECTOR: 12, LEAP_QUATERNION: 16, LEAP_BONE: 44,
                LEAP_DIGIT: 184, LEAP_PALM: 80, LEAP_HAND: 1084,
                LEAP_FRAME_HEADER: 24, LEAP_CONNECTION_MESSAGE: 20}
    bad = {t.__name__: (sizeof(t), size) for t, size in expected.items()
           if sizeof(t) != size}
    if bad:
        raise LeapError(f"LeapC struct layout mismatch (got, want): {bad}")


def load_library(path=None):
    tried = []
    for candidate in ([path] if path else LIB_CANDIDATES):
        if not candidate:
            continue
        tried.append(candidate)
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise LeapError(
        "libLeapC not found. Install the Ultraleap Hand Tracking service, or "
        "point LEAPC_LIB at the dylib.\n  tried: " + "\n         ".join(tried))


def _bind(lib):
    lib.LeapCreateConnection.argtypes = [c_void_p, POINTER(c_void_p)]
    lib.LeapCreateConnection.restype = c_uint32
    lib.LeapOpenConnection.argtypes = [c_void_p]
    lib.LeapOpenConnection.restype = c_uint32
    lib.LeapPollConnection.argtypes = [c_void_p, c_uint32,
                                       POINTER(LEAP_CONNECTION_MESSAGE)]
    lib.LeapPollConnection.restype = c_uint32
    lib.LeapSetTrackingMode.argtypes = [c_void_p, c_uint32]
    lib.LeapSetTrackingMode.restype = c_uint32
    lib.LeapCloseConnection.argtypes = [c_void_p]
    lib.LeapCloseConnection.restype = None
    lib.LeapDestroyConnection.argtypes = [c_void_p]
    lib.LeapDestroyConnection.restype = None
    return lib


# ------------------------------------------------------------------ frames
@dataclass
class Hand:
    """One tracked hand, copied out of the SDK's memory.

    Copied and not referenced on purpose: LeapPollConnection documents that
    the pointers in an event stay valid only until the *next* poll, so a Hand
    that borrowed them would quietly start describing a later frame.

    Positions are millimetres in the device frame. `bones[digit]` is a (5, 3)
    array: the four bone joints of that digit plus the fingertip, base first.
    """

    id: int
    type: str                       # "left" or "right"
    palm: np.ndarray
    normal: np.ndarray              # out of the front of the palm
    direction: np.ndarray           # palm toward the fingers
    width: float
    pinch_strength: float
    grab_strength: float
    pinch_distance: float
    bones: dict                     # digit name -> (5, 3) joint positions
    # The SDK's own estimate of how well it can see this hand, 0 to 1. Not
    # recorded before September 2026 - the recorder dropped it - so it reads
    # NaN off older sessions. It is the one field that says whether the
    # fingers in a fist were observed or filled in by the tracker's model.
    confidence: float = float("nan")

    def tip(self, digit):
        return self.bones[digit][4]

    def bone_direction(self, digit, bone):
        """Unit vector along one bone, base to tip. Zero for a zero-length
        bone - the thumb's metacarpal is deliberately one of those."""
        joints = self.bones[digit]
        index = BONES.index(bone)
        vector = joints[index + 1] - joints[index]
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-6 else np.zeros(3)


def _read_hand(hand):
    bones = {}
    for slot, name in enumerate(DIGITS):
        digit = hand.digits[slot]
        joints = [digit.bones[0].prev_joint.array()]
        joints += [digit.bones[i].next_joint.array() for i in range(4)]
        bones[name] = np.array(joints)
    return Hand(
        id=int(hand.id),
        type="right" if hand.type == 1 else "left",
        palm=hand.palm.position.array(),
        normal=hand.palm.normal.array(),
        direction=hand.palm.direction.array(),
        width=float(hand.palm.width),
        pinch_strength=float(hand.pinch_strength),
        grab_strength=float(hand.grab_strength),
        pinch_distance=float(hand.pinch_distance),
        bones=bones,
        confidence=float(hand.confidence),
    )


@dataclass
class Frame:
    frame_id: int
    timestamp: int
    framerate: float
    hands: list

    def hand(self, prefer="right"):
        """One hand: the preferred chirality if it is there, else whatever is.

        With two hands in view the choice has to be stable - picking "the
        first one" makes the robot swap which hand it is following whenever
        the service reorders the array.
        """
        if not self.hands:
            return None
        for hand in self.hands:
            if hand.type == prefer:
                return hand
        return self.hands[0]


class LeapConnection:
    """An open connection to the tracking service, polled frame by frame.

    Poll returns the newest tracking event and drops the rest: a stale
    fingertip is worth nothing to a hand that is trying to follow a live one.
    """

    def __init__(self, mode="desktop", library=None, timeout_ms=100):
        self_test()
        self.lib = _bind(load_library(library))
        self.mode = mode
        self.timeout_ms = timeout_ms
        self.handle = c_void_p()
        self.connected = False
        self.device_count = 0
        self.last_event = None

    # ------------------------------------------------------------ lifecycle
    def open(self):
        status = self.lib.LeapCreateConnection(None, byref(self.handle))
        if status != RS_SUCCESS:
            raise LeapError(f"LeapCreateConnection failed: 0x{status:08X}")
        status = self.lib.LeapOpenConnection(self.handle)
        if status != RS_SUCCESS:
            raise LeapError(
                f"LeapOpenConnection failed: 0x{status:08X} - is the Ultraleap "
                f"Hand Tracking service running?")
        return self

    def wait_for_device(self, seconds=5.0):
        """Pump events until the service says it is connected and has a camera.

        Returns whether a device turned up. It is not fatal if none does: the
        connection stays open and a camera plugged in later will just start
        producing frames.
        """
        deadline = time.time() + seconds
        while time.time() < deadline and not (self.connected and self.device_count):
            self.poll()
        return self.connected and self.device_count > 0

    def no_device_reason(self):
        """Why `wait_for_device` came back empty, in words that point at a fix.

        TWO DIFFERENT FAILURES used to get one message - "is one plugged in,
        and the Ultraleap service running?" - written out at three call sites.
        They need different hands on different things:

          * not connected: the service is down or unreachable. Start it.
          * connected, no device: the service is up and talking - it even
            negotiates policy - but has not bound a camera. That is what
            happened on 12 Sep with the camera on USB and `libtrack_server`
            running; the old message sent the search to the service, and a
            replug fixed it.

        The last event is named, because "stuck after policy" is the whole
        diagnosis and a bare event id is not.
        """
        last = ("nothing received" if self.last_event is None
                else EVENT_NAMES.get(self.last_event, f"id {self.last_event}"))
        if not self.connected:
            return ("cannot reach the Ultraleap tracking service - is it "
                    f"running? (last event: {last})")
        return ("the tracking service is up but has no camera bound - unplug "
                "the Leap, wait a few seconds and plug it back in, or restart "
                f"the service (last event: {last})")

    def close(self):
        if self.handle:
            self.lib.LeapCloseConnection(self.handle)
            self.lib.LeapDestroyConnection(self.handle)
            self.handle = c_void_p()
        self.connected = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    # ----------------------------------------------------------------- poll
    def poll(self, timeout_ms=None):
        """The newest tracking frame, or None if none arrived.

        Drains whatever the service has queued rather than returning the first
        one: only the latest frame describes where the hand is now.
        """
        message = LEAP_CONNECTION_MESSAGE()
        timeout = self.timeout_ms if timeout_ms is None else timeout_ms
        newest = None
        while True:
            status = self.lib.LeapPollConnection(self.handle, timeout,
                                                 byref(message))
            if status == RS_TIMEOUT:
                return newest
            if status != RS_SUCCESS:
                raise LeapError(f"LeapPollConnection failed: 0x{status:08X}")
            self.last_event = int(message.type)
            if message.size != sizeof(LEAP_CONNECTION_MESSAGE):
                # The SDK stamps its own struct size into every message. If it
                # disagrees with ours, the union pointer is not where we think
                # it is and reading through it would be a wild dereference.
                raise LeapError(
                    f"LEAP_CONNECTION_MESSAGE is {message.size} bytes to the "
                    f"SDK, {sizeof(LEAP_CONNECTION_MESSAGE)} here - the binding "
                    f"does not match the installed LeapSDK")

            if message.type == EVENT_TRACKING:
                newest = self._tracking(message)
            elif message.type == EVENT_CONNECTION:
                self.connected = True
            elif message.type in (EVENT_CONNECTION_LOST, EVENT_DEVICE_FAILURE):
                self.connected = False
            elif message.type == EVENT_DEVICE:
                self.device_count += 1
                self.connected = True
                self.lib.LeapSetTrackingMode(self.handle,
                                             MODES.get(self.mode, MODE_DESKTOP))
            elif message.type == EVENT_DEVICE_LOST:
                self.device_count = max(0, self.device_count - 1)
            # after the first message, take only what is already queued
            timeout = 0

    def _tracking(self, message):
        event = ctypes.cast(message.pointer,
                            POINTER(LEAP_TRACKING_EVENT)).contents
        hands = [_read_hand(event.pHands[i]) for i in range(event.nHands)]
        return Frame(frame_id=int(event.info.frame_id),
                     timestamp=int(event.info.timestamp),
                     framerate=float(event.framerate), hands=hands)
