"""The one place settings come from: `settings.json` at the project root.

Every tunable in the project is declared once, in DEFAULTS below. Code never
keeps its own copy of a default; it asks here:

    from src import settings
    port = settings.get("hardware.port")
    hw = settings.section("hardware")          # a copy of the whole block
    settings.set("hardware.port", "/dev/cu.usbserial-XYZ")
    settings.save()

The file is created from DEFAULTS on first run and is meant to be edited by
hand. Whatever is in it is checked on load, and the same check guards every
`set` (the GUI edits through it too): a value of the wrong type, or not in
its CHOICES, falls back to the default; a number outside its RANGES is
clamped; unknown keys are dropped and missing ones filled in. A bad edit
costs one setting, never the run.

There are no command-line switches. To change how the world loads, change a
value here; every way of starting the simulator reads the same file.
"""

from __future__ import annotations

import copy
import json
import os
import re

from .paths import SETTINGS_FILE

NOTE = ("handlab settings. Edit freely; a wrong value falls back to its "
        "default. Delete the file to restore every default.")

# The measured motor map of the bench h4nd_left: which DYNAMIXEL ID drives
# which URDF joint, which way it is mounted (sign -1: a positive joint angle
# needs a negative tick delta), and which tick is joint zero. Every flexion
# servo is mounted backwards; the four spread joints are not - by what the
# joint moves, so the thumb's spread is `thumb_proximal`.
_MOTORS = {
    "index_adduction_joint": (1, 1), "index_proximal_joint": (2, -1),
    "index_middle_joint": (3, -1), "index_distal_joint": (4, -1),
    "middle_adduction_joint": (5, 1), "middle_proximal_joint": (6, -1),
    "middle_middle_joint": (7, -1), "middle_distal_joint": (8, -1),
    "ring_adduction_joint": (9, 1), "ring_proximal_joint": (10, -1),
    "ring_middle_joint": (11, -1), "ring_distal_joint": (12, -1),
    "thumb_proximal_joint": (13, 1), "thumb_adduction_joint": (14, -1),
    "thumb_middle_joint": (15, -1), "thumb_distal_joint": (16, -1),
}

DEFAULTS = {
    "robot": {
        # The robot, as exported. Relative paths are relative to the project.
        "urdf": "assets/h4nd_left/h4nd_left.urdf",
    },
    "world": {
        "timestep": 0.002,
        "hand_position": [0.0, 0.0, 0.2],      # where the palm is mounted, m
        # How the hand is mounted, i.e. which way gravity pulls on it. Set this
        # to match the real hand on the bench. The six choices are the six
        # directions gravity can take in the hand's frame (world.ORIENTATIONS).
        "hand_orientation": "upright",
        # Encoder hand: the real robot, posed from its encoders. OPAQUE, and
        # the command hand translucent, on purpose: two translucent hands in
        # one place flicker, because translucent objects are drawn sorted by
        # distance and overlapping links tie. One opaque hand is drawn first
        # and depth-tested, so there is no sort to tie. Measured with Studio's
        # renderer under +-2 ticks of encoder noise: both translucent 1.4-2.5 %
        # of pixels flickering; this way 0.14-0.39 %, of which 0.04-0.10 % is
        # the real hand genuinely moving. Command opaque instead: 1.6-3.3 %.
        # Emission (self-lighting, 0..1) on the opaque hand makes the purple
        # over it flicker more (up to 1.6 %), so it stays 0.
        "encoder_rgba": [0.92, 0.92, 0.95, 1.0],
        "encoder_emission": 0.0,
        # Command hand: the joint state the software sends out. Translucent.
        "command_rgba": [0.60, 0.28, 0.95, 0.40],
        "command_emission": 0.0,
    },
    "control": {
        # Slew limit on the command, rad/s. The default is the "smooth"
        # preset's: what the servo's own profile delivers at 200 counts x
        # 0.229 rpm = 4.8 rad/s, so the command cannot run away from the hand
        # on a large move. The motion presets below change it.
        "max_speed": 4.8,
    },
    "hardware": {
        # Serial port of the U2D2. "" = no hardware, "mock" = a fake bus that
        # echoes commands back as encoder readings.
        "port": "",
        "baudrate": 4_000_000,
        "command_hz": 100.0,        # goal writes per second
        "read_hz": 20.0,            # encoder reads per second (16 ms each)
        "current_limit_ma": 300,
        "profile_velocity": 200,    # 0.229 rpm per count; 0 = unlimited
        "profile_acceleration": 20,
        "position_p": 250,
        "position_i": 0,
        "position_d": 600,
        # Temperature protection (hardware/thermal.py): above derate_c a joint's
        # current cap falls linearly, in 10 % steps; at cutoff_c torque goes
        # off. The servo's own hard limit is 70 C, which latches an error that
        # needs a reboot.
        "temp_derate_c": 45.0,
        "temp_cutoff_c": 60.0,
        "motors": {name: {"id": motor_id, "sign": sign, "zero": 2048}
                   for name, (motor_id, sign) in _MOTORS.items()},
    },
    # Quick settings for HOW the hand moves: each is a named set of the three
    # values that shape the motion. Picking one in the GUI copies them into
    # control.max_speed and hardware.profile_*. PID and the current limit are
    # left alone - they are how the servo holds, not how the motion is shaped.
    # Add your own here; it shows up in the dropdown.
    "motion_presets": {
        # The bench-measured defaults: the servo eases in over 0.64 s, so a
        # goal nobody meant cannot arrive before it can be seen. Lags up to
        # ~28 deg behind the command on a large move.
        "smooth": {"max_speed": 4.8, "profile_velocity": 200,
                   "profile_acceleration": 20},
        # 7.2 rad/s, ramping in 0.24 s.
        "fast": {"max_speed": 7.2, "profile_velocity": 300,
                 "profile_acceleration": 80},
        # No servo profile: the servo goes straight at each goal, so the
        # 100 Hz command stream IS the trajectory and the only speed limit is
        # max_speed. Least lag; target jumps (Zero all) arrive hard.
        "real time": {"max_speed": 8.0, "profile_velocity": 0,
                      "profile_acceleration": 0},
    },
    # A calibration object to grasp and measure against: a box in front of the
    # palm. Fixed = welded where it is placed; floating = free under gravity,
    # it falls, can be pushed, and rests on the palm or the floor. It collides
    # with the real (white) hand and the floor, never with the command ghost.
    "object": {
        "enabled": True,
        "mode": "fixed",
        # Edge lengths along the hand's x (out of the palm), y (across the
        # palm) and z (along the fingers), mm.
        "size_mm": [55.0, 55.0, 55.0],
        "mass_g": 20.0,
        # Sliding friction coefficient of every contact the object makes (with
        # the hand and the floor). Default: PLA, an ESTIMATE - published FDM-PLA
        # values are 0.35-0.55, all measured against steel (0.492, Pawlak 2018,
        # Engineering Mechanics, pin on disc); no PLA-on-PLA measurement was
        # found. A tilt test on the real parts gives the true value: tilt a
        # printed surface with the printed object on it, and mu = tan(angle at
        # which the object starts to slide).
        "friction": 0.5,
        # Centre, in the hand's mount frame, mm: 8 mm clear of the palm,
        # centred across and along the fingers. Turns with the hand.
        "position_mm": [35.0, 0.0, 70.0],
        "rgba": [0.95, 0.62, 0.20, 1.0],
    },
    # A camera watching the human hand, and a hand-tracking model run on it
    # (src/vision). The preview is the Camera window in the viewer.
    "vision": {
        "enabled": False,               # the camera only turns on when asked
        "camera_index": 0,              # which camera OpenCV opens
        "capture_width": 640,
        "capture_height": 480,
        # Show the preview flipped left-right, like a mirror. Preview only:
        # tracking and recordings always use the camera's true image.
        "mirror": True,
        "max_hands": 2,
        "min_detection": 0.5,           # confidence to find a new hand
        "min_tracking": 0.5,            # confidence to keep following one
        # MediaPipe's hand landmarker (Apache-2.0), from
        # storage.googleapis.com/mediapipe-models/hand_landmarker/
        "model": "assets/models/hand_landmarker.task",
        # The calibration checkerboard: inner corners across and down (a
        # board of 10 x 7 squares has 9 x 6), and one square's width and
        # height in mm, measured on the board as shown - a board on a screen
        # can be stretched, and squares that are not square must be said so.
        "checkerboard": [9, 6],
        "checkerboard_square_mm": [25.0, 25.0],
        # ArUco markers, for testing: looked for in every frame - drawn on
        # the preview, and where one is seen Fit 3D takes the object's pose
        # from it (a marker in the middle of one face of the object) and
        # measures the markerless fit against it. aruco_mm: the printed
        # marker's black square, side; print with scripts/aruco_markers.py.
        "aruco": True,
        "aruco_dictionary": "DICT_4X4_50",
        "aruco_mm": 40.0,
    },
    # Each camera's own numbers, by camera name, from its calibration
    # (src/vision/calibration.py; Camera section > Calibrate). fx, fy, cx, cy
    # in px at width x height; distortion: OpenCV's k1, k2, p1, p2, k3.
    "camera_calibration": {},
    # Recordings of a person doing a task (src/recording): one folder per
    # recording, under <folder>/<task>/.
    "recording": {
        "folder": "data/tasks",
        "task": "untitled",
    },
    # Masks of each object the task context names, through a whole
    # recording (src/processing/segment.py). SAM 2.1 (open) starts from the
    # context's box for each object; SAM 3 (gated: needs HF_TOKEN in the
    # environment or handlab/.env) finds each object by its name.
    "segment": {
        "model": "facebook/sam2.1-hiera-small",
    },
    # A vision-language model that says what task the frames show, as
    # structured data (src/assistant). API keys are NOT settings: they come
    # from ANTHROPIC_API_KEY / OPENAI_API_KEY, handlab/.env, or a session-only
    # field in the window, and are never written to this file.
    "assistant": {
        "provider": "claude",
        "claude_model": "claude-opus-5-5",
        "claude_effort": "medium",
        "openai_model": "gpt-6-astra",
        "max_tokens": 16000,
        # px; frames are scaled down to this. At most 2000: Claude's limit
        # for each image when a request has more than 20.
        "image_max_side": 1024,
        "clip_frames": 16,              # frames sent from a recording, spread evenly
        "system_prompt": (
            "You describe what a person is doing in camera footage of their hands "
            "handling objects. Your description becomes the task specification for "
            "teaching a robot hand the same task: object names are used as prompts "
            "for a segmentation model, boxes start tracking the objects, and phases "
            "split the recording into steps. Describe the task by what happens to "
            "the objects - where they go, how they turn, how they end up - not by "
            "how the fingers move. Name objects by what can be seen. When the "
            "images do not show something clearly, say so under uncertain instead "
            "of guessing."),
    },
    "viewer": {
        "width": 1500,
        "height": 950,
        # Studio redraws as fast as it can (~300 frames/s measured) - it has
        # no vsync setting - which keeps the GPU and fans busy for nothing.
        "max_fps": 60,
    },
}

# Dicts whose KEYS are data rather than fixed names: any key is allowed, and
# each value must look like the shape given. In RANGES and CHOICES below such
# a key is written "*".
MAPS = {
    "hardware.motors": {"id": 1, "sign": 1, "zero": 2048},
    "camera_calibration": {"width": 640, "height": 480, "fx": 0.0, "fy": 0.0,
                           "cx": 0.0, "cy": 0.0, "distortion": [0.0] * 5,
                           "rms_px": 0.0, "views": 0, "made": "",
                           "square_mm": [0.0, 0.0]},
    "motion_presets": {"max_speed": 4.8, "profile_velocity": 200,
                       "profile_acceleration": 20},
}
# Which setting each preset field is copied into.
PRESET_FIELDS = {"max_speed": "control.max_speed",
                 "profile_velocity": "hardware.profile_velocity",
                 "profile_acceleration": "hardware.profile_acceleration"}

# What a number may be, beyond its type. Out of range is clamped.
RANGES = {
    "world.timestep": (0.0001, 0.05),
    "world.encoder_emission": (0.0, 1.0),
    "world.command_emission": (0.0, 1.0),
    "control.max_speed": (0.1, 20.0),
    "hardware.command_hz": (1.0, 500.0),
    "hardware.read_hz": (1.0, 100.0),
    # XL330-M288 register limits.
    "hardware.current_limit_ma": (0, 1750),
    "hardware.profile_velocity": (0, 32767),
    "hardware.profile_acceleration": (0, 32767),
    "hardware.position_p": (0, 16383),
    "hardware.position_i": (0, 16383),
    "hardware.position_d": (0, 16383),
    "hardware.temp_derate_c": (20.0, 70.0),
    "hardware.temp_cutoff_c": (25.0, 70.0),
    "hardware.motors.*.id": (0, 252),
    "hardware.motors.*.zero": (0, 4095),
    "motion_presets.*.max_speed": (0.1, 20.0),
    "motion_presets.*.profile_velocity": (0, 32767),
    "motion_presets.*.profile_acceleration": (0, 32767),
    "object.size_mm[]": (5.0, 300.0),
    "object.mass_g": (0.1, 5000.0),
    "object.friction": (0.0, 1.5),
    "object.position_mm[]": (-200.0, 300.0),
    "vision.camera_index": (0, 9),
    "vision.capture_width": (160, 1920),
    "vision.capture_height": (120, 1080),
    "vision.max_hands": (1, 4),
    "vision.checkerboard[]": (3, 30),
    "vision.checkerboard_square_mm[]": (1.0, 200.0),
    "vision.aruco_mm": (5.0, 300.0),
    "vision.min_detection": (0.05, 0.95),
    "vision.min_tracking": (0.05, 0.95),
    "assistant.max_tokens": (256, 64000),
    "assistant.image_max_side": (256, 2000),
    "assistant.clip_frames": (1, 100),
    "viewer.width": (400, 7680),
    "viewer.height": (300, 4320),
    "viewer.max_fps": (10, 240),
}
# What a value may be, from a list. Anything else falls back to the default.
CHOICES = {
    "hardware.baudrate": (57_600, 115_200, 1_000_000, 2_000_000, 3_000_000, 4_000_000),
    "hardware.motors.*.sign": (1, -1),
    "world.hand_orientation": ("upright", "palm up", "palm down", "fingers down",
                               "thumb up", "thumb down"),
    "object.mode": ("fixed", "floating"),
    "assistant.provider": ("claude", "openai"),
    "vision.aruco_dictionary": ("DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250",
                                "DICT_ARUCO_ORIGINAL"),
    "segment.model": ("facebook/sam2.1-hiera-tiny", "facebook/sam2.1-hiera-small",
                      "facebook/sam2.1-hiera-base-plus", "facebook/sam2.1-hiera-large",
                      "facebook/sam3"),
    "assistant.claude_effort": ("low", "medium", "high", "xhigh", "max"),
}


def motion_preset(values=None):
    """Name of the preset the current settings match, or None (custom)."""
    values = values or current()
    for name, preset in values.get("motion_presets").items():
        if all(values.get(PRESET_FIELDS[field]) == value
               for field, value in preset.items()):
            return name
    return None


def apply_motion_preset(name, values=None):
    """Copy a preset into the settings it names. Call save() to keep it."""
    values = values or current()
    preset = values.get("motion_presets")[name]
    for field, value in preset.items():
        values.set(PRESET_FIELDS[field], value)


def limits(key):
    """(low, high) for a numeric setting - or for each element of a list
    setting - or None."""
    rule = _shape(key)[1]
    return RANGES.get(rule, RANGES.get(f"{rule}[]"))


def choices(key):
    """The allowed values of a setting, or None."""
    return CHOICES.get(_shape(key)[1])


# ------------------------------------------------------------------ checks
# Problems are (fatal, message). Fatal: the value was unusable and the default
# took its place. Not fatal: it was repaired (clamped, unknown key dropped).

def _shape(key):
    """(default, rule path) for a dotted key. KeyError if there is no such setting."""
    node, rule = DEFAULTS, ""
    for part in key.split("."):
        if rule in MAPS:
            node, rule = MAPS[rule], _join(rule, "*")
        elif isinstance(node, dict) and part in node:
            node, rule = node[part], _join(rule, part)
        else:
            raise KeyError(f"no such setting: {key}")
    return node, rule


def _merge(default, stored, path, rule, problems):
    """`stored` checked against `default`. Never raises."""
    if isinstance(default, dict):
        if not isinstance(stored, dict):
            problems.append((True, f"{path}: expected an object"))
            return copy.deepcopy(default)
        if rule in MAPS:
            return {key: _merge(MAPS[rule], value, _join(path, key),
                                _join(rule, "*"), problems)
                    for key, value in stored.items()}
        for key in stored:
            if key not in default and not key.startswith("_"):
                problems.append((False, f"{_join(path, key)}: unknown, dropped"))
        return {key: _merge(value, stored.get(key, value), _join(path, key),
                            _join(rule, key), problems)
                for key, value in default.items()}
    return _leaf(default, stored, path, rule, problems)


def _leaf(default, value, path, rule, problems):
    ok = False
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif isinstance(default, str):
        ok = isinstance(value, str)
    elif isinstance(default, list):
        if isinstance(value, list) and len(value) == len(default):
            return [_leaf(d, v, f"{path}[{i}]", f"{rule}[]", problems)
                    for i, (d, v) in enumerate(zip(default, value))]
    if not ok:
        problems.append((True, f"{path}: {value!r} is not like {default!r}"))
        return copy.deepcopy(default)
    if rule in CHOICES and value not in CHOICES[rule]:
        problems.append((True, f"{path}: {value!r} is not one of "
                               f"{list(CHOICES[rule])}"))
        return copy.deepcopy(default)
    if rule in RANGES:
        low, high = RANGES[rule]
        clamped = type(default)(min(max(value, low), high))
        if clamped != value:
            problems.append((False, f"{path}: {value!r} clamped to {clamped!r}"))
        value = clamped
    return value


def _join(path, key):
    return f"{path}.{key}" if path else key


# ---------------------------------------------------------------- settings
class Settings:
    """Validated settings, backed by one JSON file."""

    def __init__(self, path=SETTINGS_FILE):
        self.path = path
        self.values = copy.deepcopy(DEFAULTS)
        self.problems = []
        self.load()

    def load(self):
        """Read the file. Writes it back when it is missing or was repaired,
        so the file on disk always lists every setting there is."""
        if not self.path.exists():
            self.values = copy.deepcopy(DEFAULTS)
            self.save()
            return self
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, ValueError) as failure:
            # Keep the file - it may be a hand edit worth fixing - and run on
            # the defaults.
            self.problems = [(True, f"{self.path.name} unreadable ({failure})")]
            self.values = copy.deepcopy(DEFAULTS)
            self._report()
            return self
        self.problems = []
        self.values = _merge(DEFAULTS, stored, "", "", self.problems)
        filled = _strip(stored) != self.values
        self._report()
        if filled:
            self.save()
        return self

    def save(self):
        """Atomic: the file is either the old one or the new one, never half."""
        text = json.dumps({"_note": NOTE, **self.values}, indent=2)
        # Lists of plain values on one line: [0.6, 0.28, 0.95, 0.4].
        text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]",
                      lambda m: "[" + ", ".join(
                          v.strip() for v in m.group(1).split(",")) + "]",
                      text) + "\n"
        staged = self.path.with_suffix(".json.tmp")
        staged.write_text(text)
        os.replace(staged, self.path)

    # -------------------------------------------------------------- access
    def get(self, key):
        """A value by dotted path: get("hardware.port")."""
        node = self.values
        for part in key.split("."):
            node = node[part]
        return copy.deepcopy(node)

    def section(self, name):
        return self.get(name)

    def set(self, key, value):
        """Store one value, checked against its default, and return what was
        stored (a number out of range comes back clamped). Raises ValueError
        for a value that cannot be used. Call save() to keep it."""
        default, rule = _shape(key)
        problems = []
        checked = _merge(default, value, key, rule, problems)
        fatal = [message for is_fatal, message in problems if is_fatal]
        if fatal:
            raise ValueError("; ".join(fatal))
        *parents, last = key.split(".")
        node = self.values
        for part in parents:
            node = node[part]
        node[last] = checked
        return copy.deepcopy(checked)

    def _report(self):
        for fatal, message in self.problems:
            print(f"[settings] {message}" + (" - using default" if fatal else ""))


def _strip(stored):
    """The stored dict without comment keys, for comparing against values."""
    if isinstance(stored, dict):
        return {k: _strip(v) for k, v in stored.items() if not k.startswith("_")}
    return stored


# One instance for the whole program.
_current = None


def current():
    global _current
    if _current is None:
        _current = Settings()
    return _current


def get(key):
    return current().get(key)


def section(name):
    return current().section(name)


def set(key, value):  # noqa: A001 - the module reads as settings.set(...)
    return current().set(key, value)


def save():
    current().save()
