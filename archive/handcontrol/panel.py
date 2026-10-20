"""The pygame control panel: sixteen sliders, or a fingertip dragged in space.

Runs as its own process and sends commands over UDP - see net.py for why it
cannot simply be a second window inside the simulator.

The controls are commands, not readings. Where things actually ended up comes
back from the simulator and is drawn alongside - a white tick on each joint
track, hollow dots for the real fingertips - so a finger held short of its
target is visible rather than silent.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from .hardware import (CURRENT_LIMIT_MA, POSITION_D, POSITION_I, POSITION_P,
                       PROFILE_ACCELERATION, PROFILE_VELOCITY,
                       list_serial_ports)
from .robot.model import (DEFAULT_HAND, FINGERS, POSES, TIP_LINKS, HandModel,
                    ensure_urdf, roles_in_order, spread_signs)
from .net import COMMAND_PORT, STATE_PORT, Channel
from .settings import DEFAULTS, Settings

WIDTH, HEIGHT = 600, 868
STRIP_TOP = 116          # the hardware strip: port, output, then a status line
STRIP_HEIGHT = 96
CARD_TOP = 224           # the mode card below it

BG = (18, 20, 24)
CARD = (26, 29, 35)
LINE = (44, 49, 58)
GRID = (36, 40, 48)
TEXT = (231, 234, 239)
DIM = (124, 132, 146)
FAINT = (74, 81, 94)
ACCENT = (91, 157, 255)
WARN = (255, 184, 107)
GOOD = (126, 231, 135)
LIVE = (255, 86, 96)
GHOST = (255, 158, 71)          # the encoder hand, in the viewer and here
FINGER_COLOR = {"index": (91, 157, 255), "middle": (126, 231, 135),
                "ring": (255, 184, 107), "thumb": (255, 123, 156)}
# The operator's hand. Same hues as the robot's fingers so the two read as
# corresponding, and a dim grey pinky - the robot has no little finger, and
# drawing what the mapping throws away is more honest than hiding it.
SKELETON_COLOR = {**FINGER_COLOR, "pinky": (108, 116, 130)}


def faded(color, amount=0.42):
    """Toward the background, for the fingers that are not being controlled."""
    return tuple(int(c * amount + b * (1.0 - amount)) for c, b in zip(color, BG))


class Slider:
    """A horizontal track with a knob, and a tick for the value really reached."""

    def __init__(self, rect, low, high, value, label, unit="deg", color=ACCENT,
                 sign=1, fmt="6.1f"):
        self.rect = pygame.Rect(rect)
        self.fmt = fmt
        self.low, self.high = low, high
        # `value` is always the true joint angle. `sign` only decides which way
        # up it is drawn, so that "spread" means the same thing on every
        # finger even where the URDF's shared adduction axis says otherwise.
        self.sign = sign
        self.value = value
        self.label = label
        self.unit = unit
        self.color = color
        self.actual = None       # what the simulation reached
        self.measured = None     # what the real hand's encoder says
        self.held = False

    @property
    def display_low(self):
        return self.low if self.sign > 0 else -self.high

    @property
    def display_high(self):
        return self.high if self.sign > 0 else -self.low

    @property
    def track(self):
        return pygame.Rect(self.rect.x + 104, self.rect.centery - 3,
                           self.rect.width - 104 - 74, 6)

    def _to_x(self, value):
        track = self.track
        low, high = self.display_low, self.display_high
        span = max(high - low, 1e-9)
        fraction = min(max((value * self.sign - low) / span, 0.0), 1.0)
        return track.x + fraction * track.width

    def hit(self, pos):
        return self.track.inflate(16, 20).collidepoint(pos)

    def set_from_pos(self, pos):
        track = self.track
        low, high = self.display_low, self.display_high
        fraction = min(max((pos[0] - track.x) / max(track.width, 1), 0.0), 1.0)
        self.value = (low + fraction * (high - low)) * self.sign

    def draw(self, surface, font, small):
        track = self.track
        pygame.draw.rect(surface, LINE, track, border_radius=3)

        zero_x = (self._to_x(0.0)
                  if self.display_low < 0 < self.display_high else None)
        if zero_x is not None:
            pygame.draw.line(surface, FAINT, (zero_x, track.y - 5),
                             (zero_x, track.bottom + 5), 1)

        knob_x = self._to_x(self.value)
        origin = zero_x if zero_x is not None else track.x
        pygame.draw.rect(surface, self.color,
                         pygame.Rect(min(origin, knob_x), track.y,
                                     abs(knob_x - origin), track.height),
                         border_radius=3)

        if self.actual is not None:
            x = self._to_x(self.actual)
            pygame.draw.line(surface, TEXT, (x, track.y - 7),
                             (x, track.bottom + 7), 1)
        if self.measured is not None:
            x = self._to_x(self.measured)
            pygame.draw.line(surface, GHOST, (x, track.y - 7),
                             (x, track.bottom + 7), 2)

        pygame.draw.circle(surface, TEXT if self.held else self.color,
                           (int(knob_x), track.centery), 7 if self.held else 6)
        pygame.draw.circle(surface, BG, (int(knob_x), track.centery), 3)

        surface.blit(small.render(self.label, True, DIM),
                     (self.rect.x, self.rect.centery - 7))
        if self.unit == "deg":
            text = f"{math.degrees(self.value * self.sign):6.1f}°"
        elif self.unit == "mm":
            text = f"{self.value * 1000:6.1f} mm"
        else:
            # `fmt` so an integer register reads as 250, not 250.00 - these
            # sliders carry servo values as often as physical quantities.
            text = f"{self.value * self.sign:{self.fmt}}"
        label = font.render(text, True, TEXT)
        surface.blit(label, (self.rect.right - label.get_width(),
                             self.rect.centery - 8))


class Pad:
    """A 2D view of the hand you can drag a fingertip around in.

    Two of these, on different pairs of axes, beat three sliders for the part
    of the job that is actually spatial: the other fingers are drawn in too, so
    a target is placed relative to the hand rather than to a number.
    """

    def __init__(self, rect, axis_x, axis_y, label):
        self.rect = pygame.Rect(rect)
        self.axis_x, self.axis_y = axis_x, axis_y
        self.label = label
        self.low = [0.0, 0.0]
        self.high = [1.0, 1.0]
        self.held = False

    @property
    def field(self):
        return self.rect.inflate(-24, -34).move(0, 5)

    def set_ranges(self, ranges):
        self.low = [ranges[self.axis_x][0], ranges[self.axis_y][0]]
        self.high = [ranges[self.axis_x][1], ranges[self.axis_y][1]]

    def to_screen(self, point):
        field = self.field
        fx = (point[self.axis_x] - self.low[0]) / max(self.high[0] - self.low[0], 1e-9)
        fy = (point[self.axis_y] - self.low[1]) / max(self.high[1] - self.low[1], 1e-9)
        return (field.x + min(max(fx, 0.0), 1.0) * field.width,
                field.bottom - min(max(fy, 0.0), 1.0) * field.height)

    def hit(self, pos):
        return self.field.collidepoint(pos)

    def apply(self, pos, target):
        """Drag writes only this pad's two axes; the third is left alone."""
        field = self.field
        fx = min(max((pos[0] - field.x) / max(field.width, 1), 0.0), 1.0)
        fy = min(max((field.bottom - pos[1]) / max(field.height, 1), 0.0), 1.0)
        target[self.axis_x] = self.low[0] + fx * (self.high[0] - self.low[0])
        target[self.axis_y] = self.low[1] + fy * (self.high[1] - self.low[1])
        return target

    def draw_skeleton(self, surface, skeleton, colors=None, width=2, joint=2):
        """The operator's hand, mapped into the same world metres as the robot.

        Drawn in the pad's own two axes, so it sits directly on top of the
        robot's fingertips - which is the whole point of the view. A hand that
        looks offset from the dots is a mapping that IS offset, and there is
        nowhere for that error to hide.
        """
        colors = SKELETON_COLOR if colors is None else colors
        for digit, points in skeleton.items():
            color = colors.get(digit, DIM)
            screen = [self.to_screen(point) for point in points]
            for start, end in zip(screen[:-1], screen[1:]):
                pygame.draw.line(surface, color, start, end, width)
            for point in screen:
                pygame.draw.circle(surface, color, point, joint)

    def draw(self, surface, small, tips, selected, target, skeleton=None,
             robot=None):
        pygame.draw.rect(surface, BG, self.rect, border_radius=8)
        pygame.draw.rect(surface, LINE, self.rect, 1, border_radius=8)
        surface.blit(small.render(self.label, True, DIM),
                     (self.rect.x + 10, self.rect.y + 7))

        field = self.field
        for i in range(1, 4):
            x = field.x + field.width * i / 4
            y = field.y + field.height * i / 4
            pygame.draw.line(surface, GRID, (x, field.y), (x, field.bottom))
            pygame.draw.line(surface, GRID, (field.x, y), (field.right, y))

        # The robot's own linkage, under everything else: a bare dot per
        # fingertip says nothing about which finger it belongs to or which way
        # that finger is bent, and picking one out of four to drag was
        # guesswork. Thin and dim, because it is the backdrop, not the reading.
        if robot:
            # The selected finger at full strength, the rest faded back. Four
            # linkages drawn identically are as hard to read as four dots
            # were; the point of the view is knowing which one the sliders
            # move before moving them.
            others = {f: pts for f, pts in robot.items() if f != selected}
            self.draw_skeleton(surface, others,
                               {f: faded(c) for f, c in FINGER_COLOR.items()},
                               width=1, joint=2)
            if selected in robot:
                self.draw_skeleton(surface, {selected: robot[selected]},
                                   FINGER_COLOR, width=3, joint=4)
        if skeleton:
            self.draw_skeleton(surface, skeleton)

        for finger, tip in tips.items():
            point = self.to_screen(tip)
            color = FINGER_COLOR[finger]
            radius = 5 if finger == selected else 3
            pygame.draw.circle(surface, color, point, radius, 1)

        if target is not None:
            x, y = self.to_screen(target)
            color = FINGER_COLOR[selected]
            pygame.draw.line(surface, color, (x - 8, y), (x + 8, y), 1)
            pygame.draw.line(surface, color, (x, y - 8), (x, y + 8), 1)
            pygame.draw.circle(surface, color, (x, y), 4)


class Button:
    def __init__(self, rect, label, color=ACCENT):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color
        self.active = False
        self.enabled = True

    def draw(self, surface, font):
        if self.active:
            pygame.draw.rect(surface, self.color, self.rect, border_radius=7)
        else:
            pygame.draw.rect(surface, CARD, self.rect, border_radius=7)
            pygame.draw.rect(surface, LINE, self.rect, 1, border_radius=7)
        color = BG if self.active else (TEXT if self.enabled else FAINT)
        label = font.render(self.label, True, color)
        surface.blit(label, (self.rect.centerx - label.get_width() // 2,
                             self.rect.centery - label.get_height() // 2))


class Dropdown:
    """Pick the serial port here rather than on the command line.

    Options are re-read every time it opens: plugging the U2D2 in after
    starting the panel is the normal case, not an edge one.
    """

    def __init__(self, rect, placeholder="no hand"):
        self.rect = pygame.Rect(rect)
        self.placeholder = placeholder
        self.options = []            # [(value, label)]
        self.value = ""
        self.open = False

    def label_for(self, value):
        for option, label in self.options:
            if option == value:
                return label
        return value or self.placeholder

    def item_rect(self, index):
        return pygame.Rect(self.rect.x, self.rect.bottom + 2 + index * 26,
                           self.rect.width, 26)

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def hit_item(self, pos):
        """The value under the cursor, or None. Only meaningful while open."""
        for index in range(len(self.options)):
            if self.item_rect(index).collidepoint(pos):
                return self.options[index][0]
        return None

    def draw(self, surface, font):
        pygame.draw.rect(surface, BG, self.rect, border_radius=7)
        pygame.draw.rect(surface, ACCENT if self.open else LINE, self.rect, 1,
                         border_radius=7)
        text = self.label_for(self.value)
        colour = TEXT if self.value else DIM
        surface.blit(font.render(text[:38], True, colour),
                     (self.rect.x + 10, self.rect.centery - 6))
        x = self.rect.right - 16
        y = self.rect.centery
        points = [(x - 5, y - 2), (x + 5, y - 2), (x, y + 4)]
        pygame.draw.polygon(surface, DIM, points)

    def draw_open(self, surface, font):
        """Drawn last, over everything - a menu that renders under the widget
        below it is a menu you cannot click."""
        if not self.open:
            return
        for index, (value, label) in enumerate(self.options):
            rect = self.item_rect(index)
            chosen = value == self.value
            pygame.draw.rect(surface, LINE if chosen else CARD, rect)
            pygame.draw.rect(surface, BG, rect, 1)
            surface.blit(font.render(label[:38], True,
                                     TEXT if chosen else DIM),
                         (rect.x + 10, rect.centery - 6))


class Panel:
    def __init__(self, command_port=COMMAND_PORT, state_port=STATE_PORT,
                 hand=DEFAULT_HAND, spread_limit=None, settings=None):
        model = self.model = HandModel.load(ensure_urdf(hand, spread_limit))
        self.joint_names = model.joint_names()
        self.chains = model.finger_chains()
        self.lower, self.upper = model.limits(self.joint_names)
        self.signs = spread_signs(model, self.joint_names)

        self.link = Channel(bind=state_port, send_to=command_port)
        self.settings = Settings(settings)
        if self.settings.error:
            print(f"[panel] {self.settings.error} - using defaults")
        self.mode = self.settings["tab"]
        self.finger = self.settings["finger"]
        self.origin = [0.0, 0.0, 0.20]
        self.tips = {}
        self.tip_error = {}
        self.skeleton = {}
        self.log = {"active": False, "rows": 0, "seconds": 0.0, "path": "",
                    "error": ""}
        # Mirrors what the simulator reports. The panel never decides whether
        # the Leap is on - it asks, and draws the answer, the same way it does
        # for the serial port.
        self.leap = {"enabled": False, "available": False, "tracked": False,
                     "mode": "angles", "thumb": "angles", "calibrated": False,
                     "calibrating": False, "error": "", "miss": {}}
        self.seen_leap = False
        self.ik_avoid = True
        self.hardware = {"available": False, "enabled": False, "armed": False,
                         "error": "", "port": "", "note": ""}
        self.seen_sim = False
        self.dragging = None
        self.tip_target = None

        pygame.init()
        pygame.display.set_caption("h4nd control")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        mono = pygame.font.match_font("menlo,dejavusansmono,consolas,couriernew")
        self.font = pygame.font.Font(mono, 13)
        self.small = pygame.font.Font(mono, 11)
        self.head = pygame.font.Font(mono, 17)

        self._robot_links = self.robot_links()
        self._robot_skeleton = {}
        self._build_widgets()
        self._build_tuning()
        self._set_tip_ranges()
        self._apply_settings()

    # ------------------------------------------------------------ widgets
    def _build_widgets(self):
        # Five tabs across 600 px, so they are narrower than the four were.
        self.tabs = [Button((24, 74, 104, 30), "JOINTS"),
                     Button((136, 74, 104, 30), "FINGERTIP"),
                     Button((248, 74, 104, 30), "MIMIC"),
                     Button((360, 74, 104, 30), "LEARNING"),
                     Button((472, 74, 104, 30), "TUNING")]
        self.tabs[0].active = True

        # Rows read spread / knuckle / middle / tip for every digit. The URDF
        # chain does not: the thumb's proximal joint comes before its adduction
        # one, so walking the chain puts the thumb's first two rows the
        # opposite way round to all three fingers, which reads as the spread
        # and knuckle controls being swapped.
        #
        # Only the LAYOUT is reordered. `joint_sliders` stays in joint-name
        # order, because that order is the wire format - the command payload
        # and the state that comes back are both positional.
        by_name = {}
        self.rows = []
        y = CARD_TOP + 8
        for finger in FINGERS:
            y += 18
            row = []
            for label, name in self.display_chain(finger):
                i = self.joint_names.index(name)
                by_name[name] = Slider(
                    (24, y, WIDTH - 48, 24), self.lower[i], self.upper[i], 0.0,
                    label, "deg", FINGER_COLOR[finger],
                    sign=self.signs[name])
                row.append(by_name[name])
                y += 25
            self.rows.append((finger, row))
            y += 7
        self.joint_sliders = [by_name[name] for name in self.joint_names]

        self.finger_tabs = [Button((24 + i * 135, CARD_TOP + 28, 126, 26), f.upper())
                            for i, f in enumerate(FINGERS)]
        self.finger_tabs[0].active = True

        # Looking at the palm (Y across, Z up) and from the side (X across,
        # Z up): between them, spread and curl are both draggable.
        self.pads = [Pad((24, CARD_TOP + 62, 268, 186), 1, 2, "front  Y →  Z ↑"),
                     Pad((308, CARD_TOP + 62, 268, 186), 0, 2, "side   X →  Z ↑")]

        self.tip_sliders = [
            Slider((24, CARD_TOP + 272 + i * 28, WIDTH - 48, 24), 0.0, 1.0, 0.0, axis,
                   "mm", ACCENT)
            for i, axis in enumerate("XYZ")]
        self.snap = Button((24, CARD_TOP + 360, 176, 26), "snap to current")

        # --- MIMIC. Every control here is a request: the simulator owns the
        # Leap, so the button states below are set from what it reports back,
        # never from the click that asked for them.
        # Two positions for the whole button row. During a capture the
        # calibration block owns the top of the card - it is the only thing
        # the operator should be reading - and these move down out of its way.
        self.mimic_button = Button((24, CARD_TOP + 28, 150, 26), "FOLLOW", GOOD)
        self.calibrate_button = Button((182, CARD_TOP + 28, 132, 26), "CALIBRATE")
        self.reset_fit_button = Button((322, CARD_TOP + 28, 74, 26), "CLEAR")
        self.mirror_button = Button((404, CARD_TOP + 28, 80, 26), "MIRROR", WARN)
        # Recording is a separate concern from following - you log to compare
        # two mappings, and switching mapping mid-log is the point - so it sits
        # beside them rather than inside either.
        self.log_button = Button((492, CARD_TOP + 28, 84, 26), "LOG", LIVE)
        # Only on screen during a capture. Advancing on a click rather than a
        # countdown means the operator races nothing: getting into the pose is
        # the whole job, and it takes as long as it takes.
        self.capture_button = Button((24, CARD_TOP + 40, 150, 26), "NEXT", GOOD)
        self.mimic_row = [self.mimic_button, self.calibrate_button,
                          self.reset_fit_button, self.mirror_button,
                          self.log_button]
        self.mimic_row_y = self.mimic_button.rect.y
        self.mimic_row_drop = 48          # how far they move while capturing
        self.mimic_mode_buttons = [
            Button((90, CARD_TOP + 84, 62, 24), "RAW"),
            Button((156, CARD_TOP + 84, 80, 24), "ANGLES"),
            Button((240, CARD_TOP + 84, 66, 24), "TIP IK"),
            Button((310, CARD_TOP + 84, 104, 24), "KEYVECTOR"),
            Button((418, CARD_TOP + 84, 98, 24), "POSEBANK")]
        # LEARNING's mode row. Two entries where MIMIC has five, because the
        # question on that tab is only which of the two learned-pipeline
        # mappings is driving: the solver, or the network trained to copy it.
        #
        # TEACHER is the LIVE solver, not the offline teacher. The offline one
        # smooths with a centred window - it reads frames that have not
        # happened - so it cannot run here at all. The label names the role,
        # and this comment names the difference.
        self.learning_mode_buttons = [
            Button((90, CARD_TOP + 84, 104, 24), "TEACHER"),
            Button((200, CARD_TOP + 84, 104, 24), "STUDENT")]
        # GESTURES plays the authored probe through the live world's own
        # lanes - the same thing `learn.py replay --synthetic
        # data/synthetic/gestures --student --loop` shows,
        # without a second window. Click again to stop.
        self.gestures_button = Button((330, CARD_TOP + 84, 104, 24), "GESTURES", LIVE)
        # REPLAY plays the newest recorded session's solved blocks the same
        # way - the teacher's own labels, block by block. With SIM + HAND on
        # and the servos armed, that is the offline solve on the real hand.
        self.replay_button = Button((440, CARD_TOP + 84, 90, 24), "REPLAY", LIVE)
        self.gestures = {"playing": False, "source": "", "label": "", "frame": 0, "frames": 0}
        self.mimic_thumb_buttons = [
            Button((90, CARD_TOP + 112, 100, 24), "ANGLES"),
            Button((194, CARD_TOP + 112, 100, 24), "PINCH")]

        # Reusing the FINGERTIP tab's pads: same two views, same world
        # coordinates, and only one tab is ever on screen.
        self.mimic_pads = [
            Pad((24, CARD_TOP + 146, 268, 160), 1, 2, "front  Y →  Z ↑"),
            Pad((308, CARD_TOP + 146, 268, 160), 0, 2, "side   X →  Z ↑")]

        # Ranges chosen for what they do, not for the number they hold:
        # smoothing below about 0.15 lags visibly, spread gain above ~1 throws
        # the fingers apart on a wobble, and scale is there to reach further
        # than your own hand does.
        self.mimic_sliders = {
            "smoothing": Slider((24, CARD_TOP + 406, WIDTH - 48, 24), 0.05, 1.0,
                                0.35, "smoothing", "raw", fmt="6.2f"),
            "spread_gain": Slider((24, CARD_TOP + 434, WIDTH - 48, 24), 0.0, 1.5,
                                  0.6, "spread gain", "raw", fmt="6.2f"),
            "scale": Slider((24, CARD_TOP + 462, WIDTH - 48, 24), 0.5, 1.6, 1.0,
                            "tip scale", "raw", fmt="6.2f")}

        self.port_menu = Dropdown((104, STRIP_TOP + 8, 330, 26))
        # FOUR BUTTONS ON ONE ROW, sized to fit rather than written down.
        # They were 118 px each with hand-written x positions, and four of
        # those plus their gaps come to 616 in a 600-pixel panel: GRIP sat 16
        # px ON TOP of SIM + HAND, and since the output buttons are tested
        # first in `_press`, that overlap toggled the output and GRIP's left
        # edge did nothing at all.
        strip_row = STRIP_TOP + 40
        span = (WIDTH - 24) - 104              # the label column to the margin
        button_w = (span - 3 * 8) // 4
        def slot(k):
            return (104 + k * (button_w + 8), strip_row, button_w, 26)
        self.output_buttons = [Button(slot(0), "SIM ONLY"),
                               Button(slot(1), "SIM + HAND")]
        self.output_buttons[0].active = True
        self.grip_holding = []      # which fingers the executor has hold of
        # GRIP: closing to contact on the real hand - see `executor.py`. Beside
        # ARM because the two are about what the REAL hand does, where the pair
        # to their left is about where commands go; both grey out together on
        # `available`. Which fingers are holding goes in the status line below,
        # not in the label - "HOLD IND MID RIN" is 128 px of text and no button
        # on this row is that wide.
        self.grip_button = Button(slot(2), "GRIP", GOOD)
        self.arm_button = Button(slot(3), "ARM", LIVE)

        labels = list(POSES) + ["reset"]
        width = (WIDTH - 48 - 8 * (len(labels) - 1)) // len(labels)
        self.presets = [Button((24 + i * (width + 8), HEIGHT - 92, width, 32), n)
                        for i, n in enumerate(labels)]

    # Everything the running session will accept live. Ranges are wide enough
    # to be useful and narrow enough that a careless drag cannot wreck the
    # hand: current_limit stops well under the XL330's stall, and nothing here
    # can turn the collision check off - that needs the command line.
    TUNING = (
        ("SIMULATOR", (
            ("kp", "stiffness", 1.0, 200.0, "6.0f"),
            ("kv", "damping", 0.1, 20.0, "6.2f"),
            ("effort", "torque clip Nm", 0.1, 10.0, "6.2f"),
            ("collision_margin", "collide mm", 0.0005, 0.015, "mm"))),
        ("IK SOLVER", (
            ("ik_damping", "damping", 0.001, 0.2, "6.3f"),
            ("ik_iters", "iterations", 1.0, 100.0, "6.0f"),
            ("ik_tol", "tolerance mm", 0.0001, 0.005, "mm"))),
        ("COMMAND STREAM", (
            ("max_speed", "slew rad/s", 0.5, 15.0, "6.2f"),
            ("command_hz", "writes/s", 10.0, 200.0, "6.0f"))),
        ("SERVO  (written to all 16, RAM, live)", (
            ("position_p", "P gain", 0.0, 2000.0, "6.0f"),
            ("position_i", "I gain", 0.0, 500.0, "6.0f"),
            ("position_d", "D gain", 0.0, 2000.0, "6.0f"),
            ("profile_velocity", "profile vel", 0.0, 500.0, "6.0f"),
            ("profile_acceleration", "profile acc", 0.0, 200.0, "6.0f"),
            ("current_limit", "current mA", 50.0, 900.0, "6.0f"))),
    )
    INTEGER_FIELDS = ("position_p", "position_i", "position_d",
                      "profile_velocity", "profile_acceleration",
                      "current_limit", "ik_iters")

    def _build_tuning(self):
        self.tune_sliders = {}
        self.tune_rows = []
        # A boolean does not belong on a slider, so it gets a button. Placed
        # after the rows are laid out, on the IK group's own heading line,
        # because that is the only thing it changes.
        self.ik_avoid_button = Button((408, CARD_TOP + 8, 168, 24),
                                      "IK AVOIDS FINGERS", GOOD)
        self.defaults_button = Button((452, CARD_TOP + 8, 124, 24),
                                      "RESTORE DEFAULTS")
        y = CARD_TOP + 8
        for heading, fields in self.TUNING:
            y += 18
            row = []
            for field, label, low, high, fmt in fields:
                unit = "mm" if fmt == "mm" else "raw"
                slider = Slider((24, y, WIDTH - 48, 24), low, high, low, label,
                                unit, ACCENT, fmt="6.1f" if unit == "mm" else fmt)
                self.tune_sliders[field] = slider
                row.append(slider)
                y += 25
            self.tune_rows.append((heading, row))
            if heading.startswith("SIMULATOR"):
                self.defaults_button.rect.top = row[0].rect.y - 20
            if heading.startswith("IK"):
                self.ik_avoid_button.rect.top = row[0].rect.y - 20
            y += 7

    def _apply_settings(self):
        """Put the remembered values on the widgets that own them.

        Clamped to each slider's own range rather than trusted, so a file
        written before the TUNING table changed cannot put a knob off the end
        of its track.
        """
        for key, slider in self.mimic_sliders.items():
            slider.value = min(max(float(self.settings["mimic"][key]),
                                   slider.low), slider.high)
        for field, value in self.settings["tuning"].items():
            slider = self.tune_sliders.get(field)
            if slider is not None:
                slider.value = min(max(float(value), slider.low), slider.high)

    def refresh_ports(self):
        """What is plugged in now, plus the two links that need no hardware."""
        self.port_menu.options = (
            [(device, label) for device, label in list_serial_ports()]
            + [("mock", "mock link (no motors)"), ("", "no hand")])
        return self.port_menu.options

    def robot_links(self):
        """Which links to string together, per finger, base to tip.

        Read off the chain rather than written down: the thumb's chain runs
        proximal-adduction where a finger's runs adduction-proximal, and a
        hardcoded list would draw the thumb inside out.
        """
        out = {}
        for finger in FINGERS:
            chain = self.chains[finger]
            out[finger] = ([self.model.joints[chain[0]].parent]
                           + [self.model.joints[j].child for j in chain]
                           + [TIP_LINKS[finger]])
        return out

    def robot_skeleton(self, q):
        """World positions along every finger, for the FINGERTIP pads.

        Computed here from the joint angles the simulator already sends, not
        asked for over the wire: the panel owns a HandModel anyway, and FK for
        24 links is 2 ms. Recomputed only when a state message lands, which is
        twenty times a second rather than sixty.
        """
        # Shifted by the mount: link_pose is in the model's own frame with the
        # palm at the origin, while the pads and the fingertip dots are in
        # world metres. Skip this and the linkage draws 200 mm below its own
        # fingertips.
        origin = np.asarray(self.origin, dtype=float)
        return {finger: [self.model.link_pose(link, q, self.joint_names)[0] + origin
                         for link in self._robot_links[finger]]
                for finger in FINGERS}

    def display_chain(self, finger):
        """(label, joint) as spread, knuckle, middle, tip.

        Labelled by what each joint moves rather than what it is called - see
        model.ROLES. The thumb is why: its `proximal` joint is the one that
        swings, and its `adduction` joint is the one that flexes, so labelling
        from the names puts the thumb's spread and knuckle controls on each
        other's joints.
        """
        return roles_in_order(self.model, finger)

    def _set_tip_ranges(self):
        """A box around the mount that comfortably contains the whole hand."""
        x, y, z = self.origin
        ranges = [(x - 0.10, x + 0.14), (y - 0.14, y + 0.20), (z - 0.12, z + 0.20)]
        for slider, (low, high) in zip(self.tip_sliders, ranges):
            slider.low, slider.high = low, high
        for pad in self.pads:
            pad.set_ranges(ranges)

    # -------------------------------------------------------------- state
    def receive(self):
        message = self.link.latest("state")
        if message is None:
            return
        if not self.seen_sim:
            self.seen_sim = True
            self.origin = message.get("origin", self.origin)
            self._set_tip_ranges()
            for slider, value in zip(self.joint_sliders, message["q"]):
                slider.value = value
            self._restore_to_sim()
        self.tips = dict(message.get("tips", {}))
        self.tip_error = message.get("error", {})
        self.skeleton = message.get("skeleton", {})
        self._robot_skeleton = self.robot_skeleton(np.asarray(message["q"],
                                                              dtype=float))
        self.leap = message.get("leap", self.leap)
        self.gestures = message.get("gestures", self.gestures)
        self.log = message.get("log", self.log)
        self._sync_mimic()
        self.hardware = message.get("hardware", self.hardware)
        self.ik_avoid = bool((message.get("tuning") or {}).get("ik_avoid", True))
        for field, value in (message.get("tuning") or {}).items():
            slider = self.tune_sliders.get(field)
            if slider is not None and slider is not self.dragging:
                slider.value = min(max(float(value), slider.low), slider.high)
        if not self.port_menu.open:
            # The simulator owns the link, so it owns the truth about which
            # port is open - a failed connect must not leave the menu claiming
            # a port that is not there.
            self.port_menu.value = self.hardware.get("port", "")
        live = bool(self.hardware.get("enabled"))
        self.output_buttons[0].active = not live
        self.output_buttons[1].active = live
        armed = bool(self.hardware.get("armed"))
        self.arm_button.label = "DISARM" if armed else "ARM"
        self.arm_button.active = armed
        self.arm_button.enabled = bool(self.hardware.get("available"))
        grip = self.hardware.get("grip") or {}
        self.grip_button.active = bool(grip.get("enabled"))
        self.grip_button.enabled = bool(self.hardware.get("available"))
        self.grip_holding = grip.get("holding") or []
        for slider, value in zip(self.joint_sliders, message["q"]):
            slider.actual = value
        measured = self.hardware.get("positions")
        for index, slider in enumerate(self.joint_sliders):
            slider.measured = measured[index] if measured else None

    def _restore_to_sim(self):
        """Push the remembered settings, once the simulator is actually there.

        Deferred to the first state message rather than sent at startup: the
        panel usually wins the race to open, and a datagram sent to a socket
        nobody has bound yet is simply gone.

        Only done when a file was actually read. With no file the simulator's
        own defaults stand, so `--mimic-mode tip` on the command line survives
        - a remembered preference should not quietly overrule an instruction
        given for this run. Delete the file to go back to flags.
        """
        if not self.settings.restored:
            return
        saved = self.settings["mimic"]
        self.send_mimic(mode=saved["mode"], thumb=saved["thumb"],
                        mirror=saved["mirror"])
        # Software only: simulator gains and stream rates. The servo registers
        # are not in the file and are not sent - see settings.py.
        values = {field: self.tune_sliders[field].value
                  for field in self.settings["tuning"]
                  if field in self.tune_sliders}
        if values:
            self.link.send({"type": "tune", "values": values})
        # A port makes the hand AVAILABLE and nothing more. Output and arm are
        # separate switches and stay off, whatever this file says.
        #
        # "mock" is honoured here though it is never written here: the only
        # way it reaches the file is somebody typing it in, and that is a
        # request, not an accident.
        port = self.settings["port"]
        if port and port != self.hardware.get("port"):
            self.link.send({"type": "connect", "port": port})
        # Our file owns these slider positions now, so do not seed them from
        # the simulator's reply - it would overwrite what we just asked for.
        self.seen_leap = True

    def _sync_mimic(self):
        """Set every MIMIC widget from what the simulator just reported.

        A slider the operator is holding is left alone - otherwise the echo
        arriving 20 times a second fights the drag and the knob will not move.
        """
        leap = self.leap
        self.mimic_button.active = bool(leap.get("enabled"))
        self.mimic_button.label = "FOLLOWING" if leap.get("enabled") else "FOLLOW"
        log = self.log.get("active")
        self.log_button.active = bool(log)
        self.log_button.label = "LOGGING" if log else "LOG"
        capture = leap.get("capture")
        self.calibrate_button.active = capture is not None
        self.calibrate_button.label = "CANCEL" if capture else "CALIBRATE"
        for button in self.mimic_row:
            button.rect.y = self.mimic_row_y + (self.mimic_row_drop
                                                if capture else 0)
        if capture:
            self.capture_button.label = "CAPTURE & FIT" if capture["last"] \
                else "CAPTURE & NEXT"
            # Greyed until the sensor has actually seen the hand for long
            # enough to average. Capturing "nothing was there" would set one
            # end of a joint's range to a number nobody produced.
            self.capture_button.enabled = bool(capture["ready"])
        self.reset_fit_button.enabled = bool(leap.get("fitted"))
        self.mirror_button.active = bool(leap.get("mirror"))
        for button, mode in zip(self.mimic_mode_buttons,
                                ("raw", "angles", "tip", "keyvector",
                                 "posebank")):
            button.active = leap.get("mode") == mode
        for button, mode in zip(self.learning_mode_buttons,
                                ("keyvector", "student")):
            button.active = leap.get("mode") == mode
        for button, thumb in zip(self.mimic_thumb_buttons, ("angles", "pinch")):
            button.active = leap.get("thumb") == thumb
        for key, slider in self.mimic_sliders.items():
            value = leap.get(key)
            if value is not None and slider is not self.dragging \
                    and not self.seen_leap:
                slider.value = float(value)
        self.seen_leap = True

    def send_mimic(self, **changes):
        """One message carrying the whole tab. Settings changed while it is off
        are therefore already in place when it goes on.

        The unchanged fields come from what this panel last ASKED for, not
        from what the simulator last reported. The echo lags by up to a state
        period: read it here and a slider dragged within 50 ms of clicking TIP
        would rebuild the message with the stale `angles` and silently undo
        the click. What the operator asked for is the panel's own to remember.
        """
        remembered = self.settings["mimic"]
        message = {"type": "mimic",
                   "mode": remembered["mode"],
                   "thumb": remembered["thumb"],
                   "mirror": remembered["mirror"],
                   "smoothing": self.mimic_sliders["smoothing"].value,
                   "scale": self.mimic_sliders["scale"].value,
                   "gains": {"spread": self.mimic_sliders["spread_gain"].value}}
        message.update(changes)
        self.link.send(message)
        self.settings.update("mimic", mode=message["mode"],
                             thumb=message["thumb"], mirror=message["mirror"],
                             smoothing=message["smoothing"],
                             scale=message["scale"],
                             spread_gain=message["gains"]["spread"])

    def send_joints(self):
        self.link.send({"type": "joints",
                        "q": [s.value for s in self.joint_sliders]})

    def send_tip(self):
        if self.tip_target is not None:
            self.link.send({"type": "tip", "finger": self.finger,
                            "target": list(self.tip_target)})

    def adopt_current_tip(self):
        tip = self.tips.get(self.finger)
        if tip is None:
            tip = [self.origin[0], self.origin[1], self.origin[2] + 0.15]
        self.tip_target = list(tip)
        for slider, value in zip(self.tip_sliders, self.tip_target):
            slider.value = value

    def select_finger(self, finger):
        self.finger = self.settings.set("finger", finger)
        for tab, name in zip(self.finger_tabs, FINGERS):
            tab.active = name == finger
        for slider in self.tip_sliders:
            slider.color = FINGER_COLOR[finger]
        self.adopt_current_tip()

    MODES = ("joints", "tip", "mimic", "learning", "tune")
    # The only two mappings the LEARNING tab offers. Entering it with anything
    # else remembered would draw a mode row that cannot select the mode that is
    # actually running, and a second row belonging to a different tab.
    LEARNING_MODES = ("keyvector", "student")
    # The two tabs that drive the hand from a live operator. Leaving either
    # stops the following, and both share the same controls and the same
    # `mimic` message - they differ only in which mappings they offer.
    FOLLOWING = ("mimic", "learning")

    def set_mode(self, mode):
        # Leaving MIMIC stops the following. A tab is a view, but this one is
        # also a live coupling between an operator's hand and sixteen servos,
        # and leaving the view is not a reason to keep driving them - the
        # controls that would stop it are on the tab you just left.
        if self.mode in self.FOLLOWING and mode not in self.FOLLOWING \
                and self.leap.get("enabled"):
            self.send_mimic(on=False)
        self.mode = self.settings.set("tab", mode)
        for tab, name in zip(self.tabs, self.MODES):
            tab.active = mode == name
        if mode == "learning" \
                and self.settings["mimic"]["mode"] not in self.LEARNING_MODES:
            # Arriving from MIMIC on ANGLES or TIP IK, say. Coerced rather than
            # left alone so the tab never shows a mode row that cannot select
            # what is running - TEACHER is the safe default, being the mapping
            # this tab is asking the student to beat.
            self.send_mimic(mode=self.LEARNING_MODES[0])
        if mode == "tip":
            self.adopt_current_tip()

    # -------------------------------------------------------------- input
    def handle(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                if self.port_menu.open:
                    self.port_menu.open = False
                    return True
                return False
            if event.key == pygame.K_TAB:
                order = self.MODES
                self.set_mode(order[(order.index(self.mode) + 1) % len(order)])
            elif event.key == pygame.K_m and self.mode in self.FOLLOWING:
                self.send_mimic(on=not self.leap.get("enabled"))
            elif event.key == pygame.K_r:
                self._preset("reset")
            elif event.key == pygame.K_h:
                self.link.send({"type": "output",
                                "hardware": not self.hardware.get("enabled")})
            elif self.mode == "tip" and pygame.K_1 <= event.key <= pygame.K_4:
                self.select_finger(FINGERS[event.key - pygame.K_1])

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._press(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging is not None:
                self.dragging.held = False
            self.dragging = None
        elif event.type == pygame.MOUSEMOTION and self.dragging is not None:
            self._drag(event.pos)
        return True

    def _press(self, pos):
        if self.port_menu.open:
            chosen = self.port_menu.hit_item(pos)
            self.port_menu.open = False
            if chosen is not None and chosen != self.port_menu.value:
                self.port_menu.value = chosen
                self.link.send({"type": "connect", "port": chosen})
                # "mock" is a test link, not a bench: remembering it would
                # silently attach a fake hand to the next real session.
                self.settings.set("port", "" if chosen == "mock" else chosen)
            return
        if self.port_menu.hit(pos):
            self.refresh_ports()
            self.port_menu.open = True
            return

        for button, wants in zip(self.output_buttons, (False, True)):
            if button.rect.collidepoint(pos):
                self.link.send({"type": "output", "hardware": wants})
                return
        if self.grip_button.rect.collidepoint(pos) and self.grip_button.enabled:
            self.link.send({"type": "grip",
                            "on": not (self.hardware.get("grip") or {}).get("enabled")})
            return
        if self.arm_button.rect.collidepoint(pos) and self.arm_button.enabled:
            # Deliberately click-only. Arming energises sixteen servos, and a
            # stray keystroke is not consent.
            self.link.send({"type": "arm",
                            "on": not self.hardware.get("armed")})
            return

        for tab, mode in zip(self.tabs, self.MODES):
            if tab.rect.collidepoint(pos):
                self.set_mode(mode)
                return
        for button in self.presets:
            if button.rect.collidepoint(pos):
                self._preset(button.label)
                return

        if self.mode in self.FOLLOWING:
            if self.mimic_button.rect.collidepoint(pos):
                self.send_mimic(on=not self.leap.get("enabled"))
                return
            if self.calibrate_button.rect.collidepoint(pos):
                self.link.send({"type": "calibrate"})
                return
            if self.reset_fit_button.rect.collidepoint(pos):
                self.link.send({"type": "calibrate", "reset": True})
                return
            if self.mirror_button.rect.collidepoint(pos):
                self.send_mimic(mirror=not self.leap.get("mirror"))
                return
            if self.log_button.rect.collidepoint(pos):
                self.link.send({"type": "log",
                                "on": not self.log.get("active")})
                return
            if self.leap.get("capture") and self.capture_button.enabled \
                    and self.capture_button.rect.collidepoint(pos):
                self.link.send({"type": "calibrate", "next": True})
                return
            if not self.leap.get("capture"):
                # Hidden during a capture, so they must not be clickable
                # either - an invisible button that still fires is worse than
                # no button.
                if self.mode == "learning":
                    for button, source in ((self.gestures_button, "probe"),
                                           (self.replay_button, "session")):
                        if button.rect.collidepoint(pos):
                            playing = (self.gestures.get("playing")
                                       and self.gestures.get("source") == source)
                            self.link.send({"type": "gestures", "on": not playing,
                                            "source": source})
                            return
                row = (zip(self.learning_mode_buttons, ("keyvector", "student"))
                       if self.mode == "learning" else
                       zip(self.mimic_mode_buttons,
                           ("raw", "angles", "tip", "keyvector", "posebank")))
                for button, mode in row:
                    if button.rect.collidepoint(pos):
                        self.send_mimic(mode=mode)
                        return
                if self.leap.get("mode") != "keyvector":
                    # Hidden under keyvector because it does nothing there, so
                    # it must not be clickable there either.
                    for button, thumb in zip(self.mimic_thumb_buttons,
                                             ("angles", "pinch")):
                        if button.rect.collidepoint(pos):
                            self.send_mimic(thumb=thumb)
                            return
            controls = list(self.mimic_sliders.values())
        elif self.mode == "tune":
            if self.defaults_button.rect.collidepoint(pos):
                self.restore_tuning_defaults()
                return
            if self.ik_avoid_button.rect.collidepoint(pos):
                self.link.send({"type": "tune",
                                "values": {"ik_avoid": not self.ik_avoid}})
                return
            controls = list(self.tune_sliders.values())
        elif self.mode == "joints":
            controls = self.joint_sliders
        else:
            for tab, finger in zip(self.finger_tabs, FINGERS):
                if tab.rect.collidepoint(pos):
                    self.select_finger(finger)
                    self.send_tip()
                    return
            if self.snap.rect.collidepoint(pos):
                self.adopt_current_tip()
                self.send_tip()
                return
            controls = self.pads + self.tip_sliders

        for control in controls:
            if control.hit(pos):
                self.dragging = control
                control.held = True
                self._drag(pos)
                return

    def _drag(self, pos):
        control = self.dragging
        if isinstance(control, Pad):
            if self.tip_target is None:
                self.adopt_current_tip()
            self.tip_target = control.apply(pos, list(self.tip_target))
            for slider, value in zip(self.tip_sliders, self.tip_target):
                slider.value = value
            self.send_tip()
            return

        control.set_from_pos(pos)
        if self.mode == "tune":
            self.send_tuning(control)
        elif self.mode in self.FOLLOWING:
            self.send_mimic()
        elif self.mode == "joints":
            self.send_joints()
        else:
            self.tip_target = [s.value for s in self.tip_sliders]
            self.send_tip()

    def _preset(self, name):
        if name == "reset":
            self.link.send({"type": "reset"})
            for slider in self.joint_sliders:
                slider.value = 0.0
        else:
            self.link.send({"type": "pose", "name": name})
            pose = POSES[name]
            for slider, joint in zip(self.joint_sliders, self.joint_names):
                slider.value = pose.get(joint, 0.0)
        self.tip_target = None          # the pose owns the hand now, not a tip

    # --------------------------------------------------------------- draw
    def draw(self):
        screen = self.screen
        screen.fill(BG)

        screen.blit(self.head.render("h4nd", True, TEXT), (24, 26))
        status = "connected" if self.seen_sim else "waiting for simulator"
        width = self.small.size(status)[0]
        pygame.draw.circle(screen, GOOD if self.seen_sim else WARN,
                           (WIDTH - 36 - width, 34), 4)
        screen.blit(self.small.render(status, True, DIM), (WIDTH - 24 - width, 28))

        for tab in self.tabs:
            tab.draw(screen, self.font)

        self._draw_hardware(screen)
        pygame.draw.rect(screen, CARD,
                         pygame.Rect(12, CARD_TOP, WIDTH - 24,
                                     HEIGHT - CARD_TOP - 116),
                         border_radius=10)

        if self.mode == "joints":
            self._draw_joints(screen)
        elif self.mode == "tune":
            self._draw_tuning(screen)
        elif self.mode == "tip":
            self._draw_tip(screen)
        else:
            self._draw_mimic(screen)

        for button in self.presets:
            button.draw(screen, self.small)

        hints = {"tip": "tab mode   1-4 finger   h hand output   r reset   esc quit",
                 "mimic": "tab mode   m follow   h hand output   r reset   esc quit",
                 "learning": "tab mode   m follow   h hand output   r reset   esc quit",
                 }.get(self.mode, "tab mode   h hand output   r reset   esc quit")
        measuring = self.hardware.get("positions") is not None
        if self.mode in self.FOLLOWING:
            legend = "solid lines are your hand    hollow dots the robot's tips"
        elif self.mode == "tip":
            legend = "bright linkage is the finger the sliders move"
        elif measuring:
            legend = "white tick: simulated joint    amber tick: real encoder"
        else:
            legend = "white tick is where the joint really is"
        screen.blit(self.small.render(hints, True, FAINT), (24, HEIGHT - 42))
        screen.blit(self.small.render(legend, True, FAINT), (24, HEIGHT - 26))
        self.port_menu.draw_open(screen, self.small)
        pygame.display.flip()

    def _draw_hardware(self, screen):
        """Whether the real hand is listening, and whether it is energised.

        Armed is drawn loud on purpose: everything else on this panel moves
        pixels, and this one moves a machine.
        """
        armed = bool(self.hardware.get("armed"))
        available = bool(self.hardware.get("available"))
        strip = pygame.Rect(12, STRIP_TOP, WIDTH - 24, STRIP_HEIGHT)
        pygame.draw.rect(screen, (44, 24, 26) if armed else CARD, strip,
                         border_radius=10)
        if armed:
            pygame.draw.rect(screen, LIVE, strip, 1, border_radius=10)

        screen.blit(self.small.render("PORT", True, DIM), (26, STRIP_TOP + 16))
        self.port_menu.draw(screen, self.small)
        screen.blit(self.small.render("OUTPUT", True, DIM), (26, STRIP_TOP + 48))
        for button in self.output_buttons:
            button.draw(screen, self.small)

        # A rule between the pairs, because the row is two different questions
        # and "OUTPUT" only labels the first. Left of it: where commands go.
        # Right of it: what the real hand does with them.
        divide = (self.output_buttons[1].rect.right + self.grip_button.rect.left) // 2
        pygame.draw.line(screen, LINE, (divide, STRIP_TOP + 42),
                         (divide, STRIP_TOP + 64))
        self.grip_button.draw(screen, self.small)
        self.arm_button.draw(screen, self.small)

        error = self.hardware.get("error") or ""
        if armed:
            note, color = "LIVE - motors energised", LIVE
        elif error:
            note, color = error[:66], WARN
        elif available:
            note, color = f"{self.hardware.get('note', '')} - torque off", DIM
        else:
            note, color = "no hand attached - pick a port above", FAINT
        # Which fingers the executor is holding with goes HERE rather than in
        # the GRIP button's label: it is up to four words and the button is
        # 112 px wide. Left of the note, in the grip colour, so a glance at
        # this line says both what the hand is doing and what it has hold of.
        if self.grip_holding:
            held = "HOLDING " + " ".join(h[:3].upper() for h in self.grip_holding)
            screen.blit(self.small.render(held, True, GOOD), (26, STRIP_TOP + 76))
            note = note[:44]
            screen.blit(self.small.render(note, True, color),
                        (26 + self.small.size(held)[0] + 12, STRIP_TOP + 76))
        else:
            screen.blit(self.small.render(note, True, color),
                        (26, STRIP_TOP + 76))

    # What RESTORE puts back. The software half comes from settings.DEFAULTS,
    # which is itself imported from the modules that own those numbers, so
    # there is still exactly one place a default is written down. The servo
    # half is not in that file on purpose - restoring it writes RAM registers
    # on sixteen motors, which is fine as an answer to a button press and not
    # fine as something a file does at launch.
    def tuning_defaults(self):
        return {**DEFAULTS["tuning"], "ik_avoid": True,
                "position_p": POSITION_P, "position_i": POSITION_I,
                "position_d": POSITION_D,
                "profile_velocity": PROFILE_VELOCITY,
                "profile_acceleration": PROFILE_ACCELERATION,
                "current_limit": CURRENT_LIMIT_MA}

    def restore_tuning_defaults(self):
        """Back to what ships, on screen, in the simulator, and on disk."""
        values = self.tuning_defaults()
        for field, value in values.items():
            slider = self.tune_sliders.get(field)
            if slider is not None:
                slider.value = min(max(float(value), slider.low), slider.high)
            if field in self.settings["tuning"]:
                self.settings.update("tuning", **{field: value})
        self.ik_avoid = True
        self.link.send({"type": "tune", "values": values})

    def send_tuning(self, control):
        """One field per message: the simulator applies whatever it is given,
        so sending only what moved keeps a drag from re-writing eleven servo
        registers that nobody touched."""
        for field, slider in self.tune_sliders.items():
            if slider is control:
                value = (round(slider.value) if field in self.INTEGER_FIELDS
                         else slider.value)
                self.link.send({"type": "tune", "values": {field: value}})
                if field in self.settings["tuning"]:
                    self.settings.update("tuning", **{field: value})
                return

    def _draw_tuning(self, screen):
        for heading, row in self.tune_rows:
            y = row[0].rect.y - 16
            screen.blit(self.small.render(heading, True, ACCENT), (24, y))
            # Stop the rule short of the toggle rather than drawing through it.
            if heading.startswith("IK"):
                right = self.ik_avoid_button.rect.left - 10
            elif heading.startswith("SIMULATOR"):
                right = self.defaults_button.rect.left - 10
            else:
                right = WIDTH - 24
            pygame.draw.line(screen, LINE, (24 + self.small.size(heading)[0] + 12,
                                            y + 6), (right, y + 6))
            for slider in row:
                slider.draw(screen, self.font, self.small)

        # Last, so nothing is drawn over it. `active` is only ever a draw
        # instruction here - the state lives in `self.ik_avoid`, because
        # storing it in the widget and then overwriting it to keep the button
        # filled made the colour alternate green/amber every other frame.
        self.defaults_button.draw(screen, self.small)
        self.ik_avoid_button.active = True
        self.ik_avoid_button.color = GOOD if self.ik_avoid else WARN
        self.ik_avoid_button.label = ("IK AVOIDS FINGERS" if self.ik_avoid
                                      else "IK IGNORES FINGERS")
        self.ik_avoid_button.draw(screen, self.small)

    def _draw_joints(self, screen):
        for finger, row in self.rows:
            y = row[0].rect.y - 16
            screen.blit(self.small.render(finger.upper(), True,
                                          FINGER_COLOR[finger]), (24, y))
            pygame.draw.line(screen, LINE, (86, y + 6), (WIDTH - 24, y + 6))
            for slider in row:
                slider.draw(screen, self.font, self.small)

    def _draw_tip(self, screen):
        screen.blit(self.small.render("FINGER", True, DIM), (24, CARD_TOP + 12))
        for tab in self.finger_tabs:
            tab.draw(screen, self.small)
        for pad in self.pads:
            pad.draw(screen, self.small, self.tips, self.finger, self.tip_target,
                     robot=self._robot_skeleton)

        screen.blit(self.small.render("TARGET  (world, millimetres)", True, DIM),
                    (24, CARD_TOP + 256))
        for slider in self.tip_sliders:
            slider.draw(screen, self.font, self.small)
        self.snap.draw(screen, self.small)

        rows = [("target", self.tip_target), ("actual tip", self.tips.get(self.finger))]
        y = CARD_TOP + 394
        for label, value in rows:
            screen.blit(self.small.render(label, True, DIM), (24, y))
            text = ("  ".join(f"{v * 1000:8.1f}" for v in value) if value
                    else "       -")
            screen.blit(self.font.render(text, True, TEXT), (150, y - 2))
            y += 22

        error = self.tip_error.get(self.finger)
        screen.blit(self.small.render("solver miss", True, DIM), (24, y))
        if error is None:
            screen.blit(self.font.render("       -", True, DIM), (150, y - 2))
        else:
            reached = error <= 0.003
            note = "solved" if reached else "out of reach, closest approach"
            screen.blit(self.font.render(f"{error * 1000:8.1f} mm   {note}", True,
                                         TEXT if reached else WARN), (150, y - 2))

        # What the IK actually asked of the four joints - the joint view of a
        # tip command, without switching tabs to read it.
        y += 34
        screen.blit(self.small.render(f"{self.finger.upper()} JOINTS NOW", True, DIM),
                    (24, y))
        y += 18
        color = FINGER_COLOR[self.finger]
        for i, (label, name) in enumerate(self.display_chain(self.finger)):
            x = 24 + i * 138
            screen.blit(self.small.render(label, True, DIM), (x, y))
            angle = self.joint_sliders[self.joint_names.index(name)].actual
            text = f"{math.degrees(angle):.1f}°" if angle is not None else "-"
            screen.blit(self.font.render(text, True, color), (x, y + 16))

    def _draw_mimic(self, screen):
        """The Leap tab: what it sees, what it is doing with it, and the knobs.

        Your hand is drawn over the robot's fingertips in the same two views
        the FINGERTIP tab uses, in world metres. That overlay is the whole
        diagnostic - a hand sitting off to one side of the dots is a mapping
        that is off to one side, which no table of angles makes as obvious.
        """
        leap = self.leap
        capture = leap.get("capture")
        tracked = bool(leap.get("tracked"))
        mismatched = bool(leap.get("mismatched"))

        # The top row belongs to whichever of the two has something to say.
        # During a capture that is the instruction, and the usual status line
        # would sit on top of it - so it stands down, having nothing to add
        # that the capture block is not already saying more usefully.
        if not capture:
            screen.blit(self.small.render("LEAP", True, DIM),
                        (24, CARD_TOP + 12))
            if not leap.get("enabled"):
                note, color = "off", FAINT
            elif leap.get("error"):
                note, color = leap["error"][:52], WARN
            elif mismatched:
                # Loudest thing on the tab, because it is the failure that
                # looks like a bug: the mapping is fine, the hand is the wrong
                # one, and the robot dutifully moves the other way.
                note = (f"that is your {leap.get('type', '').upper()} hand - "
                        f"use your {leap.get('expected', '').upper()}")
                color = LIVE
            elif tracked:
                note = (f"{leap.get('type', '')} hand   "
                        f"pinch {leap.get('pinch', 0):.2f}   "
                        f"grab {leap.get('grab', 0):.2f}")
                color = GOOD
            else:
                note, color = "no hand over the sensor", WARN
            width = self.small.size(note)[0]
            pygame.draw.circle(screen, color,
                               (WIDTH - 36 - width, CARD_TOP + 18), 4)
            screen.blit(self.small.render(note, True, color),
                        (WIDTH - 24 - width, CARD_TOP + 12))
        elif mismatched:
            # The one thing still worth interrupting a capture for: fitting
            # the wrong hand's range produces a fit that is wrong everywhere.
            warning = f"WRONG HAND - use your {leap.get('expected', '').upper()}"
            width = self.small.size(warning)[0]
            screen.blit(self.small.render(warning, True, LIVE),
                        (WIDTH - 24 - width, CARD_TOP + 14))

        if capture:
            # The calibration block, above everything else it could be
            # confused with. Nothing here is a setting - it is one
            # instruction and one button, and the operator's hands are busy
            # doing what it asked.
            screen.blit(self.small.render(
                f"{capture['index'] + 1} of {capture['total']}", True, DIM),
                (24, CARD_TOP + 14))
            screen.blit(self.font.render(capture["hint"], True, TEXT),
                        (72, CARD_TOP + 12))
            self.capture_button.draw(screen, self.small)
            note, color = ((f"press when you are in the pose "
                            f"({capture['samples']} frames)"), DIM) \
                if capture["ready"] else \
                ("no hand over the sensor", WARN)
            screen.blit(self.small.render(note, True, color),
                        (186, CARD_TOP + 47))

        for button in self.mimic_row:
            button.draw(screen, self.small)

        if self.log.get("error"):
            note, colour = self.log["error"][:60], LIVE
        elif self.log.get("active"):
            note = (f"recording  {self.log['rows']} rows  "
                    f"{self.log['seconds']:.0f} s  ->  {self.log['path']}")
            colour = LIVE
        else:
            note, colour = "", None
        if colour is not None:
            screen.blit(self.small.render(note, True, colour),
                        (24, HEIGHT - 58))

        if not capture:
            fitted = leap.get("fitted") or 0
            if fitted and leap.get("calibrated"):
                fit = (f"fitted to your hand: {fitted} joints, "
                       f"and the tip mapping", GOOD)
            elif fitted:
                fit = (f"fitted to your hand: {fitted} joints", GOOD)
            elif leap.get("calibrating"):
                fit = ("fitting the tip mapping - hold your hand flat", WARN)
            else:
                fit = ("fixed gains - press CALIBRATE to fit your own range",
                       WARN)
            if leap.get("mismatched"):
                fit = (f"this robot is a {leap.get('expected')} hand. Swap "
                       f"hands, or MIRROR to drive it with that one.", LIVE)
            screen.blit(self.small.render(fit[0], True, fit[1]),
                        (24, CARD_TOP + 62))

            screen.blit(self.small.render("MODE", True, DIM),
                        (24, CARD_TOP + 90))
            for button in (self.learning_mode_buttons
                           if self.mode == "learning"
                           else self.mimic_mode_buttons):
                button.draw(screen, self.small)
            if self.mode == "learning":
                playing = bool(self.gestures.get("playing"))
                source = self.gestures.get("source", "")
                self.gestures_button.active = playing and source == "probe"
                self.replay_button.active = playing and source == "session"
                self.gestures_button.draw(screen, self.small)
                self.replay_button.draw(screen, self.small)
                if playing:
                    note = (f"{self.gestures.get('label', '')}  "
                            f"{self.gestures.get('frame', 0)}/{self.gestures.get('frames', 0)}")
                    screen.blit(self.small.render(note, True, LIVE),
                                (536, CARD_TOP + 88))
            # The second row belongs to whichever control is live. Under
            # KEYVECTOR the solver moves all four thumb joints itself, so the
            # THUMB choice does nothing - and a control that does nothing has
            # no business sitting there looking as though it does. The row
            # carries the solver's own state instead, which is the thing you
            # actually want to see in that mode.
            solver = leap.get("solver") or {}
            if self.mode == "learning" and leap.get("mode") == "student":
                # Under STUDENT the thumb row does nothing - the network emits
                # all sixteen joints - so it carries the network's own state
                # instead, the same way KEYVECTOR's row carries the solver's.
                student = leap.get("student") or {}
                screen.blit(self.small.render("NET", True, DIM),
                            (24, CARD_TOP + 118))
                if student.get("error"):
                    note, colour = student["error"], LIVE
                elif not student.get("loaded"):
                    note, colour = "loading...", DIM
                elif not student.get("aligned"):
                    note = ("hold your hand flat and open - fitting the rest "
                            "pose the network was trained against")
                    colour = DIM
                elif not student.get("ready"):
                    note = (f"filling the window: {student.get('window', 0)}"
                            f"/{student.get('needs', 0)} frames")
                    colour = DIM
                else:
                    note = (f"{student.get('window', 0)}-frame window, one "
                            f"forward pass, {student.get('ms', 0):.1f} ms")
                    if student.get("repair_deg", 0) > 0.05:
                        # The repair layer moved this command: by how much,
                        # and what it cost. Zero means the network's own
                        # answer was already clear of itself.
                        note += (f"   repair {student['repair_deg']:.1f} deg, "
                                 f"{student.get('repair_ms', 0):.1f} ms")
                    colour = DIM
                screen.blit(self.small.render(note, True, colour),
                            (90, CARD_TOP + 118))
            elif leap.get("mode") == "raw":
                screen.blit(self.small.render("RAW", True, DIM),
                            (24, CARD_TOP + 118))
                screen.blit(self.small.render(
                    "your angle straight onto the joint - no gain, no fit, "
                    "no smoothing", True, DIM), (90, CARD_TOP + 118))
            elif leap.get("mode") == "posebank":
                screen.blit(self.small.render("BANK", True, DIM),
                            (24, CARD_TOP + 118))
                bank = leap.get("bank") or {}
                if bank.get("poses"):
                    note = (f"{bank['poses']} vetted poses, nearest match "
                            f"every frame - no solver, no optimiser")
                    colour = GOOD
                else:
                    note, colour = "building the bank - a few seconds, once", WARN
                screen.blit(self.small.render(note, True, colour),
                            (90, CARD_TOP + 118))
            elif leap.get("mode") == "keyvector":
                screen.blit(self.small.render("SOLVER", True, DIM),
                            (24, CARD_TOP + 118))
                if solver.get("error"):
                    note, colour = solver["error"][:58], LIVE
                elif solver.get("ready"):
                    note = (f"{solver.get('ms', 0):.1f} ms/frame   "
                            f"{solver.get('answered', 0)}/{solver.get('sent', 0)} "
                            f"answered   thumb solved with the fingers")
                    colour = GOOD
                elif solver.get("alive"):
                    note, colour = "starting - loading the URDF", WARN
                else:
                    note, colour = "not running - following on angles", WARN
                screen.blit(self.small.render(note, True, colour),
                            (90, CARD_TOP + 118))
            elif self.mode == "learning":
                # The thumb row belongs to MIMIC. Both learning mappings solve
                # all four thumb joints themselves, so there is nothing here to
                # choose - and a control that does nothing has no business
                # looking as though it does.
                pass
            else:
                screen.blit(self.small.render("THUMB", True, DIM),
                            (24, CARD_TOP + 118))
                for button in self.mimic_thumb_buttons:
                    button.draw(screen, self.small)

        # Drawn only while a hand is actually being tracked. The state message
        # is the last one that arrived, so without this the skeleton from the
        # moment tracking dropped stays on screen looking live.
        drawn = self.skeleton if (tracked and (leap.get("enabled")
                                                or leap.get("capture"))) else None
        for pad in self.mimic_pads:
            pad.set_ranges([(s.low, s.high) for s in self.tip_sliders])
            pad.draw(screen, self.small, self.tips, None, None, skeleton=drawn)

        y = CARD_TOP + 318
        rows = [("following", (f"your {leap.get('type')} hand"
                               f"   (wants your {leap.get('expected')})"
                               if tracked else "-")),
                ("frames", f"{leap.get('frames', 0)}"
                           f"   ({leap.get('lost', 0)} with no hand)")]
        if leap.get("mode") == "tip":
            # The miss is the honest part of tip mode: an unreachable target
            # does not fail, it returns the closest approach.
            misses = leap.get("miss") or {}
            shown = "   ".join(f"{f[:3]} {v * 1000:.0f}mm"
                               for f, v in misses.items() if v is not None)
            rows.append(("tip miss", shown or "-"))
            rows.append(("thumb", f"on the angle map ({leap.get('thumb')}) - "
                                  f"it cannot reach"))
        for label, value in rows:
            screen.blit(self.small.render(label, True, DIM), (24, y))
            screen.blit(self.font.render(value, True, TEXT), (150, y - 2))
            y += 20

        for slider in self.mimic_sliders.values():
            slider.draw(screen, self.font, self.small)
        if leap.get("fitted"):
            # A fitted range already carries the scaling the gain was standing
            # in for, so the slider now trims a fit rather than being the
            # whole mapping. Worth saying, or 1.00 looks like a reset knob.
            screen.blit(self.small.render(
                "gains trim the fitted range; CLEAR to go back to fixed gains",
                True, FAINT), (24, CARD_TOP + 492))

    # --------------------------------------------------------------- loop
    def run(self):
        self.refresh_ports()
        self.select_finger(self.finger)
        self.set_mode(self.mode)
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle(event) and running
            self.receive()
            self.draw()
            # Debounced inside: a slider drag fires on every mouse-motion
            # event, and none of those is worth a syscall.
            self.settings.save()
            self.clock.tick(60)
        self.settings.close()
        pygame.quit()


def main(command_port=COMMAND_PORT, state_port=STATE_PORT,
         hand=DEFAULT_HAND, spread_limit=None):
    Panel(command_port, state_port, hand, spread_limit).run()


if __name__ == "__main__":
    main()
