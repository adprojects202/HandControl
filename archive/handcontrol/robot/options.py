"""The simulator's settings, in three groups instead of twenty-two arguments.

WHY THIS FILE EXISTS. `HandSim.__init__` grew to twenty-two parameters and
`ensure_sim_mjcf` to twelve, most of them the same values threaded from one to
the other by hand. Seven call sites each passed a different subset - three of
them, six, eight, seventeen - so reading any one of them told you nothing
about what the other six had chosen, and adding a knob meant editing both
signatures, every call site that cared, and the panel's table of names.

They were never twenty-two independent choices. They are three groups, and a
caller almost always wants the defaults for two of them:

    Physics   what makes the hand move: the timestep, gravity, the actuator
              gains, whether the links collide at all.
    Guard     what refuses a pose: the clamp before a command is applied and
              the one inside the IK, and how much daylight each insists on.
    Scene     what is drawn: the viewer, where the hand stands, the encoder
              ghost, the camera, and any extra posed copies beside it.

Frozen, so a group handed to two simulators cannot be edited through one of
them, and so `Physics()` can be a default argument without the usual mutable
-default trap. `replace(physics, kp=40)` is how you vary one.

The MJCF generator takes the same objects rather than the values unpacked out
of them, which is what stops a knob from having two spellings on the way
down.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import ARMATURE


@dataclass(frozen=True)
class Physics:
    """What makes the hand move. `None` means "whatever the model says"."""

    dt: float = 0.002
    gravity: bool = True
    self_collision: bool = True     # contact between the hand's own links
    kp: float | None = None         # position gain; model.KP
    kv: float | None = None         # damping; model.KV
    effort: float | None = None     # torque clip, Nm; model.EFFORT
    armature: float = ARMATURE


@dataclass(frozen=True)
class Guard:
    """What refuses a pose. See `robot/collide.py` for what the clamp does."""

    avoid: bool = True              # clamp commands that self-intersect
    ik_avoid: bool = True           # and the poses the tip IK proposes
    margin: float | None = None     # metres of daylight it insists on; collide.MARGIN


@dataclass(frozen=True)
class Scene:
    """What is drawn, and where.

    `copies` are extra posed hands beside the main one: (prefix, rgba, offset),
    where a scalar offset is sideways and a pair is (forward, sideways). The
    encoder ghost is a copy like any other and is added for you when `ghost`.
    `layout.Row` and `layout.Grid` compute these; nothing else should be doing
    the arithmetic by hand.
    """

    show_viewer: bool = True
    position: tuple[float, float, float] = (0.0, 0.0, 0.20)
    ghost: bool = True
    ghost_offset: float = 0.24
    camera_pos: tuple[float, float, float] | None = None
    camera_lookat: tuple[float, float, float] | None = None
    copies: tuple = ()
    main_rgba: tuple[float, float, float, float] | None = None
