"""Where several hands stand when more than one is on screen.

THE ARITHMETIC WAS WRITTEN THREE TIMES. The replay viewer, the gallery and
`main.py` each stood a set of hands beside each other, and each worked out for
itself which way `side` points, where the middle of the row is, how far each
copy sits from the driven one, and where to put the camera so the whole thing
is in frame. `LANE = 0.32` was defined twice, independently. They are one
idea with two shapes - a line of lanes, and a grid of cells - so they live
here, and a caller says how many hands and gets back where they go.

WHAT A CALLER STILL OWNS: which hands, what colour, and what the camera is
looking for. This file knows about metres, not about meaning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .model import DEFAULT_HAND

# Sideways metres between neighbouring hands. Wide enough for the widest
# nameplate the replay viewer writes on the floor, and narrow enough that six
# lanes still fit in one camera frame.
LANE = 0.32

# Between cells of a grid. Wider than LANE because a grid has hands in front
# of each other as well as beside, and a hand is deeper than it is wide.
PITCH = 0.36

# How high off the floor a hand is bolted, when nothing says otherwise.
HEIGHT = 0.20


def side_of(hand):
    """+1 or -1: which way "sideways" points for this chirality.

    Everything laid out here is mirrored with the robot, so a row that reads
    left-to-right for a left hand reads right-to-left for a right one and the
    camera stays on the same side of it.
    """
    return 1.0 if hand == "right" else -1.0


@dataclass(frozen=True)
class Row:
    """`slots` hands in a line, centred on the origin.

    Slot 0 is conventionally the operator's own skeleton and slot 1 the hand
    the simulator actually drives - the copies are placed relative to THAT,
    because `HandSim` poses copies at an offset from wherever it stands. So
    moving slot 1 moves the whole row, which is why `anchor` is slot 1 and not
    slot 0.

    CENTRED, not marching off the origin. The row used to start at zero and
    run one way, which put the whole thing to one side of the floor grid it
    stands on; two hands left of centre and two right reads as a row rather
    than as a drift.
    """

    slots: int
    hand: str = DEFAULT_HAND
    spacing: float = LANE
    height: float = HEIGHT
    forward: float = 0.0

    @property
    def side(self):
        return side_of(self.hand)

    def place(self, slot):
        """Where `slot` stands, in world metres - what to add to something
        drawn in a hand's own frame to put it in that lane."""
        centre = (self.slots - 1) / 2.0
        return (self.forward, self.side * self.spacing * (slot - centre), self.height)

    @property
    def anchor(self):
        """Where the driven hand stands, which places the row. Slot 1."""
        return self.place(1)

    def offset(self, slot):
        """Sideways metres from the driven hand to `slot`, as `Scene.copies`
        wants it - so slot 1 is 0 and each one after is a spacing further."""
        return (slot - 1) * self.spacing

    def copies(self, entries, start=2):
        """[(prefix, rgba)] -> the tuple `Scene.copies` takes, placed in order
        from `start`."""
        return tuple((prefix, rgba, self.offset(slot))
                     for slot, (prefix, rgba) in enumerate(entries, start=start))

    def camera(self, back=0.95, lift=0.52, look=HEIGHT - 0.04):
        """(position, look-at) framing the row, both in world metres.

        In front of the palms and a little to the side the row does not run,
        so no hand comes between the camera and its neighbour.
        """
        return ((back, -self.side * 0.05, lift), (0.0, 0.0, look))


@dataclass(frozen=True)
class Grid:
    """`slots` hands in as square a block as they will make.

    The gallery's shape: sixteen solved runs at once, where the point is the
    spread of a whole set rather than any one hand. Cell 0 is the driven hand
    and every other cell is a copy offset from it, forward as well as
    sideways - which is why `Scene.copies` takes a pair here and a scalar for
    a `Row`.
    """

    slots: int
    hand: str = DEFAULT_HAND
    pitch: float = PITCH
    height: float = HEIGHT

    @property
    def side(self):
        return side_of(self.hand)

    @property
    def shape(self):
        """(rows, columns)."""
        cols = math.ceil(math.sqrt(self.slots))
        return math.ceil(self.slots / cols), cols

    @property
    def cells(self):
        """(forward, sideways) of every cell, centred on the origin."""
        rows, cols = self.shape
        return [((r - (rows - 1) / 2) * self.pitch,
                 (c - (cols - 1) / 2) * self.pitch)
                for r in range(rows) for c in range(cols)][:self.slots]

    @property
    def anchor(self):
        """Where cell 0 stands, which places the grid."""
        forward, sideways = self.cells[0]
        return (forward, self.side * sideways, self.height)

    def copies(self, prefix, rgba):
        """Every cell but the first, named `prefix0`, `prefix1`, ... and
        offset from cell 0."""
        first_forward, first_sideways = self.cells[0]
        return tuple((f"{prefix}{k}_", rgba,
                      (forward - first_forward, sideways - first_sideways))
                     for k, (forward, sideways) in enumerate(self.cells) if k > 0)

    def camera(self):
        """(position, look-at) framing the whole block."""
        rows, cols = self.shape
        span = max(rows, cols) * self.pitch
        return ((0.9 * span + 0.6, -self.side * 0.15, 0.55 * span + 0.45),
                (0.0, 0.0, self.height * 0.6))
