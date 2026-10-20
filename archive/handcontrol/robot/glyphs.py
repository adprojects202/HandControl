"""Drawing short words into a MuJoCo scene, out of small spheres.

WHY THIS EXISTS RATHER THAN A LABEL. `mjvGeom` carries a `label` field and it
looks like the answer, but MuJoCo's passive viewer does not draw it: rendered
offscreen, adding a label to a visible marker changes exactly zero pixels. The
viewer's own label options draw every body in the scene, which for four hands
is eighty-four labels and unreadable.

So a lane's name is drawn the way anything else is drawn - as geometry. Five
rows by three columns per character, one small marker per lit cell, standing in
the floor beneath the hand it names.

Colour alone was not enough, and that is the whole reason: teal and green share
a green channel and read as one colour under this scene's light, so two lanes
looked identical. A word does not have that failure mode.
"""

from __future__ import annotations

import numpy as np

# 3 wide, 5 tall, top row first.
#
# THE WHOLE ALPHABET, not only the letters the current lane names happen to
# use. It was the subset, and "ROBOT" came out as "RO OT" - an unknown
# character falls back to a blank, so a missing glyph is invisible rather than
# loud. Nine letters were absent and nobody could have known which.
FONT = {
    "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "111", "100", "111"),
    "F": ("111", "100", "111", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("111", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "011"),
    "R": ("111", "101", "111", "110", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
    "+": ("000", "010", "111", "010", "000"),
    "-": ("000", "000", "111", "000", "000"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    " ": ("000", "000", "000", "000", "000"),
}

CELL = 0.007          # metres per dot
GAP = 1               # blank columns between characters

# How high the middle of a nameplate sits above the floor, in glyph rows. Three
# clears the descender; more than that lifts the word off the floor and closer
# to the hand it names, which is where it is actually readable.
LIFT = 20


def unknown(text):
    """Characters `text` would silently render as blanks."""
    return sorted({c for c in text.upper() if c not in FONT})


def cells(text):
    """(row, column) of every lit dot in `text`, origin at its top left."""
    lit, column = [], 0
    for character in text.upper():
        rows = FONT.get(character, FONT[" "])
        for row, pattern in enumerate(rows):
            for offset, value in enumerate(pattern):
                if value == "1":
                    lit.append((row, column + offset))
        column += len(rows[0]) + GAP
    return lit, column - GAP


def draw(scene, text, centre, colour, cell=CELL, marker=None):
    """Add `text` to a MuJoCo scene, standing upright and centred on `centre`.

    UPRIGHT, NOT FLAT ON THE FLOOR. Lying down, a word's reading direction
    depends on which side the camera is - orbit past ninety degrees and it
    mirrors - and its rows map to depth rather than height, which rendered the
    glyphs upside down: measured, a glyph's top row landed 60 pixels BELOW its
    bottom row. Standing it in the y-z plane makes rows height and columns
    width, which is what a reader expects and cannot be turned upside down by a
    camera move.

    `marker` is the callable that adds one box - `HandSim` owns the geom buffer
    and its bounds check, so this asks rather than reaching into the scene.
    """
    lit, width = cells(text)
    if not lit:
        return
    origin = np.asarray(centre, dtype=float)
    step = cell * 1.4
    for row, column in lit:
        # Columns along +y, which is screen-right from the default camera;
        # rows along -z, so row 0 is the top.
        position = origin + np.array([0.0,
                                      (column - width / 2.0) * step,
                                      (2 - row) * step])
        marker(scene, position, cell * 0.62, colour)
