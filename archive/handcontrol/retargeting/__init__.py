"""Turning a tracked human hand into h4nd joint targets.

Four files, layered so the parts that are not about this robot can be lifted
out of it:

    filters.py      sensor conditioning. Arrays in, arrays out. No robot.
    geometry.py     frames, hinges, signed angles. No files, no state.
    calibration.py  the fit to one operator, and its file format. No robot.
    retargeter.py   the mapping onto THIS hand. The only file that binds.

That split is not tidiness for its own sake. The mapping in `retargeter.py` is
angle-based, and angle-based mapping has a ceiling: equal angles on unequal
segments do not put fingertips in equal places. Replacing it - with keyvector
retargeting, which matches the vectors BETWEEN fingertips rather than the
angles or the positions - means writing a sibling to that one file. The
filtering, the geometry and the calibration are the same work either way.
"""

from .calibration import CALIBRATION, fit_ranges, load_fit, save_fit
from .filters import Hampel, HandFilter, OneEuro
from .geometry import (flex_angle, flexion_axis, hand_frame, hand_length,
                       robot_axes, spread_angle, unit)
from .retargeter import (CURLED, DIGIT_FOR, GAINS, THUMB_REST_SPREAD,
                         Retargeter)

__all__ = [
    "CALIBRATION", "CURLED", "DIGIT_FOR", "GAINS", "THUMB_REST_SPREAD",
    "Hampel", "HandFilter", "OneEuro", "Retargeter",
    "fit_ranges", "flex_angle", "flexion_axis", "hand_frame", "hand_length",
    "load_fit", "robot_axes", "save_fit", "spread_angle", "unit",
]
