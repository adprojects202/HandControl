"""The turns that leave a box looking the same - what "the same pose" means
for an object that is a box (a cube looks alike 24 ways)."""

import numpy as np


def box_symmetries(size):
    """Rotations that map a box of `size` onto itself (a cube: 24)."""
    found = []
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        if not np.allclose(np.asarray(size)[list(order)], size):
            continue
        for signs in np.ndindex(2, 2, 2):
            turn = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                turn[row, column] = -1.0 if sign else 1.0
            if np.linalg.det(turn) > 0:
                found.append(turn)
    return found
