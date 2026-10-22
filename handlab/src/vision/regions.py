"""An object's own region in a mask, without the specks segmentation leaves.

SAM's masks can carry small stray pieces far from the object - on the first
SAM 3 recording, 345 of 411 frames had up to 31 of them, many on the image's
edges. They are few pixels but far away, so they move the mask's centre (by
17-28 px there) and swamp its axis angle. The object's region is its largest
piece, plus any piece big enough (KEEP_SHARE of the largest) and close enough
(within the largest piece's box grown by GROW on every side) to be part of
the same object seen past an occluding finger.
"""

from __future__ import annotations

import cv2
import numpy as np

KEEP_SHARE = 0.1
GROW = 0.5


def main_region(mask):
    """`mask` (H, W) bool with the stray pieces removed."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 2:                                   # background + at most one piece
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x, y, w, h = stats[largest, :4]
    x0, y0, x1, y1 = x - GROW * w, y - GROW * h, x + w + GROW * w, y + h + GROW * h
    keep = [largest]
    for piece in range(1, count):
        px, py, pw, ph, area = stats[piece]
        if (piece != largest and area >= KEEP_SHARE * areas.max()
                and px >= x0 and py >= y0 and px + pw <= x1 and py + ph <= y1):
            keep.append(piece)
    return np.isin(labels, keep)
