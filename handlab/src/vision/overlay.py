"""Drawing what was found onto a frame: object masks and boxes, then the
hands' skeletons on top.

Everything is in the true (unmirrored) image. `mirror` flips the picture,
masks, boxes and landmarks together before drawing, so a mirrored view still
has readable labels.
"""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import CONNECTIONS, TIPS

# RGB. Left and right hands in different colours, so a swapped label shows.
COLOURS = {"Left": (80, 200, 255), "Right": (255, 150, 60)}
OTHER = (220, 220, 220)
# Objects, by their place in the task context's list.
OBJECT_COLOURS = ((60, 220, 120), (255, 90, 200), (255, 215, 60), (150, 120, 255),
                  (60, 200, 230))
MASK_ALPHA = 0.45
BONE = 2                 # px
JOINT = 3                # px radius; fingertips one larger
FONT = cv2.FONT_HERSHEY_SIMPLEX


def object_colour(index):
    return OBJECT_COLOURS[index % len(OBJECT_COLOURS)]


def draw(rgb, hands, fps=None, mirror=False, masks=(), boxes=(), axes=(), lines=()):
    """A copy of `rgb`, mirrored left-right first if `mirror`, with drawn on:
    `masks` - (mask (H, W) bool, colour, label) - tinted and outlined;
    `boxes` - ((left, top, right, bottom) in 0..1, colour, label);
    `axes` - ((x, y) px centre, angle rad, half length px, colour): an
    object's long axis through its centre;
    `lines` - ((x0, y0), (x1, y1), colour, thickness) px: e.g. a 3D fit's
    wireframe, projected;
    then each hand's skeleton, side and score."""
    out = cv2.flip(rgb, 1) if mirror else rgb.copy()
    height, width = out.shape[:2]
    for mask, colour, label in masks:
        if mirror:
            mask = mask[:, ::-1]
        if not mask.any():
            continue
        out[mask] = (out[mask] * (1 - MASK_ALPHA)
                     + np.array(colour) * MASK_ALPHA).astype(np.uint8)
        outline, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, outline, -1, colour, 2, cv2.LINE_AA)
        ys, xs = np.nonzero(mask)
        _label(out, label, (int(np.median(xs)) - 4 * len(label), max(14, int(ys.min()) - 6)),
               colour)
    for (left, top, right, bottom), colour, label in boxes:
        if mirror:
            left, right = 1.0 - right, 1.0 - left
        corner = (int(left * width), int(top * height))
        cv2.rectangle(out, corner, (int(right * width), int(bottom * height)),
                      colour, 2, cv2.LINE_AA)
        _label(out, label, (corner[0], max(14, corner[1] - 6)), colour)
    for (x0, y0), (x1, y1), colour, thickness in lines:
        if mirror:
            x0, x1 = width - 1 - x0, width - 1 - x1
        cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), colour, thickness, cv2.LINE_AA)
    for (x, y), angle, half, colour in axes:
        if mirror:
            x, angle = width - 1 - x, np.pi - angle
        dx, dy = half * np.cos(angle), half * np.sin(angle)
        cv2.line(out, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)),
                 (255, 255, 255), 6, cv2.LINE_AA)
        cv2.line(out, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)),
                 colour, 3, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 4, colour, -1, cv2.LINE_AA)
    for hand in hands:
        colour = COLOURS.get(hand.side, OTHER)
        pixels = [(int(((1.0 - x) if mirror else x) * width), int(y * height))
                  for x, y, _ in hand.image]
        for start, end in CONNECTIONS:
            cv2.line(out, pixels[start], pixels[end], colour, BONE, cv2.LINE_AA)
        for i, point in enumerate(pixels):
            radius = JOINT + 1 if i in TIPS else JOINT
            cv2.circle(out, point, radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, point, radius, colour, 1, cv2.LINE_AA)
        x, y = pixels[0]
        score = f" {hand.score:.2f}" if np.isfinite(hand.score) else " (filled)"
        _label(out, hand.side + score, (x + 8, y + 18), colour)
    if fps is not None:
        _label(out, f"{fps:.0f} fps", (8, 22), OTHER)
    return out


def _label(image, text, origin, colour):
    """Text with a dark outline, readable on any background."""
    cv2.putText(image, text, origin, FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, 0.55, colour, 1, cv2.LINE_AA)
