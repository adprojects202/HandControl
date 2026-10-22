"""The object step: how each segmented object moves, by itself and relative
to the hand holding it -> object.npz.

    measure_objects(episode, progress)

Needs the segment step (masks.npz); uses the hands step's hands when it has
run (hands_clean.npz), else the hands as recorded.

In the image plane for now - positions in pixels, angles about the camera's
axis. Depth (and so 3D position and full rotation) is the next step.

Per mask track (O) and frame (N):

    visible      (O, N) bool: the mask has any pixels
    centroid     (O, N, 2) px;  area (O, N) fraction of the image
    angle        (O, N) rad: the mask's long axis, from its second moments,
                 unwrapped through time (it has no head or tail: it is only
                 known modulo pi)
    elongation   (O, N) long/short axis ratio - near 1 (a cube: 1.05) the
                 long axis means nothing; use `turn`
    turn         (O, N) rad: how far the object has turned in the image since
                 it was first seen, counter-clockwise as the camera sees it.
                 Each frame is compared with a reference frame: the turn is
                 the rotation that best lays the reference's mask (centred,
                 slightly blurred) over this one, so it works for any shape.
                 The reference moves on when the object has turned KEY_TURN
                 from it or no longer matches it well (KEY_MATCH) - comparing
                 with the last frame instead loses the small steps (0.5 deg a
                 frame measured as 0.06). NaN where unseen; after a gap longer
                 than MAX_GAP frames it carries on from the last value.
    heading      (O, N) rad: the long axis at first sight, carried round by
                 `turn` - in image angles like `angle` (y down, so clockwise
                 on screen is positive: heading = angle at first sight - turn).
                 Drawn on playback; for a cube the first axis is arbitrary
                 but it then turns with the cube.
    contact      (O, N, 2) bool, per hand slot: at least CONTACT_POINTS of
                 that hand's landmarks lie within CONTACT_PX of the mask
    hand         (O,) the slot (0 left, 1 right) in contact most often; -1 none
    in_hand      (O, N, 2): centroid relative to that hand's palm, in hand
                 lengths (wrist to middle knuckle), image axes; NaN unseen
    in_hand_angle  (O, N) rad: angle relative to that hand's own axis
                 (wrist to middle knuckle), unwrapped like `angle`
    in_hand_turn (O, N) rad: `turn` less the holding hand's own turn in the
                 image - how far the object has turned in the hand
"""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np

from ..recording.episode import OBJECT

CONTACT_PX = 12          # a landmark this close to the mask touches it
TURN_STEP_MAX = np.radians(30)    # most an object turns between two frames
TURN_SEARCH = np.radians(1)       # search step, then refined
TURN_CROP = 128                   # px, side of the masks compared
TURN_BLUR = 1.5                   # px (in the crop): a smooth match to refine
KEY_TURN = np.radians(20)         # a new reference after this much turn ...
KEY_MATCH = 0.85                  # ... or when the match falls below this
MAX_GAP = 5                       # frames unseen before a turn starts over
CONTACT_POINTS = 3       # landmarks touching, for the hand to be in contact
PALM = [0, 5, 9, 13, 17]
WRIST, MIDDLE_KNUCKLE = 0, 9


def measure_objects(episode, progress=lambda done, total, text="": None):
    """Measure every mask track of the episode and save object.npz;
    returns {name: share of frames held} as a summary."""
    masks = episode.masks
    if not masks:
        raise ValueError("segment the recording first: there are no masks")
    hands = episode.clean_hands or episode.hands
    if not hands:
        raise ValueError("the recording has no hands")
    tracks, count = len(masks["names"]), episode.frames
    width, height = int(masks["width"]), int(masks["height"])
    landmarks = hands["image"][..., :2] * np.array([width, height])   # (N, 2, 21, 2) px
    present = hands["present"]
    out = {"visible": np.zeros((tracks, count), bool),
           "centroid": np.full((tracks, count, 2), np.nan, np.float32),
           "area": np.zeros((tracks, count), np.float32),
           "angle": np.full((tracks, count), np.nan, np.float32),
           "elongation": np.full((tracks, count), np.nan, np.float32),
           "contact": np.zeros((tracks, count, 2), bool),
           "turn": np.full((tracks, count), np.nan, np.float32),
           "heading": np.full((tracks, count), np.nan, np.float32)}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CONTACT_PX + 1,) * 2)
    for track in range(tracks):
        for index in range(count):
            mask = episode.mask(track, index)
            if not mask.any():
                continue
            out["visible"][track, index] = True
            moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
            out["centroid"][track, index] = (moments["m10"] / moments["m00"],
                                             moments["m01"] / moments["m00"])
            out["area"][track, index] = moments["m00"] / (width * height)
            out["angle"][track, index], out["elongation"][track, index] = _axis(moments)
            near = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            for slot in range(2):
                if present[index, slot]:
                    x, y = np.clip(landmarks[index, slot].round().astype(int).T,
                                   0, [[width - 1], [height - 1]])
                    out["contact"][track, index, slot] = near[y, x].sum() >= CONTACT_POINTS
            progress(track * count + index + 1, tracks * count,
                     f"{masks['names'][track]}: frame {index}")
        out["angle"][track] = _unwrap_axis(out["angle"][track])
        out["turn"][track] = _turns([episode.mask(track, i) if out["visible"][track, i]
                                     else None for i in range(count)],
                                    out["centroid"][track], out["area"][track] * width * height)
        first = np.flatnonzero(out["visible"][track])
        if first.size:
            out["heading"][track] = out["angle"][track, first[0]] - out["turn"][track]
    holds = out["contact"].sum(axis=1)                               # (O, 2)
    out["hand"] = np.where(holds.max(axis=1) > 0, holds.argmax(axis=1), -1)
    out["in_hand"], out["in_hand_angle"], out["in_hand_turn"] = _relative(
        out, landmarks, present)
    episode.save_arrays(OBJECT, dict(
        out, names=masks["names"], made=np.array(datetime.now().isoformat(timespec="seconds"))))
    return {str(name): float(out["contact"][k, :, out["hand"][k]].mean())
            if out["hand"][k] >= 0 else 0.0 for k, name in enumerate(masks["names"])}


def _axis(moments):
    """(angle of the long axis, long/short ratio) from a mask's moments."""
    mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    spread = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    long, short = (mu20 + mu02 + spread) / 2, (mu20 + mu02 - spread) / 2
    return angle, np.sqrt(long / short) if short > 0 else np.inf


def _unwrap_axis(angle):
    """An axis angle (known modulo pi) made continuous through time; NaN gaps
    are kept and do not break the continuity."""
    seen = ~np.isnan(angle)
    out = angle.copy()
    if seen.any():
        out[seen] = np.unwrap(2 * angle[seen]) / 2
    return out


def _turns(masks, centroids, areas):
    """The object's turn in each frame (see the module docstring)."""
    turn = np.full(len(masks), np.nan, np.float32)
    reference = None              # (crop, side, turn at the reference)
    last, total, from_reference = None, 0.0, 0.0
    for index, mask in enumerate(masks):
        if mask is None:
            continue
        if reference is None or index - last > MAX_GAP + 1:
            side = 2.5 * np.sqrt(areas[index])
            reference, from_reference = (_crop(mask, centroids[index], side), side, total), 0.0
        else:
            crop, side, base = reference
            from_reference, match = _rotation(crop, _crop(mask, centroids[index], side),
                                              around=from_reference)
            total = base + from_reference
            if abs(from_reference) > KEY_TURN or match < KEY_MATCH:
                side = 2.5 * np.sqrt(areas[index])
                reference, from_reference = (_crop(mask, centroids[index], side), side,
                                             total), 0.0
        turn[index], last = total, index
    return turn


def _crop(mask, centre, side):
    """A TURN_CROP-pixel square of `mask` around `centre`, `side` px across,
    blurred a little so the match below varies smoothly with the angle."""
    scale = TURN_CROP / side
    move = np.array([[scale, 0, TURN_CROP / 2 - scale * centre[0]],
                     [0, scale, TURN_CROP / 2 - scale * centre[1]]])
    crop = cv2.warpAffine(mask.astype(np.float32), move, (TURN_CROP, TURN_CROP))
    return cv2.GaussianBlur(crop, (0, 0), TURN_BLUR)


def _rotation(reference, crop, around=0.0):
    """(rotation in rad, counter-clockwise on screen, that best lays
    `reference` over `crop` - searched within TURN_STEP_MAX of `around`, then
    refined by a parabola; how well they then match, 0..1)."""
    angles = around + np.arange(-TURN_STEP_MAX, TURN_STEP_MAX + TURN_SEARCH / 2,
                                TURN_SEARCH)
    centre = (TURN_CROP / 2, TURN_CROP / 2)
    overlap = np.array([
        (cv2.warpAffine(reference, cv2.getRotationMatrix2D(centre, np.degrees(a), 1.0),
                        (TURN_CROP, TURN_CROP)) * crop).sum() for a in angles])
    best = int(np.argmax(overlap))
    angle = angles[best]
    if 0 < best < len(angles) - 1:
        left, mid, right = overlap[best - 1:best + 2]
        curve = left - 2 * mid + right
        if curve < 0:
            angle += 0.5 * (left - right) / curve * TURN_SEARCH
    match = overlap[best] / max(np.sqrt((reference ** 2).sum() * (crop ** 2).sum()), 1e-9)
    return angle, match


def _relative(out, landmarks, present):
    """Centroid, angle and turn in the holding hand's own image frame."""
    tracks, count = out["visible"].shape
    in_hand = np.full((tracks, count, 2), np.nan, np.float32)
    in_hand_angle = np.full((tracks, count), np.nan, np.float32)
    in_hand_turn = np.full((tracks, count), np.nan, np.float32)
    for track in range(tracks):
        slot = out["hand"][track]
        if slot < 0:
            continue
        hand = landmarks[:, slot]                                    # (N, 21, 2)
        axis = hand[:, MIDDLE_KNUCKLE] - hand[:, WRIST]
        length = np.linalg.norm(axis, axis=1)
        seen = out["visible"][track] & present[:, slot] & (length > 0)
        palm = hand[:, PALM].mean(axis=1)
        in_hand[track, seen] = ((out["centroid"][track, seen] - palm[seen])
                                / length[seen, None])
        hand_angle = np.arctan2(axis[:, 1], axis[:, 0])
        relative = np.full(count, np.nan)
        relative[seen] = out["angle"][track, seen] - hand_angle[seen]
        in_hand_angle[track] = _unwrap_axis(relative)
        hand_turn = np.full(count, np.nan)                   # image y is down: a
        hand_turn[seen] = -hand_angle[seen]                  # turn on screen is -angle
        if seen.any():
            hand_turn[seen] = np.unwrap(hand_turn[seen]) - hand_turn[seen][0]
            in_hand_turn[track, seen] = out["turn"][track, seen] - hand_turn[seen]
    return in_hand, in_hand_angle, in_hand_turn
