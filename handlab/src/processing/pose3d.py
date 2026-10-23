"""The 3D step: the hands and the object in metres, in the camera's frame,
and the object relative to the palm holding it -> pose3d.npz.

    fit_3d(episode, progress)

Needs the camera's calibration (Episode.calibration), the hands step and the
object step. The object is taken to be the World section's calibration
object: a box of `object.size_mm`.

Hands. MediaPipe gives each hand's shape in metres (world landmarks) and
where its points are in the image; with the camera's numbers the two give
the hand's pose (OpenCV solvePnP, lens distortion included). The palm frame:
origin at the palm's centre (wrist and four knuckles), x from the wrist to
the middle knuckle, z out of the palm (the side that holds things), y = z x x.
The hand's size is MediaPipe's guess, and the distance scales with it. The
guess wobbles from frame to frame (wrist to middle knuckle 85-102 mm on the
first recording, while the hand moved 1 cm), which made the distance swing
by 5 cm. A hand does not change size, so each frame's shape is first scaled
to the recording's median size. `hand_scale` is what the object's known
size says that size should be (see below).

Object. A box of known size is posed so that its outline - the convex hull
of its corners, projected with the camera's numbers - covers the mask
(overlap, softened at the edges so the fit can feel its way). Its size in
the image fixes its distance; the shape of its outline, its rotation. Each
frame starts from the last frame's pose (Nelder-Mead). The first frame - and
any frame whose fit ends below RETRY_BELOW, i.e. the object turned faster
than one frame's refinement follows - searches instead: SEARCH
orientations spread evenly over all of them, each placed so its outline has
the mask's centre and area, are compared, and the best SEARCH_KEEP refined.
(Guessing a few tilts at one position left a test cube 39-46 degrees off:
every guess was misplaced, so none looked right.) On test cubes: within 1
degree and 5 mm - except square-on to the camera, where a tilt of up to
about 10 degrees leaves the outline the same square. A
cube's outline cannot tell a tilt towards the camera from the same tilt
away; following it frame to frame keeps one reading throughout. A box also
looks the same turned onto any of its like faces (a cube: 24 ways), so of
those equal poses each frame keeps the one nearest the last frame's - the
rotation then changes only as the object really turns.

Both are then smoothed through time - a median over SMOOTH_FRAMES frames
of positions and of rotations (as unit quaternions kept in one hemisphere),
within each unbroken run. MediaPipe's hand shape jumps for a frame or a few
while the object hides the hand (the palm turned up to 64 degrees in one
frame on the first recording; median 1.8), and the fit wavers where a cube
faces the camera squarely; the median takes out both and keeps real turns.

A still hand. The robot's hand does not move, so demonstrations are made
with the hand held still, and then its median pose over the recording is a
better palm frame than each frame's: the palm's orientation is the least
sure thing here while the object hides it (on the first recording, a fifth
of the frames were over 20 degrees off the palm's 2-second median, and the
fit could not tell which were right). When the palm's centre moves less
than STILL_M across the image (10-90% of frames, at its median distance)
the hand counts as still and `in_palm_*` use its median pose; otherwise
each frame's (`still_hand` says which). Across, not in depth: the depth is
the least sure part of the palm's position.

ArUco markers (testing). With `vision.aruco`, every frame of the recording
is searched for markers; the largest one seen is taken to sit in the middle
of a face of the first object, so its pose gives the object's - its centre
half the box's depth behind the marker (vision/aruco.py). Where a marker is
seen its pose is used (`object_source` 2); elsewhere the outline fit (1).
The outline fit also runs on every frame on its own (`outline_*`, never
told about the markers), and where both exist the two are compared:
`outline_error_mm`, `outline_error_deg` (up to the box's symmetries) - a
measure of the markerless fit. Measured on rendered 40 mm markers at 0.3 m
with this camera: 1-2 mm, and 1-2 degrees tilted - but 4-5 square-on, where
a marker's own rotation is as unsure as an outline's. Markers whose
corners fit their pose worse than MARKER_ERROR_SHARE of their side in the
image are not used (a fixed 2 px threw out most of a close-up recording,
where markers were 105 px across and fitted to 2.4 px).

hand_scale: an object held against the palm has its centre half its
thinnest side out from the palm, along the palm's normal, so the palm's true
distance follows from the object's. The factor that puts it there, over the
frames the hand holds it (median), is `hand_scale` (1 = MediaPipe's hand is
the right size; scaling the hand scales its distance from the camera). It is
reported, and `in_palm_*` use the palm moved to match. "The palm" is the
plane of MediaPipe's wrist and knuckle points - the joints, not the skin.

Per hand slot (2) and frame (N); per object track (O):

    palm_position (N, 2, 3) m, palm_rotation (N, 2, 3, 3) - columns are the
                 palm's x, y, z in camera axes; NaN where the hand is unseen
    hand_fit_px  (N, 2) mean distance of the hand's points from their fit
    object_position (O, N, 3) m, object_rotation (O, N, 3, 3), object_fit
                 (O, N) overlap 0..1 of outline and mask
    in_palm_position (O, N, 3) m, in_palm_rotation (O, N, 3, 3) - the object
                 in its holding hand's palm frame
    in_palm_turn (O, N) deg: how far the object has turned in the palm since
                 the first frame it was seen there - the smallest such angle
                 over the box's symmetries, which no outline can tell apart
    box_mm (3,), hand (O,) slot, hand_scale (O,), still_hand (O,) bool, made
    object_source (O, N) 0 unseen, 1 outline, 2 marker
    outline_position (O, N, 3), outline_rotation (O, N, 3, 3): the outline
                 fit alone, unsmoothed
    marker_id (N,) the marker used (-1 none); marker_error_px (N,)
    outline_error_mm, outline_error_deg (N,): outline fit against the marker
"""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .. import settings
from ..recording.episode import OBJECT
from ..vision import aruco
from ..vision.tracker import SIDES
from ..sim.symmetry import box_symmetries as _symmetries
from .files import POSE3D

PALM = [0, 5, 9, 13, 17]
WRIST, INDEX_KNUCKLE, MIDDLE_KNUCKLE, PINKY_KNUCKLE = 0, 5, 9, 17
# cross(index - wrist, pinky - wrist) points into the palm of a left hand,
# out of it on a right (measured on a recorded left hand, palm to camera).
OUT_OF_PALM = {"Left": -1.0, "Right": 1.0}
CROP = 120                    # px, the longer side of the window the fit is done in
EDGE_SOFTNESS = 1.5           # px blur of the outline and the mask
SEARCH = 2000
SEARCH_KEEP = 8
SMOOTH_FRAMES = 9
MARKER_ERROR_SHARE = 0.03
# Overlap under which a frame is searched afresh. At 0.9 the fit lagged a
# fast-turning cube for a few frames, stuck near 0.90, then jumped 40-47
# degrees when the search found it again (0.97-0.98).
RETRY_BELOW = 0.95
SWITCH_GAIN = 0.02            # ... and the search's pose taken only if this much better
STILL_M = 0.03


def fit_3d(episode, progress=lambda done, total, text="": None):
    """Fit the episode's hands and objects in 3D and save pose3d.npz;
    returns a summary."""
    calibration = episode.calibration
    if calibration is None:
        raise ValueError("calibrate the camera this was recorded with first "
                         f"({episode.meta.get('camera', {}).get('name', '?')})")
    hands = episode.clean_hands
    motion = episode.arrays(OBJECT)
    if not hands or not motion:
        raise ValueError("run the hands and object steps first")
    width, height = episode.meta["camera"]["width"], episode.meta["camera"]["height"]
    calibration = calibration.scaled(width, height)
    if calibration is None:
        raise ValueError("the calibration is for another image shape")
    camera = (calibration.matrix(), np.array(calibration.distortion))
    box_mm = np.array(settings.get("object.size_mm"), float)
    count, tracks = episode.frames, len(motion["names"])

    out = _hands(hands, camera, width, height)
    for slot in range(2):
        _smooth(out["palm_position"][:, slot], out["palm_rotation"][:, slot])
    progress(0, tracks * count, "hands done")
    out.update(object_position=np.full((tracks, count, 3), np.nan),
               object_rotation=np.full((tracks, count, 3, 3), np.nan),
               object_fit=np.full((tracks, count), np.nan),
               in_palm_position=np.full((tracks, count, 3), np.nan),
               in_palm_rotation=np.full((tracks, count, 3, 3), np.nan),
               in_palm_turn=np.full((tracks, count), np.nan),
               hand_scale=np.full(tracks, np.nan),
               still_hand=np.zeros(tracks, bool))
    corners = _box_corners(box_mm / 1000)
    symmetries = _symmetries(box_mm)
    markers = _markers(episode, camera, box_mm.min() / 1000, progress)
    out.update(markers)
    out.update(object_source=np.zeros((tracks, count), np.uint8),
               outline_position=np.full((tracks, count, 3), np.nan),
               outline_rotation=np.full((tracks, count, 3, 3), np.nan),
               outline_error_mm=np.full(count, np.nan),
               outline_error_deg=np.full(count, np.nan))
    for track in range(tracks):
        pose, last = None, None
        for index in range(count):
            if motion["visible"][track, index]:
                mask = episode.mask(track, index)
                pose = _fit_box(mask, corners, camera, pose, box_mm.max() / 1000,
                                symmetries)
                if pose is not None:
                    rotation, position, overlap = pose
                    out["outline_position"][track, index] = position
                    out["outline_rotation"][track, index] = rotation
                    out["object_fit"][track, index] = overlap
                    out["object_source"][track, index] = 1
            if track == 0 and out["marker_id"][index] >= 0:    # a marker: the object's pose
                rotation, position = (out["marker_rotation"][index],
                                      out["marker_position"][index])
                out["object_source"][track, index] = 2
                if pose is not None:
                    out["outline_error_mm"][index] = 1000 * np.linalg.norm(pose[1] - position)
                    out["outline_error_deg"][index] = _turn_between(rotation, pose[0],
                                                                    symmetries)
            elif out["object_source"][track, index] == 0:
                progress(track * count + index + 1, tracks * count,
                         f"{motion['names'][track]}: frame {index}")
                continue
            if last is not None:                    # the same look, nearest the last
                rotation = _nearest_equivalent(rotation, last, symmetries)
            out["object_position"][track, index], out["object_rotation"][track, index] = \
                position, rotation
            last = rotation
            progress(track * count + index + 1, tracks * count,
                     f"{motion['names'][track]}: frame {index}")
        _smooth(out["object_position"][track], out["object_rotation"][track])
        _in_palm(out, track, int(motion["hand"][track]), motion, box_mm / 1000, symmetries)
    out.update(box_mm=box_mm, hand=motion["hand"], names=motion["names"],
               made=np.array(datetime.now().isoformat(timespec="seconds")))
    episode.save_arrays(POSE3D, {k: np.asarray(v, np.float32) if isinstance(v, np.ndarray)
                                 and v.dtype == np.float64 else v for k, v in out.items()})
    seen = np.isfinite(out["palm_position"][..., 2])
    compared = np.isfinite(out["outline_error_mm"])
    return {"palm_distance_m": float(np.nanmedian(out["palm_position"][..., 2][seen]))
            if seen.any() else None,
            "object_fit": [float(np.nanmedian(f)) for f in out["object_fit"]],
            "hand_scale": [float(s) for s in out["hand_scale"]],
            "still_hand": [bool(s) for s in out["still_hand"]],
            "marker_frames": int((out["marker_id"] >= 0).sum()),
            "outline_vs_marker_mm": float(np.median(out["outline_error_mm"][compared]))
            if compared.any() else None,
            "outline_vs_marker_deg": float(np.median(out["outline_error_deg"][compared]))
            if compared.any() else None}


def load(episode):
    """An episode's pose3d.npz ({} before Fit 3D), with the fields an older
    file lacks filled in as they would have been: files made before ArUco
    markers were looked for have no markers, and every posed frame came from
    the outline fit."""
    pose = dict(episode.arrays(POSE3D))
    if not pose:
        return pose
    tracks, count = pose["object_position"].shape[:2]
    posed = np.isfinite(pose["object_position"][..., 0])
    defaults = {"object_source": lambda: posed.astype(np.uint8),
                "marker_id": lambda: np.full(count, -1, np.int32),
                "marker_error_px": lambda: np.full(count, np.nan),
                "outline_error_mm": lambda: np.full(count, np.nan),
                "outline_error_deg": lambda: np.full(count, np.nan)}
    for key, make in defaults.items():
        if key not in pose:
            pose[key] = make()
    return pose


# ------------------------------------------------------------------ markers
def _markers(episode, camera, depth, progress):
    """Per frame, the largest ArUco marker's id, error, and the pose of the
    box it sits on (centre `depth`/2 behind it); id -1 where none is used."""
    count = episode.frames
    found = {"marker_id": np.full(count, -1, np.int32),
             "marker_error_px": np.full(count, np.nan),
             "marker_position": np.full((count, 3), np.nan),
             "marker_rotation": np.full((count, 3, 3), np.nan)}
    if not settings.get("vision.aruco"):
        return found
    name, size = settings.get("vision.aruco_dictionary"), settings.get("vision.aruco_mm") / 1000
    matrix, distortion = camera
    progress(0, count, "looking for ArUco markers")
    for index, rgb in enumerate(episode.read_frames(list(range(count)))):
        seen = aruco.detect(rgb, name)
        pose = aruco.marker_pose(seen[0][1], size, matrix, distortion) if seen else None
        if pose is None or pose[2] > MARKER_ERROR_SHARE * aruco.side_px(seen[0][1]):
            continue
        rotation, position, error = pose
        found["marker_id"][index], found["marker_error_px"][index] = seen[0][0], error
        found["marker_rotation"][index] = rotation
        found["marker_position"][index] = aruco.box_centre(rotation, position, depth)
    return found


# ------------------------------------------------------------------ hands
def _hands(hands, camera, width, height):
    matrix, distortion = camera
    count = len(hands["present"])
    out = {"palm_position": np.full((count, 2, 3), np.nan),
           "palm_rotation": np.full((count, 2, 3, 3), np.nan),
           "hand_fit_px": np.full((count, 2), np.nan)}
    for slot, side in enumerate(SIDES):
        present = np.flatnonzero(hands["present"][:, slot])
        if not present.size:
            continue
        sizes = np.array([_hand_size(hands["world"][i, slot]) for i in present])
        typical = float(np.median(sizes))
        for index, size in zip(present, sizes):
            shape = hands["world"][index, slot].astype(np.float64) * (typical / size)
            seen = (hands["image"][index, slot, :, :2] * [width, height]).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(shape, seen, matrix, distortion,
                                          flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            fitted, _ = cv2.projectPoints(shape, rvec, tvec, matrix, distortion)
            out["hand_fit_px"][index, slot] = np.linalg.norm(fitted[:, 0] - seen, axis=1).mean()
            rotation, _ = cv2.Rodrigues(rvec)
            points = shape @ rotation.T + tvec.ravel()           # camera frame, m
            out["palm_position"][index, slot] = points[PALM].mean(axis=0)
            out["palm_rotation"][index, slot] = _palm_axes(points, side)
    return out


def _hand_size(shape):
    """Mean distance from the wrist to the four knuckles, m."""
    return float(np.linalg.norm(shape[[5, 9, 13, 17]] - shape[WRIST], axis=1).mean())


def _palm_axes(points, side):
    """Columns: the palm's x (to the middle knuckle), y, z (out of the palm)."""
    x = points[MIDDLE_KNUCKLE] - points[WRIST]
    x /= np.linalg.norm(x)
    z = OUT_OF_PALM[side] * np.cross(points[INDEX_KNUCKLE] - points[WRIST],
                                     points[PINKY_KNUCKLE] - points[WRIST])
    z -= (z @ x) * x
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


# ------------------------------------------------------------------ object
def _nearest_equivalent(rotation, last, symmetries):
    """Of the poses that look the same as `rotation`, the nearest `last`."""
    return max((rotation @ turn for turn in symmetries),
               key=lambda candidate: np.trace(last.T @ candidate))


def edges(size):
    """The 12 edges of a box of `size` (m), as pairs of points in its frame."""
    corners = _box_corners(size)
    return [(corners[a], corners[b]) for a in range(8) for b in range(a + 1, 8)
            if bin(a ^ b).count("1") == 1]


def project(segments, rotation, position, calibration):
    """3D segments in a body's frame (posed by rotation, position in the
    camera's frame) as pixel segments, with the camera's lens."""
    points = np.array([point for segment in segments for point in segment], float)
    pixels, _ = cv2.projectPoints(points @ np.asarray(rotation).T + position, np.zeros(3),
                                  np.zeros(3), calibration.matrix(),
                                  np.array(calibration.distortion))
    pixels = pixels[:, 0]
    return [(tuple(pixels[k]), tuple(pixels[k + 1])) for k in range(0, len(pixels), 2)]


def _box_corners(size):
    half = np.asarray(size) / 2
    return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half


def _fit_box(mask, corners, camera, last, edge, symmetries=()):
    """(rotation, position, overlap) of the box whose outline best covers
    `mask`, starting from `last` (or from guesses when None)."""
    matrix, distortion = camera
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return None
    window = _window(xs, ys, mask.shape)
    target = _soft(_crop(mask.astype(np.float32), window))
    centre = np.array([xs.mean(), ys.mean()])
    area = float(xs.size)

    def cost(params):
        rotation = Rotation.from_rotvec(params[:3]).as_matrix()
        outline = _outline(corners, rotation, params[3:], camera, window)
        if outline is None:
            return 1.0
        return 1.0 - _overlap(outline, target)

    def search():
        distance = matrix[0, 0] * edge / np.sqrt(area)
        guesses = []
        for turn in Rotation.random(SEARCH, random_state=0):
            rotation = turn.as_matrix()
            position = _place(corners, rotation, centre, area, distance, camera)
            guesses.append(np.r_[turn.as_rotvec(), position])
        starts = sorted(guesses, key=cost)[:SEARCH_KEEP]
        return min((_refine(cost, start) for start in starts), key=lambda r: r.fun)

    if last is None:
        best = search()
    else:
        rotation, position, _ = last
        ray = _ray(centre, matrix, distortion)
        best = _refine(cost, np.r_[Rotation.from_matrix(rotation).as_rotvec(),
                                   ray * position[2] / ray[2]])
        if 1.0 - best.fun < RETRY_BELOW:
            # Only for a clearly better fit: the search can land on the tilt
            # mirrored through the image, which fits as well and would jump.
            fresh = search()
            if fresh.fun < best.fun - SWITCH_GAIN:
                best = fresh
    rotation = Rotation.from_rotvec(best.x[:3]).as_matrix()
    if last is not None and symmetries:
        rotation = _nearest_equivalent(rotation, last[0], symmetries)
    return rotation, best.x[3:].copy(), 1.0 - float(best.fun)


def _place(corners, rotation, centre, area, distance, camera):
    """A position for the box turned by `rotation` whose outline has the
    mask's centre and area (two rounds of moving and scaling)."""
    matrix, distortion = camera
    position = _ray(centre, matrix, distortion) * distance
    for _ in range(2):
        pixels, _ = cv2.projectPoints(corners @ rotation.T + position, np.zeros(3),
                                      np.zeros(3), matrix, distortion)
        hull = cv2.convexHull(pixels[:, 0].astype(np.float32))
        moments = cv2.moments(hull)
        if moments["m00"] <= 0:
            break
        outline_centre = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
        depth = position[2] * np.sqrt(moments["m00"] / area)
        here, _ = cv2.projectPoints(position[None], np.zeros(3), np.zeros(3), matrix, distortion)
        target = here[0, 0] + (centre - outline_centre)
        ray = _ray(target, matrix, distortion)
        position = ray * depth / ray[2]
    return position


def _refine(cost, start):
    """Nelder-Mead from `start`: rotation vector (rad), position (m)."""
    scale = np.r_[0.15, 0.15, 0.15, 0.01, 0.01, 0.02]
    simplex = np.vstack([start] + [start + np.eye(6)[k] * scale[k] for k in range(6)])
    return minimize(cost, start, method="Nelder-Mead",
                    options={"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-5,
                             "maxfev": 1500})


def _window(xs, ys, shape):
    """(x0, y0, scale): the crop around the mask the fit is done in."""
    extent = max(np.ptp(xs), np.ptp(ys))
    margin = 0.4 * extent + 10
    x0, y0 = xs.min() - margin, ys.min() - margin
    side = extent + 2 * margin
    return x0, y0, CROP / side


def _crop(image, window):
    x0, y0, scale = window
    move = np.array([[scale, 0, -scale * x0], [0, scale, -scale * y0]])
    return cv2.warpAffine(image, move, (CROP, CROP))


def _soft(image):
    return cv2.GaussianBlur(image, (0, 0), EDGE_SOFTNESS)


def _outline(corners, rotation, position, camera, window):
    """The box's outline, filled, in the crop - or None behind the camera."""
    matrix, distortion = camera
    points = corners @ rotation.T + position
    if (points[:, 2] <= 0.01).any():
        return None
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)
    x0, y0, scale = window
    crop = (pixels[:, 0] - (x0, y0)) * scale
    hull = cv2.convexHull(np.round(crop * 16).astype(np.int32))
    canvas = np.zeros((CROP, CROP), np.float32)
    cv2.fillConvexPoly(canvas, hull, 1.0, cv2.LINE_AA, 4)
    return _soft(canvas)


def _overlap(a, b):
    """Soft intersection over union."""
    return float(np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-9))


def _ray(pixel, matrix, distortion):
    """The direction (z = 1) through `pixel`, distortion undone."""
    x, y = cv2.undistortPoints(np.array([[pixel]], np.float64), matrix, distortion)[0, 0]
    return np.array([x, y, 1.0])


def _smooth(positions, rotations):
    """Median-filter positions (N, 3) and rotations (N, 3, 3) in place,
    within each run of frames where they are known."""
    known = np.isfinite(positions[:, 0]) & np.isfinite(rotations[:, 0, 0])
    edges = np.diff(np.r_[0, known.astype(np.int8), 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start < 3:
            continue
        positions[start:end] = median_filter(positions[start:end], size=(SMOOTH_FRAMES, 1),
                                             mode="nearest")
        quats = Rotation.from_matrix(rotations[start:end]).as_quat()
        for k in range(1, len(quats)):                     # one hemisphere
            if quats[k] @ quats[k - 1] < 0:
                quats[k] = -quats[k]
        quats = median_filter(quats, size=(SMOOTH_FRAMES, 1), mode="nearest")
        rotations[start:end] = Rotation.from_quat(
            quats / np.linalg.norm(quats, axis=1, keepdims=True)).as_matrix()


def _turn_between(first, rotation, symmetries):
    """Degrees between two box poses, the least over the box's symmetries."""
    return min(np.degrees(np.arccos(np.clip((np.trace(first.T @ rotation @ turn) - 1) / 2,
                                            -1, 1)))
               for turn in symmetries)


def _in_palm(out, track, slot, motion, box, symmetries=()):
    """The object in its hand's palm frame, the palm moved to the distance
    the object's known size implies (see `hand_scale`)."""
    if slot < 0:
        return
    palm_at = out["palm_position"][:, slot]
    palm_axes = out["palm_rotation"][:, slot]
    position = out["object_position"][track]
    held = motion["contact"][track, :, slot] & np.isfinite(palm_at[:, 2]) \
        & np.isfinite(position[:, 2])
    if held.any():
        # normal . (object - scale * palm) = half the object's thinnest side
        normal = palm_axes[held][:, :, 2]
        ratios = (((normal * position[held]).sum(1) - box.min() / 2)
                  / (normal * palm_at[held]).sum(1))
        out["hand_scale"][track] = float(np.median(ratios))
    scale = out["hand_scale"][track] if np.isfinite(out["hand_scale"][track]) else 1.0
    seen = np.isfinite(palm_at[:, 2]) & np.isfinite(position[:, 2])
    if seen.sum() >= 3:
        across = palm_at[seen, :2] / palm_at[seen, 2:3] * np.median(palm_at[seen, 2])
        spread = np.percentile(across, 90, axis=0) - np.percentile(across, 10, axis=0)
        if np.linalg.norm(spread) < STILL_M:                 # a still hand: one palm frame
            out["still_hand"][track] = True
            quats = Rotation.from_matrix(palm_axes[seen]).as_quat()
            quats[quats @ quats[0] < 0] *= -1
            middle = np.median(quats, axis=0)
            palm_axes = np.broadcast_to(Rotation.from_quat(middle / np.linalg.norm(middle))
                                        .as_matrix(), palm_axes.shape)
            palm_at = np.broadcast_to(np.median(palm_at[seen], axis=0), palm_at.shape)
    first = None
    for index in np.flatnonzero(seen):
        axes = palm_axes[index]
        out["in_palm_position"][track, index] = axes.T @ (position[index] - scale * palm_at[index])
        out["in_palm_rotation"][track, index] = axes.T @ out["object_rotation"][track, index]
        if first is None:
            first = out["in_palm_rotation"][track, index]
        out["in_palm_turn"][track, index] = _turn_between(
            first, out["in_palm_rotation"][track, index], symmetries or [np.eye(3)])
