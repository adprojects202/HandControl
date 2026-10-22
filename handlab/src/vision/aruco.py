"""ArUco markers: printed squares whose pose a calibrated camera measures
directly - here a reference to check the markerless 3D fit against, and the
object's pose wherever one is seen.

    found = detect(rgb, "DICT_4X4_50")         # [(id, corners (4, 2) px)]
    pose = marker_pose(corners, 0.04, matrix, distortion)
    rotation, position, error_px = pose        # or None
    centre = box_centre(rotation, position, 0.055)

A marker's frame (OpenCV's): origin at its centre, x to the right and y up
as the marker reads, z out of the paper towards whoever sees it. Its corners
come top-left, top-right, bottom-right, bottom-left. A marker stuck in the
middle of one face of a box has the box's centre half the box's depth behind
it, along -z; and since every face of a cube looks alike, which face it is on
does not matter when poses are compared up to the cube's symmetries.

Print markers with scripts/aruco_markers.py (`vision.aruco_dictionary`,
`vision.aruco_mm`: the black square's side as printed).
"""

from __future__ import annotations

import cv2
import numpy as np

DICTIONARIES = {name: getattr(cv2.aruco, name) for name in
                ("DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL")}
_detectors = {}


def detector(name):
    """OpenCV's detector for dictionary `name`, corners refined to sub-pixel."""
    if name not in _detectors:
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _detectors[name] = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(DICTIONARIES[name]), parameters)
    return _detectors[name]


def detect(rgb, name):
    """The markers in `rgb`: [(id, corners (4, 2) px)], largest first."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector(name).detectMarkers(gray)
    if ids is None:
        return []
    found = [(int(i), c.reshape(4, 2).astype(np.float64)) for i, c in zip(ids.ravel(), corners)]
    return sorted(found, key=lambda m: -cv2.contourArea(m[1].astype(np.float32)))


def marker_pose(corners, size, matrix, distortion):
    """(rotation, position m, reprojection error px) of a marker of side
    `size` (m) seen at `corners`, or None."""
    half = size / 2
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    # A square has two poses that fit it nearly alike; IPPE gives both. Seen
    # square-on, the one solvePnP returned alone could be the wrong one (42 px
    # off on a test marker), so both are scored and the better kept.
    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(square, corners, matrix, distortion,
                                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
    best = None
    for rvec, tvec in zip(rvecs[:count], tvecs[:count]):
        seen, _ = cv2.projectPoints(square, rvec, tvec, matrix, distortion)
        error = float(np.linalg.norm(seen[:, 0] - corners, axis=1).mean())
        if best is None or error < best[2]:
            best = (cv2.Rodrigues(rvec)[0], tvec.ravel(), error)
    return best


def side_px(corners):
    """A marker's mean side in the image, px."""
    return float(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1).mean())


def box_centre(rotation, position, depth):
    """The centre of a box whose face carries the marker, `depth` (m) deep."""
    return position - rotation[:, 2] * depth / 2


def draw(image, found, mirror=False):
    """Each marker's outline and id, on `image` (in place)."""
    width = image.shape[1]
    for marker_id, corners in found:
        points = corners.copy()
        if mirror:
            points[:, 0] = width - 1 - points[:, 0]
        cv2.polylines(image, [points.astype(np.int32)], True, (255, 60, 200), 2, cv2.LINE_AA)
        cv2.circle(image, tuple(points[0].astype(int)), 4, (255, 60, 200), -1, cv2.LINE_AA)
        corner = tuple(points.min(axis=0).astype(int))
        cv2.putText(image, f"id {marker_id}", (corner[0], max(12, corner[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, f"id {marker_id}", (corner[0], max(12, corner[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 200), 1, cv2.LINE_AA)
    return image
