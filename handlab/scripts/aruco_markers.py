"""A sheet of ArUco markers to print, at the size settings.json says.

    python scripts/aruco_markers.py

Writes renders/aruco_markers.png: markers 0-5 of `vision.aruco_dictionary`,
each black square `vision.aruco_mm` across at 300 dpi, with a 50 mm bar to
check the print. Print at 100% (no "fit to page"), measure the bar, and stick
one marker in the middle of a face of the object. If the black square did
not come out at `vision.aruco_mm`, set it to what was printed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src import settings  # noqa: E402
from src.paths import ROOT  # noqa: E402
from src.vision.aruco import DICTIONARIES  # noqa: E402

DPI = 300
MARKERS = range(6)
COLUMNS = 3


def main():
    name, side_mm = settings.get("vision.aruco_dictionary"), settings.get("vision.aruco_mm")
    px_per_mm = DPI / 25.4
    side = round(side_mm * px_per_mm)
    margin = round(12 * px_per_mm)
    cell = side + 2 * margin
    rows = -(-len(MARKERS) // COLUMNS)
    sheet = np.full((rows * cell + round(25 * px_per_mm), COLUMNS * cell), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])
    for n, marker in enumerate(MARKERS):
        row, column = divmod(n, COLUMNS)
        x, y = column * cell + margin, row * cell + margin
        sheet[y:y + side, x:x + side] = cv2.aruco.generateImageMarker(dictionary, marker, side)
        cv2.putText(sheet, f"id {marker}  {name}  {side_mm:g} mm", (x, y + side + round(6 * px_per_mm)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    bar_y = rows * cell + round(8 * px_per_mm)
    bar = round(50 * px_per_mm)
    sheet[bar_y:bar_y + round(2 * px_per_mm), margin:margin + bar] = 0
    cv2.putText(sheet, "50 mm - print at 100% and check this bar", (margin, bar_y + round(9 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    out = ROOT / "renders" / "aruco_markers.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(sheet).save(out, dpi=(DPI, DPI))
    print(f"wrote {out} - {len(MARKERS)} markers of {name}, {side_mm:g} mm, at {DPI} dpi")


if __name__ == "__main__":
    main()
