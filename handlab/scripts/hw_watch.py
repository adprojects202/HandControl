"""Live joint readout from the encoders, for checking motor signs and IDs.
Read-only: torque is not touched.

    python scripts/hw_watch.py

Turn torque off (or power the hand limp), bend one finger, and watch which
row moves and which way. A bar running opposite to the finger means that
joint's `sign` in settings.json is wrong; the wrong row moving means its `id`
is. Ctrl-C to stop.
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import robot, settings  # noqa: E402
from src.hardware import DynamixelLink  # noqa: E402


def main():
    config = settings.section("hardware")
    if config["port"] in ("", "mock"):
        print("set hardware.port in settings.json (scripts/hw_scan.py lists ports)")
        return 1
    names = robot.load().names
    link = DynamixelLink(names, config).connect()
    start = link.read_positions()
    try:
        while True:
            now = link.read_positions()
            lines = []
            for name, angle, moved in zip(names, now, now - start):
                degrees = math.degrees(moved)
                bar = "#" * int(min(abs(degrees), 60.0) / 3.0)
                left, right = (bar.rjust(20), " " * 20) if degrees < 0 \
                    else (" " * 20, bar.ljust(20))
                lines.append(f"{name:24s} {math.degrees(angle):7.1f} deg "
                             f"{left}|{right} {degrees:+7.1f}")
            print("\033[H\033[J" + "\n".join(lines)
                  + "\n\nbars: movement since start (left = negative)")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
