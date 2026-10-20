"""Read-only bus check. Moves nothing, changes no register.

    python scripts/hw_scan.py

Lists the USB serial ports, then pings every motor on `hardware.port` from
settings.json and prints where each joint is.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import robot, settings  # noqa: E402
from src.hardware import DynamixelLink, list_ports  # noqa: E402


def main():
    print("serial ports:")
    for device, description in list_ports() or [("(none)", "")]:
        print(f"  {device:36s} {description}")

    config = settings.section("hardware")
    if config["port"] in ("", "mock"):
        print("\nset hardware.port in settings.json to one of the ports above")
        return 1

    names = robot.load().names
    link = DynamixelLink(names, config).connect()
    try:
        print(f"\n{config['port']} @ {config['baudrate']} baud")
        print(f"{'joint':26s} {'id':>3} {'model':>6} {'ticks':>6} {'deg':>8}")
        for name in names:
            motor = link.motors[name]
            if link.found[name] is None:
                print(f"{name:26s} {motor['id']:>3}  no reply")
                continue
            ticks = link.read(motor["id"], "present_position")
            print(f"{name:26s} {motor['id']:>3} {link.found[name]:>6} {ticks:>6} "
                  f"{math.degrees(link.to_radians(name, ticks)):>8.2f}")
        print(f"torque is {'ON' if link.torque_on else 'off'}")
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
