#!/usr/bin/env python3
"""The learned-retargeting pipeline: survey, fit, teach, synthesise, watch.

Everything about learning the mapping instead of solving it every frame lives
in `handcontrol/learning/`. This is the one command line over it -
subcommands rather than five scripts, the same shape `main.py` already has.

    python learn.py survey --operator akshay --hand left     record the sensor
    python learn.py noise                                    fit its error
    python learn.py teacher                                  solve for labels
    python learn.py synth                                    generate pairs
    python learn.py train                                    train the student
    python learn.py gestures                                 probe it on shapes
                                                             it never trained on
    .venv/bin/mjpython learn.py replay --student             watch it
    .venv/bin/mjpython learn.py gallery                      16 hands at once

`replay` and `gallery` open the MuJoCo viewer, so they want **mjpython** - the same
main-thread requirement, and the same fix, as `main.py`. Everything else runs
on plain python.

Read `handcontrol/learning/README.md` first; it says what each stage
is for and what has already been measured.
"""

from __future__ import annotations

import argparse
import sys

from handcontrol.learning.cli import add_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="learn.py",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[5:15]))
    add_arguments(parser.add_subparsers(dest="stage", required=True))
    args = parser.parse_args(argv)
    return args.handler(args, parser)


if __name__ == "__main__":
    sys.exit(main())
