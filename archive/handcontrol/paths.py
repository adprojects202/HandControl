"""Where everything lives, in one place.

WHY THIS FILE EXISTS. The project root used to be recomputed in eight separate
modules, each with its own count of `.parent` - two, three, or `parents[3]`
depending how deep the file sat. That is fine until a file moves, and then it
is silently wrong: the path still resolves, just to somewhere that is not
there, and the failure surfaces as a missing calibration or a URDF that cannot
be read rather than as an import error.

Moving `model.py` one directory deeper is exactly that move, so the counting
stops here. Everything else asks.
"""

from __future__ import annotations

from pathlib import Path

# handcontrol/paths.py -> handcontrol -> the project
ROOT = Path(__file__).resolve().parents[1]

# The robot and its scene: what it IS, as opposed to the code that drives it.
# Vendored input, generated output and the three hand-edited calibration files
# all sit together because they describe one physical setup.
WORLD = ROOT / "world"
# h4nd_left, not h4nd_2. The first export was a RIGHT hand that had to be
# mirrored to get the hand on the bench, and it carried a spurious 15-degree
# roll on three of the thumb's joint origins. This one is modelled as the left
# hand directly - see `model.MIRRORED_FOR`, which inverts because of it.
URDF_PACKAGE = WORLD / "h4nd_left"
BUILD = WORLD / "build"          # generated; delete it and it comes back
CONFIG = WORLD / "config"        # hand, operator and panel calibration

# Recordings and everything fitted from them. Regenerable, and not in git.
DATA = ROOT / "data"
# The pooled sensor-noise fit, NAMED rather than spelled out per caller. There
# are two constants called DATA - this one and `sessions.DATA`, which is
# data/leap - and three call sites reached for the wrong one, so `--belief`
# and `--intent` died on a missing file the moment they were queued. Import
# this and there is nothing left to get wrong.
NOISE_FIT = DATA / "leap" / "noise.json"
# Trained models. Under data/ and NOT under world/build/, because build/ is
# documented as "delete it and it comes back" and a student does not - it is
# ten minutes of training, not a regenerated mesh.
MODELS = DATA / "models"
LOGS = ROOT / "logs"
