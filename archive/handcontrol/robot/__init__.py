"""The robot and the MuJoCo world it lives in.

`model.py` reads the URDF and is the authority on chains, limits, joint ROLES
and chirality. `mjcf.py` generates the simulation models from it. `sim.py` runs
them. `options.py` carries the settings both of those take, in three groups
rather than as twenty-two arguments. `layout.py` says where several hands
stand when more than one is on screen. `collide.py` decides which poses put
the hand inside itself. `glyphs.py` draws short words out of markers, for
labelling things in a scene.

Everything above this layer - the retargeting, the learning, the panel - asks
these files what the hand is rather than carrying its own copy of the answer.
The assets themselves live in `world/`: the vendored URDF, the generated MJCF
and meshes, and the calibration files. `handcontrol/paths.py` says where.
"""

from .model import HandModel, ensure_urdf
from .sim import HandSim

__all__ = ["HandModel", "HandSim", "ensure_urdf"]
