"""What the robot is, read from its URDF and nothing else.

Joint names, order and limits all come from the URDF through MuJoCo's own
importer, so the simulator, the sliders and the hardware map can never
disagree about them. Change the URDF in settings.json and all three follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import settings
from .paths import resolve


@dataclass(frozen=True)
class Joint:
    name: str
    lower: float          # rad
    upper: float          # rad
    group: str            # "index", "thumb", ... - the name up to its first "_"

    @property
    def label(self):
        """The name inside its group: index_proximal_joint -> proximal."""
        short = self.name.removesuffix("_joint")
        return short[len(self.group) + 1:] if short != self.group else short


@dataclass(frozen=True)
class Robot:
    urdf: Path
    joints: tuple[Joint, ...]

    @property
    def names(self):
        return [j.name for j in self.joints]

    @property
    def lower(self):
        return np.array([j.lower for j in self.joints])

    @property
    def upper(self):
        return np.array([j.upper for j in self.joints])

    def __len__(self):
        return len(self.joints)

    def groups(self):
        """group -> joint indices, in URDF order."""
        out = {}
        for index, joint in enumerate(self.joints):
            out.setdefault(joint.group, []).append(index)
        return out

    def clamp(self, q):
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def spec(self):
        """A fresh MjSpec of the robot alone, for building a world from."""
        return mujoco.MjSpec.from_file(str(self.urdf))


def load(urdf=None):
    """The robot named in settings.json (or `urdf`)."""
    urdf = resolve(urdf or settings.get("robot.urdf"))
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf} (robot.urdf in settings.json)")
    model = mujoco.MjSpec.from_file(str(urdf)).compile()
    joints = []
    for index in range(model.njnt):
        if int(model.jnt_type[index]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                              int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        name = model.joint(index).name
        lower, upper = model.jnt_range[index] if model.jnt_limited[index] \
            else (-np.pi, np.pi)
        joints.append(Joint(name, float(lower), float(upper),
                            name.split("_")[0]))
    return Robot(urdf, tuple(joints))
