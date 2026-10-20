"""A bank of poses the robot is allowed to be in, and the nearest one to yours.

WHY THIS EXISTS. The solver matches vectors; nothing in it holds the hand in a
configuration a hand would actually adopt. Four joints reaching a three
coordinate target leaves one degree of freedom per finger unconstrained, and
the solver spends it on whatever the objective cannot see - measured on a real
recording, it spent it on a hook, with the operator's DIP at 10 degrees and the
robot's at 106.

`dex_retargeting` regularises only toward the PREVIOUS frame, which keeps
motion continuous and does nothing at all about whether the pose is sane. This
was written to supply the missing term: the nearest pose from a bank of vetted
ones.

WHAT IT ACTUALLY IS TODAY: a MODE, not a regulariser. `Mimic._solve_bank`
commands the retrieved pose directly, because the keyvector bridge takes
keypoints and a config and has no channel for a preference. The consequence is
the ceiling `Mimic` documents at its call site - the hand can never do better
than what is in the bank, and it steps between entries rather than moving
continuously, with the rate limiter smoothing the steps. Handing it to the
solver as a prior instead is the design this was written for and is not
built; it would need a new message on the bridge.

THE DESCRIPTOR is what makes human and robot comparable: four fingertips in the
wrist frame plus the six distances between them, divided by the hand's own
reach. Scale-free, so this robot's 1.27x thumb stops mattering, and computed by
the identical formula on both sides - which is the only reason a nearest
neighbour means anything across two different hands.

That descriptor was checked against 1178 distinct frames of real tracking:
samples that agree on it have individual bone directions within 0.4 degrees on
average. It discriminates human posture almost perfectly, which is what makes
retrieval sound here. The ambiguity this fixes is on the ROBOT's side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..paths import BUILD

from ..robot.model import FINGERS, TIP_LINKS, roles_in_order


# build/, not config/: this is generated from the URDF and is ten megabytes of
# it. config/ holds the small hand-edited files that describe one setup, and a
# regenerable blob has no business sitting among them. Delete it and it comes
# back, like everything else in build/.
CACHE = BUILD / "pose_bank.npz"

# Human finger joints do not move independently: the DIP follows the PIP at
# roughly two thirds, and the PIP follows the knuckle. Sampling those ratios
# rather than the joints themselves is what makes the bank a bank of plausible
# hands instead of a uniform grid over sixteen numbers, most of which is poses
# no hand has ever made.
PIP_OF_KNUCKLE = 1.15
DIP_OF_PIP = 0.66
COUPLING_SLACK = 0.16        # how far a real hand departs from those ratios
BONE_WEIGHT = 0.35           # bone directions against tip placement, in the key
DEFAULT_SIZE = 20000


def descriptor(tips, wrist=None, bones=None):
    """What a hand looks like, in a form both hands can be measured in.

    Four fingertips in the wrist frame and their six mutual distances, divided
    by the hand's own reach so scale drops out - plus, if given, the direction
    of every bone.

    THE BONES ARE NOT OPTIONAL IN PRACTICE. Fingertips alone describe human
    posture almost perfectly - 1178 real frames that agree on the tip part
    have bone directions within 0.4 degrees - because human kinematics forbid
    two different postures reaching the same point. The ROBOT has no such
    scruples: four joints for a three-coordinate target, so a bank indexed on
    tips alone contains hyperextended and neutral knuckles at the same
    address, and retrieval averages them. Measured, that put the knuckle 15 to
    27 degrees out while every other joint landed within four.
    """
    tips = np.asarray(tips, dtype=float)
    single = tips.ndim == 2
    if single:
        tips = tips[None]
    if wrist is not None:
        tips = tips - np.asarray(wrist, dtype=float)[..., None, :]

    reach = np.linalg.norm(tips, axis=2).mean(axis=1, keepdims=True)
    reach = np.maximum(reach, 1e-9)
    scaled = tips / reach[..., None]

    gaps = [np.linalg.norm(scaled[:, i] - scaled[:, j], axis=1)
            for i in range(len(FINGERS)) for j in range(i + 1, len(FINGERS))]
    parts = [scaled.reshape(len(scaled), -1), np.stack(gaps, axis=1)]
    if bones is not None:
        bones = np.asarray(bones, dtype=float)
        if bones.ndim == 2:
            bones = bones[None]
        unit = bones / np.maximum(np.linalg.norm(bones, axis=2, keepdims=True),
                                  1e-9)
        # weighted so posture and placement both matter rather than the 36
        # direction numbers drowning the 18 placement ones
        parts.append(unit.reshape(len(unit), -1) * BONE_WEIGHT)
    out = np.concatenate(parts, axis=1)
    return out[0] if single else out


class PoseBank:
    """Plausible robot poses, indexed by what the hand looks like in them."""

    def __init__(self, model, joint_names=None):
        self.model = model
        self.joint_names = list(joint_names or model.joint_names())
        self.roles = {f: dict(roles_in_order(model, f)) for f in FINGERS}
        self.index = {n: i for i, n in enumerate(self.joint_names)}
        self.lower, self.upper = model.limits(self.joint_names)
        # Three bones per digit, matching the solver's set. The obvious fourth
        # - palm to knuckle - is a mounting bracket on this robot, not a
        # metacarpal, and it points 27 degrees away from the operator's. It was
        # in this key, and it is why retrieval used to miss the knuckle by 20
        # degrees while landing every other joint within four.
        self._bone_pairs = [(a, b) for f in FINGERS for a, b in
                            ((f"{f}_proximal_link", f"{f}_middle_link"),
                             (f"{f}_middle_link", f"{f}_distal_link"),
                             (f"{f}_distal_link", f"{f}_tip_link"))]
        self._bone_links = sorted({l for pair in self._bone_pairs for l in pair}
                                  | set(TIP_LINKS.values()))
        self.poses = None            # (N, 16) joint vectors
        self.keys = None             # (N, 54) descriptors
        self.tree = None

    # ------------------------------------------------------------- building
    def _sample(self, count, rng):
        """Joint vectors drawn from a hand's own degrees of freedom.

        Each joint is sampled across its WHOLE declared range, biased toward
        the open end, with the flexion joints pulled toward the ratios a real
        finger keeps - PIP about 1.15 times the knuckle, DIP about two thirds
        of the PIP - and allowed to depart from them.

        Both halves of that matter, and each was learned by getting it wrong.
        Sampling the joints independently fills the bank with reachable
        nonsense. Coupling them rigidly makes it unable to represent a hand
        that hyperextends its knuckle while flexing the PIP, which is what
        this operator's hand does at rest. And parameterising the knuckle as a
        fraction of its flexed range silently capped hyperextension at -5
        degrees when the hand being matched sits at -15, so the nearest
        neighbour was never near.
        """
        q = np.zeros((count, len(self.joint_names)))
        for finger in FINGERS:
            role = self.roles[finger]
            top = {r: self.upper[self.index[role[r]]] for r in role}
            bottom = {r: self.lower[self.index[role[r]]] for r in role}

            openness = rng.random(count) ** 1.3      # more open hands than fists
            knuckle = bottom["knuckle"] + openness * (top["knuckle"]
                                                      - bottom["knuckle"])
            slack = COUPLING_SLACK * (top["middle"] - bottom["middle"])
            pip = np.clip(knuckle * PIP_OF_KNUCKLE + rng.normal(0, slack, count),
                          bottom["middle"], top["middle"])
            tip = np.clip(pip * DIP_OF_PIP + rng.normal(0, slack * 0.6, count),
                          bottom["tip"], top["tip"])
            span = top["spread"] - bottom["spread"]
            spread = ((rng.random(count) - 0.5) * 0.5 * span
                      * (1.0 - 0.7 * openness))

            q[:, self.index[role["knuckle"]]] = knuckle
            q[:, self.index[role["middle"]]] = pip
            q[:, self.index[role["tip"]]] = tip
            q[:, self.index[role["spread"]]] = spread
        return np.clip(q, self.lower, self.upper)

    def build(self, count=DEFAULT_SIZE, seed=0):
        """Fill the bank with sampled poses and index them by fingertip.

        The poses are NOT screened for self-collision here. They were, against
        an optional `SelfCollision` nobody ever passed, so the screen never
        ran; and the bank feeds `Mimic`'s tip-IK seed, which goes through the
        live guard on its way to the hand anyway.
        """
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(seed)
        poses = self._sample(count, rng)
        tips, bones = [], []
        for q in poses:
            place = {l: self.model.link_pose(l, q, self.joint_names)[0]
                     for l in self._bone_links}
            tips.append([place[TIP_LINKS[f]] for f in FINGERS])
            bones.append([place[b] - place[a] for a, b in self._bone_pairs])
        self.poses = poses
        self.keys = descriptor(np.array(tips), bones=np.array(bones))
        self.tree = cKDTree(self.keys)
        return len(poses)

    # ------------------------------------------------------------ retrieval
    def key_for(self, tips, bones):
        """The descriptor for a tracked hand, by the same formula as the bank.

        `tips` and `bones` must already be in the robot's axes - that is what
        `Retargeter.rotation` produces - and `bones` must be the same three
        per digit, in the same order, as `_bone_pairs`. Nothing checks that,
        and getting it wrong retrieves confidently wrong poses.
        """
        return descriptor(np.asarray(tips), bones=np.asarray(bones))

    def ready(self, path=CACHE, count=DEFAULT_SIZE):
        """Load the cached bank, or build and cache one. Seconds, once."""
        if self.tree is not None:
            return True
        if self.load(path):
            return True
        self.build(count)
        try:
            self.save(path)
        except OSError:
            pass          # a bank that cannot be cached still works this run
        return True

    def nearest(self, key, k=4):
        """The blended joint vector of the k nearest poses to `key`.

        Blended rather than snapped: the single nearest pose changes
        discontinuously as the operator moves, and a prior that jumps is a
        prior that makes the hand jump. Weighted by inverse distance, so a
        close match dominates and the blend degrades gracefully into it.
        """
        if self.tree is None:
            return None
        distance, where = self.tree.query(np.asarray(key, dtype=float),
                                          k=min(k, len(self.poses)))
        distance = np.atleast_1d(distance)
        where = np.atleast_1d(where)
        weight = 1.0 / np.maximum(distance, 1e-6)
        weight /= weight.sum()
        return (self.poses[where] * weight[:, None]).sum(axis=0)

    # ---------------------------------------------------------------- files
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, poses=self.poses, keys=self.keys,
                            joints=np.array(self.joint_names))
        return path

    def load(self, path):
        from scipy.spatial import cKDTree

        path = Path(path)
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            if list(data["joints"]) != self.joint_names:
                return False          # a different robot; rebuild rather than lie
            self.poses, self.keys = data["poses"], data["keys"]
        except (ValueError, OSError, KeyError):
            return False
        self.tree = cKDTree(self.keys)
        return True
