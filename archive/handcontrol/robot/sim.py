"""The hand in MuJoCo: physics, fingertip IK, and a viewer.

Replaces the Genesis simulator this project started on. The reasons were
measured, on this hand, not chosen on taste:

    step cost            Genesis 1.2 ms      MuJoCo ~0.02 ms
    rise to 90%          0.26 s              0.10 s
    IK time per frame    9.2 ms (worst 1005) 1.4 ms (worst 2.1)
    IK joint jitter      6.85 deg/frame      0.78 deg/frame
                         (worst 54 deg)      (worst 1.5 deg)
    IK and collisions    none: 11.2 mm       avoids: 0.17 mm
    self-collision       unusable            works

The jitter and the 1005 ms spike share one cause: Genesis restarts its IK from
random seeds when it struggles, so it lands on a DIFFERENT valid branch between
frames. mink is differential - it starts from where the hand already is and
takes the shortest joint-space step - so smoothness is structural here rather
than tuned.

MACOS: `mujoco.viewer.launch_passive` must run under `mjpython`, which ships in
the venv beside `python`. Headless needs neither.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import mujoco

from .mjcf import GHOST, collision_geoms, ensure_mjcf, ensure_sim_mjcf
from .options import Guard, Physics, Scene
from .model import (DEFAULT_HAND, EFFORT, FINGERS, IK, KP, KV,
                    HandModel, ensure_urdf)

# How close mink is allowed to bring two geoms before it steers away, and how
# far out it starts caring. Metres.
IK_CLEARANCE = 0.002
IK_DETECT = 0.010


class HandSim:
    """One h4nd_left, bolted in mid-air, driven by joint position targets."""

    def __init__(self, hand=DEFAULT_HAND, spread_limit=None,
                 physics=Physics(), guard=Guard(), scene=Scene()):
        """The hand, and three groups of settings - see `robot/options.py`.

        A caller that wants the model and nothing unusual passes only `hand`.
        Vary one knob with `replace(Physics(), kp=40)` rather than by adding
        an argument here: this signature was twenty-two parameters long and
        every one of them had to be spelled again in `ensure_sim_mjcf`.
        """
        self.hand = hand
        self.physics, self.guard, self.scene = physics, guard, scene
        self.urdf = str(ensure_urdf(hand, spread_limit))
        self.model = HandModel.load(self.urdf)
        self.joint_names = self.model.joint_names()
        self.chains = self.model.finger_chains()
        self.lower, self.upper = self.model.limits(self.joint_names)
        self.position = tuple(scene.position)
        self.show_viewer = scene.show_viewer

        # The gains are read out of the group once, here, because the panel
        # writes them live through `set_gains` and the group is frozen.
        self.kp = KP if physics.kp is None else physics.kp
        self.kv = KV if physics.kv is None else physics.kv
        self.effort = EFFORT if physics.effort is None else physics.effort
        self.ik = dict(IK)

        # Nothing reaches the joints except through `command`, so that is where
        # self-collision is refused. MuJoCo does resolve contacts now, unlike
        # Genesis, but a contact only pushes back in the SIMULATOR - the real
        # hand has no physics to save it, and it takes the same targets.
        #
        # `guard.ik_avoid` is a different job from `guard.avoid`: it decides
        # whether the SOLVER steers around the hand's own fingers, where
        # `avoid` is the clamp on whatever the solver produces. Turning it off
        # does not let the hand pass through itself - it makes the IK drive
        # straight at a neighbour and stop dead at the clamp instead of
        # finding a way round, which is how to see what it is buying.
        self.ik_avoid = guard.ik_avoid
        self._collision_margin = guard.margin
        self.collision = None
        self._requested = None
        self._safe = None
        self.blocked = False

        self.copy_qadr = {}
        self._spread_limit = spread_limit
        default_pos, default_lookat = self._default_view()
        self.camera_pos = (default_pos if scene.camera_pos is None
                           else tuple(scene.camera_pos))
        self.camera_lookat = (default_lookat if scene.camera_lookat is None
                              else tuple(scene.camera_lookat))
        self.viewer = None
        self._built = False

    # The settings moved into `Physics` when this constructor was cut down from
    # twenty-two arguments, but `dt` and its neighbours were being read from
    # OUTSIDE - the mimic loop paces itself against `sim.dt`. They stay
    # readable here, delegating rather than copied, so there is still one
    # place each value lives.
    @property
    def dt(self):
        return self.physics.dt

    @property
    def gravity(self):
        return self.physics.gravity

    @property
    def self_collision(self):
        return self.physics.self_collision

    @property
    def armature(self):
        return self.physics.armature

    def _default_view(self):
        """(camera position, look-at point) framing both hands, in world metres.

        Derived from where the hands actually are rather than written down as
        one pair of constants. The encoder hand sits `ghost_offset` to one
        side, and WHICH side follows the chirality - so a fixed camera that
        frames the pair for a left hand is standing on top of the ghost for a
        right one.

        In front of the palms (the hand's +X face), a little above, and offset
        to the side the encoder hand is NOT on, so it cannot come between the
        camera and the simulated hand.
        """
        x, y, z = self.position
        side = 1.0 if self.hand == "right" else -1.0
        shift = side * self.scene.ghost_offset if self.scene.ghost else 0.0
        lookat = (x, y + shift / 2.0, z + 0.02)
        return (x + 0.50, lookat[1] - side * 0.18, z + 0.24), lookat

    # --------------------------------------------------------------- build
    def build(self):
        # The generator gets the same objects, not the values unpacked out of
        # them - see `robot/options.py`. The gains are the exception: the panel
        # can have moved them since the group was made.
        path = ensure_sim_mjcf(self.hand, self._spread_limit,
                               replace(self.physics, kp=self.kp, kv=self.kv,
                                       effort=self.effort),
                               self.scene)
        self.mj = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.mj)
        self.mj.opt.timestep = self.physics.dt
        if not self.physics.gravity:
            self.mj.opt.gravity[:] = 0.0
        if not self.physics.self_collision:
            self.mj.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        def jid(name):
            return mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_JOINT, name)

        self.qadr = np.array([self.mj.jnt_qposadr[jid(n)]
                              for n in self.joint_names])
        self.vadr = np.array([self.mj.jnt_dofadr[jid(n)]
                              for n in self.joint_names])
        self.actuators = np.array([
            mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.joint_names])
        self.sites = {f: mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_SITE,
                                           f"{f}_tip")
                      for f in FINGERS}
        self.ghost_qadr = None
        if self.scene.ghost:
            ghost = [jid(GHOST + n) for n in self.joint_names]
            if all(i >= 0 for i in ghost):
                self.ghost_qadr = np.array([self.mj.jnt_qposadr[i]
                                            for i in ghost])
                self.ghost_vadr = np.array([self.mj.jnt_dofadr[i]
                                            for i in ghost])
        self._ghost_q = np.zeros(len(self.joint_names))
        # What each posed copy is holding. Zero is the open hand, so a lane
        # nobody has posed yet sits flat rather than sagging.
        self._copy_q = {}

        # Address every extra copy the same way the ghost is addressed. Stored
        # by prefix rather than positionally so a caller names what it is
        # posing instead of counting.
        for prefix, _, _ in self.scene.copies:
            ids = [jid(prefix + n) for n in self.joint_names]
            if all(i >= 0 for i in ids):
                self.copy_qadr[prefix] = (
                    np.array([self.mj.jnt_qposadr[i] for i in ids]),
                    np.array([self.mj.jnt_dofadr[i] for i in ids]))
                self._copy_q[prefix] = np.zeros(len(self.joint_names))

        self._build_ik()
        self.targets = np.zeros(len(self.joint_names))
        self.reset()
        if self.show_viewer:
            self._open_viewer()
        self._built = True
        return self

    def _build_ik(self):
        """mink, on the kinematic model rather than this one.

        The simulation model carries a second, ghost copy of every joint. Those
        DOFs are not the robot's and the solver must not be able to spend them,
        so the IK gets its own model with the hand in it once.
        """
        import mink

        self.ik_mj = mujoco.MjModel.from_xml_path(
            str(ensure_mjcf(self.hand, self._spread_limit)))
        self.ik_cfg = mink.Configuration(self.ik_mj)
        order = [mujoco.mj_id2name(self.ik_mj, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(self.ik_mj.njnt)]
        self.ik_order = [order.index(n) for n in self.joint_names]

        self.ik_tasks = {
            f: mink.FrameTask(f"{f}_tip", "site", position_cost=1.0,
                              orientation_cost=0.0, lm_damping=1e-3)
            for f in FINGERS}
        # Holds the redundant fourth joint of each finger near a sane pose.
        # Without it a 4-DOF finger chasing a 3-DOF target wanders.
        self.ik_posture = mink.PostureTask(self.ik_mj, cost=1e-3)
        self.ik_posture.set_target(np.zeros(self.ik_mj.nq))
        self.ik_damping = mink.DampingTask(self.ik_mj,
                                           cost=self.ik["damping"])

        groups = collision_geoms(self.ik_mj)
        keys = [k for k in groups if groups[k]]
        pairs = [(groups[a], groups[b])
                 for i, a in enumerate(keys) for b in keys[i + 1:]]
        self._ik_joint_limit = mink.ConfigurationLimit(self.ik_mj)
        self._ik_collision_limit = mink.CollisionAvoidanceLimit(
            self.ik_mj, geom_pairs=pairs,
            minimum_distance_from_collisions=IK_CLEARANCE,
            collision_detection_distance=IK_DETECT)
        self._apply_ik_limits()
        self._mink = mink
        self.ik_dofs = {
            f: [self.ik_order[self.joint_names.index(n)] for n in chain]
            for f, chain in self.chains.items()}

    def _apply_ik_limits(self):
        """Joint limits always; the collision term only when asked for."""
        self.ik_limits = [self._ik_joint_limit]
        if self.ik_avoid:
            self.ik_limits.append(self._ik_collision_limit)

    def set_ik_avoid(self, on):
        self.ik_avoid = bool(on)
        if getattr(self, "_ik_collision_limit", None) is not None:
            self._apply_ik_limits()
        return self.ik_avoid

    def _open_viewer(self):
        import mujoco.viewer

        try:
            self.viewer = mujoco.viewer.launch_passive(
                self.mj, self.data, show_left_ui=False, show_right_ui=False)
        except RuntimeError as failure:
            # The usual cause is `python` instead of `mjpython`, and it is
            # worth saying so rather than letting the stack trace speak.
            raise SystemExit(
                f"cannot open the MuJoCo viewer: {failure}\n"
                f"  on macOS the viewer must run under mjpython:\n"
                f"    .venv/bin/mjpython main.py ...\n"
                f"  or pass --headless") from failure
        with self.viewer.lock():
            camera = self.viewer.cam
            camera.lookat[:] = self.camera_lookat
            offset = np.asarray(self.camera_pos) - np.asarray(self.camera_lookat)
            distance = max(float(np.linalg.norm(offset)), 1e-9)
            camera.distance = distance
            camera.azimuth = float(np.degrees(np.arctan2(-offset[1],
                                                         -offset[0])))
            camera.elevation = -float(np.degrees(np.arcsin(
                np.clip(offset[2] / distance, -1.0, 1.0))))

    # ---------------------------------------------------------- commanding
    def clamp(self, q):
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def reset(self, q=None):
        q = np.zeros(len(self.joint_names)) if q is None else self.clamp(q)
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.vadr] = 0.0
        self.data.ctrl[self.actuators] = q
        mujoco.mj_forward(self.mj, self.data)
        self.targets = q.copy()
        self._requested = None
        self._safe = q.copy()
        self.blocked = False
        return q

    def command(self, q):
        q = self.clamp(q)
        if self.guard.avoid:
            # Only re-solve when the target actually changes: the loop calls
            # this every step with the same vector. Measured: a clear check
            # is under 2 ms, a blocked one bisects through the meshes at
            # about fifteen - and a student trained on collision-free labels
            # is blocked on none of its frames.
            if self._requested is None or not np.array_equal(q, self._requested):
                self._requested = q.copy()
                self._safe, self.blocked = self.safe(q)
            q = self._safe
        self.targets = q
        self.data.ctrl[self.actuators] = q

    def set_copy(self, prefix, q):
        """Pose one extra copy by writing qpos directly.

        Written, not simulated: these copies exist to be looked at, and running
        a controller on them would make them lag the thing they are meant to
        show."""
        if prefix not in self.copy_qadr:
            return
        # Stored as well as written, because `step` re-applies it every tick -
        # otherwise gravity takes the copy back between poses.
        self._copy_q[prefix] = self.clamp(q)
        qadr, vadr = self.copy_qadr[prefix]
        self.data.qpos[qadr] = self._copy_q[prefix]
        self.data.qvel[vadr] = 0.0

    def set_ghost(self, q):
        """Pose the encoder hand. Angles are the real hand's, in URDF terms."""
        self._ghost_q = self.clamp(q)

    def step(self):
        if self.ghost_qadr is not None:
            # Written, never simulated. It has mass and would otherwise sag.
            self.data.qpos[self.ghost_qadr] = self._ghost_q
            self.data.qvel[self.ghost_vadr] = 0.0
        for prefix, (qadr, vadr) in self.copy_qadr.items():
            # Every posed copy, held the same way and for the same reason. The
            # ghost had this and the copies did not, and the difference is not
            # subtle: a copy with joints and no actuators sags 25 degrees in
            # two seconds. Before any mapping had an answer to pose them with,
            # the comparison lanes simply fell over.
            self.data.qpos[qadr] = self._copy_q[prefix]
            self.data.qvel[vadr] = 0.0
        mujoco.mj_step(self.mj, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    # --------------------------------------------------------------- state
    def joint_positions(self):
        return np.array(self.data.qpos[self.qadr])

    def tip_position(self, finger):
        """The fingertip itself, not the distal link origin.

        The distal link's frame sits on its own joint axis and so does not move
        when that joint rotates; `<finger>_tip` is a site at the end of the
        distal mesh, which does.
        """
        return np.array(self.data.site_xpos[self.sites[finger]])

    def tip_positions(self):
        return {f: self.tip_position(f) for f in FINGERS}

    # ------------------------------------------------------------------ IK
    def solve_tip(self, finger, target, init=None, iterations=None, dt=0.05,
                  **_):
        """Joint angles putting `finger`'s tip at `target` (world metres).

        Differential, and warm-started from `init` - that is the whole reason
        it does not jitter. Only that finger's DOFs are freed; the rest are
        frozen, so solving one finger cannot rearrange another.

        Returns (16-vector, position error in metres). Out of reach returns the
        closest approach and an honest residual rather than failing.
        """
        mink = self._mink
        start = self.targets if init is None else self.clamp(init)
        q = np.zeros(self.ik_mj.nq)
        q[self.ik_order] = start
        self.ik_cfg.update(q)

        # World metres in, model metres to the solver: the MJCF the IK runs on
        # has the palm at the origin, while everything outside is mounted.
        local = np.asarray(target, dtype=float) - np.asarray(self.position)
        self.ik_tasks[finger].set_target(mink.SE3.from_translation(local))

        frozen = [i for i in range(self.ik_mj.nv)
                  if i not in self.ik_dofs[finger]]
        tasks = [self.ik_tasks[finger], self.ik_posture, self.ik_damping,
                 mink.DofFreezingTask(self.ik_mj, frozen, gain=1.0)]

        # Iterate to the tolerance, not to a fixed count. Warm started - which
        # every frame of a moving target is - it exits after one or two steps;
        # a cold solve from far away runs to the cap. A fixed six iterations
        # tracks beautifully and solves a standing target to 38 mm.
        cap = int(self.ik["max_solver_iters"] if iterations is None
                  else iterations)
        tolerance = float(self.ik["pos_tol"])
        for _ in range(max(cap, 1)):
            reached = np.array(self.ik_cfg.get_transform_frame_to_world(
                f"{finger}_tip", "site").translation())
            if float(np.linalg.norm(reached - local)) <= tolerance:
                break
            # 1e-12 is the QP's numerical regulariser and wants to stay
            # negligible: at 1e-2 it dominates the objective and the solve goes
            # from 4 iterations and 0.39 mm to 150 iterations and 1.45 mm. The
            # tunable `damping` is the DampingTask's cost instead, which is the
            # knob that actually trades convergence speed for steadiness.
            velocity = mink.solve_ik(self.ik_cfg, tasks, dt, "daqp",
                                     damping=1e-12, limits=self.ik_limits)
            self.ik_cfg.integrate_inplace(velocity, dt)

        reached = np.array(self.ik_cfg.get_transform_frame_to_world(
            f"{finger}_tip", "site").translation())
        solved = np.array(self.ik_cfg.q)[self.ik_order]
        return self.clamp(solved), float(np.linalg.norm(reached - local))

    def reachable_tip(self, finger, target, **kwargs):
        """solve_tip, but keeping every other finger where it already is."""
        q = self.targets.copy()
        solved, error = self.solve_tip(finger, target, init=q, **kwargs)
        for name in self.chains[finger]:
            i = self.joint_names.index(name)
            q[i] = solved[i]
        return q, error

    # -------------------------------------------------------------- tuning
    def set_gains(self, kp=None, kv=None, effort=None):
        """Retune the actuators on a running model."""
        if kp is not None:
            self.kp = float(kp)
        if kv is not None:
            self.kv = float(kv)
        if effort is not None:
            self.effort = float(effort)
        for index in self.actuators:
            # position actuator: gainprm[0] is kp, biasprm[1] is -kp,
            # biasprm[2] is -kv. Setting all three keeps it a PD servo.
            self.mj.actuator_gainprm[index, 0] = self.kp
            self.mj.actuator_biasprm[index, 1] = -self.kp
            self.mj.actuator_biasprm[index, 2] = -self.kv
            self.mj.actuator_forcerange[index] = (-self.effort, self.effort)

    def set_ik(self, **values):
        for key, value in values.items():
            if key in self.ik:
                self.ik[key] = (int(value) if key == "max_solver_iters"
                                else float(value))
        if "damping" in values and getattr(self, "_mink", None) is not None:
            self.ik_damping = self._mink.DampingTask(
                self.ik_mj, cost=self.ik["damping"])

    def set_collision_margin(self, margin):
        self._collision_margin = float(margin)
        collider = self.collider()
        if collider is not None:
            collider.margin = float(margin)
        self._requested = None

    def tuning(self):
        """Every knob the panel draws, with a number for each.

        `collision_margin` falls back to the default rather than to None: with
        `--allow-collisions` no collider is ever built, so the stored margin
        stays None, and the panel's slider does `float(value)` on whatever
        arrives.
        """
        from .collide import MARGIN

        margin = (self.collision.margin if self.collision is not None
                  else self._collision_margin)
        margin = MARGIN if margin is None else margin
        return {"kp": self.kp, "kv": self.kv, "effort": self.effort,
                "collision_margin": margin,
                "ik_avoid": bool(self.ik_avoid),
                "ik_damping": self.ik["damping"],
                "ik_iters": self.ik["max_solver_iters"],
                "ik_tol": self.ik["pos_tol"]}

    # ---------------------------------------------------------- collisions
    def collider(self):
        if self.collision is None and self.guard.avoid:
            from .collide import MARGIN, SelfCollision
            self.collision = SelfCollision(
                self.model, self.urdf,
                MARGIN if self._collision_margin is None
                else self._collision_margin)
        return self.collision

    def safe(self, q, start=None):
        collider = self.collider()
        if collider is None:
            return self.clamp(q), False
        return collider.clamp(self.clamp(start if start is not None
                                         else self.targets), self.clamp(q))

    # -------------------------------------------------------------- viewer
    def draw_debug(self, targets=None, skeleton=None, radius=0.005,
                   bone=0.0022, joint=0.0045, extra=None,
                   skeleton_rgba=(0.30, 0.55, 1.0, 0.95), colors=None,
                   draw_joints=True, scene=None, capsules=None):
        """Tip targets and the operator's hand, overlaid on the scene.

        `capsules` is a list of (start, end, radius, rgba) in world metres:
        the objects a replayed hand is holding, drawn in every robot lane.

        ONE call for both: the overlay buffer is rebuilt from scratch each
        time, so two independent draw methods would each wipe the other's work
        and only the last would ever be seen.

        `skeleton` maps any key to a chain of points; `colors` may override the
        colour per key, which is how a gallery turns one hand red while the
        rest stay blue. `scene` draws into an offscreen renderer's scene
        instead of the viewer's - on top of the model, without resetting it -
        so a headless render shows the same overlay a viewer would.
        """
        own = scene is None
        if own:
            if self.viewer is None:
                return
            scene = self.viewer.user_scn
            scene.ngeom = 0
        for target in (targets or {}).values():
            self._marker(scene, np.asarray(target, dtype=float), radius,
                         (1.0, 0.35, 0.2, 0.8))
        for key, points in (skeleton or {}).items():
            colour = (colors or {}).get(key, skeleton_rgba)
            points = [np.asarray(p, dtype=float) for p in points]
            if draw_joints:
                for point in points:
                    self._marker(scene, point, joint, colour)
            for a, b in zip(points[:-1], points[1:]):
                self._bone(scene, a, b, bone, colour)
        for start, end, radius_, colour in (capsules or ()):
            self._bone(scene, np.asarray(start, dtype=float),
                       np.asarray(end, dtype=float), radius_, colour)
        for text, centre, colour, cell in (extra or ()):
            # Words on the floor, drawn as boxes. MuJoCo's viewer will not
            # render a geom's `label` - measured, it changes zero pixels - so a
            # name has to be geometry like everything else.
            from . import glyphs
            glyphs.draw(scene, text, centre, colour, cell=cell,
                        marker=self._marker)

    def _marker(self, scene, position, radius, color):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0.0, 0.0]), position,
                            np.eye(3).flatten(), np.array(color, dtype=np.float32))
        scene.ngeom += 1

    def _bone(self, scene, start, end, radius, color):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.array(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             start, end)
        scene.ngeom += 1

    def viewer_alive(self):
        return (not self.show_viewer) or (self.viewer is not None
                                          and self.viewer.is_running())

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
