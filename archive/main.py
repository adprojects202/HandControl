#!/usr/bin/env python
"""Drive the h4nd_left hand in MuJoCo, by joint or by fingertip.

    python main.py                       viewer + pygame control panel
    python main.py --port PORT           the same, with the real hand attached
    python main.py --no-panel            viewer only
    python main.py --headless            no viewer (for the IK, not the show)
    python main.py panel                 just the panel (the simulator spawns this)
    python main.py ik index 0.01 0.02 0.30      solve one fingertip, print angles
    python main.py pose fist             hold a named pose in the viewer
    python main.py poses                 list the named poses
    python main.py hw scan --port PORT   what is on the bus (reads only)

Naming a port only makes the hand *available*. Sending it commands, and
energising it, are two further switches in the panel, both off at startup.

The panel is a separate process on purpose: macOS will not run MuJoCo's viewer
off the main thread, and it will not run pygame's window off the main thread
either. They talk over UDP (handcontrol/net.py).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time

import numpy as np

from handcontrol.hardware import (BAUDRATE, COMMAND_HZ, CURRENT_LIMIT_MA, DynamixelHand,
                                  HardwareOutput, MAX_SPEED, MockHand, load_calibration,
                                  save_calibration, ticks_to_radians)
from handcontrol.mimic import FUTURE_HORIZON, Mimic, skeleton_offset
from handcontrol.robot.model import (DEFAULT_HAND, EFFORT, FINGERS, KP, KV, POSES,
                               HandModel, ensure_urdf, joint_role, pose_vector)
from handcontrol.net import COMMAND_PORT, STATE_PORT, Channel
from handcontrol.robot.layout import Row
from handcontrol.robot.options import Guard, Physics, Scene
from handcontrol.tracelog import TraceLog

STATE_HZ = 20.0
# The skeleton is rebuilt and re-uploaded every time it is drawn, so it is
# drawn less often than the hand is simulated. 30 Hz reads as continuous and
# leaves the physics step the frame it is entitled to.
SKELETON_HZ = 30.0


# panel field -> the argument HandSim.set_ik takes
IK_TUNING = {"ik_damping": "damping", "ik_iters": "max_solver_iters",
             "ik_tol": "pos_tol"}
SERVO_TUNING = ("position_p", "position_i", "position_d",
                "profile_velocity", "profile_acceleration", "current_limit")


def _apply_tuning(sim, hardware, values):
    """Retune a running session. Everything here takes effect immediately.

    The servo fields are RAM registers on the XL330, so they do not need the
    torque dropped; the simulator's gains MuJoCo accepts at any time; and the
    stream rates are just numbers the bus thread reads next tick.
    """
    gains = {k: v for k, v in values.items() if k in ("kp", "kv", "effort")}
    if gains:
        sim.set_gains(**gains)
    if "collision_margin" in values:
        sim.set_collision_margin(values["collision_margin"])
    if "ik_avoid" in values:
        sim.set_ik_avoid(values["ik_avoid"])
    ik = {arg: values[field] for field, arg in IK_TUNING.items()
          if field in values}
    if ik:
        sim.set_ik(**ik)
    if "max_speed" in values:
        hardware.limiter.max_speed = float(values["max_speed"])
    if "command_hz" in values:
        hardware.interval = 1.0 / max(float(values["command_hz"]), 1.0)
    servo = {k: int(v) for k, v in values.items() if k in SERVO_TUNING}
    if servo and hardware.available:
        try:
            hardware.set_servo(servo)
        except Exception as failure:
            hardware.error = f"gains: {failure}"


def _tuning_state(sim, hardware):
    """Current values, read back rather than remembered, so the panel cannot
    drift out of step with what is actually running."""
    state = sim.tuning()
    state["max_speed"] = hardware.limiter.max_speed
    state["command_hz"] = 1.0 / hardware.interval
    state.update(hardware.servo_tuning())
    return state


def open_link(port, joint_names, args):
    """A connected link for `port`, or None for "no hardware".

    Raises on a port that will not talk, so the caller can decide whether that
    is fatal (the `hw` commands) or just something to report (the panel, which
    has to stay up so you can pick a different one).
    """
    if port == "mock":
        return MockHand(joint_names).connect()
    if not port:
        return None
    return DynamixelHand(port, joint_names, baudrate=args.baud,
                         current_ma=args.current_ma).connect()


def describe(link):
    return (f"{link.status()['note']}"
            + (", TORQUE ALREADY ON" if link.torque_on else ", torque off"))


def hardware_link(args, joint_names):
    """The link named on the command line, or nothing.

    A hand that will not talk is reported and stepped over: losing the bus
    should cost you the hardware, not the session.
    """
    port = "mock" if getattr(args, "mock_hardware", False) else \
        getattr(args, "port", None)
    try:
        link = open_link(port or "", joint_names, args)
    # Broad on purpose, and for the same reason as the run loop: a port that is
    # gone, busy, or held by another program raises out of pyserial, not out of
    # the DYNAMIXEL SDK, and none of it deserves a traceback.
    except Exception as failure:
        print(f"[handcontrol] hardware unavailable on {port}: {failure}")
        return None
    if link is not None:
        print(f"[handcontrol] hardware on {port}: {describe(link)}")
    return link


def spawn_panel(args):
    """The panel has to build the same model this process did - same chirality,
    same limits - or its sliders describe a different robot."""
    command = [sys.executable, __file__, "panel",
               "--command-port", str(args.command_port),
               "--state-port", str(args.state_port),
               "--hand", args.hand]
    if args.spread_limit is not None:
        command += ["--spread-limit", str(args.spread_limit)]
    return subprocess.Popen(command)


# The live world, left to right.
#
# WHAT EACH LANE ACTUALLY SHOWS, because they are not the same kind of thing
# and reading them as if they were is misleading:
#
#   TEACHER and STUDENT are posed with the RAW output of each mapping - the
#   solver's answer and the network's, straight out. No smoothing, no physics.
#
#   AHEAD is the student's forward head - where it says the hand will be
#   ~80 ms from now (`Mimic.FUTURE_HORIZON`). It leads STUDENT on purpose;
#   a lane that did not run ahead would be showing nothing.
#
#   ROBOT is the simulated hand, and the only one with physics. It shows where
#   MuJoCo's position controller has got to, having been commanded the selected
#   mapping's answer AFTER `Retargeter.accept` smooths and rate-limits it, the
#   joint limits clamp it, and the self-collision clamp holds back anything
#   that would put the hand through itself.
#
#   ENCODER is the real robot, read back off the bus.
#
# So ROBOT lags whichever coloured lane is selected, always, and that lag is
# the smoothing and the controller rather than a disagreement between mappings.
# What reaches the servos is `sim.targets` - the commanded mapping's joints
# after all three of those - so a divergence between the violet lane and the
# grey one can be the collision clamp rather than the model.
#
# MuJoCo's IK is in none of this: `solve_tip` belongs to the FINGERTIP tab and
# is unreachable from either learning mode.
#
# A lane that only updated while it was selected would be showing history, so
# `Mimic` runs both mappings every frame and this poses both every frame.
LANES = ("YOU", "ROBOT", "TEACHER", "STUDENT", "AHEAD", "ENCODER")


def run(args):
    from handcontrol.robot.sim import HandSim
    from handcontrol.robot.mjcf import (FUTURE, FUTURE_RGBA, STUDENT,
                                        STUDENT_RGBA, TEACHER, TEACHER_RGBA)

    # Lane 1 is the commanded hand, so placing it places the row; the copies
    # and the encoder ghost are offset from it. `layout.Row` owns that
    # arithmetic - see `robot/layout.py`.
    lanes = args.lanes and not args.headless
    x, _, z = args.position
    row = Row(len(LANES), args.hand, height=z, forward=x)
    side = row.side
    camera_pos, camera_lookat = row.camera(back=1.12, lift=0.60, look=0.16)

    sim = HandSim(
        hand=args.hand, spread_limit=args.spread_limit,
        physics=Physics(gravity=not args.no_gravity,
                        self_collision=not args.no_self_collision,
                        kp=args.kp, kv=args.kv, effort=args.effort),
        guard=Guard(avoid=not args.allow_collisions,
                    ik_avoid=not args.ik_allow_collisions,
                    margin=args.collision_margin),
        scene=Scene(
            show_viewer=not args.headless,
            position=row.anchor if lanes else tuple(args.position),
            ghost=not args.no_ghost,
            ghost_offset=(row.offset(LANES.index("ENCODER")) if lanes
                          else args.ghost_offset),
            copies=(row.copies([(TEACHER, TEACHER_RGBA),
                                (STUDENT, STUDENT_RGBA),
                                (FUTURE, FUTURE_RGBA)]) if lanes else ()),
            camera_pos=camera_pos if lanes else None,
            camera_lookat=camera_lookat if lanes else None)).build()

    hardware = HardwareOutput(hardware_link(args, sim.joint_names),
                              len(sim.joint_names), max_speed=args.max_speed,
                              hz=args.command_hz, joint_names=sim.joint_names)


    # Claim the port BEFORE spawning the panel. The other order leaves an
    # orphan behind on a clash: the panel comes up, grabs the state port, and
    # outlives the simulator that could not bind the command port - so the next
    # attempt fails on a port held by a window from the attempt before.
    trace = TraceLog()
    link = Channel(bind=args.command_port, send_to=args.state_port)

    panel = None
    if not args.no_panel:
        panel = spawn_panel(args)

    # Built whether or not it is ever used: constructing it opens no camera,
    # so a session that never touches the MIMIC tab pays nothing, and one that
    # does can switch it on without a restart.
    mimic = Mimic(sim.model, mode=args.mimic_mode, thumb=args.mimic_thumb,
                  follow=args.follow, tracking_mode=args.tracking_mode,
                  library=args.leap_lib, mirror=args.mirror)
    print(f"[handcontrol] this is a {mimic.retarget.chirality()} hand - hold "
          f"your {mimic.expected.upper()} hand over the sensor"
          + (" (mirrored)" if args.mirror else ""))
    if args.mimic and not mimic.enable(True):
        print(f"[handcontrol] leap unavailable: {mimic.error}")
    # Where the operator's hand is drawn in the viewer: above the robot, not
    # on it. Superimposed they interpenetrate and neither shape is readable -
    # the panel's pad views are where the overlay belongs, and there it is the
    # diagnostic. Derived from the model unless overridden.
    viewer_offset = (tuple(args.skeleton_offset) if args.skeleton_offset
                     else skeleton_offset(mimic.retarget))
    if lanes and not args.skeleton_offset:
        # In the row the operator gets lane 0 of their own rather than floating
        # above the robot: it is being compared against every mapping at once
        # now, not against one hand. `mimic.skeleton` places it relative to
        # `sim.position`, which is lane 1, so the offset is one lane back.
        viewer_offset = (0.0, -side * row.spacing, -(sim.position[2] - z))

    targets = np.zeros(len(sim.joint_names))
    tip_targets = {}
    tip_errors = {}
    skeleton = {}
    viewer_skeleton = {}
    # Which posed copy each mapping drives, and the words under the row.
    lane_copies = ([("keyvector", TEACHER), ("student", STUDENT),
                    ("future", FUTURE)] if lanes else [])
    nameplates = []
    if lanes:
        from handcontrol.robot import glyphs
        from handcontrol.robot.mjcf import GHOST_RGBA
        colours = {"YOU": (0.30, 0.55, 1.0, 0.95),
                   "ROBOT": (0.72, 0.74, 0.78, 1.0),
                   "TEACHER": tuple(float(v) for v in TEACHER_RGBA.split()),
                   "STUDENT": tuple(float(v) for v in STUDENT_RGBA.split()),
                   "AHEAD": tuple(float(v) for v in FUTURE_RGBA.split()),
                   "ENCODER": tuple(float(v) for v in GHOST_RGBA.split())}
        widest = max(glyphs.cells(tag)[1] for tag in LANES)
        cell = min(glyphs.CELL, row.spacing * 0.82 / (widest * 1.4))
        base = z + glyphs.LIFT * cell * 1.4
        for index, tag in enumerate(LANES):
            # Slot 0's own sideways place, which is one lane back from the
            # anchor - the row is centred, so this is not index * spacing.
            nameplates.append((tag, (0.10, sim.position[1] + side * row.offset(index),
                                     base), colours[tag], cell))
    markers_drawn = None
    last_frame = None
    last_state = last_skeleton = 0.0
    # Where a loop iteration's time goes, exponential averages in ms, and
    # the worst iteration. Reported in the state message and at exit.
    loop = {"mapping": 0.0, "guard": 0.0, "physics": 0.0, "draw": 0.0,
            "total": 0.0, "worst_total": 0.0}
    tick_began = time.perf_counter()

    def timed(key, began):
        took = (time.perf_counter() - began) * 1000.0
        loop[key] += 0.05 * (took - loop[key])
        return time.perf_counter()
    # The gesture probe playing through the lanes - see `learning.playback`.
    playback = None

    print("[handcontrol] running. close the viewer window to stop.")
    try:
        started = time.time()
        while sim.viewer_alive() and (not args.seconds
                                      or time.time() - started < args.seconds):
            for message in link.drain():
                kind = message.get("type")
                if kind == "joints":
                    targets = sim.clamp(message["q"])
                    tip_targets.clear()
                    tip_errors.clear()
                elif kind == "tip":
                    finger = message["finger"]
                    target = np.asarray(message["target"], dtype=float)
                    targets, error = sim.reachable_tip(finger, target)
                    tip_targets[finger] = target.tolist()
                    tip_errors[finger] = error
                elif kind == "pose":
                    targets = sim.clamp(
                        pose_vector(POSES.get(message["name"], {}),
                                    sim.joint_names))
                    tip_targets.clear()
                    tip_errors.clear()
                elif kind == "reset":
                    targets = sim.reset()
                    tip_targets.clear()
                    tip_errors.clear()
                elif kind == "connect":
                    port = message.get("port", "")
                    try:
                        hardware.attach(open_link(port, sim.joint_names, args))
                        if hardware.available:
                            print(f"[handcontrol] hardware on {port}: "
                                  f"{describe(hardware.link)}")
                    # Broad on purpose: a wrong port raises anything from
                    # DynamixelError to OSError, and none of it should take
                    # the session down - the point is to pick another one.
                    except Exception as failure:
                        hardware.detach()
                        hardware.error = f"{port}: {failure}"
                        print(f"[handcontrol] {hardware.error}")
                elif kind == "tune":
                    _apply_tuning(sim, hardware, message.get("values", {}))
                elif kind == "output":
                    hardware.enable(bool(message.get("hardware")))
                elif kind == "arm":
                    hardware.arm(bool(message.get("on")))
                elif kind == "mimic":
                    # The panel sends the whole tab state, so settings changed
                    # while it was off take effect the moment it goes on.
                    mimic.configure(**{k: v for k, v in message.items()
                                       if k != "type" and k != "on"})
                    if "on" in message:
                        # enable() returns the resulting state, so switching
                        # OFF returns False too - only a failure to switch ON
                        # is worth reporting.
                        wanted = bool(message["on"])
                        if not mimic.enable(wanted) and wanted:
                            print(f"[handcontrol] leap unavailable: "
                                  f"{mimic.error}")
                    if not mimic.enabled:
                        tip_targets.clear()
                        tip_errors.clear()
                        skeleton = viewer_skeleton = {}
                elif kind == "grip":
                    hardware.enable_grip(bool(message.get("on")))
                    print("[handcontrol] closing to contact "
                          + ("on" if hardware.executor.enabled else "off"))
                elif kind == "log":
                    path = trace.toggle(sim.joint_names, message.get("on"))
                    print(f"[handcontrol] logging to {path}" if path else
                          f"[handcontrol] log closed: {trace.path} "
                          f"({trace.rows} rows)" if trace.path else
                          f"[handcontrol] {trace.error}")
                elif kind == "gestures":
                    if message.get("on") and playback is None:
                        from handcontrol.learning import playback as probe
                        source = message.get("source", "probe")
                        target = (probe.newest_session() if source == "session"
                                  else probe.PROBE)
                        if target is None or not (probe.available(target)
                                                  or (target / "teacher.npz").exists()):
                            print("[handcontrol] nothing to play - run "
                                  "`learn.py gestures` or `learn.py teacher` first")
                        else:
                            try:
                                playback = probe.GesturePlayback(
                                    target, blocks=message.get("blocks"))
                            except FileNotFoundError as failure:
                                print(f"[handcontrol] {failure}")
                            else:
                                if mimic.student is None and not mimic.load_student():
                                    print(f"[handcontrol] playing without a student: "
                                          f"{mimic.student_error}")
                                print(f"[handcontrol] playing the {playback.source}, "
                                      f"{len(playback)} frames at {playback.hz:.0f} Hz")
                    elif not message.get("on"):
                        playback = None
                elif kind == "calibrate":
                    if message.get("reset"):
                        mimic.reset_calibration()
                    elif message.get("next"):
                        mimic.capture_next()
                    elif mimic.capture is not None:
                        mimic.cancel_capture()
                    elif not mimic.start_capture():
                        print(f"[handcontrol] leap unavailable: {mimic.error}")

            # The hand follows the operator only while the tab is on. Sliders,
            # tip drags and presets all still write `targets`, so switching
            # MIMIC off leaves the hand exactly where the operator left it and
            # hands control straight back to whatever tab you switch to.
            stage = time.perf_counter()
            frame_stamp = time.time()
            if playback is not None:
                # THE PROBE STANDS IN FOR THE SENSOR. The teacher's labels
                # pose TEACHER, the student runs live on the probe's frames
                # for STUDENT and AHEAD, and whichever of the two the LEARNING
                # tab has selected commands the robot - through the same
                # clamp and guard as a tracked hand would.
                playback.advance()
                mimic.alternates["keyvector"] = playback.teacher()
                command = mimic.alternates["keyvector"]
                if mimic.student is not None:
                    heads = mimic.student_heads(*playback.windows())
                    mimic.alternates["student"] = heads[0]
                    if FUTURE_HORIZON in heads:
                        mimic.alternates["future"] = heads[FUTURE_HORIZON]
                    if mimic.mode == "student":
                        command = mimic.repair(heads[0])
                targets = sim.clamp(command)
                tip_targets = {}
                tip_errors = {}
            elif mimic.enabled:
                # WALL CLOCK, not sim.dt. The rate limiter converts rad/s into
                # a per-call step, so it needs the time between CALLS - and
                # sim.dt is the physics timestep, which is 0.002 s while this
                # loop comes round every 0.006. Handing it sim.dt made the
                # limiter believe a third of the real time had passed, so a
                # 6 rad/s setting delivered 1.9, and every mode felt like it
                # was dragging behind the hand.
                #
                # Capped, for the same reason the bus thread caps its own: a
                # stalled frame must not buy a large sudden movement.
                now = time.time()
                frame_dt = min(now - (last_frame or now - sim.dt), 0.05)
                last_frame = now
                targets = sim.clamp(mimic.step(sim, max(frame_dt, sim.dt)))
                tip_targets = {f: t.tolist()
                               for f, t in mimic.tip_targets.items()}
                tip_errors = dict(mimic.tip_errors) if mimic.mode == "tip" else {}
            elif mimic.capture is not None:
                # Calibrating reads the sensor without driving the hand. It has
                # to: measuring your range with sixteen servos mirroring your
                # fist is not an improvement, and gating this on `enabled` used
                # to freeze the capture between poses whenever FOLLOW was off.
                mimic.poll()

            # After the mapping has run and before the frame is committed, so
            # the row records the command that was actually issued alongside
            # the hand that produced it.
            trace.write(mimic, sim, hardware, targets)
            stage = timed("mapping", stage)

            sim.command(targets)
            stage = timed("guard", stage)
            # `sim.targets`, not `targets`: the simulator holds back anything
            # that would put the hand through itself, and the real hand has
            # far more to lose by it than the picture does. The stamp is when
            # the sensor frame behind this command arrived, so the bus can
            # report how old a command is by the time it is written.
            # The contact head rides along so the executor can tell a pinch
            # from a grasp; None whenever the student is not the one driving.
            hardware.command(sim.targets, stamp=frame_stamp,
                             pinching=getattr(mimic, "student_contact", None))
            # The encoder hand: whatever the bus thread last read, redrawn as
            # the second, amber hand. It updates whenever a port is open -
            # armed or not - so pushing the real fingers about by hand shows
            # up in the viewer, which is the cheapest calibration check there
            # is.
            if hardware.positions is not None:
                sim.set_ghost(hardware.positions)
            # The two mappings that are NOT commanding, posed beside the one
            # that is. `Mimic` runs both every frame precisely so this lane
            # pair is a comparison rather than a record of the last time each
            # happened to be selected.
            for name, prefix in lane_copies:
                spare = mimic.alternates.get(name)
                if spare is not None:
                    sim.set_copy(prefix, sim.clamp(spare))
            sim.step()
            stage = timed("physics", stage)

            now = time.time()
            # Targets are redrawn only when they move, but the operator's hand
            # moves constantly, so with mimic on the overlay is redrawn on a
            # clock instead. Both go through one call: `draw_debug` clears
            # every debug object, so two callers would erase each other.
            drawn = tuple(sorted((f, tuple(t)) for f, t in tip_targets.items()))
            due = ((mimic.reading or playback is not None)
                   and now - last_skeleton > 1.0 / SKELETON_HZ)
            # Debug objects persist until something clears them, so a hand
            # that stops being read cannot just be left alone: the last frame
            # before FOLLOW went off would hang in the viewer looking like a
            # hand that is still there. It is REDRAWN rather than blanked -
            # with nothing tracked `mimic.skeleton` gives a flat hand, so lane
            # 0 holds its place in the row instead of leaving a hole where the
            # thing every other lane is compared against should be.
            untracked = bool(viewer_skeleton) and not mimic.reading and playback is None
            if untracked:
                skeleton = mimic.skeleton(sim.position)
                viewer_skeleton = {d: (np.asarray(p) + viewer_offset).tolist()
                                   for d, p in skeleton.items()}
            if drawn != markers_drawn or due or untracked:
                if due:
                    last_skeleton = now
                    # Two placements of one transform: the panel superimposes
                    # its copy on the robot's fingertips, the viewer lifts its
                    # copy clear so the two hands read as two shapes.
                    skeleton = (playback.skeleton(sim.position) if playback is not None
                                else mimic.skeleton(sim.position))
                    viewer_skeleton = {d: (np.asarray(p) + viewer_offset).tolist()
                                       for d, p in skeleton.items()}
                sim.draw_debug(targets=tip_targets, skeleton=viewer_skeleton,
                               extra=nameplates)
                markers_drawn = drawn
            stage = timed("draw", stage)
            whole = (time.perf_counter() - tick_began) * 1000.0
            tick_began = time.perf_counter()
            loop["total"] += 0.05 * (whole - loop["total"])
            loop["worst_total"] = max(loop["worst_total"], whole)

            if now - last_state > 1.0 / STATE_HZ:
                last_state = now
                link.send({
                    "type": "state",
                    "origin": list(sim.position),
                    "q": sim.joint_positions().tolist(),
                    "tips": {f: p.tolist() for f, p in sim.tip_positions().items()},
                    "error": tip_errors,
                    "blocked": bool(getattr(sim, "blocked", False)),
                    "tuning": _tuning_state(sim, hardware),
                    "hardware": hardware.state(),
                    "leap": mimic.status(),
                    "log": trace.status(),
                    "loop": {k: round(v, 2) for k, v in loop.items()},
                    "gestures": {"playing": playback is not None,
                                 "source": playback.source if playback is not None else "",
                                 "label": playback.label if playback is not None else "",
                                 "frame": playback.index if playback is not None else 0,
                                 "frames": len(playback) if playback is not None else 0},
                    # Rounded to the tenth of a millimetre. Full float repr
                    # triples the datagram to draw a line five pixels wide.
                    "skeleton": {d: np.round(p, 4).tolist()
                                 for d, p in skeleton.items()},
                })
    except KeyboardInterrupt:
        pass
    finally:
        print("[handcontrol] loop ms, averaged: "
              + "  ".join(f"{k} {v:.2f}" for k, v in loop.items())
              + f"   bus latency {hardware.latency_ms:.1f} ms "
              f"(worst {hardware.worst_latency_ms:.1f})")
        if panel is not None:
            panel.terminate()
        if trace.active:
            print(f"[handcontrol] log closed: {trace.stop()} "
                  f"({trace.rows} rows)")
        mimic.close()
        if hardware.armed:
            # Left holding on purpose. Cutting torque on the way out drops
            # whatever the hand is gripping; `main.py hw relax` is the
            # deliberate way to let go.
            print("[handcontrol] hardware still armed - `hw relax` to release")
        hardware.close()
    return 0


def run_pose(args):
    from handcontrol.robot.sim import HandSim

    if args.name not in POSES:
        print(f"unknown pose {args.name!r}; known: {', '.join(POSES)}")
        return 1
    sim = HandSim(hand=args.hand, spread_limit=args.spread_limit,
                  scene=Scene(show_viewer=not args.headless)).build()
    q = sim.clamp(pose_vector(POSES[args.name], sim.joint_names))
    steps = args.steps if args.headless else 10 ** 9
    for _ in range(steps):
        if not sim.viewer_alive():
            break
        sim.command(q)
        sim.step()
    print(f"{args.name}: tracking error "
          f"{np.degrees(np.abs(sim.joint_positions() - q)).max():.2f} deg")
    for finger, tip in sim.tip_positions().items():
        print(f"  {finger:7s} tip {np.round(tip, 4)}")
    return 0


def run_ik(args):
    from handcontrol.robot.sim import HandSim

    sim = HandSim(hand=args.hand, spread_limit=args.spread_limit,
                  scene=Scene(show_viewer=False)).build()
    target = np.array(args.target, dtype=float)
    q, error = sim.reachable_tip(args.finger, target)
    print(f"target      {np.round(target, 4)}")
    print(f"solver miss {error * 1000:.2f} mm"
          + ("  (out of reach - closest approach)" if error > 0.003 else ""))
    for name in sim.chains[args.finger]:
        i = sim.joint_names.index(name)
        print(f"  {name:26s} {np.degrees(q[i]):8.2f} deg")
    for _ in range(400):
        sim.command(q)
        sim.step()
    reached = sim.tip_position(args.finger)
    print(f"settled tip {np.round(reached, 4)}  "
          f"({np.linalg.norm(reached - target) * 1000:.2f} mm from target)")
    return 0


def _hw_sign(args, joint_names):
    """Which way each motor turns for a positive joint angle.

    No serial port involved: this edits config/hand_calibration.json, and the
    only reason it lives under `hw` is that it describes the hardware.
    """
    calibration = load_calibration(joint_names)
    if args.group:
        # Grouped by what the joint moves, not by its name - `thumb_proximal`
        # is a spread and `thumb_adduction` is a flexion, whatever they are
        # called. The joints are listed back so there is no guessing which
        # ones a group covered.
        groups = {"flexion": [n for n in joint_names
                              if joint_role(n) != "spread"],
                  "spread": [n for n in joint_names
                             if joint_role(n) == "spread"],
                  "all": list(joint_names)}
        chosen = groups[args.group]
    elif args.joint:
        if args.joint not in joint_names:
            print(f"no such joint: {args.joint}")
            return 1
        chosen = [args.joint]
    else:
        print("which joints? --joint index_proximal_joint, "
              "or --group flexion|spread|all")
        return 1

    for name in chosen:
        calibration["motors"][name]["sign"] = int(args.value)
    path = save_calibration(calibration)
    print(f"sign {int(args.value):+d} on {len(chosen)} joint(s) -> {path}")
    for name in chosen:
        print(f"  {name}")
    return 0


def _hw_id(args, joint_names):
    """Which motor ID drives which joint.

    Also a file edit, no serial port. The default map is IDs 1..16 down the
    URDF's chain order, and the URDF's chain puts the thumb's knuckle before
    its spread while every finger is the other way round - so if the hand was
    wired spread-first throughout, the thumb's two IDs are the pair to swap.
    Nothing has confirmed the map either way; `hw watch` does.
    """
    calibration = load_calibration(joint_names)
    motors = calibration["motors"]
    if args.swap:
        first, second = args.swap
        for name in (first, second):
            if name not in motors:
                print(f"no such joint: {name}")
                return 1
        motors[first]["id"], motors[second]["id"] = \
            motors[second]["id"], motors[first]["id"]
        print(f"swapped: {first} -> id {motors[first]['id']}, "
              f"{second} -> id {motors[second]['id']}")
    elif args.joint:
        if args.joint not in motors:
            print(f"no such joint: {args.joint}")
            return 1
        motors[args.joint]["id"] = int(args.value)
        print(f"{args.joint} -> id {int(args.value)}")
    else:
        print(f"{'joint':26s} {'id':>3}")
        for name in joint_names:
            print(f"{name:26s} {motors[name]['id']:>3}")
        print("\nchange one with --joint NAME --value N, "
              "or exchange a pair with --swap A B")
        return 0
    print(f"wrote {save_calibration(calibration)}")
    return 0


def run_hw(args):
    """Bus-level commands. `scan` and `status` read only; the rest do not."""
    model = HandModel.load(ensure_urdf(args.hand, args.spread_limit))
    joint_names = model.joint_names()
    lower, upper = model.limits(joint_names)
    limits = dict(zip(joint_names, zip(lower, upper)))
    if args.action == "sign":
        return _hw_sign(args, joint_names)      # a file edit, not a bus command
    if args.action == "id":
        return _hw_id(args, joint_names)

    link = hardware_link(args, joint_names)
    if link is None:
        print("no hardware link - pass --port (or --mock-hardware)")
        return 1

    try:
        if args.action == "scan":
            print(f"{'joint':26s} {'id':>3} {'model':>6} {'ticks':>6} {'deg':>8}")
            found = link.ping_all()
            for name in joint_names:
                motor = link.motors[name]
                if found[name] is None:
                    print(f"{name:26s} {motor['id']:>3}  no reply")
                    continue
                ticks = link._read(motor["id"], "present_position")
                print(f"{name:26s} {motor['id']:>3} {found[name]:>6} {ticks:>6} "
                      f"{np.degrees(link.to_radians(name, ticks)):>8.2f}")
            print(f"torque is currently {'ON' if link.torque_on else 'off'}")

        elif args.action == "status":
            for row in link.health():
                print("  ".join(f"{k}={v}" for k, v in row.items()))

        elif args.action == "zero":
            # Whatever pose the hand is in right now becomes joint zero.
            #
            # YOU SHOULD NOT NEED THIS ON THIS HAND. It was assembled the
            # standard way: every servo centred at tick 2048 before its horn
            # went on, so motor centre IS joint zero by construction, and the
            # default calibration - 2048 on all sixteen - is already correct.
            # It survives a power cycle: the zero lives in the file, and the
            # XL330's encoder is absolute.
            #
            # Capturing zero from a hand-held pose can only ADD error to that.
            # It did: run on 12 Sep against a posed hand, it scattered all
            # sixteen zeros from -24 to +15 deg off centre and put the thumb
            # knuckle at +45.5, because a thumb posed to LOOK right is not at
            # q = 0 on a URDF with a 30 deg mount roll and a 45 deg pitch in
            # the middle segment. The simulator looked perfect; the real hand
            # did not. All sixteen went back to 2048.
            #
            # Run it only for a joint whose horn has been re-mounted, a
            # replaced servo, or a linkage that has slipped - and then against
            # the simulator at q = 0, never against a description of the pose.
            #
            # `--joint` re-zeroes ONE joint and leaves the rest alone, so a
            # single bad capture can be fixed without re-posing the whole hand
            # and risking the fifteen that are already right.
            calibration = load_calibration(joint_names)
            targets = [args.joint] if args.joint else list(joint_names)
            unknown = [n for n in targets if n not in calibration["motors"]]
            if unknown:
                print(f"no such joint: {unknown[0]}")
                return 1
            for name in targets:
                before = calibration["motors"][name]["zero"]
                ticks = link._read(calibration["motors"][name]["id"],
                                   "present_position")
                calibration["motors"][name]["zero"] = int(ticks)
                if args.joint:
                    print(f"  {name}: zero {before} -> {int(ticks)} "
                          f"({np.degrees(ticks_to_radians(int(ticks) - before)):+.1f} deg)")
            calibration["calibrated"] = True
            path = save_calibration(calibration)
            print(f"wrote {path}")
            print("check the signs by hand: a joint that reads more negative as "
                  "it flexes needs sign -1 in that file.")

        elif args.action == "watch":
            # The fastest way to answer "is this joint's direction flipped?":
            # relax the hand, bend one finger, and read which joint moved and
            # which way. A wrong `sign` shows up here as a bar going the
            # opposite way to the finger.
            print("torque off, then bend the fingers by hand. ctrl-c to stop.")
            print("bars are movement SINCE START, not absolute angle.\n")
            start = link.read_positions()
            try:
                while True:
                    now = link.read_positions()
                    delta = np.degrees(now - start)
                    lines = []
                    for name, value, absolute in zip(joint_names, delta,
                                                     np.degrees(now)):
                        width = int(min(abs(value), 60.0) / 3.0)
                        left = ("#" * width).rjust(20) if value < 0 else " " * 20
                        right = ("#" * width).ljust(20) if value >= 0 else " " * 20
                        lines.append(f"{name:24s} {absolute:8.1f}d "
                                     f"{left}|{right} {value:+7.1f}")
                    print("\033[H\033[J" + "\n".join(lines)
                          + "\n\nleft of the bar is negative, right is positive")
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nstopped")

        elif args.action == "gains":
            values = {f: getattr(args, f)
                      for f in ("position_p", "position_i", "position_d",
                                "profile_velocity", "profile_acceleration",
                                "current_limit")
                      if getattr(args, f, None) is not None}
            if values:
                # RAM registers on the XL330, so this takes effect immediately
                # and does not need the torque turned off.
                link.write_gains(values)
                print("set on all 16:",
                      "  ".join(f"{k}={v}" for k, v in values.items()))
            rows = link.read_gains()
            head = ("position_p", "position_i", "position_d",
                    "profile_velocity", "profile_acceleration", "current_limit")
            width = {h: max(6, len(h.split("_")[-1])) for h in head}
            print(f"{'joint':26s} "
                  + " ".join(f"{h.split('_')[-1]:>{width[h]}s}" for h in head))
            for name in joint_names:
                r = rows[name]
                print(f"{name:26s} "
                      + " ".join(f"{r[h]:>{width[h]}d}" for h in head))
            print("\nA joint that overshoots is short of D, not of P: the current "
                  "limit clips\nthe drive, so more P only arrives faster. A joint "
                  "that buzzes at rest has\ntoo much D for a 0.088 deg/tick "
                  "encoder. `hw step` measures which.")

        elif args.action == "step":
            # Moves the hand. One joint, a small amount, and back.
            if not args.joint:
                print("which joint? --joint index_proximal_joint")
                return 1
            amp = math.radians(args.degrees)
            here = link.read_positions()
            i = joint_names.index(args.joint)
            link.arm()
            goal = here.copy()
            goal[i] = float(np.clip(here[i] + amp, *limits[args.joint]))
            print(f"stepping {args.joint} by {math.degrees(goal[i] - here[i]):+.1f} "
                  f"deg and back")
            trace = []
            link.write_positions(goal)
            start = time.time()
            while time.time() - start < args.seconds:
                trace.append((time.time() - start,
                              link.read_positions()[i]))
            link.write_positions(here)
            time.sleep(args.seconds)

            reached = np.array([v for _, v in trace])
            target = goal[i]
            travel = abs(target - here[i])
            overshoot = (reached.max() - target if target > here[i]
                         else target - reached.min())
            tail = reached[-max(len(reached) // 4, 2):]
            print(f"  travel      {math.degrees(travel):6.1f} deg")
            print(f"  overshoot   {math.degrees(overshoot):+6.2f} deg "
                  f"({100 * overshoot / max(travel, 1e-9):+.0f}% of travel)")
            print(f"  settle ring {math.degrees(tail.max() - tail.min()):6.2f} "
                  f"deg peak-to-peak")
            print(f"  samples     {len(trace)} in {args.seconds:.1f} s")
            if overshoot > 0.1 * travel:
                print("  -> overshooting: raise position_d")
            elif math.degrees(tail.max() - tail.min()) > 0.5:
                print("  -> buzzing at rest: lower position_d")
            else:
                print("  -> clean")

        elif args.action == "relax":
            link.disarm()
            print("torque off - the hand is limp and will drop what it holds")

        elif args.action == "hold":
            here = link.arm()
            print("torque on, holding where it already was:")
            for name, value in zip(joint_names, here):
                print(f"  {name:26s} {np.degrees(value):8.2f} deg")

        elif args.action == "grip":
            return bench_grip(args, link, joint_names)
    finally:
        link.close()
    return 0


def bench_grip(args, link, joint_names):
    """Close ONE finger onto whatever is in front of it, and say what the
    executor's thresholds should be.

    THE NUMBERS IN `executor.py` CAME FROM A DATASHEET AND A MOCK. This is how
    they meet a real object. It closes the chosen finger at the rate the bus
    normally runs, logs what the encoders and the currents actually did, and
    reads the answer back out of the trace:

      * where the encoder stopped following, which is the contact,
      * what the current rose to and when it crossed `STALL_MA`, which is
        the one thing the executor latches on,
      * and how far the finger drives into the object before that happens.

    It also prints the command lead and the encoder speed, which are NOT
    tests: both were tried as detectors and both failed on this hand - see
    `executor.STALL_MA`.

    Put something in the finger's path first - a cylinder taped to the bench
    is what this was written for. Run it with nothing there too: the trace
    should show the finger reaching its limit instead, and the thresholds it
    prints from THAT run are the ones that would false-trigger.
    """
    from handcontrol.executor import STALL_MA, CLOSING_ROLES
    from handcontrol.robot.model import joint_role

    finger = args.finger
    joints = [k for k, n in enumerate(joint_names)
              if n.startswith(finger + "_") and joint_role(n) in CLOSING_ROLES]
    if not joints:
        print(f"no closing joints for {finger!r}")
        return 1
    if args.obstacle is not None and hasattr(link, "obstacle"):
        # A virtual stop, so the whole procedure can be rehearsed - and
        # checked - without a hand on the bench. See `MockHand.read_state`.
        link.obstacle = {k: float(args.obstacle) for k in joints}
        print(f"mock stop at {args.obstacle} rad")
    print(f"closing {finger} ({', '.join(joint_names[k] for k in joints)})")
    print("put the object in its path, or run with nothing there for the "
          "false-trigger case. ctrl-c to stop early.\n")

    here = link.arm()
    target = np.asarray(here, dtype=float).copy()
    dt, rate = 1.0 / args.command_hz, args.close_speed
    trace = []
    try:
        for tick in range(int(args.seconds / dt)):
            target[joints] = np.minimum(target[joints] + rate * dt, args.close_to)
            link.write_positions(target)
            time.sleep(dt)
            if hasattr(link, "read_state"):
                positions, currents = link.read_state()
            else:
                positions, currents = link.read_positions(), None
            trace.append((tick * dt, target[joints].mean(),
                          float(np.mean(np.asarray(positions)[joints])),
                          0.0 if currents is None
                          else float(np.max(np.abs(np.asarray(currents)[joints])))))
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        link.disarm()
        print("torque off")

    if len(trace) < 10:
        print("too short to read anything from")
        return 1
    clock = np.array([row[0] for row in trace])
    wanted = np.array([row[1] for row in trace])
    reached = np.array([row[2] for row in trace])
    current = np.array([row[3] for row in trace])
    ahead = wanted - reached
    speed = np.gradient(reached, clock)

    # Contact is where the command started running away from the encoder and
    # never caught up again - the last frame the two were within a hair.
    close = np.flatnonzero(ahead < 0.02)
    contact = int(close[-1]) if len(close) else 0
    print(f"  {len(trace)} samples over {clock[-1]:.1f}s")
    print(f"  contact at {clock[contact]:.2f}s, {np.degrees(reached[contact]):.1f} deg")
    # THE DETECTOR, and it is the only one: present current. See
    # `executor.STALL_MA` for the run that removed the position test.
    fired = np.flatnonzero(current[contact:] > STALL_MA)
    when = f"{clock[contact + fired[0]] - clock[contact]:.2f}s after contact" \
        if len(fired) else "NEVER - nothing here reads as a stall"
    print(f"  current peaked at {current.max():.0f} mA   STALL_MA is {STALL_MA}, "
          f"crossed {when}")
    if len(fired):
        print(f"  the finger travels "
              f"{np.degrees(reached[contact + fired[0]] - reached[contact]):.1f} deg "
              f"into the object before it latches")
    # The two below are MEASUREMENTS, not tests. Both were tried as detectors
    # and both failed: free air produces a LARGER command lead than contact
    # does (28.5 deg against 17.9), and the encoder keeps creeping into the
    # object, so "not moving" is true on a tenth of post-contact frames.
    print(f"  command lead after contact {np.degrees(ahead[contact:].max()):.1f} deg "
          f"(free air reaches 28.5 - the lead does not separate contact)")
    print(f"  encoder speed in the 0.1s before contact "
          f"{np.abs(speed[max(0, contact - int(0.1 / dt)):contact]).max():.2f} rad/s, "
          f"after {np.abs(speed[contact:]).max():.2f} - it never stops")
    if args.out:
        np.savetxt(args.out, np.column_stack([clock, wanted, reached, current]),
                   header="t_s command_rad encoder_rad current_ma", comments="")
        print(f"  trace written to {args.out}")
    return 0


def main(argv=None):
    def _hardware_flags(target):
        target.add_argument("--port", help="U2D2 serial port, e.g. "
                                           "/dev/cu.usbserial-XXXXXXXX")
        target.add_argument("--baud", type=int, default=BAUDRATE)
        target.add_argument("--current-ma", type=int, default=CURRENT_LIMIT_MA,
                            help="per-motor current limit; what stops a stalled "
                                 "finger from pulling until something gives")
        target.add_argument("--max-speed", type=float, default=MAX_SPEED,
                            help="rad/s ceiling on how fast a command may move; "
                                 "the servo's own profile velocity is usually "
                                 "tighter still - see `hw gains`")
        target.add_argument("--command-hz", type=float, default=COMMAND_HZ,
                            help="goal writes per second down the bus")
        target.add_argument("--mock-hardware", action="store_true",
                            help="exercise the hardware path with no bus")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--hand", default=DEFAULT_HAND, choices=("right", "left"),
                        help="which hand is on the bench. `left` is the default "
                             "and is a mirrored copy of the vendored URDF; "
                             "`right` is the package untouched. See "
                             "model.MIRRORED_FOR for why the label and the "
                             "geometry disagree")
    common.add_argument("--spread-limit", type=float, default=None, metavar="DEG",
                        help="clamp the finger adduction joints to +/-DEG. The "
                             "URDF gives index and ring a 130 degree spread, the "
                             "same span as a flexion joint, against 40 for the "
                             "middle finger; 20 makes all three match")
    common.add_argument("--command-port", type=int, default=COMMAND_PORT)
    common.add_argument("--state-port", type=int, default=STATE_PORT)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     parents=[common])
    sub = parser.add_subparsers(dest="command")

    def add(name, help):
        return sub.add_parser(name, help=help, parents=[common])

    run_parser = add("run", "viewer + control panel (the default)")
    run_parser.add_argument("--no-panel", action="store_true")
    run_parser.add_argument("--headless", action="store_true")
    run_parser.add_argument("--seconds", type=float, default=0.0,
                            help="stop after this long; 0 runs until the window closes")
    run_parser.add_argument("--no-gravity", action="store_true")
    run_parser.add_argument("--no-self-collision", action="store_true",
                            help="turn off contact between the hand's own links. "
                                 "The clamp that refuses self-intersecting poses "
                                 "is separate and stays on - see collide.py")
    run_parser.add_argument("--position", type=float, nargs=3,
                            default=(0.0, 0.0, 0.20))
    # Responsiveness. The defaults are the real servo's numbers; every one of
    # these buys speed by relaxing something true. See README.
    run_parser.add_argument("--kp", type=float, default=None,
                            help=f"simulator stiffness (default {KP:g})")
    run_parser.add_argument("--kv", type=float, default=None,
                            help=f"simulator damping (default {KV:g})")
    run_parser.add_argument("--effort", type=float, default=None,
                            help=f"simulator torque clip in Nm (default "
                                 f"{EFFORT:g}, the XL330's stall). Raising it "
                                 f"buys speed by pretending the servo is "
                                 f"stronger than it is; at the shipped gains "
                                 f"the rise to 90%% is already 0.10 s")
    run_parser.add_argument("--collision-margin", type=float, default=None,
                            metavar="M",
                            help="metres of penetration past as-built to allow "
                                 "(default 0.006; below 0.004 the hand can no "
                                 "longer pinch)")
    run_parser.add_argument("--ik-allow-collisions", action="store_true",
                            help="let the IK solve straight through the hand's "
                                 "own fingers. The clamp still refuses the "
                                 "result, so the finger stops at the boundary "
                                 "instead of routing around it")
    run_parser.add_argument("--allow-collisions", action="store_true",
                            help="stop refusing poses that put the hand through "
                                 "itself; see handcontrol/collide.py")
    run_parser.add_argument("--no-lanes", dest="lanes", action="store_false",
                            help="the old two-hand world instead of the row: "
                                 "the commanded hand and the encoder ghost "
                                 "only, with no comparison lanes")
    run_parser.add_argument("--no-ghost", action="store_true",
                            help="do not draw the second, encoder-driven hand")
    run_parser.add_argument("--ghost-offset", type=float, default=0.24,
                            help="how far to the side the encoder hand sits; "
                                 "0 overlays it on the simulated one")
    run_parser.add_argument("--mimic", action="store_true",
                            help="start following a hand over the Leap "
                                 "immediately; the MIMIC tab does the same")
    run_parser.add_argument("--mimic-mode", default="keyvector",
                            choices=("raw", "angles", "tip", "keyvector", "posebank", "student"),
                            help="drive the joints from your joint angles, from "
                                 "the fingertip IK, or from the keyvector "
                                 "solver (needs .venv-dex; see the README)")
    run_parser.add_argument("--mimic-thumb", default="angles",
                            choices=("angles", "pinch"),
                            help="drive the robot thumb from your thumb's "
                                 "angles, or from the Leap's pinch estimate")
    run_parser.add_argument("--follow", default=None,
                            choices=("right", "left"),
                            help="which of your hands to follow when both are "
                                 "in view. Defaults to the one the robot IS, "
                                 "measured off the model")
    run_parser.add_argument("--mirror", action="store_true",
                            help="drive with the OPPOSITE hand: world motion "
                                 "then matches, at the cost of your index "
                                 "spreading the robot's index the other way")
    run_parser.add_argument("--tracking-mode", default="desktop",
                            choices=("desktop", "hmd", "screentop"),
                            help="desktop is the sensor flat, looking up")
    run_parser.add_argument("--leap-lib",
                            help="path to libLeapC.dylib, if the SDK is not "
                                 "where it installs itself")
    run_parser.add_argument("--skeleton-offset", type=float, nargs=3,
                            default=None, metavar=("X", "Y", "Z"),
                            help="where to draw your hand relative to the "
                                 "robot's palm; the default clears its "
                                 "fingertips. 0 0 0 overlays them")
    _hardware_flags(run_parser)
    run_parser.set_defaults(func=run)

    hw_parser = add("hw", "talk to the real hand over the U2D2")
    hw_parser.add_argument("action",
                           choices=("scan", "status", "watch", "zero", "sign",
                                    "id", "gains", "step", "relax", "hold",
                                    "grip"))
    hw_parser.add_argument("--joint", help="for `sign`: which joint to flip; "
                                          "for `zero`: re-zero only this joint")
    hw_parser.add_argument("--finger", default="index", choices=FINGERS,
                           help="for `grip`: which digit to close")
    hw_parser.add_argument("--close-to", type=float, default=1.2,
                           help="for `grip`: radians to close towards")
    hw_parser.add_argument("--close-speed", type=float, default=0.6,
                           help="for `grip`: rad/s the command closes at")
    hw_parser.add_argument("--out", default=None,
                           help="for `grip`: write the trace to this file")
    hw_parser.add_argument("--obstacle", type=float, default=None,
                           help="for `grip --mock-hardware`: put a virtual stop "
                                "this many radians into the close, to rehearse "
                                "the bench run without a hand")
    # `--seconds` is shared with `step`, below.
    hw_parser.add_argument("--group", choices=("flexion", "spread", "all"),
                           help="for `sign`: flip a whole set at once - the "
                                "twelve flexion actuators on this build are "
                                "mounted backwards together")
    hw_parser.add_argument("--value", type=int, default=-1,
                           help="for `sign`: 1 or -1. for `id`: the motor ID")
    hw_parser.add_argument("--swap", nargs=2, metavar=("JOINT", "JOINT"),
                           help="for `id`: exchange two joints' motor IDs")
    for field in ("position-p", "position-i", "position-d",
                  "profile-velocity", "profile-acceleration", "current-limit"):
        hw_parser.add_argument(f"--{field}", type=int, default=None,
                               help=f"for `gains`: set {field.replace('-', ' ')} "
                                    "on every motor")
    hw_parser.add_argument("--degrees", type=float, default=15.0,
                           help="for `step`: how far to move the joint")
    hw_parser.add_argument("--seconds", type=float, default=1.5,
                           help="for `step`: how long to record")
    _hardware_flags(hw_parser)
    hw_parser.set_defaults(func=run_hw)

    panel_parser = add("panel", "the pygame panel on its own")
    panel_parser.set_defaults(func=lambda a: _panel(a))

    pose_parser = add("pose", "hold a named pose")
    pose_parser.add_argument("name")
    pose_parser.add_argument("--headless", action="store_true")
    pose_parser.add_argument("--steps", type=int, default=400)
    pose_parser.set_defaults(func=run_pose)

    ik_parser = add("ik", "solve one fingertip target")
    ik_parser.add_argument("finger", choices=FINGERS)
    # A plain string metavar, not ("X", "Y", "Z"): argparse only formats a
    # tuple metavar for OPTIONALS, and on a positional it raises out of
    # `--help` itself - so `main.py ik --help` was the one help page that
    # could not be printed.
    ik_parser.add_argument("target", type=float, nargs=3, metavar="COORD",
                           help="fingertip target in world metres: X Y Z")
    ik_parser.set_defaults(func=run_ik)

    add("poses", "list the named poses").set_defaults(
        func=lambda a: (print("\n".join(POSES)), 0)[1])

    argv = list(sys.argv[1:] if argv is None else argv)
    known = set(sub.choices)
    if not argv or (argv[0] not in known and argv[0] not in ("-h", "--help")):
        argv = ["run"] + argv          # bare `python main.py [flags]` means run
    args = parser.parse_args(argv)
    return args.func(args)


def _panel(args):
    from handcontrol.panel import main as panel_main
    panel_main(args.command_port, args.state_port, args.hand,
               args.spread_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
