#!/usr/bin/env python
"""Mimic a hand over a Leap Motion with the h4nd_left, without the panel.

    mjpython leap_mimic.py                   Leap -> MuJoCo viewer
    python leap_mimic.py --mode tip          the same, through the fingertip IK
    python leap_mimic.py --print             angles in the terminal, no viewer
    python leap_mimic.py --udp               drive a main.py that is already up
    python leap_mimic.py --port PORT --arm   and move the real hand

`python main.py` has all of this in its MIMIC tab, with your hand drawn in the
viewer, and is the normal way in. This is the same machinery with no panel and
no UDP hop: one flag per setting, one loop, and `--print`, which needs no
simulator at all and is the fastest way to see whether the retargeting is
sane. Both drive `handcontrol/mimic.py`; neither has its own copy of it.

Start with `--print`. It puts the retargeted angle next to the human angle it
came from, joint by joint - which is where a wrong sign or a silly gain is
obvious, and where in the viewer it just looks like a finger doing something
strange.

Three fingers and a thumb against a human five, sixteen joints against a human
twenty-odd, limits of [-20, 110] degrees that no knuckle obeys: the mapping is
a resemblance, not a measurement. `handcontrol/retargeting/` says which angle
drives which joint and why.

  --mode angles (default) drives each robot joint from the matching human
  angle. Always answers, never surprises, and puts fingertips approximately
  nowhere - the segments are not the same lengths.

  --mode tip maps the fingertips into the robot's palm frame and lets the IK
  find the joints, after fitting scale and offset to your open hand. Pinches
  land where you meant them, to a couple of millimetres. It needs the
  simulator, so `--print` shows its targets rather than solving them and
  `--udp` refuses it. It solves the three fingers only: the robot's thumb
  cannot reach where a human thumb goes - see `mimic.TIP_FINGERS` - so the
  thumb stays on the angle mapping unless you ask for `--tip-thumb ik`.

THE HARDWARE IS OFF unless `--port` and `--arm` are both given. `--arm`
energises sixteen servos that will then follow a hand waved over a camera:
have `python main.py hw relax` ready in another terminal.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from handcontrol.hardware import (BAUDRATE, CURRENT_LIMIT_MA, MAX_SPEED,
                                  DynamixelHand, HardwareOutput, MockHand)
from handcontrol.mimic import TIP_FINGERS, Mimic
from handcontrol.robot.model import (DEFAULT_HAND, FINGERS, ROLES, HandModel,
                               ensure_urdf)
from handcontrol.net import COMMAND_PORT, Channel
from handcontrol.retargeting import GAINS


def open_link(args, joint_names):
    """A connected hardware link, or None. Never fatal: this session is worth
    having with no hand attached, and the port is the part most likely to be
    wrong."""
    port = "mock" if args.mock_hardware else args.port
    if not port:
        return None
    try:
        if port == "mock":
            return MockHand(joint_names).connect()
        return DynamixelHand(port, joint_names, baudrate=args.baud,
                             current_ma=args.current_ma).connect()
    # Broad on purpose: a port that is gone, busy, or held by another program
    # raises out of pyserial rather than the DYNAMIXEL SDK.
    except Exception as failure:
        print(f"[leap] hardware unavailable on {port}: {failure}")
        return None


def build_mimic(args, model):
    gains = dict(GAINS)
    for entry in args.gain or []:
        role, _, value = entry.partition("=")
        if role not in gains:
            raise SystemExit(f"unknown gain {role!r}; known: {', '.join(gains)}")
        gains[role] = float(value)
    mimic = Mimic(model, mode=args.mode, thumb=args.thumb, follow=args.follow,
                  on_lost=args.on_lost, tip_thumb=args.tip_thumb,
                  tracking_mode=args.tracking_mode, library=args.leap_lib,
                  gains=gains, smoothing=args.smoothing, max_rate=args.max_rate,
                  scale=args.scale, calibrate=args.calibrate, mirror=args.mirror)
    print(f"[leap] this is a {mimic.retarget.chirality()} hand - hold your "
          f"{mimic.expected.upper()} hand over the sensor"
          + (" (mirrored)" if args.mirror else ""))
    return mimic


# ------------------------------------------------------------------- output
def record_loop(args, mimic):
    """Dump raw frames to JSON, for working out why a mapping looks wrong.

    Bone positions exactly as the SDK reports them, plus the angles this code
    measures from them. With both in one file the question "is the sensor
    wrong or is the retargeting wrong?" is answerable away from the bench,
    which is the only way anyone but the operator can debug the mapping.
    """
    import json

    frames = []
    deadline = time.time() + args.seconds
    print(f"[leap] recording {args.seconds:.0f}s to {args.record} - "
          f"move your hand through the poses you care about.")
    while time.time() < deadline:
        hand = mimic.poll()
        if hand is None:
            continue
        angles = mimic.retarget.measure(hand)
        frames.append({
            "type": hand.type,
            "palm": hand.palm.tolist(),
            "normal": hand.normal.tolist(),
            "direction": hand.direction.tolist(),
            "pinch": hand.pinch_strength, "grab": hand.grab_strength,
            "bones": {d: p.tolist() for d, p in hand.bones.items()},
            "angles": {f"{f}.{role}": v for (f, role), v in angles.items()},
        })
        time.sleep(1.0 / args.rate)
    path = args.record
    with open(path, "w") as handle:
        json.dump({"hand": args.hand, "follow": args.follow,
                   "frames": frames}, handle)
    print(f"[leap] wrote {len(frames)} frames to {path}")
    return 0


def print_loop(args, mimic):
    """No simulator: the angles, as text, refreshed in place.

    Under `--mode tip` there is no IK to run without a model, so it prints the
    fingertip targets instead - which is the half of tip mode that can be
    wrong on its own.
    """
    retarget = mimic.retarget
    print("[leap] printing. ctrl-c to stop.")
    last = 0.0
    try:
        while True:
            hand = mimic.poll()
            now = time.time()
            if now - last < 1.0 / args.print_hz:
                time.sleep(0.002)
                continue
            last = now
            if hand is None:
                print(f"\033[H\033[J[leap] no hand over the sensor "
                      f"({mimic.frames} frames, {mimic.lost} without one)")
                continue
            if args.mode == "tip":
                targets = retarget.tips(hand, origin=args.position)
                rows = [f"{f:8s} target {np.round(targets[f] * 1000, 1)} mm"
                        + ("" if f in TIP_FINGERS or args.tip_thumb == "ik"
                           else "   (not solved - it cannot reach)")
                        for f in FINGERS]
                rows.append("" if retarget.calibrated else
                            "not fitted yet - hold your hand flat and open")
            else:
                angles = retarget.measure(hand)
                q = retarget.joints(hand)
                rows = []
                for finger in FINGERS:
                    parts = []
                    for role in ROLES:
                        name = retarget.joint(finger, role)
                        parts.append(
                            f"{role:7s} "
                            f"{math.degrees(angles[finger, role]):6.1f}"
                            f"->{math.degrees(q[retarget.index[name]]):6.1f}")
                    rows.append(f"{finger:8s} " + "  ".join(parts))
            warning = ([f"!! that is your {hand.type.upper()} hand and this "
                        f"robot is a {mimic.expected.upper()} one - it will "
                        f"appear to move the wrong way. Swap hands, or "
                        f"--mirror.", ""] if mimic.mismatched else [])
            print("\033[H\033[J" + "\n".join([
                f"[leap] {hand.type} hand   pinch {hand.pinch_strength:.2f}   "
                f"grab {hand.grab_strength:.2f}   frames {mimic.frames}", ""]
                + warning + rows
                + ["", "human angle -> commanded joint, degrees"]))
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def udp_loop(args, mimic):
    """Drive a `main.py` that is already up, over its command port.

    Joint vectors, not `tip` messages, even in tip mode: main.py applies one
    tip message at a time against its own last target, so four in a frame
    leaves three fingers behind. The IK belongs to whoever owns the model, so
    tip mode is refused here rather than half done - and `main.py --mimic`
    does the whole job in one process anyway.
    """
    if args.mode == "tip":
        raise SystemExit("--mode tip needs the simulator for its IK. Use "
                         "`python main.py --mimic --mimic-mode tip`, or drop "
                         "--udp, or use --mode angles.")
    link = Channel(send_to=args.command_port)
    print(f"[leap] sending joint targets to 127.0.0.1:{args.command_port}. "
          f"ctrl-c to stop.")
    try:
        while True:
            q = mimic.step(dt=1.0 / args.rate)
            link.send({"type": "joints", "q": list(map(float, q))})
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        link.close()
    return 0


def sim_loop(args, mimic):
    """The simulator, the viewer, and optionally the hand on the bench."""
    from handcontrol.robot.options import Physics, Scene
    from handcontrol.robot.sim import HandSim

    sim = HandSim(hand=args.hand, spread_limit=args.spread_limit,
                  physics=Physics(gravity=not args.no_gravity),
                  scene=Scene(show_viewer=not args.headless,
                              position=tuple(args.position),
                              ghost=bool(args.port or args.mock_hardware))).build()

    hardware = HardwareOutput(open_link(args, sim.joint_names),
                              len(sim.joint_names), max_speed=args.max_speed)
    if hardware.available:
        hardware.enable(True)
        if args.arm:
            hardware.arm(True)
            print("[leap] HARDWARE ARMED - the servos are live and following "
                  "your hand. `python main.py hw relax` cuts torque.")
        else:
            print("[leap] hardware attached but not armed; --arm energises it")

    last_draw = last_report = 0.0
    print("[leap] running. close the viewer window to stop.")
    try:
        last_frame = None
        while sim.viewer_alive():
            # wall clock, not the physics timestep - see main.py's run loop
            now = time.time()
            frame_dt = min(now - (last_frame or now - sim.dt), 0.05)
            last_frame = now
            q = mimic.step(sim, max(frame_dt, sim.dt))
            sim.command(q)
            hardware.command(q)
            if hardware.positions is not None:
                sim.set_ghost(hardware.positions)
            sim.step()

            # Rebuilt and re-uploaded each time, so drawn on a clock rather
            # than every step - 30 Hz reads as continuous and leaves the
            # physics the frame it is entitled to.
            now = time.time()
            if now - last_draw > 1.0 / args.skeleton_hz:
                last_draw = now
                sim.draw_debug(targets=mimic.tip_targets,
                               skeleton=mimic.skeleton(sim.position))
            if args.mode == "tip" and mimic.tip_errors \
                    and now - last_report > 1.0:
                last_report = now
                # The miss is the honest part: an unreachable target does not
                # fail, it comes back as the closest approach.
                print("[leap] tip miss " + "  ".join(
                    f"{f} {e * 1000:5.1f}mm" for f, e in mimic.tip_errors.items()))
    except KeyboardInterrupt:
        pass
    finally:
        mimic.close()
        if hardware.armed:
            # Left holding, as main.py does: cutting torque on the way out
            # drops whatever the hand is gripping.
            print("[leap] hardware still armed - `main.py hw relax` to release")
        hardware.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:9]))

    parser.add_argument("--mode", default="keyvector",
                        choices=("raw", "angles", "tip", "keyvector", "posebank", "student"),
                        help="drive the joints from your joint angles, from the "
                             "fingertip IK, or from the keyvector solver "
                             "(needs .venv-dex; see the README)")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="no simulator: print the retargeted angles")
    parser.add_argument("--udp", action="store_true",
                        help="send joint targets to a running main.py instead "
                             "of simulating here")
    parser.add_argument("--record", metavar="FILE",
                        help="record raw frames and measured angles to JSON, "
                             "for diagnosing a mapping that looks wrong")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to record for")
    parser.add_argument("--print-hz", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=60.0,
                        help="send rate for --udp")

    leap = parser.add_argument_group("leap")
    leap.add_argument("--tracking-mode", default="desktop",
                      choices=("desktop", "hmd", "screentop"),
                      help="desktop is the sensor flat on the bench looking up")
    leap.add_argument("--follow", default=None, choices=("right", "left"),
                      help="which of your hands to follow when both are in "
                           "view. Defaults to the one the robot IS, measured "
                           "off the model")
    leap.add_argument("--mirror", action="store_true",
                      help="drive with the OPPOSITE hand: world motion then "
                           "matches, at the cost of your index spreading the "
                           "robot's index the other way")
    leap.add_argument("--leap-lib", help="path to libLeapC.dylib, if the SDK "
                                         "is not where it installs itself")
    leap.add_argument("--wait", type=float, default=5.0,
                      help="seconds to wait for the camera before starting "
                           "anyway")

    mapping = parser.add_argument_group("retargeting")
    mapping.add_argument("--thumb", default="angles", choices=("angles", "pinch"),
                         help="drive the robot thumb from your thumb's angles, "
                              "or from the Leap's pinch estimate, which "
                              "survives the thumb being occluded")
    mapping.add_argument("--tip-thumb", default="angles", choices=("angles", "ik"),
                         help="under --mode tip, whether to send the thumb "
                              "through the IK too. It cannot reach where a "
                              "human thumb goes, so by default it is not.")
    mapping.add_argument("--gain", action="append", metavar="ROLE=VALUE",
                         help="robot radians per human radian, by role: "
                              + ", ".join(GAINS))
    mapping.add_argument("--smoothing", type=float, default=0.35,
                         help="per-frame blend toward the new pose; lower is "
                              "steadier and later")
    mapping.add_argument("--max-rate", type=float, default=6.0,
                         help="rad/s ceiling on the commanded angles")
    mapping.add_argument("--scale", type=float, default=1.0,
                         help="multiplier on the tip mapping, to reach further "
                              "than your hand does")
    mapping.add_argument("--no-calibrate", dest="calibrate",
                         action="store_false",
                         help="skip fitting the neutral pose to your open hand")
    mapping.add_argument("--on-lost", default="hold", choices=("hold", "open"),
                         help="what to do when tracking drops: hold the last "
                              "pose, or unroll the fingers")

    robot = parser.add_argument_group("robot")
    robot.add_argument("--hand", default=DEFAULT_HAND, choices=("left", "right"))
    robot.add_argument("--spread-limit", type=float, default=None, metavar="DEG")
    robot.add_argument("--headless", action="store_true")
    robot.add_argument("--no-gravity", action="store_true")
    robot.add_argument("--position", type=float, nargs=3,
                       default=(0.0, 0.0, 0.20))
    robot.add_argument("--command-port", type=int, default=COMMAND_PORT)
    robot.add_argument("--skeleton-hz", type=float, default=30.0,
                       help="how often to redraw your hand in the viewer")

    bench = parser.add_argument_group("hardware")
    bench.add_argument("--port", help="U2D2 serial port")
    bench.add_argument("--baud", type=int, default=BAUDRATE)
    bench.add_argument("--current-ma", type=int, default=CURRENT_LIMIT_MA)
    bench.add_argument("--max-speed", type=float, default=MAX_SPEED)
    bench.add_argument("--mock-hardware", action="store_true")
    bench.add_argument("--arm", action="store_true",
                       help="energise the servos. They will follow your hand.")

    args = parser.parse_args(argv)

    mimic = build_mimic(args, HandModel.load(ensure_urdf(args.hand,
                                                         args.spread_limit)))
    if not mimic.enable(True):
        print(f"[leap] {mimic.error}")
        return 1
    if mimic.connection is not None and not mimic.connection.wait_for_device(
            args.wait):
        print(f"[leap] {mimic.connection.no_device_reason()} - carrying on; "
              "it will pick one up if it appears.")
    try:
        if args.record:
            return record_loop(args, mimic)
        if args.print_only:
            return print_loop(args, mimic)
        if args.udp:
            return udp_loop(args, mimic)
        return sim_loop(args, mimic)
    finally:
        mimic.close()


if __name__ == "__main__":
    sys.exit(main())
