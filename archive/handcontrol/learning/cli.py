"""Argument parsing for the learned-retargeting pipeline.

One file, nine subcommands, and no logic: everything here reads flags, calls
into the modules beside it, and prints. That split is the same one
`leap_mimic.py` has from `mimic.py`, and it is what lets the pipeline be
driven from a notebook or a test without going through a command line.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..robot.model import FINGERS, HandModel, ensure_urdf
from ..leap import DIGITS, LeapConnection, LeapError
from ..retargeting import Retargeter
from . import report as reporting
from . import replay as playback
from . import survey as recording
from . import teacher as offline
from . import validate
from .noise import analyse
from ..paths import MODELS, NOISE_FIT
from . import objective
from . import geometry as distance_table   # `geometry` is a sampler dict in run_gestures
from .sessions import (DATA, finish_keypoints, fit_bone_offsets, raw_synthetic,
                       fit_proportions, hands_from, keypoints_for, load,
                       sessions_under)
from .synthetic import DIVERSITY, sampler_for


SYNTHETIC = DATA.parent / "synthetic"


def show(number, total, block, frames, done, all_frames):
    """Progress for a run-by-run solve, one line rewritten in place."""
    print(f"\r    [{number}/{total}] {block:<15} {frames:6d} frames   "
          f"{done}/{all_frames} done", end="", flush=True)


# Solver steps per synthetic run. The recorded session gets objective.STEPS
# (600); a generated hand starts from a warm start that already fits it and
# has no sensor noise to fight, so it settles sooner. Set from a measurement
# on the same six hands - see the README's synthetic section.
SYNTH_STEPS = 300

def add_arguments(parsers):
    """Wire every subcommand onto an argparse subparser group."""
    survey = parsers.add_parser(
        "survey", help="record a scripted survey of the sensor (Phase 0)")
    survey.add_argument("--operator", default="")
    survey.add_argument("--hand", default="left", choices=("left", "right"),
                        help="which of YOUR hands to record")
    survey.add_argument("--out", default=None)
    survey.add_argument("--blocks", default="")
    survey.add_argument("--list", action="store_true")
    survey.add_argument("--append", metavar="DIR", default=None,
                        help="add to an existing session; `latest` for the "
                             "most recent")
    survey.add_argument("--missing", action="store_true",
                        help="with --append, only the blocks it lacks")
    survey.add_argument("--tracking-mode", default="desktop",
                        choices=("desktop", "hmd", "screentop"))
    survey.add_argument("--library", default=None)
    survey.set_defaults(handler=run_survey)

    noise = parsers.add_parser(
        "noise", help="fit the sensor's error model from recordings")
    noise.add_argument("sessions", nargs="*")
    noise.add_argument("--out", default=None)
    noise.add_argument("--no-plot", action="store_true")
    noise.set_defaults(handler=run_noise)

    teacher = parsers.add_parser(
        "teacher", help="solve recorded sessions offline, for labels (Phase 1)")
    teacher.add_argument("sessions", nargs="*")
    teacher.add_argument("--robot-hand", default="left",
                         choices=("left", "right"),
                         help="which hand the ROBOT is, as main.py means it")
    teacher.add_argument("--spread-limit", type=float, default=None)
    teacher.add_argument("--smooth-window", type=int, default=21)
    teacher.add_argument("--maxeval", type=int, default=2000)
    teacher.add_argument("--limit", type=int, default=0)
    teacher.add_argument("--blocks", default="",
                         help="solve only these blocks. NOTE: the result "
                              "REPLACES teacher.npz rather than merging into "
                              "it, so a partial run leaves partial labels - "
                              "pass --out to keep them separate")
    teacher.add_argument("--out", default=None,
                         help="write teacher.npz/json here instead of into the "
                              "session - use it with --blocks")
    teacher.set_defaults(handler=run_teacher)

    synth = parsers.add_parser(
        "synth", help="synthetic hands through the teacher (Phase 2)")
    synth.add_argument("sessions", nargs="*",
                       help="sessions to fit the template from")
    synth.add_argument("--hands", type=int, default=100)
    synth.add_argument("--seconds", type=float, default=6.0)
    synth.add_argument("--hz", type=float, default=125.0)
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument("--robot-hand", default="left", choices=("left", "right"))
    synth.add_argument("--maxeval", type=int, default=2000)
    synth.add_argument("--steps", type=int, default=SYNTH_STEPS,
                       help="solver steps per run (default %(default)s; the "
                            "recorded session uses objective.STEPS)")
    synth.add_argument("--out", default=None)
    synth.add_argument("--diversity", default="measured",
                       choices=tuple(DIVERSITY),
                       help="how independent the sampler lets the fingers be. "
                            "'measured' fits the recording and so cannot "
                            "reach a shape the operator never made; 'wide' "
                            "oversamples those on purpose")
    synth.add_argument("--object-rate", type=float, default=None,
                       metavar="FRACTION",
                       help="how often a keyframe closes on a cylinder "
                            "(synthetic.OBJECT_RATE). 0 removes objects from "
                            "the labels; MEASURED AND WORSE - three seeds "
                            "score 14.70 deg against 13.02 with objects kept, "
                            "and 17-19 on contact frames against 13-15")
    synth.add_argument("--pinch", type=float, default=None, metavar="RATE",
                       help="share of keyframes solved to a thumb-fingertip "
                            "contact (default PINCH_RATE, 0.3). The recording "
                            "touches on 8%% of frames; the student needs more")
    synth.add_argument("--compare", action="store_true",
                       help="only check the sample against the real hands")
    synth.set_defaults(handler=run_synth)

    student = parsers.add_parser(
        "train", help="train the student on the pairs (Phase 3)")
    student.add_argument("--synthetic", default=None,
                         help="pairs.npz (default: data/synthetic/seed0)")
    student.add_argument("--session", default=None,
                         help="session with teacher.npz (default: newest)")
    student.add_argument("--out", default=None,
                         help="where to write the model (default data/models/student.pt)")
    student.add_argument("--epochs", type=int, default=12)
    student.add_argument("--batch", type=int, default=256)
    student.add_argument("--lr", type=float, default=2e-3)
    student.add_argument("--basis", type=int, default=None,
                         help="PCA components on the output. 16 is the default "
                              "and the identity; 10 is the constrained basis, "
                              "kept for the comparison")
    student.add_argument("--channels", type=int, default=None)
    student.add_argument("--warmup", type=int, default=0, metavar="EPOCHS",
                         help="ramp the learning rate from zero over this many "
                              "pretraining epochs. A wider network (--channels "
                              "192) opened far deeper in self-collision than the "
                              "default and one seed of two never recovered")
    student.add_argument("--direct", type=float, default=None,
                         metavar="WEIGHT",
                         help="weight on the TEACHER'S OWN OBJECTIVE, scored on "
                              "the student's pose (train.W_DIRECT). 0 is pure "
                              "distillation; the label stays as a trust region "
                              "either way. The objective sits near 40 where the "
                              "other terms sit near 0.05, so try 0.01-0.05")
    student.add_argument("--belief", action="store_true",
                         help="feed the sensor's own fitted error as 21 more "
                              "channels, and train a head that predicts the "
                              "network's own joint error. Changes the input "
                              "width; the checkpoint records it, so it loads "
                              "and plays back without the flag")
    student.add_argument("--contact-gate", action="store_true",
                         help="suppress the belief channels as a pinch closes "
                              "(belief.contact_gate). Implies --belief. The "
                              "channels distrust a folded digit, and a folded "
                              "digit is usually one about to touch")
    student.add_argument("--intent", action="store_true",
                         help="add the 3-D grasp head and its decoder, "
                              "so the live path can hand "
                              "authority from the fingertips to the grasp where "
                              "the sensor stops being worth believing. Implies "
                              "--belief: the handover is gated on sigma")
    student.add_argument("--seed", type=int, default=0,
                         help="weight init, shuffling, dropout and the noise "
                              "draw. Two seeds of one setting say how much of "
                              "a difference between settings is the setting")
    student.add_argument("--robot-hand", default="left", choices=("left", "right"),
                         help="which robot the labels were solved for; the FK, "
                              "the joint limits and the collision proxy all come from it")
    student.add_argument("--device", default=None)
    student.add_argument("--workers", type=int, default=0)
    student.add_argument("--accel", type=float, default=None,
                         help="weight on the student's own roughness. Already "
                              "0 by default - measured worse at every weight "
                              "tried; this is here to try it again")
    student.add_argument("--pair", type=float, default=None,
                         help="weight on thumb-to-finger distances against the "
                              "teacher's (default train.W_PAIR); 0 turns it off")
    student.add_argument("--no-noise", action="store_true",
                         help="skip the measured sensor noise on synthetic "
                              "input - the Phase 4 ablation")
    student.set_defaults(handler=run_train)

    probe = parsers.add_parser(
        "gestures", help="hand-authored gestures the student never trained on")
    probe.add_argument("sessions", nargs="*",
                       help="sessions to fit the hand template from")
    probe.add_argument("--student", default=str(MODELS / "student.pt"))
    probe.add_argument("--robot-hand", default="left", choices=("left", "right"))
    probe.add_argument("--maxeval", type=int, default=2000)
    probe.add_argument("--sequence", default="probe",
                       choices=("probe", "slow_curl", "wrap"),
                       help="which authored sequence (default probe; slow_curl is the "
                            "continuity test, run it with --move 3)")
    probe.add_argument("--hold", type=float, default=0.8)
    probe.add_argument("--wrap-radius", type=float, default=None, metavar="MM",
                       help="cylinder radius the wrap sequence is authored for (default gestures.WRAP_RADIUS_MM)")
    probe.add_argument("--move", type=float, default=0.9)
    probe.add_argument("--hz", type=float, default=125.0)
    probe.add_argument("--out", default=None,
                       help="default data/synthetic/gestures")
    probe.set_defaults(handler=run_gestures)

    checks = parsers.add_parser(
        "selftest", help="check the invariants this package keeps breaking")
    checks.add_argument("--robot-hand", default="left",
                        choices=("left", "right"))
    checks.set_defaults(handler=run_selftest)

    play = parsers.add_parser(
        "replay", help="play a solved session in the MuJoCo viewer")
    play.add_argument("session", nargs="?", default=None)
    play.add_argument("--synthetic", default=None)
    play.add_argument("--blocks", default="")
    play.add_argument("--speed", type=float, default=1.0)
    play.add_argument("--loop", action="store_true")
    play.add_argument("--robot-hand", default="left", choices=("left", "right"))
    play.add_argument("--skeleton-offset", type=float, nargs=3,
                      default=playback.SKELETON_OFFSET)
    play.add_argument("--student", nargs="?",
                      const=str(MODELS / "student.pt"),
                      default=None,
                      help="drive the hand from a trained student instead of "
                           "the teacher (default data/models/student.pt)")
    play.add_argument("--future", type=int, default=0, metavar="FRAMES",
                      help="add a lane driven by the student's forward head, "
                           "N frames ahead (needs --student; 5, 10 or 15)")
    play.add_argument("--scaled-skeleton", action="store_true",
                      help="draw the hand the SOLVER sees - each digit "
                           "stretched to robot proportions - instead of the "
                           "operator's own")
    play.add_argument("--video", metavar="PATH",
                      help="render the span to an mp4 instead of opening a "
                           "window - same lanes, same camera, no viewer")
    play.add_argument("--fps", type=int, default=30,
                      help="frame rate for --video (default 30)")
    play.add_argument("--live-lane", dest="ghost", default="live",
                      choices=("live", "none"),
                      help="whether to stand the live solver's answer in the "
                           "row (the teacher's is always there)")
    play.set_defaults(handler=run_replay)

    wall = parsers.add_parser(
        "gallery", help="a grid of hands, each replaying a different sequence")
    wall.add_argument("target", nargs="?", default=None,
                      help="a pair file's directory or a session; default the "
                           "latest synthetic set")
    wall.add_argument("--hands", type=int, default=16)
    wall.add_argument("--pitch", type=float, default=None, help="metres between cells")
    wall.add_argument("--speed", type=float, default=1.0)
    wall.add_argument("--seed", type=int, default=0, help="which runs are picked")
    wall.add_argument("--robot-hand", default="left", choices=("left", "right"))
    wall.add_argument("--png", default=None,
                      help="render one frame headless to this file instead of "
                           "opening a window")
    wall.add_argument("--frame", type=int, default=0, help="which frame, with --png")
    wall.add_argument("--once", action="store_true", help="stop after one pass")
    wall.set_defaults(handler=run_gallery)


def run_survey(args, parser):

    if args.list:
        total = sum(b.seconds for b in recording.BLOCKS)
        print(f"\n{len(recording.BLOCKS)} blocks, {total / 60:.1f} minutes of recording\n")
        for block in recording.BLOCKS:
            print(f"  {block.key:<15} {block.seconds:5.0f}s  "
                  f"{block.phase:<7} {block.title}")
        print()
        return 0

    chosen = recording.BLOCKS
    if args.blocks:
        keys = [k.strip() for k in args.blocks.split(",") if k.strip()]
        unknown = [k for k in keys if k not in recording.BY_KEY]
        if unknown:
            parser.error(f"unknown block(s): {', '.join(unknown)}")
        chosen = tuple(recording.BY_KEY[k] for k in keys)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    resumed, resumed_meta = None, None
    if args.append:
        target = args.append
        if target == "latest":
            sessions = sorted(p for p in DATA.glob("*")
                              if (p / "session.npz").exists())
            if not sessions:
                parser.error(f"no sessions under {DATA} to append to")
            target = sessions[-1]
        resumed, resumed_meta = recording.Recorder.load(target)
        if resumed is None:
            parser.error(f"{target} is not a session directory")
        out = Path(target)
        done = {s["key"] for s in resumed_meta.get("recorded", [])}
        print(f"\n  appending to {out}")
        print(f"  already recorded: {', '.join(sorted(done)) or 'nothing'}")
        if args.missing:
            chosen = tuple(b for b in chosen if b.key not in done)
            if not chosen:
                print("  nothing missing - session is complete.")
                return 0
        # A resumed session keeps the operator and hand it was recorded with.
        # Relabelling half a file from a flag typed today would put two
        # people's noise, or two chiralities, under one name.
        for field in ("operator", "hand"):
            stored = resumed_meta.get(field)
            if stored and getattr(args, field) != stored:
                print(f"  session was recorded with {field}={stored!r}; "
                      f"keeping that")
                setattr(args, field, stored)
    else:
        out = Path(args.out) if args.out else DATA / stamp

    print("\n" + "=" * 70)
    print("  PHASE 0 - LEAP SURVEY")
    print("=" * 70)
    print(f"  operator   {args.operator or '(unnamed)'}")
    print(f"  hand       {args.hand}")
    print(f"  blocks     {len(chosen)}, "
          f"{sum(b.seconds for b in chosen) / 60:.1f} min of recording")
    print(f"  writing to {out}")
    print("\n  Nothing here is filtered. Raw SDK output only.")
    print("  Sensor flat on the bench, hand about 20 cm above it.")

    try:
        connection = LeapConnection(mode=args.tracking_mode,
                                    library=args.library).open()
    except LeapError as failure:
        print(f"\n  cannot open the sensor: {failure}")
        return 1
    with connection:
        if not connection.wait_for_device(5.0):
            print(f"\n  {connection.no_device_reason()}")
            return 1
        if not recording.sensor_check(connection, args.hand):
            return 1

        recorder = resumed if resumed is not None else recording.Recorder()
        meta = resumed_meta if resumed_meta is not None else {
                "operator": args.operator, "hand": args.hand,
                "tracking_mode": args.tracking_mode,
                "started": datetime.now().isoformat(timespec="seconds"),
                "filtered": False,
                "digits": list(DIGITS),
                "bones_layout": "(frame, digit, 5 joint positions, xyz) mm, "
                                "device frame; index 4 is the fingertip",
                "blocks": [b.key for b in recording.BLOCKS],
                "recorded": []}

        for number, block in enumerate(chosen, start=1):
            while True:
                choice = recording.prompt(block, number, len(chosen))
                if choice == "quit":
                    print("\n  Stopping here. Keeping what is recorded.")
                    recording.save(out, recorder, meta)
                    return 0
                if choice == "skip":
                    break
                recording.countdown()
                # Numbered against THIS session's own table, not today's.
                # Every consumer downstream turns a name into a row with
                # `meta["blocks"].index(...)`, so a block appended to an older
                # recording has to take the number that table gives it. Keyed
                # on `recording.BLOCKS` instead, appending the three blocks
                # added after the first session was recorded wrote them as 6,
                # 7 and 8 - which that session's table calls spread_cycle,
                # finger_isolate and thumb_sweep - and every later run would
                # have trained on frames labelled as three different blocks.
                if block.key not in meta["blocks"]:
                    meta["blocks"].append(block.key)
                stats = recording.record_block(connection, recorder, block,
                                     meta["blocks"].index(block.key), args.hand)
                meta["recorded"].append(stats)
                answer = recording.after()
                if answer == "quit":
                    # Ctrl-C at the redo prompt used to fall through to the
                    # next block, because only "redo" was tested for.
                    print("\n  Stopping here. Keeping what is recorded.")
                    recording.save(out, recorder, meta)
                    return 0
                if answer == "redo":
                    # Drop the redone block's rows rather than keeping both.
                    # Two takes of one block at one label is a set with an
                    # unannounced weighting in it.
                    del recorder.rows[stats["row_start"] - recorder.loaded:]
                    meta["recorded"].pop()
                    continue
                break

        print("\n" + "=" * 70)
        print("  DONE")
        print("=" * 70)
        for stats in meta["recorded"]:
            print(f"  {stats['key']:<15} {stats['frames']:6d} frames  "
                  f"{stats['hz']:6.1f} Hz  {stats['tracked_pct']:5.1f}% tracked")
        recording.save(out, recorder, meta)
    return 0




def run_noise(args, parser):

    paths = [Path(p) for p in args.sessions] or sessions_under()
    if not paths:
        print(f"no sessions under {DATA} - record one with `learn.py survey`")
        return 1

    data, metas = load(paths)
    # Pooling across chirality would fit one model to two hands and describe
    # neither; across operators it fits the average of two people's noise.
    hands = {m["hand"] for m in metas}
    if len(hands) > 1:
        print(f"  refusing to pool {'/'.join(sorted(hands))} hands - "
              f"run them separately")
        return 1
    operators = {m.get("operator", "") for m in metas}
    if len(operators) > 1:
        print(f"  NOTE: pooling {len(operators)} operators "
              f"({', '.join(sorted(operators))}); this fit is their average.")

    fit = analyse(data, metas[0])
    fit["sessions"] = [m["path"] for m in metas]
    fit["operators"] = sorted(operators)

    print(f"\n  pooling {len(paths)} session(s):")
    for meta in metas:
        print(f"    {meta['path']}  {len(meta.get('recorded', []))} blocks  "
              f"{meta.get('operator') or 'unnamed'}")
    recorded = {s["key"] for m in metas for s in m.get("recorded", [])}
    missing = [b for b in metas[0]["blocks"] if b not in recorded]
    if missing:
        print(f"  NOT RECORDED: {', '.join(missing)}  -  "
              f"learn.py survey --append latest --missing")

    reporting.report(fit, metas[0]["digits"])
    out = Path(args.out) if args.out else NOISE_FIT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fit, indent=2))
    print(f"  fit written to {out}")
    if not args.no_plot:
        reporting.plot(fit, data, metas[0], out.with_suffix(".png"))
    return 0


def run_teacher(args, parser):
    offline.SMOOTH_WINDOW = args.smooth_window

    paths = [Path(p) for p in args.sessions] or sessions_under()
    if not paths:
        print(f"no sessions under {DATA} - record one with `learn.py survey`")
        return 1

    model = HandModel.load(ensure_urdf(args.robot_hand, args.spread_limit))
    retarget = Retargeter(model=model)

    for path in paths:
        print(f"\n=== {path}")
        data, metas = load([path])
        meta = metas[0]
        expected = retarget.chirality()
        if meta["hand"] != expected:
            print(f"  NOTE: this robot is a {expected} hand and the session "
                  f"was recorded with the {meta['hand']}. Solving anyway - the "
                  f"mapping stays anatomical - but check --robot-hand.")

        result = offline.solve_session(
            data, meta, model, retarget, FINGERS,
            blocks=[b for b in args.blocks.split(",") if b] or None,
            limit=args.limit, maxeval=args.maxeval, progress=show)
        if result is None:
            print("  no runs long enough to solve")
            continue
        points = result["points"]
        print(f"\r    {result['runs']} runs, {len(points)} frames solved"
              + " " * 30)
        print("  digit scale: " + "  ".join(
            f"{k} {v:.3f}" for k, v in result["digit_scale"].items()))

        print("  self-collision (MuJoCo, past as-built): " + "  ".join(
            f"{n}: {v}" for n, v in (("penetration", f"{result['metrics'][1]['penetration_pct']}%"),
                                       ("worst", f"{result['metrics'][1]['worst_penetration_mm']}mm"))))
        # The thumb is printed split rather than pooled - see `thumb_free_deg`
        # in `teacher.measure`. Pooled, it moves with how much of the block
        # touches, which reads as the mapping changing when it has not.
        print(f"\n  {'':<18}{'fingers':>9}{'thumb free':>12}{'thumb touch':>13}"
              f"{'contact':>9}{'gap@touch':>11}{'jerk':>12}{'max step':>10}{'ms/frame':>10}")
        for row in result["metrics"]:
            contact = f"{row['contact_pct']:.0f}%" if row["contact_pct"] is not None else "-"
            gap = f"{row['gap_at_touch_mm']:.1f}mm" if row["gap_at_touch_mm"] is not None else "-"
            thumb = lambda key: (f"{row[key]:.1f}d" if row.get(key) is not None else "-")
            print(f"  {row['name']:<18}{row['posture_deg']:>8.2f}d"
                  f"{thumb('thumb_free_deg'):>12}{thumb('thumb_contact_deg'):>13}"
                  f"{contact:>9}{gap:>11}{row['jerk']:>12.3g}{row['worst_step_deg']:>9.2f}d"
                  f"{row['ms_per_frame']:>10.2f}")
        live, taught = result["metrics"]
        print("  contact by finger: " + "  ".join(
            f"{f} {v:.0f}%" for f, v in taught["contact_by_finger"].items() if v is not None))
        print(f"\n  headroom: {live['posture_deg'] - taught['posture_deg']:+.2f} deg fingers, "
              f"contact {live['contact_pct'] or 0:.0f}% -> {taught['contact_pct'] or 0:.0f}%, "
              f"jerk x{live['jerk'] / max(taught['jerk'], 1e-9):.1f}")

        out = Path(args.out) / "teacher.npz" if args.out \
            else path / "teacher.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, q_star=result["q_star"].astype(np.float32),
            q_live=result["q_live"].astype(np.float32),
            keypoints=points.astype(np.float32),
            keypoints_raw=result["raw"].astype(np.float32),
            pinch_strength=result["intent"].astype(np.float32),
            # The hand as distances, and the one label only a teacher can
            # compute: seconds until each cell next touches. See geometry.py.
            distances=result["distances"].astype(np.float32),
            time_to_contact=result["time_to_contact"].astype(np.float32),
            row=result["rows"].astype(np.int64),
            run_bounds=result["bounds"],
            block=data["block"][result["rows"]],
            t=data["t"][result["rows"]],
            joint_names=np.array(result["joint_names"]),
            digit_scale=np.array([result["digit_scale"].get(f, 1.0)
                                  for f in FINGERS]),
            fingers=np.array(list(FINGERS)))
        (out.with_suffix(".json")).write_text(json.dumps(
            {"session": str(path), "frames": int(len(points)),
             "hz": round(1 / result["dt"], 1), "robot_hand": args.robot_hand,
             "maxeval": args.maxeval, "smooth_window": args.smooth_window,
             "digit_scale": result["digit_scale"],
             "metrics": result["metrics"]}, indent=2))
        print(f"  wrote {out} ({len(points)} frames)")
    return 0


def run_synth(args, parser):

    paths = [Path(p) for p in args.sessions] or sessions_under()
    if not paths:
        print(f"no sessions under {DATA} - the template is fitted from one")
        return 1

    data, metas = load(paths)
    model = HandModel.load(ensure_urdf(args.robot_hand, None))
    retarget = Retargeter(model=model)

    # Per-digit scaling first, because the sampler's thumb band is fitted
    # against finished keypoints and finishing needs it.
    tracked = np.flatnonzero(data["tracked"])
    scale = fit_proportions(retarget,
                            [hands_from(data, i) for i in tracked[::200]],
                            FINGERS)
    offsets = fit_bone_offsets(retarget, model,
                               [hands_from(data, i) for i in tracked[::200]],
                               FINGERS)
    # The recorded hands as the TEACHER sees them - same alignment, same
    # scaling, same axes - which is what the synthetic ones have to match.
    real = np.stack([keypoints_for(retarget, data, i, scale, offsets)
                     for i in tracked[::50]])
    # Rest pose, pose ranges and thumb band all fitted - see `sampler_for`.
    diversity = dict(DIVERSITY[args.diversity])
    if args.pinch is not None:
        diversity["pinch_rate"] = args.pinch
    if args.object_rate is not None:
        diversity["object_rate"] = args.object_rate
    sampler = sampler_for(data, metas[0], retarget, args.seed,
                          diversity=diversity)

    print(f"\n  drawing {args.hands} hands x {args.seconds}s @ {args.hz} Hz")
    print(f"  diversity '{args.diversity}': "
          + "  ".join(f"{k}={v:g}" for k, v in sampler.settings().items()))
    began = time.perf_counter()
    sequences, hand_objects = [], []
    for _ in range(args.hands):
        sequences.append(sampler.sequence(args.seconds, args.hz))
        hand_objects.append(list(sampler.objects))
    points = np.concatenate(sequences)
    print(f"  {sum(len(o) for o in hand_objects)} objects in "
          f"{sum(1 for o in hand_objects if o)} of {args.hands} hands")
    print(f"  {len(points)} frames in {time.perf_counter() - began:.1f}s")

    scaled = finish_keypoints(points, scale, retarget, offsets)

    # Both sides as the TEACHER receives them - metres, robot axes, scaled and
    # rest-aligned - rather than one raw and one finished. Comparing a
    # half-processed sample against a finished one was how a 90 degree rotation
    # went unnoticed for 150,000 frames.
    validate.compare(real * 1000.0, scaled * 1000.0)
    validate.check_frame(real, scaled)
    if args.compare:
        return 0

    # The raw view and an intent signal. A generated hand has no sensor behind
    # it, so intent is read off its own geometry - see `intent_from_distance`.
    raw = raw_synthetic(points, retarget)
    intent = objective.intent_from_distance(np.stack(
        [np.linalg.norm(raw[:, objective.TIP["thumb"]] - raw[:, objective.TIP[f]], axis=1)
         for f in objective.OTHERS], axis=1) * 1000.0)

    warm = offline.solver(model.path, retarget.joint_names, alpha=1.0,
                          maxeval=args.maxeval)
    from . import gestures as authored
    teacher = offline.Teacher(model, retarget.joint_names, scale, hand=args.robot_hand)
    reach = float(np.mean([scale.get(f, 1.0) for f in objective.OTHERS]))
    palm = authored.palm_of(teacher.fk)
    print(f"\n  solving {len(scaled)} frames through the teacher...")
    began, at = time.perf_counter(), 0
    q_star = np.empty((len(scaled), len(retarget.joint_names)))
    bounds, placed_objects = [], []
    from . import collision
    for number, sequence in enumerate(sequences, start=1):
        span = slice(at, at + len(sequence))
        seed = offline.pass_over(warm, scaled[span])
        # This hand's objects, placed from the hand (`gestures.cylinder_to_robot`),
        # on absolute frames, present per frame; the solve may nudge them a
        # few millimetres - see `objective.W_OBJECT_PRIOR`.
        items = [authored.cylinder_to_robot(
                     dict(item, frames=[item["frames"][0] + at, item["frames"][1] + at]),
                     retarget, reach, palm)
                 for item in hand_objects[number - 1]]
        objects = collision.Objects(items) if items else None
        present = (None if objects is None
                   else objects.presence(at, at + len(sequence)).numpy())
        q_star[span] = teacher.teach(seed, scaled[span], raw[span], intent[span],
                                     steps=args.steps, objects=objects, present=present)
        if objects is not None:
            placed_objects.extend(objects.items)
        bounds.append((at, at + len(sequence)))
        at += len(sequence)
        if number % 5 == 0 or number == len(sequences):
            rate = at / (time.perf_counter() - began)
            print(f"\r    {number}/{len(sequences)} hands   {rate:.0f} frames/s"
                  f"   {(len(scaled) - at) / rate:5.0f}s left    ",
                  end="", flush=True)
    print(f"\r    {len(sequences)} hands, {len(scaled)} frames solved" + " " * 24)
    distances = distance_table.pairwise(raw)
    ttc = distance_table.time_to_contact(distances, np.arange(len(raw)) / args.hz,
                                   bounds, objective.TOUCH_MM)

    out = Path(args.out) if args.out else SYNTHETIC / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "pairs.npz",
        keypoints=scaled.astype(np.float32),        # metres, robot axes
        keypoints_raw=raw.astype(np.float32),
        pinch_strength=intent.astype(np.float32),
        distances=distances.astype(np.float32),
        time_to_contact=ttc.astype(np.float32),
        q_star=q_star.astype(np.float32),
        run_bounds=np.array(bounds, dtype=np.int64),
        joint_names=np.array(retarget.joint_names))
    (out / "objects.json").write_text(json.dumps(placed_objects, indent=1))
    (out / "pairs.json").write_text(json.dumps(
        {"hands": args.hands, "seconds": args.seconds, "hz": args.hz,
         "seed": args.seed, "frames": int(len(scaled)), "objects": len(placed_objects),
         "robot_hand": args.robot_hand, "maxeval": args.maxeval,
         "template_from": [m["path"] for m in metas],
         "diversity": args.diversity, "sampler": sampler.settings(),
         "digit_scale": scale}, indent=2))
    size = (out / "pairs.npz").stat().st_size / 1e6
    print(f"\n  wrote {out}/pairs.npz  ({len(scaled)} pairs, {size:.1f} MB)")
    return 0


def run_replay(args, parser):
    return playback.play(args.session, synthetic=args.synthetic,
                         blocks=args.blocks,
                         speed=args.speed, loop=args.loop,
                         robot_hand=args.robot_hand,
                         skeleton_offset=args.skeleton_offset,
                         student=args.student, ghost=args.ghost,
                         scaled_skeleton=args.scaled_skeleton,
                         future=args.future, video=args.video, fps=args.fps)


def run_gallery(args, parser):
    from . import gallery

    target = args.target or str(SYNTHETIC / "seed0")
    return gallery.play(target, hands=args.hands, pitch=args.pitch or gallery.PITCH, speed=args.speed,
                        loop=not args.once, robot_hand=args.robot_hand, seed=args.seed,
                        png=args.png, frame=args.frame)


def run_train(args, parser):
    from . import train as training

    if args.intent:
        # alpha is gated on sigma, so the grasp head needs the belief channels
        args.belief = True          # see `student.authority`
    if getattr(args, "contact_gate", False):
        args.belief = True          # it gates those channels; see `belief`
    from .augment import NoiseModel
    from .student import BASIS, CHANNELS

    synthetic_path = Path(args.synthetic or (SYNTHETIC / "seed0" / "pairs.npz"))
    if synthetic_path.is_dir():
        synthetic_path = synthetic_path / "pairs.npz"
    if not synthetic_path.exists():
        parser.error(f"no synthetic pairs at {synthetic_path} - "
                     f"run `learn.py synth` first")

    session = Path(args.session) if args.session else None
    if session is None:
        found = [p for p in sessions_under() if (p / "teacher.npz").exists()]
        if not found:
            parser.error("no session has teacher.npz - run `learn.py teacher`")
        session = found[-1]
    meta = json.loads((session / "meta.json").read_text())

    noise = None
    fit = NOISE_FIT
    if not args.no_noise:
        if not fit.exists():
            parser.error(f"no noise fit at {fit} - run `learn.py noise`, "
                         f"or pass --no-noise")
        noise = NoiseModel.load(fit)

    print(f"\n  synthetic  {synthetic_path}")
    print(f"  real       {session}")
    print(f"  noise      {'measured, on synthetic input' if noise else 'OFF (ablation)'}")

    model, history, _, _, device = training.train(
        training.load_pairs(synthetic_path),
        training.load_pairs(session / "teacher.npz"),
        meta["blocks"],
        Path(args.out) if args.out else MODELS / "student.pt",
        noise=noise, epochs=args.epochs, batch=args.batch,
        learning_rate=args.lr, basis=args.basis or BASIS,
        channels=args.channels or CHANNELS, device=args.device,
        workers=args.workers,
        w_accel=training.W_ACCEL if args.accel is None else args.accel,
        w_pair=training.W_PAIR if args.pair is None else args.pair,
        robot_hand=args.robot_hand, seed=args.seed,
        w_direct=training.W_DIRECT if args.direct is None else args.direct,
        belief=bool(args.belief), authority=bool(args.intent),
        contact_gate=bool(getattr(args, "contact_gate", False)),
        warmup=args.warmup)

    real = training.load_pairs(session / "teacher.npz")
    hand_model = HandModel.load(ensure_urdf(args.robot_hand, None))
    joint_names = [str(n) for n in real["joint_names"]]
    rows = training.compare(model, real, meta["blocks"], device, hand_model,
                            joint_names, robot_hand=args.robot_hand)
    if rows:
        print(f"\n  HELD-OUT BLOCKS ({', '.join(training.HELD_OUT)})")
        print(f"  {'':<18}{'fingers':>9}{'thumb':>8}{'contact':>9}"
              f"{'gap@touch':>11}{'collide':>9}{'jerk':>12}{'max step':>10}")
        for row in rows:
            contact = f"{row['contact_pct']:.0f}%" if row["contact_pct"] is not None else "-"
            gap = f"{row['gap_at_touch_mm']:.1f}mm" if row["gap_at_touch_mm"] is not None else "-"
            print(f"  {row['name']:<18}{row['posture_deg']:>8.2f}d{row['thumb_deg']:>7.1f}d"
                  f"{contact:>9}{gap:>11}{row['penetration_pct']:>8.1f}%"
                  f"{row['jerk']:>12.3g}{row['worst_step_deg']:>9.2f}d")

    skill = training.lookahead_skill(model, real, meta["blocks"], device)
    if skill:
        print("\n  LOOKAHEAD - is predicting ahead better than holding still?")
        print(f"  {'horizon':<12}{'student':>10}{'hold':>9}{'better by':>12}")
        for row in skill:
            if not row["horizon"]:
                # Holding the present pose IS the present pose, so this row is
                # zero by definition and is not a comparison - it is here as
                # the student's own present-tense error, which the rows below
                # are read against.
                print(f"  +{row['horizon']:<3}f {row['ms']:>4} ms"
                      f"{row['student_deg']:>9.2f}d{'--':>8}"
                      f"{'(reference)':>12}")
                continue
            print(f"  +{row['horizon']:<3}f {row['ms']:>4} ms"
                  f"{row['student_deg']:>9.2f}d{row['hold_deg']:>8.2f}d"
                  f"{row['hold_deg'] - row['student_deg']:>+11.2f}d")

    out = Path(args.out) if args.out else MODELS / "student.pt"
    out.with_suffix(".json").write_text(json.dumps(
        {"synthetic": str(synthetic_path), "session": str(session),
         "noise": bool(noise), "epochs": args.epochs, "batch": args.batch,
         "lr": args.lr, "basis": args.basis or BASIS,
         "accel": training.W_ACCEL if args.accel is None else args.accel,
         "pair": training.W_PAIR if args.pair is None else args.pair,
         "held_out": list(training.HELD_OUT), "lookahead": skill,
         "history": history, "comparison": rows}, indent=2))
    print(f"\n  wrote {out}")
    return 0


def run_gestures(args, parser):
    """Author the gestures, solve them, and see what the student does with them.

    The teacher solves the same frames, so there is a reference for a shape
    nobody has ever tracked - which is the only way to say whether the student
    is wrong or merely somewhere the teacher also declines to go.
    """
    from . import gestures as authored
    from . import teacher as offline
    from .replay import student_rollout

    paths = [Path(p) for p in args.sessions] or sessions_under()
    if not paths:
        parser.error(f"no sessions under {DATA} - the hand template is fitted "
                     f"from one")
    data, metas = load(paths)
    model = HandModel.load(ensure_urdf(args.robot_hand, None))
    retarget = Retargeter(model=model)
    tracked = np.flatnonzero(data["tracked"])
    scale = fit_proportions(retarget,
                            [hands_from(data, i) for i in tracked[::200]],
                            FINGERS)
    offsets = fit_bone_offsets(retarget, model,
                               [hands_from(data, i) for i in tracked[::200]],
                               FINGERS)
    sampler = sampler_for(data, metas[0], retarget)
    geometry = {key: sampler.template[key]
                for key in ("knuckle", "length", "direction")}

    problems = authored.verify(sampler, geometry)
    if problems:
        print("  authored gestures do not hold their shape:")
        for line in problems:
            print(f"    {line}")
        return 1
    print(f"  {len(authored.GESTURES)} gestures verify")

    names = authored.SEQUENCES[args.sequence]
    if args.wrap_radius:
        authored.WRAP_RADIUS_MM = float(args.wrap_radius)
    objects = None
    if args.sequence == "wrap":
        # The cylinder the operator closed on, in the robot's palm, for the
        # teacher to hold and the census to score. See `gestures.wrap_pose`.
        from . import collision
        out_dir = Path(args.out) if args.out else SYNTHETIC / (args.sequence + (f"{int(args.wrap_radius)}" if args.wrap_radius else ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        reach = float(np.mean([scale.get(f, 1.0) for f in objective.OTHERS]))
        palm = authored.palm_of(offline.Teacher(model, retarget.joint_names, scale,
                                               hand=args.robot_hand).fk)
        placed = authored.cylinder_to_robot(
            authored.cylinder_in_hand(geometry, args.wrap_radius or authored.WRAP_RADIUS_MM),
            retarget, reach, palm)
        print(f"  cylinder placed from the hand at x={placed['p0'][0] * 1000:.0f} "
              f"z={placed['p0'][2] * 1000:.0f} mm, radius {placed['radius'] * 1000:.0f}")
        (out_dir / "objects.json").write_text(json.dumps([placed], indent=1))
        objects = collision.Objects.load(out_dir / "objects.json")
    points, labels = authored.sequence(sampler, geometry, names=names, hold=args.hold,
                                       move=args.move, hz=args.hz)
    scaled = finish_keypoints(points, scale, retarget, offsets)

    raw = raw_synthetic(points, retarget)
    intent = objective.intent_from_distance(np.stack(
        [np.linalg.norm(raw[:, objective.TIP["thumb"]] - raw[:, objective.TIP[f]], axis=1)
         for f in objective.OTHERS], axis=1) * 1000.0)

    bounds = np.array([[0, len(scaled)]], dtype=np.int64)
    print(f"  {len(scaled)} frames, {len(scaled) / args.hz:.1f}s")
    print("  solving with the teacher...")
    warm = offline.solver(model.path, retarget.joint_names, alpha=1.0,
                          maxeval=args.maxeval)
    teacher = offline.Teacher(model, retarget.joint_names, scale, hand=args.robot_hand)
    present = None if objects is None else np.ones((len(scaled), len(objects)), dtype=np.float32)
    q_star = teacher.teach(offline.pass_over(warm, scaled), scaled, raw, intent,
                           objects=objects, present=present)
    live_solver = offline.solver(model.path, retarget.joint_names,
                                 alpha=offline.LIVE_ALPHA)
    q_live = offline.pass_over(live_solver, scaled)

    out = Path(args.out) if args.out else SYNTHETIC / ("gestures" if args.sequence == "probe" else args.sequence
                                                     + (f"{int(args.wrap_radius)}" if args.wrap_radius else ""))
    out.mkdir(parents=True, exist_ok=True)
    distances = distance_table.pairwise(raw)
    ttc = distance_table.time_to_contact(distances, np.arange(len(raw)) / args.hz,
                                   bounds, objective.TOUCH_MM)
    np.savez_compressed(out / "pairs.npz",
                        keypoints=scaled.astype(np.float32),
                        keypoints_raw=raw.astype(np.float32),
                        pinch_strength=intent.astype(np.float32),
                        distances=distances.astype(np.float32),
                        time_to_contact=ttc.astype(np.float32),
                        q_star=q_star.astype(np.float32),
                        q_live=q_live.astype(np.float32),
                        run_bounds=bounds, block=labels,
                        t=np.arange(len(scaled)) / args.hz,
                        joint_names=np.array(retarget.joint_names),
                        digit_scale=np.array([scale.get(f, 1.0)
                                              for f in FINGERS]),
                        fingers=np.array(list(FINGERS)))
    (out / "pairs.json").write_text(json.dumps(
        {"gestures": list(names), "sequence": args.sequence, "frames": int(len(scaled)),
         "hz": args.hz, "hold": args.hold, "move": args.move,
         "note": "authored gestures: outside the recorded half of the training "
                  "set, and drawn only as a deliberately oversampled corner of "
                  "the synthetic half - see gestures.py"},
        indent=2))

    student = Path(args.student)
    if not student.exists():
        print(f"\n  no student at {student} - wrote the reference only")
        return 0
    q_student, valid, stored, _ = student_rollout(student, scaled, raw, bounds)
    if stored != list(retarget.joint_names):
        parser.error("the student was trained on a different joint order")
    np.savez_compressed(out / "student.npz",
                        q_student=q_student.astype(np.float32), valid=valid)

    print("\n  PER GESTURE - how far the student lands from the teacher")
    print(f"  {'gesture':<12}{'joint deg':>11}{'tip mm':>9}"
          f"{'live deg':>10}{'live mm':>9}")
    tips = offline.TIP_LINKS
    def tip_error(a, b, rows):
        total = 0.0
        for index in rows:
            for finger in FINGERS:
                pa = model.link_pose(tips[finger], a[index],
                                     retarget.joint_names)[0]
                pb = model.link_pose(tips[finger], b[index],
                                     retarget.joint_names)[0]
                total += float(np.linalg.norm(pa - pb))
        return 1000.0 * total / max(len(rows) * len(FINGERS), 1)

    for index, name in enumerate(names):
        rows = np.flatnonzero((labels == index) & valid)
        if not len(rows):
            continue
        rows = rows[::5]
        print(f"  {name:<12}"
              f"{np.degrees(np.abs(q_student[rows] - q_star[rows]).mean()):>10.2f}d"
              f"{tip_error(q_student, q_star, rows):>9.1f}"
              f"{np.degrees(np.abs(q_live[rows] - q_star[rows]).mean()):>9.2f}d"
              f"{tip_error(q_live, q_star, rows):>9.1f}")
    if objects is not None:
        # WHAT THE HAND DOES WITH THE THING IN IT, scored in MuJoCo with the
        # cylinder as a real geom rather than through the proxy the solve
        # optimised. The wrap probe exists to answer this and used to answer
        # it only in a scratch script.
        held = np.flatnonzero((labels == names.index("wrap")) & valid)
        held = held[len(held) // 4: len(held) // 4 + max(1, len(held) // 2)][::5]
        census = collision.Census(retarget.joint_names, args.robot_hand, objects=objects)
        print(f"\n  ON THE CYLINDER ({objects.radius[0] * 1000:.0f} mm radius), over the held wrap")
        print(f"  {'source':<12}{'links on it':>12}{'digits':>8}{'thumb':>7}"
              f"{'penetration':>13}{'worst':>8}")
        for label, q in (("teacher", q_star), ("student", q_student), ("live", q_live)):
            deep = [census.against_objects(q[i]) for i in held]
            links = np.mean([len(t) for _, t in deep])
            digits = np.mean([len({n.split("_")[0] for n in t}) for _, t in deep])
            thumb = np.mean([any(n.startswith("thumb") for n in t) for _, t in deep])
            print(f"  {label:<12}{links:>12.1f}{digits:>8.1f}{100 * thumb:>6.0f}%"
                  f"{np.mean([p for p, _ in deep]):>10.1f} mm{max(p for p, _ in deep):>7.1f}")

    print(f"\n  wrote {out}/pairs.npz and student.npz")
    print(f"  watch it:  .venv/bin/mjpython learn.py replay "
          f"--synthetic {out} --student --loop")
    return 0


def run_selftest(args, parser):
    from . import selftest

    print("\n  invariants, against the recordings on disk\n")
    failures = selftest.run(args.robot_hand)
    print()
    if failures:
        print(f"  {len(failures)} FAILED")
        return 1
    print("  all pass")
    return 0
