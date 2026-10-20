"""Phase 0: a survey of what the Leap actually reports, block by block.

This is NOT a labelling exercise. Labels for the learned retargeter come from
the offline solver, not from a hand waved over a sensor. What only a real hand
can supply is two things, and this file exists to collect exactly those:

  THE NOISE MODEL. Leap noise has at least three modes, each augmented
  separately by `augment.NoiseModel`: per-keypoint jitter, per-finger
  correlated drift when the
  tracker's model of a hidden finger is slightly wrong, and frame-level snaps
  when a finger is re-acquired. A static hold measures the first, an occluded
  hold the second, and the occlusion block the third. Fitting one gaussian to
  all of it would leave the two that actually break grasps unmodelled.

  THE INPUT DISTRIBUTION. Synthetic hands are cheap and perfect, which is the
  problem: they are drawn from an assumption about how hands move. These
  recordings are what that assumption gets checked against.

NOTHING IS FILTERED HERE. No Hampel, no One Euro, no smoothing, no calibration
fit. The filter is part of what the learned model is meant to replace, so
recording through it would hide the signal being surveyed and bake today's
filter constants into tomorrow's training set. `handcontrol.leap` is the only
import from the project, deliberately - this file cannot go stale when the
retargeting changes, because it does not know the retargeting exists.

TIMESTAMPS ARE RECORDED, NOT ASSUMED. The Leap does not deliver frames at a
constant rate - it drops under load and under poor tracking, which is exactly
when the interesting samples happen. A recording stamped at a nominal rate
would smear those gaps into the signal, and both the one-euro filter and any
temporal model need the true dt.

Driven by `learn.py survey`; nothing here parses arguments.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..leap import DIGITS

# Frames arrive at ~119 Hz on this sensor. Polling with a zero timeout takes whatever is
# already queued and returns immediately, so the sleep is what stops this from
# spinning a core; it is well under a frame interval, so nothing is missed.
POLL_SLEEP = 0.001
COUNTDOWN = 3


@dataclass(frozen=True)
class Block:
    """One scripted recording. `phase` is what the block is FOR, and it is
    stored per frame because the analysis differs: a hold is measured for
    variance about a constant, a motion for spectrum and for tracking loss."""

    key: str
    phase: str                  # hold | motion | free
    seconds: float
    title: str
    instruction: str


# The script. Ordered so the still blocks come first - they are the ones that
# need a settled operator and an unhurried sensor, and doing them after five
# minutes of waving gets a tired hand's tremor into the noise floor.
#
# The last four blocks are the ones people skip, and they carry most of the
# signal: rotation because the sensor's error is strongly a function of viewing
# angle, occlusion because that is where the snaps live, and free motion
# because a scripted gesture set only ever covers the corners of the space.
BLOCKS = (
    Block("flat_hold", "hold", 30,
          "FLAT PALM, HELD STILL",
          "Palm flat and open, fingers together, about 20 cm above the sensor.\n"
          "Then hold as still as you can. Do not follow the numbers - watching\n"
          "the readout makes people correct, and a corrected hand is not a\n"
          "still one."),
    Block("fist_hold", "hold", 30,
          "FIST, HELD STILL",
          "Closed fist, knuckles up, held still. This is the noise floor WITH\n"
          "self-occlusion: the ring and middle DIPs are hidden behind the\n"
          "fingers in front and the SDK is inferring them."),
    Block("spread_hold", "hold", 30,
          "FLAT, FINGERS SPREAD WIDE, HELD STILL",
          "Flat palm, fingers spread as far as they go, held still. A third\n"
          "occlusion regime: nothing hidden, but the fingers are at the edge\n"
          "of their range where the tracker's prior pulls hardest."),
    Block("curl_slow", "motion", 60,
          "FLAT -> FIST -> FLAT, SLOW, x10",
          "Ten full cycles, about six seconds each. Deliberate and smooth.\n"
          "This is the transition distribution the model spends most of its\n"
          "time in."),
    Block("curl_fast", "motion", 30,
          "FLAT -> FIST -> FLAT, FAST, x10",
          "Ten cycles as fast as you can make them cleanly. Speed is where\n"
          "tracking breaks, and a model trained only on slow motion will meet\n"
          "this for the first time on the hardware."),
    Block("thumb_tap", "motion", 45,
          "THUMB TAPS EACH FINGERTIP",
          "Thumb to index, index away; thumb to middle; thumb to ring. Round\n"
          "and round. Contact all the way, not a near miss - pinch distance is\n"
          "the metric the whole retargeting is judged on."),
    # THE BLOCKS THE FIRST SESSION WAS MISSING. Its thumb taps were 74%
    # index: 1.2% of frames touched the ring and 2.6% the middle, and the
    # student learned ring contact at 0% from it. A pinch per finger, held
    # and repeated, is the only data that fixes that.
    Block("middle_pinch", "motion", 45,
          "THUMB TO MIDDLE, OVER AND OVER",
          "Thumb tip to middle fingertip, hold a beat, open, again. Index and\n"
          "ring stay where they like. Contact every time - the middle finger\n"
          "is the one the network cannot learn from the taps above."),
    Block("ring_pinch", "motion", 45,
          "THUMB TO RING, OVER AND OVER",
          "Same, thumb tip to ring fingertip. This is mostly adduction and it\n"
          "is the pinch the sensor sees worst; take it slow and make it land."),
    Block("object_grasp", "motion", 60,
          "PICK UP, HOLD, PUT DOWN",
          "A pen, then something the size of a cup. Reach, close on it, lift,\n"
          "hold still, put it back, let go. The sensor will lose fingers behind\n"
          "the object - that is the point; the labels have to survive it."),
    Block("spread_cycle", "motion", 30,
          "SPREAD WIDE <-> TOGETHER",
          "Fingers flat throughout, spreading and closing sideways. The fist\n"
          "capture cannot reach this joint at all, because the fingers are\n"
          "together for the whole of it."),
    Block("finger_isolate", "motion", 60,
          "ONE FINGER AT A TIME",
          "Curl the index alone five times, then the middle, then the ring,\n"
          "then the thumb. Keep the others as straight as you can. This is\n"
          "what says how far a real hand departs from the tendon coupling the\n"
          "synthetic sampler assumes."),
    Block("thumb_sweep", "motion", 45,
          "THUMB OPPOSITION SWEEP",
          "Thumb from flat beside the index, out and across the palm to the\n"
          "base of the ring, and back. Slowly. The thumb is the digit that\n"
          "tracks worst and matters most."),
    Block("rot45", "motion", 30,
          "HAND ROTATED 45 DEGREES, FIST CYCLES",
          "Same flat->fist->flat, but with the hand rolled about 45 degrees\n"
          "off palm-down. Sensor error is a function of viewing angle, and a\n"
          "training set taken only palm-down learns one slice of it."),
    Block("rot90", "motion", 30,
          "HAND EDGE-ON, FIST CYCLES",
          "Rolled to 90 degrees - thumb up, palm facing sideways. Tracking\n"
          "will be visibly worse. That is the point: these frames are what\n"
          "teaches the model what bad input looks like."),
    Block("occlusion", "motion", 45,
          "DELIBERATE OCCLUSION",
          "Hide fingers behind fingers, on purpose. Fingers straight and\n"
          "pointing at the sensor. Cross them. Let the tracker lose a finger\n"
          "and re-acquire it. You are hunting single-frame snaps - the sample\n"
          "class that no amount of smoothing fixes."),
    Block("free", "free", 300,
          "FREE MOTION, NO SCRIPT",
          "Five minutes. Grasp imaginary objects, gesture, count on your\n"
          "fingers, whatever a hand does. Do not perform the blocks above.\n"
          "The scripted set covers the corners of the space; this is the only\n"
          "block that covers the middle."),
)
BY_KEY = {block.key: block for block in BLOCKS}

# Per frame, in the order they are written. `bones` is (5, 5, 3): five digits,
# and per digit the base of the metacarpal plus the far end of each of its four
# bones, so the fingertip is index 4. Millimetres in the device frame, exactly
# as the SDK reports them.
NAN3 = np.full(3, np.nan, dtype=np.float32)
NAN_BONES = np.full((5, 5, 3), np.nan, dtype=np.float32)


class Recorder:
    """Accumulates frames across every block of one session.

    Can be re-opened from a session already on disk, which is what makes a
    block skipped on the day recoverable later: the missing block is recorded
    into the same directory and the file rewritten whole, rather than leaving
    two half-sessions for something downstream to remember to join.

    WHAT IS ALREADY ON DISK STAYS AN ARRAY. The obvious implementation reads
    the old session back into the same list of per-frame dicts that new frames
    go into, and it is unusably slow: an npz decompresses the whole array on
    every element access, so fifty thousand frames times sixteen fields is
    nearly a million decompressions of a multi-megabyte block. The loaded
    session is held as arrays and joined to the new frames once, at save.
    """

    def __init__(self, prior=None):
        self.prior = prior or {}          # arrays already on disk
        self.rows = []                    # frames recorded this run
        self.started = time.monotonic()

    @property
    def loaded(self):
        return len(self.prior["t"]) if self.prior else 0

    @property
    def total(self):
        return self.loaded + len(self.rows)

    @classmethod
    def load(cls, path):
        """Re-open a session directory. Returns (recorder, meta), or (None, None).

        Stored timestamps are kept as they are rather than rebased: they are
        seconds since that session's own start, and a block recorded a week
        later genuinely did not happen at minute three. What anything
        downstream reads is dt WITHIN a contiguous run, and appending never
        puts two takes inside one run."""
        path = Path(path)
        if not (path / "session.npz").exists():
            return None, None
        with np.load(path / "session.npz") as handle:
            prior = {name: handle[name] for name in handle.files}
        meta = json.loads((path / "meta.json").read_text())
        recorder = cls(prior)
        # Carry the clock on from where the file ended so the joined `t` stays
        # monotonic, which is what lets a run be found by index at all.
        last = float(prior["t"][-1]) if len(prior["t"]) else 0.0
        recorder.started = time.monotonic() - last
        return recorder, meta

    def add(self, frame, hand, block_index):
        """One tracking frame. `hand` may be None - that is a tracking LOSS and
        it is recorded, because how often and for how long the sensor loses a
        hand is one of the things this session is measuring. A frame that never
        arrived is a different thing entirely and is simply not here."""
        self.rows.append({
            "t": time.monotonic() - self.started,
            "device_us": frame.timestamp,
            "frame_id": frame.frame_id,
            "framerate": frame.framerate,
            "n_hands": len(frame.hands),
            "block": block_index,
            "tracked": hand is not None,
            "hand_id": hand.id if hand else -1,
            "bones": (np.stack([hand.bones[d] for d in DIGITS])
                      if hand else NAN_BONES),
            "palm": hand.palm if hand else NAN3,
            "normal": hand.normal if hand else NAN3,
            "direction": hand.direction if hand else NAN3,
            "width": hand.width if hand else np.nan,
            "pinch_strength": hand.pinch_strength if hand else np.nan,
            "grab_strength": hand.grab_strength if hand else np.nan,
            "pinch_distance": hand.pinch_distance if hand else np.nan,
            "confidence": hand.confidence if hand else np.nan,
        })

    def arrays(self):
        if not self.rows:
            return dict(self.prior)
        column = lambda name, dtype: np.array(  # noqa: E731
            [row[name] for row in self.rows], dtype=dtype)
        fresh = {
            "t": column("t", np.float64),
            "device_us": column("device_us", np.int64),
            "frame_id": column("frame_id", np.int64),
            "framerate": column("framerate", np.float32),
            "n_hands": column("n_hands", np.int8),
            "block": column("block", np.int16),
            "tracked": column("tracked", bool),
            "hand_id": column("hand_id", np.int64),
            "bones": column("bones", np.float32),
            "palm": column("palm", np.float32),
            "normal": column("normal", np.float32),
            "direction": column("direction", np.float32),
            "width": column("width", np.float32),
            "pinch_strength": column("pinch_strength", np.float32),
            "grab_strength": column("grab_strength", np.float32),
            "pinch_distance": column("pinch_distance", np.float32),
            "confidence": column("confidence", np.float32),
        }
        if not self.prior:
            return fresh
        # A column this build records that the file on disk predates is filled
        # with NaN for the old rows rather than raising: `confidence` arrived
        # after the first session, and appending to it died with a KeyError
        # here - AFTER the operator had sat through up to five minutes of
        # recording, with nothing written. A column the file has and this build
        # no longer records is carried through unchanged, for the same reason.
        out = {}
        for name in set(fresh) | set(self.prior):
            older = self.prior.get(name)
            newer = fresh.get(name)
            if older is None:
                older = np.full((len(self.prior["t"]),) + newer.shape[1:],
                                np.nan, dtype=newer.dtype)
            if newer is None:
                newer = np.full((len(fresh["t"]),) + older.shape[1:],
                                np.nan, dtype=older.dtype)
            out[name] = np.concatenate([older, newer])
        return out


def sensor_check(connection, prefer):
    """Prove the sensor works and the right hand is over it, before the script.

    Worth its own step because every failure it catches is one that would
    otherwise be discovered after thirteen minutes of recording: no service,
    no camera, or the operator holding up the hand the session is not for."""
    print("\n  Hold your %s hand over the sensor." % prefer.upper())
    deadline = time.time() + 30
    seen = 0
    while time.time() < deadline:
        frame = connection.poll(0)
        if frame is not None and frame.hands:
            types = {h.type for h in frame.hands}
            if prefer in types:
                seen += 1
                span = np.linalg.norm(frame.hands[0].bones["middle"][4]
                                      - frame.hands[0].palm)
                print(f"\r  seen: {seen:3d} frames   {frame.framerate:5.1f} Hz"
                      f"   palm->middle tip {span:5.1f} mm      ", end="")
                if seen >= 60:
                    print("\n  Sensor good.")
                    return True
            else:
                print(f"\r  that is a {'/'.join(sorted(types))} hand, this "
                      f"session is for the {prefer}      ", end="")
                seen = 0
        time.sleep(POLL_SLEEP)
    print("\n  Gave up waiting for a hand.")
    return False


def countdown(seconds=COUNTDOWN):
    for remaining in range(seconds, 0, -1):
        print(f"\r  starting in {remaining}...   ", end="", flush=True)
        time.sleep(1.0)
    print("\r  RECORDING          ")


def record_block(connection, recorder, block, block_index, prefer):
    """Record one block. Returns its stats, or None if it was interrupted.

    Ctrl-C ends the BLOCK and keeps what it captured, rather than ending the
    session: a block that went wrong is worth redoing, and thirteen minutes of
    earlier blocks are not worth losing to say so."""
    start = len(recorder.rows)
    began = time.monotonic()
    last_print = 0.0
    interrupted = False
    try:
        while True:
            elapsed = time.monotonic() - began
            if elapsed >= block.seconds:
                break
            frame = connection.poll(0)
            if frame is not None:
                recorder.add(frame, frame.hand(prefer), block_index)
            if elapsed - last_print > 0.2:
                last_print = elapsed
                rows = recorder.rows[start:]
                tracked = sum(1 for r in rows if r["tracked"])
                bar = int(30 * elapsed / block.seconds)
                print(f"\r  [{'#' * bar}{'.' * (30 - bar)}] "
                      f"{block.seconds - elapsed:5.1f}s left   "
                      f"{len(rows):6d} frames   "
                      f"{100.0 * tracked / max(len(rows), 1):5.1f}% tracked",
                      end="", flush=True)
            time.sleep(POLL_SLEEP)
    except KeyboardInterrupt:
        interrupted = True

    rows = recorder.rows[start:]
    tracked = sum(1 for row in rows if row["tracked"])
    seconds = time.monotonic() - began
    stats = {
        "key": block.key, "phase": block.phase,
        "frames": len(rows), "tracked": tracked,
        "seconds": round(seconds, 2),
        "hz": round(len(rows) / max(seconds, 1e-6), 1),
        "tracked_pct": round(100.0 * tracked / max(len(rows), 1), 1),
        "interrupted": interrupted,
        "row_start": recorder.loaded + start,
        "row_end": recorder.total,
    }
    print(f"\r  {'INTERRUPTED' if interrupted else 'done'}: "
          f"{stats['frames']} frames, {stats['hz']} Hz, "
          f"{stats['tracked_pct']}% tracked" + " " * 30)
    if stats["tracked_pct"] < 80 and block.key not in ("rot90", "occlusion"):
        print("  NOTE: under 80% tracked. Worth redoing unless that was "
              "deliberate.")
    return stats


def prompt(block, number, total):
    """Show the block and wait. Returns 'go', 'skip' or 'quit'."""
    print("\n" + "=" * 70)
    print(f"  BLOCK {number}/{total}   {block.title}   [{block.seconds:.0f}s]")
    print("=" * 70)
    for line in block.instruction.splitlines():
        print("  " + line)
    print()
    while True:
        try:
            answer = input("  [enter] record   [s] skip   [q] stop here > ")
        except (EOFError, KeyboardInterrupt):
            return "quit"
        answer = answer.strip().lower()
        if answer in ("", "g", "go"):
            return "go"
        if answer in ("s", "skip"):
            return "skip"
        if answer in ("q", "quit"):
            return "quit"


def after():
    """Offer a redo. A block recorded badly is worth one minute now and is
    worth nothing at all once the operator has gone home."""
    while True:
        try:
            answer = input("  [enter] next   [r] redo this block > ")
        except (EOFError, KeyboardInterrupt):
            return "quit"
        answer = answer.strip().lower()
        if answer in ("", "n", "next"):
            return "next"
        if answer in ("r", "redo"):
            return "redo"


def save(path, recorder, meta):
    arrays = recorder.arrays()
    if not arrays:
        print("\nNothing recorded, nothing written.")
        return None
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "session.npz", **arrays)
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    size = (path / "session.npz").stat().st_size / 1e6
    print(f"\nWrote {len(arrays['t'])} frames to {path}/session.npz "
          f"({size:.1f} MB)")
    print(f"      metadata in {path}/meta.json")
    return path


