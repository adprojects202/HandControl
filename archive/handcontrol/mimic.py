"""Following a human hand: the Leap, the retargeting, and the IK, in one object.

This is the piece both entry points drive. `main.py` owns it so that the MIMIC
tab can switch it on inside the running simulator, and `leap_mimic.py` owns it
for the standalone loop; neither has a copy of the logic.

It lives in the SIMULATOR's process, not the panel's, and that is not an
accident of layout. The IK belongs to whoever owns the MuJoCo model, the
skeleton has to be drawn in that same scene, and the panel already receives a
state message twenty times a second - so the hand goes in the direction the
data already flows, and the panel stays what it is everywhere else: a control
surface that sends requests and draws what comes back.

Nothing here raises at the caller. A Leap that is unplugged, a service that is
not running, a hand that leaves the cone mid-grip - all of it is ordinary, and
all of it turns into a status string and a held pose rather than a traceback
that takes the session down with it.
"""

from __future__ import annotations

import time
from pathlib import Path
from collections import deque

import numpy as np

from .leap import DIGITS, LeapConnection, LeapError
from .robot.model import DISTAL_LINKS, FINGERS, TIP_LINKS
from .retargeting import GAINS, HandFilter, Retargeter
from .retargeting.bridge import KeyvectorBridge
from .retargeting.keyvector import (digit_lengths, mano_keypoints,
                                    rest_offsets)
from .retargeting.posebank import PoseBank

# The robot thumb does not have the human one's workspace, and no number of
# solver iterations changes that. Gridding its four joints puts its tip between
# z = -165 and +56 mm while the fingertips sit at +150: it cannot come up the
# front of the hand to meet them the way a thumb does. Handed a scaled human
# thumb position it therefore returns its closest approach - 40 to 60 mm out,
# saturated against a limit, no longer tracking anything.
#
# So tip mode solves the three fingers, whose targets are genuinely reachable
# to a couple of millimetres, and leaves the thumb on the angle mapping, which
# at least moves when the operator's thumb moves.
TIP_FINGERS = ("index", "middle", "ring")

# How long a hand may go without its numbers changing before it stops counting
# as tracked. The sensor can deliver frames at full rate that all carry the
# same pose - a 15 second recording arrived that way, every row flagged
# tracked, the operator's hand frozen and the robot drifting on its own. There
# is nothing in the SDK that reports this: the frame id keeps incrementing and
# `confidence` is always 1.0. Comparing the numbers to the last ones is the
# only way to notice.
STALE_SECONDS = 0.6

# An open hand, by the Leap's own grab estimate. Calibration waits for one:
# fitting the neutral pose to a half-closed hand teaches the mapping that
# half-closed is neutral, and every finger is short for the rest of the run.
OPEN_GRAB = 0.15
CALIBRATION_FRAMES = 20

# How many open frames to average before the rest alignment is fitted. More
# than the calibration above, because this one is fitted per BONE and a single
# tremor spreads further than it does through a joint range.
REST_FRAMES = 40

# The guided fit. Three poses, because between them they pin every degree of
# freedom the mapping has: open sets where zero is, the fist sets how far each
# joint travels, and the spread sets the sideways band - which cannot come from
# the other two, since a fist has the fingers together.
CAPTURES = (
    ("open", "hold your hand FLAT and OPEN, fingers together"),
    ("fist", "now close it into a FIST"),
    ("spread", "flat again, fingers SPREAD WIDE"),
)
# Each pose is captured when the operator says so, not when a timer says so.
# A countdown makes the machine's clock the thing you are racing; a button
# makes getting into the pose the only job, which is the one that matters.
#
# What is stored is the mean of the last `SAMPLE_WINDOW` frames rather than the
# single frame the click landed on, so a tremor or one bad frame of tracking
# does not become the operator's range. At 119 Hz that is about a quarter of a
# second of history, already in hand the moment the button is pressed.
SAMPLE_WINDOW = 30
MIN_SAMPLES = 10             # below this there is nothing worth averaging


def skeleton_offset(retarget):
    """How far above the robot to draw the operator's hand, by default.

    Above it, not on it: overlaid, the two hands interpenetrate and neither is
    readable. The panel's pad views are where they are superimposed, and there
    that is the whole point - a hand offset from the dots is a mapping that is
    offset. In the viewer you want to compare two shapes, so they need to be
    two shapes. Measured off the model, so it clears the robot's own
    fingertips whatever the URDF says they are.
    """
    return (0.0, 0.0, retarget.robot_length + 0.06)


# Which of the student's forward heads the live world shows as AHEAD, in
# frames: ten is ~80 ms at the sensor's rate, the nearest horizon at which
# predicting beat holding still (train.py's lookahead table), and about the
# pipeline's own delay.
FUTURE_HORIZON = 10

class Mimic:
    """One Leap, one retargeter, and the state of following a hand with them."""

    def __init__(self, model, mode="keyvector", thumb="angles", follow=None,
                 on_lost="hold", tip_thumb="angles", tracking_mode="desktop",
                 library=None, gains=None, smoothing=0.35, max_rate=6.0,
                 scale=1.0, calibrate=True, mirror=False):
        self.retarget = Retargeter(model=model, gains=dict(gains or GAINS),
                                   smoothing=smoothing, max_rate=max_rate,
                                   thumb_mode=thumb, scale=scale, mirror=mirror)
        self.mode = mode
        # Default to the hand the robot actually IS, measured off the model.
        # Following the other one is not broken - the mapping stays anatomical,
        # your index is still its index - but two opposite hands cannot also
        # agree about direction in the room, so it reads as the robot moving
        # the wrong way. Guessing "right" here was that bug.
        self.follow = follow or self.retarget.chirality()
        self.on_lost = on_lost
        self.tip_thumb = tip_thumb
        self.tracking_mode = tracking_mode
        self.library = library
        self.autocalibrate = calibrate

        # Conditioning goes at the sensor boundary, not in the mapping: a
        # millimetre of jitter on a bone joint is degrees of jitter on the
        # angle it defines, and filtering here cleans the angles, the
        # fingertip targets and the drawn skeleton in one place.
        self.filter = HandFilter()
        self._last_poll = None
        # The keyvector solver, in its own interpreter. Not started until the
        # mode is actually selected: it costs a process and a few seconds of
        # Pinocchio, and a session that never asks for it should pay neither.
        self.bridge = KeyvectorBridge()
        self.urdf = getattr(model, "path", None)
        self._scaled = False
        self.digit_scale = None        # per-digit, see keyvector.mano_keypoints
        # Per bone, the rotation making this operator's flat hand the robot's
        # zero pose. Fitted live from open frames, exactly as the offline
        # teacher fits it from the flat-hold block.
        #
        # WITHOUT IT THE SOLVER REPRODUCES THE OPERATOR'S REST as an error it
        # spends joint range on: measured on a recording, a flat hand commanded
        # -17, -18 and -13 degrees of adduction on the three fingers, all the
        # same sign, which swings the whole finger set off to one side.
        # Corrected, the same hand commands -2.3, -0.6 and -0.3.
        self.bone_offsets = None
        self._rest_frames = []
        # The trained student, and the window of keypoints it reads. Loaded on
        # first use, like the solver, so a session that never selects it pays
        # nothing - and a missing model costs the better mapping, not the
        # session, exactly as a missing solver does.
        # Both learned mappings, every frame, whichever one is commanding.
        # The viewer shows them side by side and the point of that view is the
        # comparison - a lane that only updates when it happens to be selected
        # is showing history, not a comparison.
        #
        # Cheap enough to be unconditional: the solver is a subprocess that is
        # sent and polled without waiting, and the network is one forward pass.
        self.alternates = {}
        self.student = None
        # Declared here and not only on a successful load. Every attribute a
        # method reads has to exist whether or not the happy path ran, or the
        # failure mode is an AttributeError two layers from the thing that
        # actually went wrong - which for a checkpoint that would not load is
        # exactly the wrong place to be told.
        self.student_horizons = (0,)
        self.student_error = ""
        self.student_ms = 0.0
        self.student_contact = None    # contact head, per thumb pair; see `student_heads`
        # The repair layer - see `learning.collision.repair`. Built with the
        # student; what it moved the last command by, for the panel.
        self._repair = None
        self.repair_deg = 0.0
        self.repair_ms = 0.0
        self._student_tried = False
        self._window = deque(maxlen=1)
        self._raw_window = deque(maxlen=1)    # the raw view of the same frames
        # Retrieval mode. Built on first use, cached to config/; needs no
        # solver process, so it works with no .venv-dex installed at all.
        self.bank = PoseBank(model)
        # Set once a start has been tried and failed, so a missing .venv-dex
        # costs one spawn attempt rather than one per frame forever.
        self._bridge_failed = False
        self.connection = None
        self.enabled = False
        self.error = ""
        self.hand = None
        self.raw_hand = None           # unfiltered, for comparing the two
        self.stale = False             # frames arriving, pose not changing
        self._fingerprint = None
        self._changed_at = None
        self.tracked = False
        self.frames = 0
        self.lost = 0
        self.open_frames = 0
        self.wants_calibration = False
        self.tip_targets = {}
        self.tip_errors = {}
        self.capture = None
        self.retarget.load()

    # ------------------------------------------------------------ lifecycle
    def enable(self, on=True):
        """Open the Leap on first use, and report rather than raise.

        Lazy because a session that never touches the MIMIC tab should not pay
        for a camera it is not using, and should not fail to start because one
        is not plugged in.
        """
        on = bool(on)
        if on == self.enabled:
            return self.enabled
        if not on:
            self.enabled = False
            if self.capture is None:
                # A capture keeps reading the sensor after the robot stops
                # following it: measuring your hand and driving the robot with
                # it are separate things, and only the second is what FOLLOW
                # switches.
                self.tracked = False
                self.hand = None
            return False
        if not self._open():
            return False
        self.enabled = True
        if self.autocalibrate and self.mode == "tip":
            self.request_calibration()
        return True

    def _open(self):
        """Open the connection if it is not already. Reports, never raises."""
        if self.connection is not None:
            return True
        try:
            self.connection = LeapConnection(
                mode=self.tracking_mode, library=self.library).open()
            self.connection.wait_for_device(2.0)
            self.error = (self.connection.no_device_reason()
                          if not self.connection.device_count else "")
        except LeapError as failure:
            self.connection = None
            self.error = str(failure).splitlines()[0]
            return False
        return True

    @property
    def reading(self):
        """Is there any reason to be polling the sensor?

        Following a hand is one; measuring one for the calibration is the
        other, and it does not need the robot to be moving. Gating the poll on
        `enabled` alone was a bug: switching FOLLOW off mid-calibration froze
        it between poses with no way to see why.
        """
        return self.connection is not None and (self.enabled
                                                or self.capture is not None)

    def start_bridge(self):
        """Bring up the keyvector solver. Reports rather than raises."""
        if self.bridge.alive:
            return True
        order = ("thumb", "index", "middle", "ring")
        started = self.bridge.start(
            self.urdf, self.retarget.joint_names,
            [TIP_LINKS[f] for f in order],
            wrist=self.retarget.model.root_link,
            # A distal link's frame sits on its own joint axis, so
            # distal -> tip IS the DIP-to-fingertip vector the orientation
            # term needs. Same order as the tips, which is the contract.
            distal_links=[DISTAL_LINKS[f] for f in order])
        if not started:
            self.error = self.bridge.error
        self._scaled = False
        return started

    def close(self):
        self.bridge.stop()
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.enabled = False

    @property
    def available(self):
        return self.connection is not None and self.connection.device_count > 0

    @property
    def expected(self):
        """The hand to hold over the sensor for motion to match direction."""
        chirality = self.retarget.chirality()
        if self.retarget.mirror:
            return "left" if chirality == "right" else "right"
        return chirality

    @property
    def mismatched(self):
        """Is the hand being tracked the mirror of the one the robot is?

        Reported rather than corrected. Silently swapping to the other hand
        would be worse - the operator may have exactly one hand free - and
        silently mirroring would trade an anatomical mapping for a directional
        one without saying so. `mirror=True` is that trade, made explicitly.
        """
        return bool(self.tracked and self.hand
                    and self.hand.type != self.expected)

    # -------------------------------------------------------------- options
    def configure(self, **options):
        """Apply whatever the panel sent, ignoring what it did not.

        Switching into tip mode re-arms calibration if the mapping has never
        been fitted, because that is the mode the fit is for and the moment
        the operator is about to need it.
        """
        for key in ("mode", "follow", "on_lost", "tip_thumb"):
            if options.get(key) is not None:
                setattr(self, key, options[key])
        if self.mode == "keyvector":
            # NOT started here. Selecting the mode is not the same as using
            # it, and a solver is a whole interpreter plus a few seconds of
            # Pinocchio - a session that leaves the mode on the default and
            # never presses FOLLOW should pay for none of that. `step` starts
            # it on the first frame that actually needs a solution.
            self._bridge_failed = False
        elif self.bridge.alive:
            # Leaving the mode stops the process. It is a whole interpreter
            # holding a UDP port; leaving it running to maybe be wanted again
            # is how ports end up occupied by things nobody remembers starting.
            self.bridge.stop()
        if options.get("mirror") is not None:
            self.retarget.mirror = bool(options["mirror"])
            # The ranges were fitted through the old frame, and spread is
            # measured relative to the thumb side, so half of them are now
            # backwards. Flexion survives; the sideways fit does not.
            for finger in FINGERS:
                self.retarget.ranges.pop((finger, "spread"), None)
        if options.get("thumb") is not None:
            self.retarget.thumb_mode = options["thumb"]
        for key in ("smoothing", "max_rate", "scale"):
            if options.get(key) is not None:
                setattr(self.retarget, key, float(options[key]))
        for role, value in (options.get("gains") or {}).items():
            if role in self.retarget.gains:
                self.retarget.gains[role] = float(value)
        if self.mode == "tip" and not self.retarget.calibrated \
                and self.autocalibrate:
            self.request_calibration()
        return self.status()

    def request_calibration(self):
        """Arm the tip mapping's fit. It lands on the next frames of an open
        hand, silently - this is the one that runs by itself when tip mode is
        switched on. The guided range fit is `start_capture`."""
        self.wants_calibration = True
        self.open_frames = 0

    # ------------------------------------------------------- guided capture
    def start_capture(self):
        """Begin the three-pose fit: open, fist, spread.

        This is the answer to "it moves, but it does not look like my hand". A
        fixed gain assumes the operator curls through the same angles the robot
        has, and nobody does - the Leap under-reports a DIP it cannot see,
        people differ by tens of degrees at full curl, and human finger spread
        is a sixth of the robot's. Fitted, a fist is a fist and flat is flat,
        whoever is doing it.

        Opens the sensor if it is not already open, but does NOT start the
        robot following: measuring a hand and being driven by one are separate,
        and calibrating with sixteen servos mirroring your fist is not an
        improvement.
        """
        if not self._open():
            return False
        self.capture = {"step": 0, "samples": deque(maxlen=SAMPLE_WINDOW),
                        "poses": {}, "open_hand": None}
        return True

    def cancel_capture(self):
        self.capture = None
        self._forget_hand()

    def _forget_hand(self):
        """Drop the last hand once nothing is reading the sensor.

        Otherwise `tracked` keeps saying yes after the last poll, and every
        readout downstream - the status line, the skeleton - describes a hand
        that stopped being watched some time ago.
        """
        self.filter.reset()
        if not self.reading:
            self.tracked = False
            self.hand = None

    def reset_calibration(self):
        """Back to fixed gains and no tip fit. The file stays until the next
        successful capture overwrites it."""
        self.capture = None
        self.retarget.clear_ranges()
        self.retarget.calibrated = False
        self.retarget.hand_scale = None
        self.retarget.offset = np.zeros(3)

    def _advance_capture(self, hand):
        """One frame of the guided fit. `hand` is None when tracking is lost.

        All this does is keep a short rolling window of what the hand is doing
        now. Nothing is committed until `capture_next`, so the operator can
        take as long as they like getting into the pose, and losing tracking
        halfway costs the window rather than the run.
        """
        if hand is None:
            self.capture["samples"].clear()
            return
        self.capture["samples"].append(self.retarget.measure(hand))
        if self.capture["step"] == 0:
            # The open hand fits the tip mapping too, so the operator is not
            # asked to hold the same pose twice for two different fits.
            self.capture["open_hand"] = hand

    def capture_next(self):
        """Commit the pose being held and move on. The NEXT button.

        Refuses rather than storing rubbish if the sensor cannot currently see
        a hand - a captured pose that is really "nothing was there" would set
        one end of a joint's range to a number nobody produced, and every
        later frame would be scaled against it.
        """
        if self.capture is None:
            return False
        samples = self.capture["samples"]
        if len(samples) < MIN_SAMPLES:
            self.error = "hold your hand where the sensor can see it"
            return False
        self.error = ""
        name = CAPTURES[self.capture["step"]][0]
        self.capture["poses"][name] = {
            key: float(np.mean([sample[key] for sample in samples]))
            for key in samples[0]}
        self.capture["step"] += 1
        if self.capture["step"] < len(CAPTURES):
            self.capture["samples"] = deque(maxlen=SAMPLE_WINDOW)
            return True
        self._finish_capture(self.capture)
        return True

    def _finish_capture(self, capture):
        fitted = self.retarget.fit_ranges(capture["poses"])
        if capture["open_hand"] is not None:
            self.retarget.calibrate(capture["open_hand"])
            self.wants_calibration = False
        self.capture = None
        self._forget_hand()
        if fitted or self.retarget.calibrated:
            self.retarget.save()
        self.error = "" if fitted else "calibration found no movement to fit"

    def capture_status(self):
        if self.capture is None:
            return None
        step, hint = CAPTURES[self.capture["step"]]
        samples = len(self.capture["samples"])
        return {"step": step, "hint": hint, "index": self.capture["step"],
                "total": len(CAPTURES), "samples": samples,
                "ready": samples >= MIN_SAMPLES,
                "last": self.capture["step"] == len(CAPTURES) - 1}

    # ----------------------------------------------------------------- poll
    def poll(self):
        """The newest hand, or None. Runs the calibration when it is armed.

        A frame that arrives with no hand in it is a loss; no frame at all
        just means the camera has not produced one since the last call, and
        the previous hand still stands.
        """
        if not self.reading:
            return None
        try:
            frame = self.connection.poll(0)
        except LeapError as failure:
            self.error = str(failure).splitlines()[0]
            self.enabled = False
            return None
        if frame is None:
            return self.hand if self.tracked else None
        self.frames += 1
        hand = frame.hand(self.follow)
        now = time.monotonic()
        dt = now - self._last_poll if self._last_poll else 1.0 / 100.0
        self._last_poll = now
        self.raw_hand = hand
        self._check_stale(hand, now)
        hand = self.filter(hand, dt)
        self.tracked = hand is not None
        self.hand = hand
        if hand is None:
            self.lost += 1
        # The RAW hand, for the same reason the capture below uses it: the
        # rest alignment measures a HELD pose, and the filter's lag is a cost
        # worth paying for live control and pure bias for a measurement.
        self._fit_rest(self.raw_hand)
        if self.capture is not None:
            # Calibrate from the RAW hand, not the filtered one. The filter
            # exists to make live control smooth, and its lag is a cost worth
            # paying there; here it is pure bias, because a pose captured
            # moments after the operator moved into it is still partly the
            # previous pose. Averaging thirty raw frames is a better estimate
            # of a held pose than one lagged sample of it, and unbiased.
            self._advance_capture(self.raw_hand)
        if hand is None:
            return None
        if self.wants_calibration:
            self._calibrate(hand)
        return hand

    def _check_stale(self, hand, now):
        """Is the sensor delivering frames that never change?

        Cheap fingerprint rather than a full comparison: the fingertips alone
        are enough, because nothing else moves while they stay put.
        """
        if hand is None:
            self._fingerprint = self._changed_at = None
            self.stale = False
            return
        mark = tuple(np.round(np.concatenate(
            [hand.bones[d][4] for d in ("thumb", "index", "middle")]), 4))
        if mark != self._fingerprint:
            self._fingerprint, self._changed_at = mark, now
            self.stale = False
        elif self._changed_at is not None:
            self.stale = (now - self._changed_at) > STALE_SECONDS

    def _calibrate(self, hand):
        """Counted over consecutive open frames, not taken from the first one,
        so a hand passing through open on its way to a fist does not get to
        define the neutral pose."""
        if hand.grab_strength > OPEN_GRAB:
            self.open_frames = 0
            return
        self.open_frames += 1
        if self.open_frames >= CALIBRATION_FRAMES \
                and self.retarget.calibrate(hand):
            self.wants_calibration = False

    def rest_keypoints(self):
        """A flat hand, as 21 MANO keypoints, taken from the ROBOT's own zero.

        WHY THE ROBOT'S GEOMETRY AND NOT A REMEMBERED HAND. After the rest
        alignment, a flat operator hand IS the robot at joint zero - that is
        precisely what `bone_offsets` is fitted to make true - and the digit
        scaling maps the operator onto these proportions anyway. So the robot's
        own forward kinematics at zero is the correct prior, and it needs no
        recording, no operator and no sensor.

        The pinky has no robot counterpart, so it is placed one ring-to-middle
        spacing further out along the same line. No solver term reads it; the
        student does, and a slot left at the wrist would be a keypoint sitting
        somewhere no little finger has ever been.
        """
        zero = np.zeros(len(self.retarget.joint_names))
        points = np.zeros((21, 3))
        chains = {}
        for finger in FINGERS:
            links = [f"{finger}_proximal_link", f"{finger}_middle_link",
                     f"{finger}_distal_link", TIP_LINKS[finger]]
            chains[finger] = [self.retarget.model.link_pose(
                link, zero, self.retarget.joint_names)[0] for link in links]
        step = np.asarray(chains["ring"][0]) - np.asarray(chains["middle"][0])
        chains["pinky"] = [np.asarray(p) + step for p in chains["ring"]]

        # NOT used to prime the student's window - see `prime`. The robot has
        # no metacarpal, so its proximal link IS its knuckle, and there is no
        # honest way to place a MANO wrist against it: anchoring on the palm
        # origin left the two hands 33 mm apart and averaging the knuckles made
        # it 60. This is for DRAWING lane 0 when nothing is tracked, where the
        # anchor is cosmetic and the shape is the point.
        for slot, digit in enumerate(DIGITS):
            for k, place in enumerate(chains[digit]):
                points[4 * slot + 1 + k] = np.asarray(place)
        return points

    def prime(self, keypoints=None, raw=None):
        """Fill what a downstream reader would otherwise find empty.

        THE POINT IS TO REMOVE A DISCONTINUITY. Left empty, the student has
        nothing to read for its first twenty frames and the caller falls back
        to the angle mapping - so the first thing the operator sees is one
        mapping and the second is a jump to another.

        PRIMED WITH THE FIRST REAL FRAME, repeated, rather than with a
        synthetic flat hand. A synthetic one has to be anchored against the
        robot's geometry, and the robot has no metacarpal to hang a MANO wrist
        on: anchored on the palm origin the two hands sat 33 mm apart, and
        averaging the knuckles made it 60. Repeating the operator's own first
        aligned frame is exact by construction and needs no anchor at all - the
        window then says "the hand has been here all along", which is the only
        history that cannot be wrong.
        """
        if keypoints is not None and self._window.maxlen:
            self._window.clear()
            self._raw_window.clear()
            for _ in range(self._window.maxlen):
                self._window.append(keypoints)
                self._raw_window.append(raw)
        self.alternates.setdefault("keyvector", np.zeros(
            len(self.retarget.joint_names)))
        self.alternates.setdefault("future", np.zeros(
            len(self.retarget.joint_names)))
        self.alternates.setdefault("student", np.zeros(
            len(self.retarget.joint_names)))

    def _fit_rest(self, hand):
        """Accumulate open frames until the rest alignment can be fitted.

        Watches the same signal the autocalibration does - a hand the SDK calls
        open - because "flat" is what both are asking for. Fitted once and then
        held: it describes this operator's hand, not this frame.
        """
        # `hand` is None on a frame the sensor delivered with nothing in it,
        # which is most of them when the operator's hand is out of the cone.
        # Tracking drops constantly; this is not an error path.
        if hand is None or self.bone_offsets is not None \
                or hand.grab_strength > OPEN_GRAB:
            return
        self._rest_frames.append(hand)
        if len(self._rest_frames) < REST_FRAMES:
            return

        robot = {}
        zero = np.zeros(len(self.retarget.joint_names))
        for finger in FINGERS:
            links = [f"{finger}_proximal_link", f"{finger}_middle_link",
                     f"{finger}_distal_link", TIP_LINKS[finger]]
            places = [self.retarget.model.link_pose(link, zero,
                                                    self.retarget.joint_names)[0]
                      for link in links]
            robot[finger] = [(b - a) / np.linalg.norm(b - a)
                             for a, b in zip(places[:-1], places[1:])]

        # Averaged through the nearest rotation, not elementwise: the mean of
        # rotation matrices is not a rotation, and feeding one to the chain
        # would scale the bones as well as turning them.
        totals = {f: [np.zeros((3, 3)) for _ in range(3)] for f in FINGERS}
        for frame in self._rest_frames:
            found = rest_offsets(frame, self.retarget.rotation(frame), robot)
            for finger in FINGERS:
                for k in range(3):
                    totals[finger][k] += found[finger][k]
        offsets = {}
        for finger in FINGERS:
            turns = []
            for k in range(3):
                left, _, right = np.linalg.svd(totals[finger][k]
                                               / len(self._rest_frames))
                turn = left @ right
                if np.linalg.det(turn) < 0:
                    left[:, -1] *= -1
                    turn = left @ right
                turns.append(turn)
            offsets[finger] = turns
        self.bone_offsets = offsets
        self._rest_frames = []

    # ----------------------------------------------------------------- step
    def step(self, sim=None, dt=1.0 / 60.0):
        """One frame: poll, retarget, and hand back a full joint vector.

        `sim` is only needed for tip mode, and tip mode without one silently
        becomes angle mode rather than failing - `--print` has no simulator to
        solve with, and the angles are still worth watching.
        """
        hand = self.poll()
        if hand is None:
            self.tip_targets = {}
            return (self.retarget.relax() if self.on_lost == "open"
                    else self.retarget.command)

        if self.mode == "raw":
            # One to one, nothing in between. See Retargeter.to_vector_raw.
            self.tip_targets = {}
            return self.retarget.direct(
                self.retarget.to_vector_raw(self.retarget.measure(hand)), dt)
        if self.mode == "posebank":
            return self.retarget.accept(self._solve_posebank(hand), dt)
        if self.mode in ("student", "keyvector"):
            # Whichever is commanding, run the other too - see `alternates`.
            other = "keyvector" if self.mode == "student" else "student"
            try:
                spare = (self._solve_keyvector(hand) if other == "keyvector"
                         else self._solve_student(hand))
            # Broad: the comparison lane is a convenience and must never be
            # able to stop the hand that is actually being driven.
            except Exception:
                spare = None
            if spare is not None and other == "keyvector":
                self.alternates[other] = np.asarray(spare)
            # The student lane is NOT set from `spare`. `_solve_student` has
            # already filed the network's RAW answer there and what it returns
            # is the repaired one, so assigning it here made the STUDENT lane
            # mean the raw pose while the student commands and the repaired
            # pose while the keyvector solver does - the same lane showing two
            # different things depending on a mode switch.

        if self.mode == "student":
            solved = self._solve_student(hand)
            if solved is not None:
                self.alternates["student"] = np.asarray(solved)
                return self.retarget.accept(solved, dt)
            # Nothing yet - the model is loading, the rest alignment is not
            # fitted, or the window is not full. The angle mapping needs none
            # of those and always answers.
            return self.retarget.joints(hand, dt)
        if self.mode == "keyvector":
            solved = self._solve_keyvector(hand)
            if solved is not None:
                self.alternates["keyvector"] = np.asarray(solved)
                return self.retarget.accept(solved, dt)
            # No solution yet - the solver is still starting, or gone. The
            # angle mapping needs nothing and always answers, so the hand
            # keeps following while that resolves rather than freezing.
            return self.retarget.joints(hand, dt)
        if self.mode == "tip" and sim is not None:
            solved = self._solve_tips(sim, hand)
            return self.retarget.accept(solved, dt)
        self.tip_targets = {}
        return self.retarget.joints(hand, dt)

    def _solve_posebank(self, hand):
        """The nearest vetted pose to the one you are making.

        No optimiser and no solver process: describe the hand, look the
        description up, blend the neighbours. Every answer is a pose that was
        checked when the bank was built, so it cannot be a contorted one -
        which is the guarantee an optimiser cannot give, since an optimiser
        will hand you any pose that minimises its residual.

        The cost of that guarantee is a ceiling: it can never do better than
        what is in the bank, and it steps between entries rather than moving
        continuously. The rate limit downstream smooths the steps.
        """
        self.tip_targets = {}
        self.bank.ready()
        rotation = self.retarget.rotation(hand)
        wrist = np.mean([hand.bones[d][0] for d in ("index", "middle", "ring")],
                        axis=0)
        tips = np.array([rotation @ (hand.tip(f) - wrist) for f in FINGERS])
        # the same three bones per digit, in the same order, as the bank's
        bones = []
        for finger in FINGERS:
            joints = hand.bones[finger]
            for index in (1, 2, 3):
                bones.append(rotation @ (joints[index + 1] - joints[index]))
        pose = self.bank.nearest(self.bank.key_for(tips, np.array(bones)))
        return self.retarget.command if pose is None else pose

    def _solve_student(self, hand):
        """One forward pass of the trained network. None until it can answer.

        CAUSAL AND WINDOWED, because that is what it was trained as: it reads
        the last `window` frames of keypoints and nothing later. Until that many
        have arrived the caller falls back to the angle mapping, which is the
        same thing the solver path does while the bridge is starting.

        The window is filled with the SAME keypoints the solver is sent - the
        rest alignment included - so the network is shown the distribution it
        was trained on rather than a differently-prepared one. Beside it, the
        raw view of the same frames: rotation only, the view a distance is
        measured in. See `student.features`.
        """
        self.tip_targets = {}
        if self.student is None:
            if self._student_tried:
                return None
            self._student_tried = True
            if not self.load_student():
                return None
        if self.bone_offsets is None:
            # No rest alignment yet means the keypoints are not yet the ones
            # the student was trained on. Feeding them anyway would look like
            # a bad model rather than an unfitted operator.
            return None
        points, raw = self.keypoints(hand), self.raw_keypoints(hand)
        if not self._window:
            # First aligned frame: fill the whole window with it, so the
            # network answers now rather than after another twenty.
            self.prime(points, raw)
        else:
            self._window.append(points)
            self._raw_window.append(raw)
        heads = self.student_heads(np.stack(self._window),
                                   np.stack(self._raw_window))
        # The forward head, kept beside the present one every frame: the
        # AHEAD lane in the live world is this, and it has to move whether
        # or not the student is the mapping in command.
        if FUTURE_HORIZON in heads:
            self.alternates["future"] = heads[FUTURE_HORIZON]
        # REPAIRED before it is commanded. The STUDENT lane keeps the raw
        # answer, so the viewer shows what the network said and the robot
        # shows what it was allowed to do - the difference is the repair.
        self.alternates["student"] = heads[0]
        return self.repair(heads[0])

    def repair(self, q):
        """The network's answer, nudged clear of itself. See
        `learning.collision.repair` - a few gradient steps on the same proxy
        the teacher solved against, so a pose four degrees off a clear
        label does not put a finger through its neighbour."""
        if self._repair is None:
            # Say so in the telemetry too: leaving the last student's figures
            # standing had the panel reporting a repair that was no longer
            # running.
            self.repair_ms, self.repair_deg = 0.0, 0.0
            return q
        import torch

        from .learning import collision
        from .learning.objective import COLLIDE_TOL_MM

        fk, proxy = self._repair
        began = time.perf_counter()
        fixed = collision.repair(fk, proxy, torch.as_tensor(
            np.asarray(q, dtype=np.float32))[None], tolerance=COLLIDE_TOL_MM)[0].numpy()
        self.repair_ms = (time.perf_counter() - began) * 1000.0
        self.repair_deg = float(np.degrees(np.abs(fixed - q)).max())
        return fixed

    def load_student(self, path=None):
        """Bring up the trained network. Reports rather than raises."""
        import torch

        from .learning.student import Student
        from .paths import MODELS

        path = Path(path or MODELS / "student.pt")
        if not path.exists():
            self.student_error = (f"no student at {path} - "
                                  f"train one with `learn.py train`")
            return False
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            # THE FLAGS FIRST. A checkpoint records whether it was trained
            # with the sensor's error model as input and with the grasp head,
            # and both change the network that has to be built to receive it -
            # the input width and the head list.
            #
            # PASSED, NOT ASSIGNED. This used to set `student.BELIEF` and
            # `student.FEATURES` on the module before constructing, which the
            # panel makes dangerous: it loads students one after another in
            # one process, so a belief student left the module claiming 102
            # features for every baseline student loaded after it. The width
            # also came from a second hand-written copy of the sum, which is
            # the kind of duplicate that drifts.
            from .learning.student import feature_width
            trained_belief = bool(blob.get("belief"))
            expected = feature_width(trained_belief)
            if int(blob["features"]) != expected:
                raise ValueError(f"it wants {int(blob['features'])} features "
                                 f"and this build makes {expected} - rerun "
                                 f"`learn.py train`")
            model = Student(blob["mean"], blob["basis"],
                            channels=blob["channels"],
                            belief=trained_belief,
                            authority=bool(blob.get("authority")))
            model.load_state_dict(blob["state"])
            # Read inside the try as well: these two used to be fetched after
            # it, so a checkpoint predating either key raised KeyError out of
            # `load_student` - and in student mode straight out of
            # `Mimic.step`, past the viewer's loop - against this module's
            # promise that nothing here raises at the caller.
            stored = [str(n) for n in blob["joint_names"]]
            window = int(blob["window"])
            horizons = tuple(blob.get("horizons", (0,)))
        # Broad: a checkpoint from an older architecture, an unreadable file
        # and a torch that cannot load it all land here, and none of them
        # should end the session.
        except Exception as failure:
            self.student_error = f"cannot load {path.name}: {failure}"
            return False
        if stored != list(self.retarget.joint_names):
            self.student_error = ("that student was trained on a different "
                                  "joint order than this model has")
            return False
        if trained_belief:
            # It was trained on the sensor's error model, so the live path has
            # to compute the same channels from the same fit.
            from .learning.belief import Belief
            from .paths import NOISE_FIT
            try:
                model.belief = Belief.load(NOISE_FIT)
                model.belief.gate = bool(blob.get("contact_gate"))
            except Exception as failure:
                self.student_error = (f"that student needs the noise fit and "
                                      f"it cannot be read: {failure}")
                return False
        model.eval()
        self.student = model
        self.student_horizons = horizons
        self._window = deque(maxlen=window)
        self._raw_window = deque(maxlen=window)
        self.student_error = ""
        # The proxy the teacher was solved against, for the repair layer.
        # Cached beside the model, so this is a file read after the first.
        from .learning import collision
        from .learning.kinematics import ChainFK
        fk = ChainFK(self.retarget.model, stored)
        lower, upper = self.retarget.model.limits(stored)
        self._repair = (fk, collision.for_hand(fk, lower, upper,
                                               self.retarget.chirality()))
        # The lanes get a pose now; the window fills from the first aligned
        # frame - see `prime`.
        self.prime()
        return True

    def student_heads(self, window, raw):
        """(frames, 21, 3) aligned and (frames, 21, 3) raw -> {horizon: q}.

        Every head from one forward pass: 0 is the present, and each larger
        horizon is a prediction that many frames ahead. Commanding a forward
        head is how the lag of any downstream smoothing gets paid for - see
        `student.HORIZONS`.
        """
        import torch

        from .learning.student import Student

        began = time.perf_counter()
        with torch.no_grad():
            batch = torch.from_numpy(
                np.asarray(window, dtype=np.float32)[None])
            raw_batch = torch.from_numpy(
                np.asarray(raw, dtype=np.float32)[None])
            heads = self.student.predict(batch, raw_batch)
            out = heads["q"][0, -1].numpy()                          # (horizons, joints)
            # The contact head, kept rather than dropped: the executor reads
            # it to tell a pinch from a grasp - see `executor.pinched`. It is
            # one probability per thumb pair, newest frame.
            self.student_contact = torch.sigmoid(
                heads[Student.CONTACT][0, -1]).numpy()
        self.student_ms = (time.perf_counter() - began) * 1000.0
        return {horizon: out[index] for index, horizon in enumerate(self.student_horizons)}


    def _solve_keyvector(self, hand):
        """One frame through the DexPilot solver. Returns None until it answers.

        Send, then read whatever has arrived - never wait. The solve is a
        couple of milliseconds on loopback, so the answer is at worst a frame
        old, and a blocking read would put the solver's worst case straight
        into the simulation loop.
        """
        self.tip_targets = {}
        if not self.bridge.alive:
            if self._bridge_failed:
                return None
            if not self.start_bridge():
                self._bridge_failed = True
            return None
        if not self._scaled:
            # Proportions need a hand to have been seen, which is after the
            # solver had to start. Measured once, on the first hand.
            self.digit_scale = self._fit_proportions(hand)
            # The keypoints are already scaled per digit, so the solver's own
            # global factor stays at one rather than scaling them twice.
            self.bridge.set_scaling(1.0)
            self._scaled = True
        self.bridge.send(self.keypoints(hand))
        return self.bridge.poll()

    def keypoints(self, hand):
        """The 21 MANO keypoints, exactly as the offline pipeline builds them.

        One method, so the live solver and the student are fed the same thing -
        and so the rest alignment cannot be applied to one and forgotten on the
        other, which is what it was."""
        return mano_keypoints(hand, self.retarget.rotation(hand),
                              digit_scale=self.digit_scale,
                              bone_offsets=self.bone_offsets)

    def raw_keypoints(self, hand):
        """The same frame in the robot's axes and nothing else: no digit
        scale, no rest offsets. The view a distance survives in - what
        `sessions.raw_keypoints_for` builds offline, built here from the live
        hand so the student's distance features are the ones it trained on."""
        return mano_keypoints(hand, self.retarget.rotation(hand))

    def _solve_tips(self, sim, hand):
        """Fingertip IK, threaded through one joint vector.

        The seed is the angle mapping's answer for this same frame, and that
        is doing two jobs. Any chain not solved here - the thumb, normally -
        keeps its angle-mapped value instead of freezing. And a four-DOF
        finger reaching a three-DOF point has a curve of solutions to choose
        from, so starting the solver at the pose the operator's own finger is
        in picks the human-looking one rather than whichever branch it drifted
        onto last frame.

        `q` then carries forward from finger to finger, so the four results
        compose into one hand rather than four hands of which only the last
        survives.
        """
        fingers = FINGERS if self.tip_thumb == "ik" else TIP_FINGERS
        targets = self.retarget.tips(hand, origin=sim.position)
        q = self.retarget.to_vector(self.retarget.measure(hand))
        errors = {}
        for finger in fingers:
            solved, error = sim.solve_tip(finger, targets[finger], init=q,
                                          iterations=12)
            for name in sim.chains[finger]:
                index = sim.joint_names.index(name)
                q[index] = solved[index]
            errors[finger] = float(error)
        self.tip_targets = {f: targets[f] for f in fingers}
        self.tip_errors = errors
        return q

    def _fit_proportions(self, hand):
        """How much longer each of the robot's digits is than the operator's.

        One number per digit, not one for the hand: the robot's fingers are
        about 1.04 times this operator's and its thumb 1.27 times, and a single
        factor cannot serve both. The pinky has no robot counterpart and is
        left alone.
        """
        human = digit_lengths(hand)
        scale = {}
        for finger in FINGERS:
            reach = float(np.linalg.norm(self.retarget.rest[finger]))
            if human.get(finger, 0.0) > 1e-6:
                scale[finger] = reach / human[finger]
        return scale

    # ----------------------------------------------------------------- draw
    def skeleton(self, origin=(0.0, 0.0, 0.0), offset=None):
        """The operator's hand in robot world metres, or {} if there is none.

        `offset` lifts it clear of the robot for the viewer. The panel asks
        without one, because there the two hands superimposed is the check.
        """
        if self.hand is None or not self.tracked:
            # The flat hand rather than nothing. An empty lane 0 is a hole in
            # the row, and every other lane is already showing its own idea of
            # a flat hand - see `rest_keypoints`.
            if origin is None:
                return {}
            points = self.rest_keypoints()
            place = np.asarray(origin, dtype=float) + (
                np.zeros(3) if offset is None else np.asarray(offset, float))
            return {digit: [(points[4 * slot + 1 + k] + place).tolist()
                            for k in range(4)]
                    for slot, digit in enumerate(DIGITS)}
        if offset is None:
            offset = (0.0, 0.0, 0.0)
        origin = np.asarray(origin, dtype=float) + np.asarray(offset, dtype=float)
        return self.retarget.skeleton(self.hand, origin)

    def status(self):
        """What the panel needs to draw the tab. JSON-safe throughout."""
        hand = self.hand if self.tracked else None
        return {
            "enabled": self.enabled,
            "available": self.available,
            "mode": self.mode,
            "thumb": self.retarget.thumb_mode,
            "tip_thumb": self.tip_thumb,
            "follow": self.follow,
            "expected": self.expected,
            "mirror": self.retarget.mirror,
            "mismatched": self.mismatched,
            "on_lost": self.on_lost,
            "tracked": bool(hand) and not self.stale,
            "stale": self.stale,
            "type": hand.type if hand else "",
            "pinch": float(hand.pinch_strength) if hand else 0.0,
            "grab": float(hand.grab_strength) if hand else 0.0,
            "frames": self.frames,
            "lost": self.lost,
            "filtered": self.filter.enabled,
            "spikes": self.filter.rejected,
            "calibrated": self.retarget.calibrated,
            "calibrating": self.wants_calibration,
            "fitted": len(self.retarget.ranges),
            "capture": self.capture_status(),
            "smoothing": self.retarget.smoothing,
            "scale": self.retarget.scale,
            "spread_gain": self.retarget.gains["spread"],
            "error": self.error,
            "miss": {f: self.tip_errors.get(f) for f in TIP_FINGERS}
                    if self.mode == "tip" else {},
            "solver": self.bridge.status(),
            "student": {
                "loaded": self.student is not None,
                # The rest alignment is fitted from open frames and gates the
                # student entirely: without it the keypoints are not the ones
                # it was trained on, and it would look like a bad model rather
                # than an unfitted operator.
                "aligned": self.bone_offsets is not None,
                "window": len(self._window),
                "needs": self._window.maxlen,
                "ready": len(self._window) >= (self._window.maxlen or 1),
                "ms": self.student_ms,
                "ahead": FUTURE_HORIZON if FUTURE_HORIZON in self.student_horizons else 0,
                "repair_deg": round(self.repair_deg, 2),
                "repair_ms": round(self.repair_ms, 2),
                "error": self.student_error,
            },
            "bank": {"poses": 0 if self.bank.poses is None else len(self.bank.poses)},
        }
