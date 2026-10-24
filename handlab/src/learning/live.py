"""Live hand control: the robot hand following the hand the camera sees, as
it moves - the trained hand-motion policy in a hand sim on this Mac, its
servo targets the joint targets.

    live = LiveHand(camera, send, names)
    live.start()          # the run learning.hand_run names, else the newest
    live.running, live.status, live.error, live.side, live.angle_deg
    live.stop()

A thread steps at RATE (the policy's 20 Hz): it takes the newest camera
frame's hand - the left one when both are seen (the robot is a left hand),
else the one seen, a right hand mirrored onto the left - as the shape to
take; the policy moves the hand sim's servo targets one step
(hand_follow.CpuHandEnv), and `send(targets)` gets them, in the robot's
joint order. With no hand in view, the hand holds where it is. Live, only
the shape now is known, so the policy is aimed at it for every step ahead
(CpuHandEnv.aim): it follows a little later than on a recording.

Nothing here moves the real hand by itself: `send` sets the controller's
targets (the purple command hand); the hardware gets them only while
streaming (Hardware section), slew-limited as always.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import torch

from .hand_follow import CpuHandEnv, load_policy
from .handmotion import FEATURES, RATE, errors, human_features
from .train import chosen_run

HAND = "hand motion"


class LiveHand:
    def __init__(self, camera, send, names):
        """`camera`: vision.Camera; `send(targets)`: the joint targets, in
        the order of `names` (the robot's)."""
        self.camera = camera
        self.send = send
        self.names = list(names)
        self.status = "off"
        self.error = ""
        self.side = None                 # the hand followed now, or None
        self.angle_deg = np.nan          # the robot's shape off the person's
        self.run = None
        self._thread = None
        self._stop = threading.Event()

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start following; False (and `error` says why) without a trained run."""
        if self.running:
            return True
        self.run = chosen_run(HAND)
        if self.run is None:
            self.error = "no hand-motion run yet (Learning, learn: hand motion)"
            return False
        self.error, self.status = "", "loading the policy"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(self.run, self._stop),
                                        name="handlab-live", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.status, self.side = "off", None

    def _loop(self, run, stop):
        try:
            env = CpuHandEnv(np.zeros((1, FEATURES)))
            if env.joint_names != self.names:
                raise RuntimeError("the hand sim's joints are not the robot's order")
            policy = load_policy(env, run)
            period = 1.0 / RATE
            due = time.monotonic()
            while not stop.is_set():
                self._step(env, policy)
                due += period
                wait = due - time.monotonic()
                if wait < 0:                       # fell behind: start counting afresh
                    due, wait = time.monotonic(), 0.0
                stop.wait(wait)
        except Exception as failure:             # live stops, the app goes on
            self.error = str(failure) or type(failure).__name__
            self.status = "stopped"

    def _step(self, env, policy):
        frame = self.camera.latest()
        if frame is None:
            self.status, self.side = "waiting for the camera", None
            return
        hand = _pick(frame.hands)
        if hand is None:
            self.status, self.side = "no hand in view - holding", None
            return
        env.aim(human_features(hand.world.astype(float), hand.side))
        with torch.inference_mode():
            env.step(policy(env.get_observations()))
        self.send(env.targets.copy())
        angle, _, _ = errors(env.shape_now(), env.goal())
        self.side, self.angle_deg = hand.side, float(np.degrees(angle[0]))
        self.status = f"following your {hand.side.lower()} hand"


def _pick(hands):
    """The hand to follow: the left one when both are seen, else the one seen."""
    if not hands:
        return None
    left = [hand for hand in hands if hand.side == "Left"]
    return (left or sorted(hands, key=lambda hand: -hand.score))[0]
