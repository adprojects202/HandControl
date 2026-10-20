"""Learning the retargeting instead of solving it every frame.

The package beside this one maps a human hand onto this robot by measuring
angles or by optimising keyvectors, once per frame, live. Both work and both
have the same ceiling: a per-frame optimiser is a DISCONTINUOUS function of
its input, and no downstream filter repairs a discontinuity. A network is
continuous by construction, which is the property being bought here.

WHAT SHAPED THIS PACKAGE, in the order it was learned:

  Solving harder does not make the mapping more accurate. Offline, with no
  time budget, the live solver's own objective scored the same pinch error
  and seven times less jerk. Smoothness is what offline buys.

  The objective was the ceiling, and the teacher's own error had never been
  measured. Scored against the operator instead of against itself: 9.35
  degrees of bone direction, competitive with published systems, and unable
  to make a single pinch - because a pinch does not survive the per-digit
  scaling and rest rotation the keypoints go through. `objective.py` carries
  that measurement and the objective built on it: two views of the hand, a
  task-space thumb, contact gated on the sensor's own intent. Fingers better
  on every block; contact from 0% to 95%, and the table that says so is in
  `learning/README.md`.

  A learned teacher does not beat a converged optimiser on that optimiser's
  objective, and six network variants on the corrected one landed far off
  the per-frame floor. So the teacher stays an optimiser; the network is the
  student.

THE FILES, layered by how much each knows about this robot:

    sessions.py    recorded Leap sessions: pooling, runs, both keypoint views.
    noise.py       what the sensor's error looks like, and how to reproduce it.
    synthetic.py   plausible human hands from nothing at all.
    gestures.py    authored shapes the student never trains on.
    kinematics.py  the finger chain in torch - one FK for teacher and student.
    collision.py   the links as boxes, offsets measured off the meshes.
    playback.py    the gesture probe through the live world's lanes.
    geometry.py    the hand as distances: every tip pair, every curl, and
                   time-to-contact - the one label only a teacher can compute.
    objective.py   what the teacher minimises, with the number behind each term.
    teacher.py     the offline solve, run by run, warm-started from the live solver.
    student.py     the causal network that copies the teacher, fast.
    train.py       pretrain on synthetic, fine-tune on real, score held-out blocks.
    replay.py      watch any of it in the MuJoCo viewer.
    gallery.py     sixteen hands at once - the quality of a set, at a glance.
    augment.py     the fitted sensor noise, replayed onto synthetic hands.
    validate.py    the frame gate: refuses to write pairs in the wrong axes.
    survey.py      the recording script and the block table it runs.
    report.py      the noise fit, printed and plotted.
    selftest.py    the invariants this package keeps breaking.

`learn.py` at the root is the one command line over all of it.
"""

from .sessions import (hands_from, keypoints_for, load,
                       runs_for, sessions_under, tracked_runs)

__all__ = ["hands_from", "keypoints_for", "load", "runs_for",
           "sessions_under", "tracked_runs"]
