#!/bin/zsh
# The measurement queue behind the five-way comparison - baseline, no-object
# labels, the direct objective, the belief channels and the contact gate.
#
#     ./experiments.sh status   what has finished, what is left
#     ./experiments.sh run      run whatever is not done yet
#
# CONCLUDED. Every result this queue produced is written up in
# handcontrol/learning/README.md, and the models are in data/archive/. It is
# kept as the exact record of how those numbers were made, and it now needs
# `run` to start: a bare `./experiments.sh` used to launch it, and with the
# outputs archived that meant sixteen stages and about five hours re-training
# a finished comparison - on labels since re-solved with four more survey
# blocks, so it would not even reproduce the numbers it is kept for.
#
# WHY A SCRIPT AND NOT THREE COMMANDS. These have to run one at a time - two
# training runs on one GPU are slower than the same two in sequence and can
# exhaust it - and each stage is worth nothing until the one before it has
# finished. Every stage checks for its own output first, so stopping this with
# ctrl-c and starting it again picks up where it left off rather than redoing
# an hour of solving.
#
# Roughly 2.5 hours from cold. Logs land in logs/experiments/.
set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
OUT=logs/experiments
PROGRESS=$OUT/progress.log
mkdir -p $OUT

note() { print -r -- "$(date +%H:%M:%S)  $*" | tee -a $PROGRESS }

# Run one stage unless the file it produces is already there.
stage() {
  local name=$1 produces=$2; shift 2
  if [[ -e $produces ]]; then
    note "skip  $name — $produces is already there"
    return 0
  fi
  note "start $name"
  if "$@" > $OUT/$name.log 2>&1; then
    note "done  $name"
  else
    note "FAILED $name — see $OUT/$name.log"
    return 1
  fi
}

if [[ "${1:-run}" == "status" ]]; then
  printf "\n  STAGE                          OUTPUT                              DONE\n"
  for row in \
    "re-solve teacher|data/leap/20260902-124152/teacher.npz" \
    "synthetic set|data/synthetic/seed0/pairs.npz" \
    "gesture probe|data/synthetic/gestures/pairs.npz" \
    "baseline seed 0|data/models/student_seed0.pt" \
    "baseline seed 1|data/models/student_seed1.pt" \
    "baseline seed 2|data/models/student_seed2.pt" \
    "no-object labels|data/synthetic/noobj/pairs.npz" \
    "no-object seed 0|data/models/noobj_seed0.pt" \
    "no-object seed 1|data/models/noobj_seed1.pt" \
    "no-object seed 2|data/models/noobj_seed2.pt" \
    "direct 0.01|data/models/direct_0.01.pt" \
    "direct 0.05|data/models/direct_0.05.pt" \
    "direct 0.20|data/models/direct_0.20.pt" \
    "belief seed 0|data/models/belief_seed0.pt" \
    "belief seed 1|data/models/belief_seed1.pt" \
    "belief seed 2|data/models/belief_seed2.pt" \
    "direct 0.01 seed 1|data/models/direct001_seed1.pt" \
    "direct 0.01 seed 2|data/models/direct001_seed2.pt" \
    "gated belief seed 0|data/models/gated_seed0.pt" \
    "gated belief seed 1|data/models/gated_seed1.pt" \
    "gated belief seed 2|data/models/gated_seed2.pt"
  do
    local label=${row%%|*} file=${row##*|}
    [[ -e $file ]] && local mark="yes" || local mark="—"
    printf "  %-30s %-35s %s\n" "$label" "$file" "$mark"
  done
  printf "\n"
  [[ -f $PROGRESS ]] && { print -r -- "  last lines of $PROGRESS:"; tail -5 $PROGRESS | sed 's/^/    /' }
  exit 0
fi

if [[ "${1:-}" != "run" ]]; then
  print -r -- ""
  print -r -- "  This comparison is concluded - see handcontrol/learning/README.md."
  print -r -- "  Its models are in data/archive/, so a run would re-train all of it."
  print -r -- ""
  print -r -- "    ./experiments.sh status   see what is on disk"
  print -r -- "    ./experiments.sh run      start it anyway"
  print -r -- ""
  exit 1
fi

note "=== queue begins"

# ------------------------------------------------------- regenerate first
# THE HAND CHANGED. `world/h4nd_left` replaced `h4nd_2`: natively left where
# the old one was a mirrored right, the thumb's spurious 15-degree rolls gone,
# and the adduction limits tightened. Every label on disk was solved against
# the old geometry - re-solving one block took thumb_tap contact from 19% back
# to 74%, which is the measurement that says the labels were stale rather than
# the hand wrong. Nothing below means anything until these three have run.
stage "teacher" data/leap/20260902-124152/teacher.npz \
  $PY -u learn.py teacher

stage "synth" data/synthetic/seed0/pairs.npz \
  $PY -u learn.py synth --seconds 6 --seed 0

stage "gestures" data/synthetic/gestures/pairs.npz \
  $PY -u learn.py gestures

# ---------------------------------------------------------------- item 1
# Three seeds of the SHIPPED setting. Nothing below is readable without this:
# two seeds already differ by 2.66 degrees, which is larger than most of the
# differences the table is being asked to explain.
for s in 0 1 2; do
  stage "baseline-seed$s" data/models/student_seed$s.pt \
    $PY -u learn.py train --epochs 16 --workers 4 --seed $s --out data/models/student_seed$s.pt
done

# ---------------------------------------------------------------- item 2
# Labels the student can actually reproduce. A fifth of synthetic frames were
# solved with a cylinder in the objective that the student cannot observe and
# never will - the Leap tracks a hand, not the world. Objects move to the
# executor, which discovers them by closing.
stage "noobj-synth" data/synthetic/noobj/pairs.npz \
  $PY -u learn.py synth --seconds 6 --seed 0 --object-rate 0 --out data/synthetic/noobj

for s in 0 1 2; do
  stage "noobj-seed$s" data/models/noobj_seed$s.pt \
    $PY -u learn.py train --epochs 16 --workers 4 --seed $s \
        --synthetic data/synthetic/noobj --out data/models/noobj_seed$s.pt
done

# ---------------------------------------------------------------- item 3
# The teacher's own objective, scored on the student's pose. One seed per
# weight first: this is choosing a SCALE, and the three weights are an order
# of magnitude apart, so a seed's worth of noise cannot hide which is right.
# Three seeds at the winner comes after, once there is a winner.
for w in 0.01 0.05 0.20; do
  stage "direct-$w" data/models/direct_$w.pt \
    $PY -u learn.py train --epochs 16 --workers 4 --seed 0 --direct $w \
        --out data/models/direct_$w.pt
done

# ------------------------------------------------------- item 3, confirmed
# ONE WEIGHT SURVIVED, and the sweep is why the other two are not here.
# `pooled` is the same bone-angle array `objective.per_frame` minimises, so
# training on it and reporting it is circular - the 13.02 to 8.11 degree drop
# at w=0.20 is the optimiser working, not the mapping improving. What is NOT
# circular says the opposite at that weight: self-penetration 2.3% against the
# baseline's 0.3-0.7, and 3.5 degrees further from the teacher.
#
# At 0.01 nothing independent gets worse and the pinch gap halves, 0.6 to
# 0.3 mm. That is one seed against a 1.38 degree spread, so it is a lead and
# not yet a result - hence three.
for s in 1 2; do
  stage "direct001-seed$s" data/models/direct001_seed$s.pt \
    $PY -u learn.py train --epochs 16 --workers 4 --seed $s --direct 0.01 \
        --out data/models/direct001_seed$s.pt
done

# ---------------------------------------------------------------- item 4
# The sensor's own error as input. Three seeds, because the gate is a
# difference and the spread on these labels is 1.38 degrees.
for s in 0 1 2; do
  stage "belief-seed$s" data/models/belief_seed$s.pt \
    $PY -u learn.py train --epochs 16 --workers 4 --seed $s --belief \
        --out data/models/belief_seed$s.pt
done

# ---------------------------------------------------------------- item 4b
# THE ONE CHANGE THE MEASUREMENTS ARGUE FOR. The belief channels bought 0.96
# degrees against the teacher over three seeds, with the two groups' ranges
# not overlapping - and made every contact number worse by the same standard:
# pinch gap 0.5-0.6 mm to 1.2-1.6, penetration 0.3-0.7% of frames to 0.8-1.6.
#
# The cause is known rather than guessed. `curl` is the live proxy for
# self-occlusion, so the channels distrust a digit exactly as it folds, and a
# folding digit is usually one about to touch something. `contact_gate` scales
# sigma down as a pinch closes, on a ramp measured to separate a real pinch
# (16-20 mm) from a fist (34) and a resting hand (64) - so `fist_hold`, where
# the gain was largest, keeps its channels at 0.81 while the pinch frames of
# `thumb_tap` are cut to 0.32.
#
# The gate is what is being measured, so three seeds again.
for s in 0 1 2; do
  stage "gated-seed$s" data/models/gated_seed$s.pt \
    $PY -u learn.py train --epochs 16 --workers 4 --seed $s --contact-gate \
        --out data/models/gated_seed$s.pt
done

# INTENT IS STILL NOT QUEUED. It was to wait on item 3, which reported that
# the direct objective does not pay at any weight tried. It now waits on this:
# the grasp head hands over ON sigma, so it should be built over whichever
# sigma turns out to be the one worth having.

note "=== queue complete"
printf "\n  Compare them:  %s compare_runs.py\n\n" "$PY"
