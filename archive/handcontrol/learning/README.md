# Learning the retargeting

> **THE NUMBERS BELOW PREDATE THE CURRENT HAND.** Every measurement in this
> file was taken against the first export (`h4nd_2`, now removed - see git
> history), which was a mirrored right hand
> carrying a spurious 15-degree roll on three of the thumb's joint origins.
> `world/h4nd_left` replaced it: natively left, rolls zeroed, adduction limits
> tightened (index −20°→−5°, ring +20°→+5°, thumb to a uniform [−20, 110]).
> The fingers are unchanged; the thumb moved 32 mm and its reach grew from 164
> to 176 mm.
>
> The pipeline is being regenerated. Re-solving one block against the new hand
> took `thumb_tap` contact from 19% back to 74%, which says the labels were
> stale rather than the hand wrong - but it also says the thumb constants
> listed under "Next" are still tuned for the old geometry. Treat every table
> here as the shape of the result, not its value, until they are re-run.

The package above this one maps a human hand onto `h4nd` by measuring angles
or by optimising keyvectors, once per frame, live. Both work, and both share a
ceiling that is **not accuracy**:

> A per-frame optimiser is a discontinuous function of its input. Arbitrarily
> small changes to a tracked fingertip flip the solve into a different basin,
> and filtering does not repair that — filtering a discontinuity gives a smeared
> discontinuity.

A network is continuous by construction. That is the property being bought.

**This IS wired into the live path now.** `learn.py` builds and measures it
offline; the panel's LEARNING tab runs it against a real hand and, once ARMed,
real servos. `Mimic` gained a `student` mode that reads the same 21 keypoints
the solver is sent, over a 20-frame causal window, and returns joints through
the same `Retargeter.accept` exit every other mapping uses.

Two things had to be true first, and only one of them was:

  THE REST ALIGNMENT had never been applied live. The offline pipeline fitted
  `bone_offsets` and the live solver did not, so the bones it was sent sat up
  to 77.8 degrees from the robot's zero pose. `Mimic` now fits it from forty
  open frames and feeds both mappings through one `keypoints` method, so it
  cannot be applied to one and forgotten on the other again.

  THE WINDOW has to be full before the network can answer. It is primed with
  the operator's first aligned frame, repeated, rather than with a synthetic
  flat hand: a synthetic one needs an anchor against the robot's geometry, and
  the robot has no metacarpal to hang a MANO wrist on - anchored on the palm
  origin the two hands sat 33 mm apart, and averaging the knuckles made it 60.

```
python learn.py survey --operator akshay --hand left    Phase 0  record
python learn.py noise                                   Phase 0  fit the error
python learn.py teacher                                 Phase 1  solve for labels
python learn.py synth                                   Phase 2  generate pairs
python learn.py train                                   Phase 3  train the student
python learn.py gestures                                         probe it
python learn.py selftest                                         check the invariants
.venv/bin/mjpython learn.py replay --student --ghost live        watch any of it
```


## The files

`docs/pipeline.html` draws all of this as one page: every stage's shapes in and
out, the teacher's objective term by term, and a flow chart of which command
writes which file. Worth opening before reading the sections below.


| | |
| --- | --- |
| `sessions.py` | recorded sessions: pooling them, cutting them into runs. No robot. |
| `survey.py` | the block script and the recorder that walks an operator through it. |
| `noise.py` | what the sensor's error is. No robot. |
| `report.py` | printing and plotting a noise fit. Presentation only. |
| `augment.py` | putting that error back onto a clean hand. No robot. |
| `synthetic.py` | plausible human hands from nothing at all. No robot. |
| `validate.py` | checking a synthetic sample against the recording. |
| `kinematics.py` | the finger chain in torch. One FK for teacher and student, every body's origin and frame. |
| `collision.py` | the hand's own links as oriented boxes, offsets measured off the meshes, and MuJoCo's contacts as the truth they are checked against. |
| `geometry.py` | the hand as distances - tip pairs, curls, derivatives, time-to-contact. One table every cell goes through. |
| `objective.py` | what the teacher minimises, with the number behind every constant. |
| `teacher.py` | the offline solve, run by run, warm-started from the live solver. |
| `student.py` | the network, and a differentiable FK to train it against. |
| `train.py` | training it, and scoring it against the solver it replaces. |
| `gestures.py` | hand-authored shapes it was never trained on. |
| `replay.py` | playing a solved session, or a student, back in the viewer. It draws lane labels with `robot/glyphs.py`. |
| `playback.py` | the gesture probe fed through the live world's own lanes - the GESTURES button. |
| `gallery.py` | sixteen hands at once, each replaying a different run - a set's quality at a glance. |
| `selftest.py` | the invariants this package keeps breaking. |
| `cli.py` | argument parsing. No logic - `finish_keypoints` and `sampler_for` used to live here and now sit with the machinery they belong to. |

Layered the same way the package above is — by how much each part knows about
this robot — for the same reason: the parts that are not about `h4nd` can be
lifted out of it, and only `teacher.py` has to be rewritten for a different
hand.

## What a run is, and why everything is cut into them

Every estimate in this package is temporal — a smoothing window, a warm start,
a frame-to-frame step — and all of them assume the previous row happened a
hundredth of a second ago. Three things break that while leaving rows adjacent
in the array: a tracking loss, the gap between two blocks while the operator
reads a prompt, and the join between two sessions recorded on different days.

This is not pedantry. Solving across block seams warm-started a fist from a
flat hand. Measuring jerk across them counted the seam as a **108 degree**
single-frame joint step, and reported the offline teacher as *worse* than the
live mapping it improves on.

`sessions.tracked_runs` is the one place that decides. Everything calls it.

## Phase 0 — what the sensor actually does

`survey.py` records still holds, curls slow and fast, thumb taps, spread,
isolated fingers, the hand rotated 45 and 90 degrees, deliberate occlusion, and
free motion. `noise.py` fits it. Measured over 55k frames of one operator:

| | |
| --- | --- |
| jitter at rest | 0.042 mm |
| jitter at 800 mm/s | 7.98 mm |
| hidden finger vs visible | ×3.1 to ×5.6 |
| error correlated **across** fingers | 0.34–0.82, after dividing out speed |
| residual power below 5 Hz | 41–83% |
| worst single-frame fingertip relocation | 98 mm |
| frames the SDK repeated and reported as tracked | none |

Four of those are reasons a single fitted gaussian would train for noise nobody
has. The noise is not a constant — it moves by a factor of 190 with hand speed.
It is not independent per keypoint — the tracker gets the whole hand slightly
wrong at once, and error sampled independently averages away down a finger
where the real error does not. It is not white, so a memoryless augmentation is
removable in a way the real thing is not. And the snaps are not noise at all;
they are wrong samples.

`augment.py` reproduces all of it. Verified against the fit:

| property | measured | reproduced |
| --- | --- | --- |
| across-finger correlation | 0.538 | 0.523 |
| within-finger | 0.705 | 0.707 |
| occlusion multiplier | ×4.88 | ×4.89 |
| AR(1) colour | 0.659 | 0.666 |

**Tracking loss is not the failure mode.** `rot90` and `occlusion` lost no
frames at all. The Leap does not drop the hand, it reports a wrong one, and
`LEAP_HAND.confidence` is documented as always 1.0 — the least trustworthy
channel reports itself as perfect.

**The measuring stick is bone length.** Every other estimate of the error needs
to know what the hand was really doing, and nothing does. A bone length does
not: it is rigid, the operator cannot change it, so every millimetre it moves is
the tracker being wrong. That makes it the one estimate valid *during motion*,
which is where the error lives and where a hold-still capture says nothing.

## Phase 1 — the teacher, and the two results that shaped the plan

`teacher.py` solves a recording run by run, offline, as a batched trajectory
against the objective in `objective.py`, warm-started from one pass of the
live keyvector solver and then centred-smoothed. The live solver still runs
here: it is the warm start, and it is the `live-equivalent` row every table
compares against.

**The first result: solving harder buys smoothness, not accuracy.** The live
solver's own objective, run offline with no time budget, scored the same pinch
error and seven times less jerk. That set the plan's shape — the teacher
supplies smooth labels — and it stood for a day.

**The second result replaced the objective.** Scored against the *operator*
rather than against itself, the teacher was 9.35° of bone direction from the
hand it was copying — competitive with published systems — and it could not
make a pinch. Not on one frame. A pinch is 21 mm raw from the sensor; after
per-digit scaling and the rest rotation it is 136 mm, and the solver matched
136 faithfully. Directions survive those transforms; distances do not. So the
teacher now reads two views of every frame — aligned for directions, rotation-
only for anything that is a distance — and the thumb, whose bone directions
are geometrically incompatible with contact on this hand (matching them to 3°
puts the tips 163 mm apart), gets a task-space objective instead. Every
constant in `objective.py` carries the measurement that set it.

Whole recording, every block, against the live solver, with the collision
terms below in place:

| | fingers | contact made | by finger | gap at touch | self-collision | jerk |
| --- | --- | --- | --- | --- | --- | --- |
| live-equivalent | 6.90° | 0% | — | 111 mm | 42% | 8.1e3 |
| teacher | **2.69°** | **95%** | 99 / 96 / 68 | **−0.1 mm** | **1.9%** | 3.4e3 |

Fingers better on every block. Self-collision is the share of frames MuJoCo
finds any link more than 3 mm into another, past as-built. Motion blocks are
rougher than the old teacher — one smoothing pass against its two plus a
re-solve — and most of that is real: the last 27 mm of a tap takes 56 ms, and
a thumb that has to arrive when the operator's does moves 23° in that time.
The old teacher looked smooth there because it never arrived.

A learned teacher was tried against the same objective — six variants, the
best at 55% contact with the fingers 33° off — and none reached the per-frame
floor the optimiser sits on. The teacher stays an optimiser.

## Self-collision — the term that arrived last

A census run after everything above had shipped: with poses set kinematically
and MuJoCo's own contacts read back, the teacher's labels put the hand
through itself on **42% of the recorded frames** and **95% of the synthetic
ones**, the authored "open" hand included. Three different faults:

- **The pinch was a penetration by construction.** The contact term drove
  the thumb and index tip *origins* to the closest the kinematics allow, and
  the tip meshes then overlapped by ten millimetres — one pair, index distal
  against thumb distal, on 20% of all frames. On hardware that pose is
  unreachable: tips press, joints strain, and the executor would read the
  stall as an object.
- **Neighbouring fingers crossed.** The sampler drew each finger's spread on
  its own; 84% of synthetic labels had a finger more than 5 mm through its
  neighbour, and the teacher copied the crossing faithfully.
- **Packing, not collision.** Neighbouring fingers sit a millimetre apart at
  rest, and the convex hulls MuJoCo collides ship 0.9 mm inside each other.
  Shallow contact is the hulls rubbing; the threshold for a fault is 3 mm past
  as-built, which is how `robot/collide.py` counts it for the servos too.

**The proxy** (`collision.py`). Neither the teacher nor the student can call
a mesh collider inside a gradient step, so every body is an oriented box —
its mesh bounds in its own frame, from an FK that now returns every body's
origin and rotation — and a pair's gap is the separating-axis gap between two
boxes, negative by the least overlap when they intersect. Capsules were tried
first and rejected by measurement: a servo link is a box, and the pinch pair
read 13.8 mm of segment distance when the pads met and 24 when the sides did.
One offset per pair absorbs what is left, fitted against MuJoCo's signed mesh
distance over a random sweep plus poses drawn to bring the thumb near a
fingertip: 112 pairs in two seconds, median error 1.1 mm, worst 2.8, cached
beside the model and keyed on the URDF. The palm is a box too, against every
body that is clear of it at rest — the mounts the digits hang from sit 22 to
29 mm inside it by construction, and charged anyway they bent the whole hand.

**The terms** (`objective.py`). Contact now drives the *surface* gap between
thumb and fingertip to zero instead of the origin distance to its floor; a
penalty charges every pair's penetration beyond a millimetre, squared, summed
over pairs — one finger through another is the whole fault, and averaging it
over ninety-five clear pairs would hide it; and the distal joint and tip of
each finger are held on their own side of the next finger's along the palm's
lateral axis, because a penalty alone pushes a body out the *shortest* way,
which for a finger that starts a solve more than halfway through its
neighbour is onward, into a hand with its fingers crossed and every pair
clear. The thumb is exempt from the ordering: it crosses under the fingers to
oppose them. The sampler stops drawing crossed hands as well — a keyframe
whose adjacent fingertips are out of order is redrawn.

Seeded from the old labels, run by run: the fist went from 100% of frames
penetrating to 0% with contact still 100% and the surface gap at 0.0 mm;
thumb tap 47% to 0.2%; spread cycle 23% to 0%. Finger posture *improved*
everywhere (6.2° to 5.3° on the tap, 7.3° to 5.0° on the fast curl). Whole
session, 42% to 1.9%; frames past the servo guard's 6 mm margin, 29% to 0.9%.
What is left is one pattern — the thumb already through the index when the
solve starts, on `rot45`, `occlusion` and `curl_fast`, tracker-garbage frames
all — and the guard clamps those live. The synthetic set, regenerated with
the sampler's rejection and every term: 92% to 15% of frames past 3 mm,
81% to 1.0% past 6 mm. What remains there is one pair, the thumb's
adduction block against the palm, three to six millimetres by the hulls -
the proxy's worst-fitted pair, and under the guard's margin.

`Teacher.measure` reports penetration from MuJoCo's contacts rather than from
the proxy the solve optimised, so the number cannot flatter the term. Solving
is five times slower for it: fifteen minutes for the session.

**The student needed two more things, because clear labels were not
enough.** Retrained on them, it still put the middle finger through a
neighbour in transit, and the reason is arithmetic: neighbouring fingers sit
a millimetre apart, the student is four to five degrees off its label, and
four degrees of adduction on a 100 mm finger is seven millimetres sideways.
The clearance is smaller than the precision, and no amount of training
changes that. So:

- **Repair, not reject** (`collision.repair`). The last thing a command goes
  through is six gradient steps of the penetration penalty on the same
  proxy the teacher solved against: the offending joints move, the way that
  clears, nothing else does. On the old student's own collisions it took
  62.8% of frames past 3 mm to 0.0%, moving a median of 3.6° and a 95th
  percentile of 7.6°; a clear frame exits in under 2 ms, a colliding one
  costs 4 to 12. `Mimic` runs it on the commanded pose and leaves the STUDENT
  lane raw, so the viewer shows what the network said and the robot what it
  was allowed to do. Training CAN run through it - `train.REPAIR_STEPS`
  makes the geometric terms score the repaired pose, straight-through, while
  the joint and collision terms read the raw one - but it ships **off**, at
  zero. Two steps of it cost three degrees of posture on the held-out blocks
  (12.5 against 9.4) and did not touch the hover it was meant to cure, so the
  student is trained to be clear on its own and the repair is a live guard
  only. (The exact second-order version produced NaN through the box test's
  degenerate axes on the first batch; the shipped one is a smooth-norm
  Gauss-Newton step.)
- **Clearance in the labels** (`objective.CLEAR_MM`). Wherever the operator's
  own neighbouring fingertips are apart, the teacher keeps 4 mm of daylight
  between the robot's; fingers held together, and fists, get only the
  tolerance. Measured on `finger_isolate`: the room between neighbours went
  from 0.6 mm (84% of frames under 2 mm) to 3.7 mm (none), at 0.05° of
  posture and no contact. The student inherits an error budget.

**The student, performing** - a causal rollout over the whole session and
the gesture probe, MuJoCo's contacts on every frame, and `robot/collide.py`'s
verdict on whether the servo path would have had to block the command:

| performing | self-collision >3 mm | past the 6 mm guard | guard would block | pinch gap at touch |
| --- | --- | --- | --- | --- |
| old student, raw | 30.5% | 26.2% | 8.1% | −7.2 mm |
| old student, repaired | 0.0% | 0.0% | 0.1% | +5.1 mm |
| collision-free labels, raw | 0.3% | 0.0% | 0.1% | +5.6 mm |
| **+ touch term (shipped), raw** | 0.7% | 0.0% | 0.4% | **+0.8 mm** |
| **+ touch term (shipped), repaired** | **0.0%** | **0.0%** | **0.0%** | +1.3 mm |

The old student commanded the thumb seven millimetres *inside* the index on
every pinch; on the probe it collided on 55% of frames and the guard would
have blocked 30% of its commands. The repair alone makes it safe, but it has
to move a third of its frames by a median 3.6° to do it and the hand rattles
for it (jerk 3.8e4). Trained on collision-free labels the student is clear
on its own - and stopped five to six millimetres short of the skin on every
pinch, the **hover**. Four retrains found it: taking the repair out of the
loss did not move it (and bought 3° of posture); a repair step that lands
on the tolerance instead of overshooting did not move it; the cause was the
collision penalty being one-sided. A network unsure of a fingertip to four
millimetres answers a penalty that only charges the inside by hedging
outward by that much. The mirror term, `W_TOUCH` - wherever the label
touches, a gap above the label's costs what penetration does - closes it:
**+0.8 mm** at the operator's touch, index and middle contact at 100%, ring
at 73% on the blocks that have it. At equal weight the thumb reached the
fingertip *through* the finger (42% of held-out frames penetrating), so the
collision term outweighs the touch term eight to one. Cost: posture on the
held-out blocks 9.4° without the touch term, 13.5° with; the trade is the
last four millimetres of the pinch, and it is the pinch the whole project
is judged on. `W_TOUCH` is the knob.

`robot/collide.py` — mesh-exact, per-pair as-built baseline, 6 mm margin,
bisection clamp — still stands between every command and the servos. It was
there before any of this, it is what `sim.command` runs, and the change here
is that the labels no longer lean on it.

## Phase 2 — synthetic hands

Labels come from the teacher, not from the bench, so pairs cost solver time
rather than operator time. `synthetic.py` draws hands and motions; the CLI
solves them.

The template is fitted off a real recording rather than published anthropometry.
Published numbers would be a second source of truth about hand geometry, free to
disagree with the operator the robot is actually driven by, and every
disagreement would land in the student as a bias. Poses are sampled *through*
the tendon coupling rather than over the joints, because a uniform grid over
twelve angles is mostly hands nobody can make.

Five things the sampler got wrong first. Each is an axis or a sign, which is
what this kind of code is made of:

- **The frame's axes.** `hand_frame` returns `(forward, normal, thumbward)`, so
  index 2 is the thumb side and index 1 is the palm normal. Built the other way
  round, a synthetic fist splayed its fingers instead of closing them — median
  index-to-middle tip distance 67 mm against a real 25.
- **Fingers curl together.** Flexion drawn per finger left one straight beside
  one folded, every frame.
- **The thumb's measured spread does not transfer.** It is signed from a fixed
  55° rest offset and runs the opposite way to a rotation about the palm normal
  — swept, +40° closes the thumb-to-index gap to 33 mm where −40° opens it to
  132. Its band is set against that gap instead, which is observable on both
  hands.
- **The whole hand was in the wrong frame.** `synthetic.py` emits in the
  operator's own hand frame, because it knows nothing about robots. A recorded
  hand does not arrive that way — it goes through `Retargeter.rotation` into the
  **robot's** axes, which is the frame the solver scores against its own forward
  kinematics. Missing that step rotated every synthetic bone direction ninety
  degrees.
- **The flexion sign.** `cross(normal, direction)` curls a finger out of the
  back of the hand; it wants `cross(direction, normal)`. Swept against the
  recording, +flex drove the index tip's normal component to −40 mm where a real
  fist puts it at +35.

**The last two passed every check that existed.** Bone lengths were right, shape
coverage was 98.5%, the solver converged with ordinary residuals — because a
rigid rotation changes none of those. They showed up on screen, as a hand
floating sideways, after 150,000 frames had been solved against them.

So `validate.check_frame` tests **absolute placement** against the recording
before anything is written. That is the one thing a rigid motion does change and
the one thing every shape check is blind to by construction. It refuses rather
than warns.

`validate.compare` is the other half, and its test is on the **support**, not
quantile against quantile: comparing a synthetic 95th percentile against a real
95th demands the synthetic be more spread at every quantile, which fails a
sampler that reaches every real shape while spending less of its time at the
extremes. What is reported is the fraction of real frames whose shape falls
inside the synthetic range — 98.5% and up across five shape measures.

**The set the student trains on now oversamples pinches.** The recording
touches on 14% of its frames, and a sampler fitted to it reproduces that; but
contact is the event the teacher labels exactly and the student has to arrive
at. `PINCH_RATE = 0.3` of keyframes are solved to a thumb-fingertip touch at
the sensor's own 20 mm, the finger drawn from index, middle and ring, the
finger free to adduct toward the thumb. `learn.py synth --pinch` sets it. The
200-hand set: 40% of frames touching, the teacher closing 83% of them, and the
coverage check still at 100% of real frames.

**Defaults, and what they cost.** `learn.py synth` draws 100 hands for six
seconds each and solves them in 300 steps where the recorded session gets
600. Measured on the same twelve hands: 300 steps against 600 is 0.7° of
posture, two points of contact and 0.8 mm of pinch gap on the labels, at
twice the speed - and the student's own error is four to five degrees, so
the set is not where its precision is lost. `--steps 600` and `--hands 200`
are there when it matters. With the collision terms in place a hand solves
at about 100 frames a second, so the default set is a quarter of an hour.

## Phase 3 — the student

```
aligned keypoints (63) + distance table (15) + thumb intent (3), x 20 frames
        ->  causal TCN  ->  16 PCA coefficients  ->  16 joints, x 4 horizons
                        ->  contact logits (3), time-to-contact (3)
```

238k parameters, ~1 ms a frame, trained on MPS in about fifteen minutes.

**Two views in.** The aligned keypoints carry pose: bone directions, which
finger is where. A pinch does not survive them - the per-digit scaling sends
two touching tips 136 mm apart - so on their own they cannot say *which*
finger the thumb is closing on, and the student fed only these learned every
pinch as an index pinch: middle-finger contact 0.4% against the teacher's
56%. The distance table (`geometry.py`) from the raw view carries the
relations, indexed the same way the teacher's objective reads them, plus a
per-pair intent ramp - the same one the teacher uses, per cell. Live, `Mimic`
builds both views from the same hand; offline, both are in every pair file.

Four constraints, each deliberate rather than a layer count:

- **Causal**, because it has to run live. Dilations 1-2-4-8 over blocks of
  two convolutions give a 61-frame receptive field over a 20-frame window
  (measured, by bumping one input frame at a time), the padding is one-sided, and
  `selftest` bumps the newest frame and checks that no earlier output of any
  head moves.
- **A cumulative norm.** GroupNorm pooled its statistics over time, which
  made the earlier supervised frames acausal; a per-frame norm is causal and
  was measured to rattle (jerk 1.99e4 against 1.66e4). The statistics now run
  cumulatively over the frames so far - GroupNorm's exactly at the newest
  frame, causal at every other.
- **A pose basis on the output** - PCA over the teacher's own solutions, at
  the full sixteen; `--basis 10` remains for the comparison.
- **A zero-initialised head**, so an untrained network emits the mean pose.

**Two auxiliary heads**, trained and never commanded: which thumb pairs the
teacher had closed at this frame - labelled off the teacher's own kinematics,
not off the operator's distance, because it is the teacher's behaviour being
copied - and seconds until each pair next touches, clipped at half a second.
They make the trunk carry "which finger, and when". Nothing reads them yet;
the executor will.

**A thumb-pair distance term** in the loss, through the same kinematics as
the tip term, weighted five-fold on frames the teacher was touching. Two tips
can each be a centimetre off with their distance exact, or each close with
their distance wrong; a pinch cares about the second. Measured off: contact
68% to 62%, gap at touch 7.0 to 9.5 mm.

Trained on the synthetic pairs with the measured sensor noise injected into
both views and the clean labels kept as the target, then fine-tuned on the
real session. Evaluation holds out whole BLOCKS (`fist_hold`, `curl_fast`,
`occlusion`) and scores every row against the OPERATOR the way the teacher
is scored.

| held-out blocks | fingers | thumb | contact | by finger | gap at touch | jerk | worst step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| live-equivalent | 7.95° | 17.2° | 0% | - | 130 mm | 8.4e3 | 17.1° |
| teacher | 5.40° | 55.3° | 78% | 90 / 56 / 0 | 4.9 mm | 3.9e3 | 10.4° |
| student, keypoints only | 9.59° | 57.3° | 70% | 94 / 0.4 / 0 | 8.7 mm | 1.75e4 | 14.2° |
| student, two views | 11.27° | 54.7° | 69% | 88 / 19 / 0 | 6.6 mm | 1.66e4 | 16.6° |
| student, collision-free labels | 12.54° | 47.5° | 85%* | 96 / 74 / 0* | 5.6 mm* | 1.21e4 | 17.4° |
| student, + touch term | 13.48° | 54.5° | 92%* | 100 / 100 / 0* | 0.8 mm* | 1.52e4 | 22.3° |
| student, + objects in the labels, replay fine-tune | 14.71° | 53.5° | 92%* | 100 / 99 / 0* | 0.7 mm* | 1.52e4 | 20.8° |
| student, + objects placed from the hand | 14.96° | 55.8° | **92%*** | **100 / 100 / 0*** | 1.3 mm* | 1.78e4 | 19.8° |
| **re-solved on the repaired proxy, seed 0 (shipped)** | 13.28° | 48.8° | **92%*** | | 0.6 mm* | 1.82e4 | 22.6° |
| the same thing, seed 1 | 15.94° | 55.4° | 92%* | | 1.2 mm* | 1.77e4 | 19.6° |

> **THIS WHOLE TABLE PREDATES `h4nd_left`.** Every row was solved and scored
> against the first export - the mirrored right hand carrying a 15-degree roll
> on three thumb joints. The labels have since been re-solved and the settings
> re-measured at three seeds each; **the five-way table above is the current
> result** and this one is kept for the shape of the argument, not its values.
> The 2.66° seed spread quoted below is also that hand's; on the current
> labels it is 1.38°.

**READ THE LAST TWO ROWS BEFORE ANY OF THE OTHERS.** They are the same
labels, the same settings and the same code, differing only in the random
seed - and they are **2.66° apart**. Every row above them was a single
unseeded run, so any two of them closer together than that are not
distinguishable, and several are. What each change bought is only readable
where the gap is large: the two-view student is genuinely worse against the
operator than the keypoint-only one, and contact genuinely went from 70% to
92%. The 1.68° the re-solve appears to have bought is inside the noise and is
not claimed. `learn.py train --seed` exists so this can be measured rather
than argued about; it should have existed from the start.

\* contact and gap under the surface definition that arrived with the
collision terms - a touch is a surface gap under 8 mm - so the last row is
not comparable to the two above it on those columns; see the performing
table further down for the like-for-like comparison. Self-collision on the
held-out blocks: 65% of frames for the two-views row, 1.2% for the last.

The by-finger column is index / middle / ring, as a percentage of the frames
the operator was touching on that pair. The ring is the labels: the teacher
never closes it on these blocks either.

**The derivatives were tried, and are off.** The plan was the distance
table's causal velocity, acceleration and jerk as channels - a cell closing
at 450 mm/s and slowing is a strike arriving. The network agreed and read
them heavily, and on a *still* fist the output rattled at 5e3 jerk against
the keypoint-only student's 626. A Jacobian attribution on those frames put
99% of the output's frame-to-frame motion on the three derivative groups
(velocity 41%, acceleration 38%, jerk 20%) and under 1% on the keypoints,
the distances and the intent together. No causal filter fixes it at the
source: a quadratic fit over 88 ms lifts the acceleration channel's
signal-to-noise from 7 to 11 and costs 17 ms of lag. With the channels
zeroed, same everything else: still-fist jerk 1.1e3, moving-block jerk down
twofold, and the prediction the derivatives were meant to buy was *better*
without them. A causal stack over twenty frames of distances computes
whatever difference it needs, trained against smooth labels rather than
fixed against a noisy input. The channels have since been removed from
`student.py` altogether - a switch nothing could set was a switch that looked
live and was not - and the measurement is kept there, under "measured, and
off"; the implementation is in git history.

**What put the smoothness back** was the loss and the norm, not the input:
velocity matching against the label's own velocity raised from 0.5 to 3 -
which, unlike an acceleration penalty (see `W_ACCEL`), cannot erase a strike,
because the strike is in the label it matches - and the cumulative norm
above. Per block, jerk on the still fist is 678 against the keypoint-only
student's 626, on `curl_fast` 2.16e4 against 2.39e4, on `occlusion` 1.87e4
against 1.89e4. The posture column paid for it: 1.7° against the operator,
almost all of it on the fist, where the operator's fingers are the tracker's
guess.

**Contact is high-variance run to run**, and that is the finding to read
every table above against. Middle-finger contact was 23%, 5%, 3%, 0.4%, 0%
and 19% across six retrains whose settings differed by one knob each; the
real session has 2.6% of its frames touching there, and it is not yet known
what the student needs to learn it reliably. One seed per setting so far.

### Phase 3b - the forward heads

`HORIZONS = (0, 5, 10, 15)` frames: now, and roughly 40, 80 and 120 ms at the
sensor's rate. One shared trunk, four heads - separate networks would each
relearn the mapping from the same data, and sharing it means the future heads
only learn how a hand carries on, which is the part that differs.

The geometric losses score ONLY the present head. Running forward kinematics on
a prediction would grade the network for a hand that has not happened.

Judged against persistence, because that is the question that matters: is
predicting ahead better than holding the last pose?

| horizon | student | hold | | keypoints only |
| --- | --- | --- | --- | --- |
| +0 f | 4.91° | -- | the present-tense error, for reference | 4.64° |
| +5 f, ~40 ms | 5.92° | 4.27° | -1.65 | -1.68 |
| +10 f, ~80 ms | 8.12° | 8.36° | **+0.24** | +0.04 |
| +15 f, ~120 ms | 11.03° | **12.38°** | **+1.35** | +0.97 |

**The distances bought prediction**: the 80 ms head is now on the right side
of persistence and the 120 ms head's margin grew by a third. Time until the
thumb touches is now a head of its own (`Student.TTC`), exactly labelled from
every pair file's `time_to_contact`; commanding the forward heads off it is
the executor's job, not the student's.

## Does it generalise? `learn.py gestures`

Held-out blocks measure interpolation: a different fist, but still a fist. The
question that decides whether this ships is what it does with a shape it has
never seen.

`gestures.py` authors ten - open, spread, fist, peace, OK, curl, wrap, and one-digit-out
for index, ring and thumb - and each carries a geometric predicate that is
checked before use, because this file has twice been where a confident claim
about a shape turned out to be wrong. A peace sign is two digits out and two
in; an OK sign is thumb and index touching, solved for the template in hand
with the same sweep the sampler's own pinches use.

**The generalisation gap turned out not to be one.** The old student scored
8.6° on these gestures against 2.6° on the real session - and 8.95° on the
synthetic set it had *pretrained on*. There is no gap between gestures it
never saw and synthetic data it trained on; both are the pretraining being
forgotten by the fine-tune, which `train.py` says in the comment above its
schedule. Widening the sampler's priors - the obvious remedy - was measured
against it and did nothing: 300x more data closed the distance from the
gestures to the nearest training pose by 4.9°, and no setting of the coupling
knobs moved it at all. `DIVERSITY["wide"]` is kept in `synthetic.py` as that
negative result.

Against the contact-making teacher the shipped student lands 11.4° from it
on these shapes on average, raw; the collision census on the same probe is
in the performing table above. The generalisation story is unchanged by the
two-view input or the collision terms.

## Checking it: `learn.py selftest`

Twenty-eight checks, written against the FAILURE MODES rather than the code's
structure, because every bug in this package has been one of five things and
none was caught by the checks already present:

  A FRAME OR A BASIS skipped or applied twice. A rigid rotation preserves every
  distance, so no coverage or distribution test can see one.

  A BOUNDARY CROSSED - a run spanning two blocks, or a difference measured
  across the seam between them.

  A STALE ARTIFACT compared against a fresh one.

  A SIGNATURE CHANGED AT ONE CALL SITE.

  A DISTANCE MEASURED IN THE WRONG VIEW. A pinch is 21 mm raw and 136 mm
  aligned, and the solver matched the 136 for a whole recording. `a pinch
  survives into the raw view` and `teacher makes contact` guard it; `geometry
  is one table` keeps every distance in the project reading the same cell.

  A PROXY THAT DRIFTED FROM THE THING IT STANDS FOR. `collision proxy
  matches mujoco` sweeps random poses through both and demands agreement on
  what is 3 mm into what; `labels do not self-collide` reads the labels on
  disk back through MuJoCo's contacts; `student is causal` bumps the newest
  frame and checks that nothing earlier moves.

  A CONSTANT DELETED BY AN EDIT AIMED AT ITS NEIGHBOUR. `QUANTILES`,
  `ROLE_FOR`, `FALLBACK` and `LENGTH_SPREAD` have each been removed by a text
  replacement targeting something beside them; Python only notices when the
  line runs, so a module imports cleanly and dies twenty minutes into a job.
  The source-only checks run FIRST and without loading anything, because that
  failure breaks the harness that would otherwise report it.

They run against the recordings on disk and take a few seconds. Run it after
touching anything here. One of them exists because the suite itself missed a
bug: `--help` proves argparse is wired and nothing else, so `every command
runs` executes the shortest real invocation of each subcommand instead.

## Watching it

```
.venv/bin/mjpython learn.py replay                      newest session
.venv/bin/mjpython learn.py replay --compare            ghost = the live solver
.venv/bin/mjpython learn.py replay --blocks thumb_tap --loop
.venv/bin/mjpython learn.py replay --synthetic data/synthetic/seed0
.venv/bin/mjpython learn.py gallery                     sixteen hands at once
```

One lane per mapping - teacher green, student violet, live solver grey, a
forward head coral - with the operator's skeleton beside them. Amber is never
used here; on the bench it means the real robot's encoders. **mjpython, not
python** — the same macOS main-thread requirement as `main.py`.

The live world (`main.py`) shows the same comparison without a second window.
Its row is YOU, ROBOT, TEACHER, STUDENT, AHEAD, ENCODER — AHEAD being the
student's 80 ms head, posed every frame whether or not the student is in
command — and the LEARNING tab's **GESTURES** button plays the authored probe
through those lanes (`playback.py`): the probe's keypoints stand in for the
sensor, the teacher's labels pose TEACHER, and the student on disk runs live
on the probe's frames for STUDENT and AHEAD. **REPLAY** does the same with
the newest recorded session's solved blocks, run by run. With SIM + HAND on
and the servos armed, REPLAY is the offline solve on the real hand, with
the encoders reading back in the lane beside it — the first bench test the
labels get. Click either again to stop.

**Where the loop's time goes** is in the state message and printed at
exit: mapping (sensor to command, the student and its repair included),
guard, physics, draw, and the bus's own figure — sensor frame arrival to
the sync write that carried it. Measured while looking for a bottleneck
that was not there: the mesh guard is under 2 ms on a clear pose, not the
20 the old comment claimed, and a blocked clamp — the only expensive case —
never happens to a student trained on collision-free labels. A screen for
it on the collision proxy was built, measured to agree with the meshes on
only 84 to 90% of poses near the margin, and removed.

`learn.py gallery` is the other view: sixteen hands in a grid, each replaying
a different run, each with its skeleton, the skeleton turning red while the
operator is touching a fingertip. It is how a set's quality is judged at a
glance, and `--png` renders one frame of it without a window.

A metrics table cannot say whether the motion looks like a hand, and that was
the complaint that started this. Two of the five sampler bugs above were found
this way and no other.

## Where the data goes

```
data/leap/<timestamp>/session.npz    raw survey, unfiltered, with timestamps
                     /meta.json      operator, hand, per-block stats
                     /teacher.npz    q*, q_live, both keypoint views, intent,
                                     distances, time-to-contact, run bounds
                     /teacher.json   the metrics table
data/leap/noise.json                 the pooled fit, including `augmentation`
data/leap/noise.png
data/synthetic/seed<n>/pairs.npz     both keypoint views, intent, distances,
                                     time-to-contact -> q*, and run bounds
data/synthetic/gestures/pairs.npz    the authored probe, and student.npz
data/models/student.pt               the trained student, and student.json
```

`world/build/` is for things that regenerate; a student is ten minutes of
training and does not, so it lives under `data/` with the other fitted
artefacts. `handcontrol/paths.py` is the only file that decides any of this.

Every file above is regenerable and none is in git.

## Objects — in the scene, in the objective

`learn.py gestures --sequence wrap --wrap-radius R` authors the operator
closing on a cylinder that rests on the palm: placed in the hand's own
frame from its knuckles, and each digit's joints swept for the pose that
lays its phalanges on the surface without passing through it
(`gestures.wrap_pose`). The same cylinder, radius and all, is placed in
the robot's palm **from the hand** (`gestures.cylinder_to_robot`,
`objects.json`), because a physical object does not scale with the hand
that holds it. Scored in MuJoCo with the cylinder as a real geom, on the
held wrap:

| 20 mm cylinder | penetration mean / max | links on it | digits on it | thumb on it | posture |
| --- | --- | --- | --- | --- | --- |
| teacher, no object in the objective | 0.0 mm | 0.0 | 0.0 | 0% | 6.9° |
| teacher, object pinned where a rule put it | 0.9 / 1.2 mm | 2.7 | 2.7 | 100% | 10.4° |
| teacher, object compliant (moves with the solve) | 1.4 / 1.6 mm | **3.7** | **2.8** | 75% | **6.6°** |
| **teacher, object placed from the hand (shipped)** | 1.0 / 1.7 mm | 2.6 | 2.6 | **78%** | 10.5° |
| live solver | 10.0 / 11.5 mm | 4.2 | 2.6 | 0% | 6.9° |

The 35 mm cylinder, placed from the hand: 1.3 / 1.9 mm, 3.3 links, 3.1
digits, thumb 72%, 10.2° - against 5.4° compliant. Four degrees of posture
is what a fixed object costs: the fingers go to it, where before it came to
them.

**What the first probe found, and why it took three tries.**
Direction-preserving retargeting keeps the operator's finger curvature and
still closes this hand on nothing: its joint axes sit at the back of links
21 mm thick where the operator's fingers are 8, and a wrap that encloses
27 mm in the operator's hand enclosed 2 on the robot. Every geometric
stand-in for "aperture" - the circle a finger's bones lie along, the peak
height of a finger above the palm - was checked against MuJoCo before any
solve and none tracked it; gating the thumb-tracking term off in a closed
hand moved nothing. The object had to be *in* the objective.

**Two terms read it** (`objective.held`). Penetration into the object costs
ten times what penetration into a neighbour does - at the same weight the
hold term parked the phalanges inside it - and every digit the operator has
bent around it, two bone bends past half a radian, is asked to lay its
three phalanges on the surface. Not its nearest body: that was the knuckle
mount, which no finger joint moves, and the term sat on it while the
fingers hovered twenty millimetres off. The hold term weighs four times
the posture term (`W_HOLD`): posture is degrees of bone direction and the
hold is millimetres of gap, averaged the same way, and at one-to-one a
finger that closes 4 mm onto a fixed object pays 6° on one bone to save
4 mm on one phalanx - it does not close, and hovered 5 to 15 mm off.

**Where the object is** (`gestures.cylinder_to_robot`) is decided by the
operator's hand and the robot's, and not by the solve. The first version
let the solve move it - compliant, a weak pull toward where it started -
and it held well (the row above) but settled somewhere different in every
training hand, so one operator wrap meant many labels, and a student with
no object input could not learn where the thing was. The operator's hand
decides the radius (kept), the axis (the knuckle line, rotated into the
robot's axes with the keypoints) and the lateral position (the offset from
the operator's knuckle-line centre, scaled by the finger-length ratio the
retargeting uses, laid off the robot's). The robot decides the rest,
because its links are thick where the operator's are thin: the object
rests on its palm face, and along the fingers it sits where the compliant
solves put the operator's own 20 and 35 mm wraps - 12 and 18 mm past the
knuckle axes, a line in the radius (`gestures.REST_ALONG_MM`). Placed where
the operator's knuckles said instead, 0.2 radii proximal of the knuckle
line, the teacher could lay one link on it: the robot's knuckle axes are
16 mm inside its palm and its proximal links 21 mm thick, and a proximal
link flexed 30 to 60° meets an object on the palm face 10 to 20 mm past
the axis, not behind it.

**Solving against a fixed object took three fixes**, each measured on the
probe. Solved against it from the start, the fingers stalled: closing a
finger onto a cylinder drives the fingertip through the surface before the
middle link lies on it, and the penetration wall stops the finger there
(1.7 links). So the object first rides with the solve for 300 steps -
the fingers close and it comes to meet them, no wall to cross - then
slides back to where the hand placed it over 100 steps while the fingers
follow (`objective.OBJECT_FREE_STEPS`, `OBJECT_RETURN_STEPS`). Riding, it
gets its own learning rate: Adam steps every parameter by its rate
whatever the gradient, and the solve's rate is in radians - on the object,
in metres, it was 20 mm a step; the cylinder leapt 70 mm up in the first
fifty steps and wandered back, and where it had been when the fingers
closed decided whether they came out clean. A millimetre a step
(`OBJECT_LR`) and it drifts to the hand and stays. And the smoothing pass
after the solve, which leaves a clean solve alone, drew a jumping thumb
straight through the cylinder - 66° in one frame on a jittering wrap, a
21-frame filter, 13 mm deep for sixty frames - so object runs get a short
polish after it (`objective.polish`): 100 steps at a quarter of the rate
with the object fixed, which pulled every frame back out (0 of 524 past
3 mm) at 0.08° of mean acceleration against the filter's 0.065.

**The synthetic hands hold objects now** (`synthetic.OBJECT_RATE`): a
fifth of keyframes close on a cylinder of 12 to 30 mm, present from
halfway out of the keyframe before to halfway into the one after, and the
teacher solves those frames with it. The sampler records the cylinder in
the hand's frame (`gestures.cylinder_in_hand`) and `learn.py synth` places
it in the robot's; 180 objects in 84 of 100 hands, and on the middle third
of each one's presence the labels lay 4.2 links and 3.5 digits on it, the
thumb 61% of the time, at 1.7 mm mean penetration with 4% of objects
having any frame past 3 mm - the compliant labels managed 3.2 links, 2.9
digits, the thumb 48%, 3.6 mm, and 48% of objects with a frame past 3 mm,
because an object that moves with the solve also moves *into* a phalanx
the smoothing pass then leaves there. The student never sees the object -
only a hand closed the way a hand closes on one - and is meant to answer
with a pose that holds it. Against the same 20 mm cylinder, as the
student answers (repaired) and then settled for its own pose and closed to
contact the way the executor would:

| 20 mm cylinder, held wrap | as answered: links / pen. | settled | after closing: links / digits / thumb | closing added |
| --- | --- | --- | --- | --- |
| teacher | 2.6 / 1.0 mm | 0 mm | 2.6 / 2.6 / 82% | 0° |
| student, compliant labels | 4.9 / **10.7 mm** | 18 mm | 0.6 / 0.5 / 18% | 33° |
| **student, placed labels (shipped)** | 0.7 / 3.5 mm | **3 mm** | **2.5 / 2.0 / 55%** | 8° |
| live solver | 1.7 / 4.0 mm | 9 mm | 0.4 / 0.4 / 0% | 0° |

The compliant-label student pushed five links *through* the cylinder and
put it 18 mm from where the hand had it; this one lands 3 mm off, 9.5°
from the teacher on the wrap against 12.4°, and stops short rather than
inside - which is the pose the executor wants, because closing to contact
from there finishes the grasp: two and a half links, two digits, the thumb
more often than not, eight degrees of closing. On the 35 mm cylinder,
outside the 12 to 30 mm the synthetic hands were drawn with, it does not
(11 mm off, half a link after closing) - a radius the labels never held is
a pose the student has never seen. Everything else is unchanged by the
placement: held-out real blocks 14.96° and 92% contact against 14.61% and
92%, the gesture probe 7.7° from the teacher against 7.4°.

**The fine-tune replays the pretraining now** (`train.REPLAY`). Fine-tuned
on the real session alone, the student drifted 11 to 13° from the synthetic
labels it had pretrained to within 6; with as many synthetic windows as
real ones in each fine-tune epoch, the gesture probe went from 11.1° to
7.4° from the teacher with the held-out real blocks unchanged (14.7°, 92%
contact, 1.5 mm gap, 0.1% self-collision). The student on disk is that one.


## The five-way measurement, on `h4nd_left` labels

Every row is three seeds unless it says otherwise. Held out: `fist_hold`,
`curl_fast`, `occlusion` - 11 runs.

| setting | pooled | vs teacher | contact | pinch gap | penetrating |
|---|---|---|---|---|---|
| baseline | 13.02° | 7.87° | 92.5-96.4% | 0.5-0.6 mm | 0.3-0.7% |
| no objects in labels | 14.70° | 8.35° | 92.3% | — | — |
| **belief channels** | 12.23° | **6.91°** | 93.5-95.7% | 1.2-1.6 mm | 0.8-1.6% |
| direct objective, w 0.01 | 9.74° | 8.00° | 92.5-94.1% | 0.4-0.6 mm | 0.7 / **23.2** / 1.2% |

**Read `pooled` with care, and never for a row trained on it.** It is
`objective.score`'s posture, and `objective.per_frame`'s FIRST TERM is the
same bone-angle array. So for the direct rows, training on it and reporting
it is circular; `vs teacher` is the column with no such argument, and it
scatters about three times less.

**The seed spread halved when the labels were re-solved against the corrected
hand** - 2.66° pooled before, 1.38° now, and 0.50° on `vs teacher`. The older
banner warning that any two rows within 2.66° say nothing was measured on the
`h4nd_2` labels and is too pessimistic here.

| **+ contact gate** | 12.2° | **6.82°** | 92.5-94.7% | 0.9-1.3 mm | 1.0-1.4% |
| + grasp head (1 seed) | — | 6.81° | 92.4% | 0.9 mm | 1.1% |

## Width and warm-up

Every setting above was measured at 96 channels, 238,150 parameters, because
nothing had ever varied the student's CAPACITY - each experiment changed the
loss or the input. Two things came out of finally doing it, and the smaller
change was the bigger win.

| recipe | width | vs teacher | contact | pinch | self-collision |
|---|---|---|---|---|---|
| none | 96 | 8.07° | 92.8% | 1.05 mm | 0.33% |
| none | 192 | 7.17° | 95.9% | 0.78 mm | 0.44%, and one seed of three collapsed |
| `--warmup 2` | 96 | 7.71° | **96.3%** | 0.73 mm | **0.21%** |
| `--warmup 2` | 192 | **7.19°** | 95.0% | **0.62 mm** | 0.33% |

Three seeds each except no-warm-up 96 (two) and the collapsed run, which is
excluded from its mean. Held out: `fist_hold`, `curl_fast`, `occlusion`.

**WARM-UP IS THE CLEAN WIN, AND IT WAS FOUND BY ACCIDENT.** It was added to
stop a collapse, and on the 96-channel baseline it improved every number:
0.36° closer to the teacher, contact up 3.5 points, the pinch gap a third
smaller, self-collision down by a third. Nothing else measured here has moved
four metrics the same way. It costs nothing at run time - it only changes the
first two epochs of training.

**WIDTH IS A TRADE, NOT A WIN.** At matched recipe, 192 is 0.52° closer to the
teacher and its range does not overlap 96's (6.85-7.41 against 7.61-7.77), and
it wins all three seed-matched pairs. But it makes contact 1.3 points *less*
often, self-collides half again as much, and its contact is erratic across
seeds (92.5-96.4 against 96.2-96.6). It buys fidelity to the solve and a
tighter pinch, and pays for them in the two numbers a grasp is judged on.

**AN EARLIER READ OF THIS WAS WRONG AND IS WORTH RECORDING.** Before the
matched comparison, width looked like a ~1° win with better contact. It was
neither: that compared no-warm-up 96 against warm-up-assisted 192 - the
recipes differed, not just the width. Warm-up lifted 96 by 0.36° and its
contact by 3.5 points, which is most of the apparent gap. Cherry-picking the
best seed of one group against the worst of the other did the rest. The
matched, three-seed numbers are the ones above.

**THE COLLAPSE.** At 192 with no warm-up, seed 1 sat at 100% held-out
self-penetration on all 21 epochs - collision loss flat at 292 from first to
last while the joint loss trained normally - and saved a checkpoint like any
other. Loaded live it would have driven a hand overlapping itself on every
frame. The cause is a learning rate that does not know the width: at 192 the
first epoch opened 5 to 40 times deeper in self-collision than at 96, and that
seed never climbed out. `--warmup 2` fixed exactly that seed, 15.92° to 6.85°,
the best single model of the study. `train.collapsed` now stops such a run
before it is saved - see the constant for why the rule is on the LOWEST epoch
so far and not the current one.

### What each one settled

**Objects in the labels: keep them.** The hypothesis was that a student which
cannot observe a cylinder cannot reproduce a label solved with one. Measured,
it is backwards - removing them costs 1.68° pooled against a 1.47° spread, and
on contact frames 17-19° against 13-15°. The objects were carrying something
the student uses.

**The direct objective: not at any weight tried.** Its posture gain is
circular, and what is not circular goes the wrong way - distance from the
teacher rises monotonically with the weight (7.87, 8.19, 9.93, 11.36 at 0,
0.01, 0.05, 0.20) and self-penetration reaches 2.3% at w=0.20. At w=0.01 one
seed of three penetrates on **23.2%** of frames, which is a coin-flip failure
rather than a scale to tune. Left in the code, defaulted off: it is still the
only route to a student not bounded by an acausal label, and `per_frame`
having no smoothness term is a concrete thing to fix before retrying it.

**The belief channels: a real gain with a specific cost.** Distance from the
teacher improves 0.96° with the two groups' ranges NOT overlapping (6.71-7.21
against 7.58-8.08), and the seed spread on posture falls from 1.38° to 0.32°,
so it also makes training markedly more repeatable. Against that, every
contact number degrades and also without overlap: pinch gap 0.5-0.6 mm becomes
1.2-1.6, penetration 0.3-0.7% becomes 0.8-1.6, contact-frame posture 13.3-15.1°
becomes 15.5-16.0.

The mechanism explains the shape. `belief.curl` is the live proxy for
self-occlusion, so the channels say "distrust this digit" exactly when a digit
folds - and a folding digit is usually one about to touch something. It backs
off where precision matters most.

Per block, against baseline seed 0: `fist_hold` improves 2.26°, `curl_fast`
0.73°, and the block actually named `occlusion` moves 0.01°. That is
consistent rather than disappointing - `occlusion_multiplier` was fitted as
fist_hold over flat_hold, so a fist is the case it was measured on, and curl
cannot see a hand hidden by being turned away. Note `fist_hold` and
`occlusion` are one run each; only `curl_fast`, at nine, carries weight.


## The thumb's 38 degrees are not a label bug

The teacher reports the thumb **38.1°** from the operator's own thumb against
the fingers' 3.99°, and that number decomposes exactly: **84.5° on the 25.4% of
frames where the operator is touching something, 22.4° on the other 74.6%**
(0.254 × 84.5 + 0.746 × 22.4 = 38.15, which `selftest` now checks against a
second route to both halves rather than taking either on trust). The cause is
visible in the objective - the thumb's direction term is multiplied by
`1 - any_touch`, so a touch removes all of it, where a touched finger keeps a
tenth of its own through `RELEASE`.

Beside the live solver on the same frames, that split is the whole trade:

|  | fingers | thumb free | thumb touching | thumb pooled | contact | gap@touch |
|--|---------|------------|----------------|--------------|---------|-----------|
| live-equivalent | 8.52° | 15.9° | **9.9°** | 14.4° | **0%** | **138.6 mm** |
| teacher | 3.99° | 22.4° | **84.5°** | 38.1° | **90%** | **1.3 mm** |

The live solver's flattering 14.4° thumb is bought by never closing: on the
operator's own touching frames it is 9.9° from their thumb direction and 138.6
mm from their fingertip. The teacher gives up 74.6° of direction on a quarter
of the frames and converts that gap into 1.3 mm. Pooled, both facts disappear
into one number that moves with how much of a block touches.

**Read the free column per block, not pooled either.** Whole-session it is
22.4° against the live solver's 15.9°, which invites the conclusion that the
teacher's thumb is worse even where the prior is fully applied. On
`flat_hold` + `thumb_tap` alone the same two numbers are **5.6° against 15.0°**
and the conclusion reverses. The pooled free column is carrying `rot90`,
`occlusion` and `free`, where the tracker's thumb is worst - which is the same
composition trap the split was introduced to get out of, one level down.

That reads like a bug with an obvious fix, and it is not. Both halves of it
were measured.

**Restoring the prior buys 5.9° of the 90 and costs contact.** `THUMB_RELEASE`
makes the release a constant, swept on `thumb_tap` + `ring_pinch` and
calibrated on the open blocks - one row each, because this solve is Adam from
a fixed warm start with no random init and the `1.0` row reproduces the shipped
labels field for field:

| release | thumb touching | contact | gap@touch | penetration |
|---------|----------------|---------|-----------|-------------|
| **1.0** (shipped) | **90.0°** | **92.8%** | **1.1 mm** | **1.5%** |
| 0.9 (as the fingers) | 89.4° | 92.9% | 1.0 mm | 1.7% |
| 0.7 | 88.3° | 92.6% | 1.0 mm | 1.9% |
| 0.5 | 87.0° | 92.5% | 1.1 mm | 2.2% |
| 0.0 (never release) | 84.1° | 91.9% | 1.3 mm | 2.4% |

With the release gone entirely the thumb still sits **84.1°** off. So the
release accounts for 5.9 of those 90 degrees and the hand accounts for 84, and
that 84 is the same wall the 60-pinch experiment hit from the other side:
matching the thumb's aligned directions to 3° puts the tips **163 mm apart**.
Buying the 5.9° costs self-penetration (1.5% of frames → 2.4%) and ring contact
(89% → 87%). No setting is worth re-solving every label for, so the default
stays 1.0 - with the constant, the sweep and a mutation-tested check that it
still reaches *both* copies of the term (`loss`, which the teacher solves, and
`per_frame`, which `--direct` scores a student against).

**And the released label is not arbitrary, so it is not unlearnable.** The
worry worth having about a released target is that the solver lands anywhere,
making a one-to-many map no causal network can fit. Measured on frames whose
keypoints agree within 13 mm and which are at least 300 frames apart - a
different tap, not the next frame - the released thumb's joints disagree by
**2.9°**, against **2.8°** for the fingers on those same frames. Contact and
the joint limits still pin the pose. (24 such pairs: a tap is slow, so a speed
gate and a contact rarely co-occur. A first pass without the gate matched 164
touching pairs and said 5.2° against the fingers' 4.6° - thinner evidence
pointing the same way.)

**What was actually wrong was the report.** The fingers were already reported
split by contact state; the thumb - the one digit released *completely* - was a
single pooled number, which moves with how much of a block touches rather than
with how good the mapping is. `score` now returns `thumb_free` and
`thumb_contact`, `learn.py teacher` prints them as two columns, and the pooled
figure is gone from the table.


## Next

The teacher makes contact and keeps the hand out of itself; the student
makes contact on the index. In the order the measurements argue for:

1. ~~**A second seed per setting.**~~ ~~**Three seeds per setting.**~~
   **Done.** Training was entirely unseeded - weight init, dropout, shuffling,
   the augmentation draw - so no two runs were the same run. On the old
   `h4nd_2` labels two seeds of the shipped setting landed **2.66° apart**; on
   the re-solved `h4nd_left` labels three seeds span **1.38°** pooled and
   **0.50°** on `vs teacher`. `learn.py train --seed` sets all of it and the
   checkpoint records which seed made it. See the five-way table above for
   what the seeds then settled.
2. ~~**Bench the executor.**~~ **Done, both halves, and it changed the
   detector.** A free-air run on the energised hand latched nothing. A run
   closing the index finger onto a cylinder then showed the position test was
   unusable on this hand: the command runs further ahead of the encoder in
   free air (28.5 deg) than when the finger is actually pushing on something
   (17.9), because free air is where the acceleration ramp is still climbing.
   Current separates them cleanly - 77 mA free, 280 on contact - so a stall is
   now present current past `STALL_MA` and nothing else. See `executor.py`.

   Still open: one finger, one rigid cylinder. A soft object that stops a
   finger without raising current is the one case current cannot see.
   `Student.CONTACT` and `Student.TTC` remain unread; the executor is where
   they would go, as an earlier warning than current gives.
3. **Record the blocks the first session lacked.** `learn.py survey` now
   carries `middle_pinch`, `ring_pinch` and `object_grasp`; the recording on
   disk has 2.6% and 1.2% of its frames touching the middle and ring, which
   is why the student learned ring contact at 0%.

   ```bash
   .venv/bin/python learn.py survey --append latest --missing
   ```

   The append path is ready and rehearsed: a block recorded into the existing
   session is numbered against THAT session's table rather than today's (they
   differ by three blocks now), and `confidence` - a column the recorder gained
   after that session was made - reads NaN on the old rows and real on the new
   instead of dying with a KeyError at save. `selftest` runs the whole
   rehearsal on a copy. What is left is the recording itself, which needs the
   operator and the sensor. Then pool and re-solve.
4. **Retune the thumb constants for `h4nd_left`.** The hand changed underneath
   them: `retargeter.THUMB_REST_SPREAD` (55°), `synthetic.THUMB_SPREAD_BAND`
   (−0.79, 0.70), `gestures.REST_ALONG_MM`/`REST_PER_RADIUS` (5.6, 0.34) and
   the thumb coupling `flex×0.9, flex×0.5` were all fitted or swept against a
   thumb that has since moved 32 mm and grown 12 mm longer. A cold re-solve of
   `thumb_tap` reaches 74% contact against the 93% the old hand managed once
   these were tuned, and the shortfall is on the ring (40%) — which is also
   the digit whose adduction travel the new export cut from 20° to 5°. Sweep
   them before reading anything into the gap.

5. **The thumb's wrong-side minimum.** The 1.9% of frames still penetrating
   are one pattern, the thumb through the index when the solve begins, on
   tracker-garbage blocks. A seed that respects the operator's own
   thumb-side would remove it; the ordering term deliberately does not.
6. **The posture cost.** The two-view student is 1.7° worse against the
   operator than the keypoint-only one, almost all on the held-out fist,
   where the fingers are the tracker's guess. Whether that is a real loss or
   a metric against a hallucination is decidable once `confidence` is on
   disk.
7. ~~**Re-solve on the repaired collision proxy.**~~ **Done, and it bought
   less than expected - which is itself the finding.** `Proxy.box_gap`
   returned exactly zero for ANY overlap: nine of its fifteen separating axes
   are cross products of parallel face normals, which two links on different
   fingers have through most of a curl, and a guard that never fired let
   those zero-length axes cap the answer at zero. The collision term saw a
   constant instead of a depth wherever a finger was already through another.
   Fixed, the proxy tracks MuJoCo's depth (7.9 mm against 8.0 on deep frames,
   r = 0.93).

   Re-solving the whole recording against it moved 2.5% of frames by more
   than a degree, and it moved the DEEP frames seventeen times further than
   the clear ones (0.23° against 0.013° median) - so the restored gradient is
   demonstrably acting. But self-collision in the labels is **unchanged**:
   2.07% of frames past 3 mm before and after, worst 20.6 mm both times.

   Why: seventeen of the nineteen deep frames are the thumb through the index
   or a finger, which is item 4 below - the thumb starts the solve on the
   wrong side and the way out goes THROUGH more penetration, not around it.
   That is a warm-start and topology problem, and no amount of gradient fixes
   it. The fix was still necessary - the proxy was lying, and every number it
   reported about depth was a floor rather than a measurement - but the
   remaining penetration needs item 4, not this.
8. **Draw the replay subset per epoch.** `train.REPLAY` mixes as many
   synthetic windows as real ones into the fine-tune, but the subset is drawn
   once for the whole stage rather than per epoch. A fresh draw shows the
   fine-tune more of the pretraining set for the same cost. Untried, because
   trying it means a retrain.
9. ~~**Read the forward heads.**~~ **Partly done, and the premise was
   wrong.** Measured, `Student.CONTACT` predicts a PINCH and not contact with
   an object: through an entire wrap on a cylinder it reads 0.00 on all three
   thumb pairs, and on the gesture probe it fires 0.26 to 1.00. So it cannot
   warn the executor that a grasp is coming, which is what this item assumed.

   What it can do is tell the executor which stalls to ignore, and that is
   wired in (`GraspExecutor.pinched`): a pinch and a grasp are identical from
   inside the executor, so latching a pinch made every pinch with GRIP on a
   grip the operator had to open out of.

   **A head that predicts OBJECT contact is the one the executor wants**, and
   it does not exist. The synthetic set has the labels for it - the census
   knows which digits touch a cylinder on every frame - but the operator's
   own session has no objects in it, so those channels would have to be
   masked through the fine-tune rather than taught as zeros. That is the real
   version of this item.
10. **The hardcodes left.** `THUMB_SPREAD_BAND` in `synthetic.py` resists
   measurement in both parameterisations tried and says so. `INDEPENDENT`,
   `SPLIT`, `PINCH_RATE`, `MARGIN` and `EXTREME` are coverage policy, stated
   as such. The synthetic sampler also turns the thumb about the palm normal
   rather than its own fitted opposition axis, which is what
   `THUMB_SPREAD_BAND` is compensating for.
11. **Objects at other angles.** Wrap labels cover cylinders across the palm,
   axis along the knuckle line, radius 12 to 30 mm. The 35 mm probe already
   fails (0.5 links after closing against the teacher's 3.4). `gestures.wrap_pose`
   sweeps against any axis, so authoring rotated wraps is a sampler change
   plus a regenerate; whether it is needed depends on whether the executor
   covers those cases on the bench.
