# handlab

One MuJoCo world for the h4nd and its DYNAMIXEL hardware, plus a camera
that tracks the human hand and a vision-language model to ask about what it
sees. Self-contained: the URDF, meshes, tracking model, settings and code all
live in this folder, and every path is relative to it. Copy the folder
anywhere and it runs.

```bash
python main.py
```

No arguments. Everything that can be tuned is in `settings.json`, so the
world is the same however it is opened.

## The world

| hand | colour | what it is |
| --- | --- | --- |
| encoder | opaque white | the real robot, posed from its encoders |
| command | translucent purple | the joint command the software sends out |
| object | orange | the calibration box, when switched on |
| demo | translucent cyan | a recording's object moved onto the robot hand (Recordings > transfer to sim); out of sight otherwise |
| plan stage | green hand, orange box, cyan ghost | a planned task replayed (Recordings > plan in sim), 0.3 m beside the real hand: the plan hand, the object as the task sim moved it, and where the recording wanted it; out of sight otherwise |

Both stand on the same mount, so the gap between them is the gap between what
was asked and what the robot did: the purple ghost sits on the solid hand
where they agree and shows ahead of it where they do not. Colours are
`world.*` settings.

Both hands are kinematic: posed, not simulated, gravity-compensated. With no
encoder reading - nothing connected, or after a disconnect - the encoder hand
is held at zero (every joint at 0 rad), never left to drift or frozen at a
stale reading.

**Hand orientation** (`world.hand_orientation`): upright, palm up, palm down,
fingers down, thumb up, thumb down - the six directions gravity can pull in
the hand's frame. Set it to match how the real hand is mounted.

**Calibration object** (`object.*`): a box, 55 mm and 20 g by default, placed
in the hand's frame (it turns with the hand) 8 mm in front of the palm.
*Fixed* holds it still where it is placed (its pose is written back every
step, like the hands'); *floating* leaves it to physics, so it falls, can be
pushed, and rests on the palm or the floor.

Its friction (`object.friction`) is the sliding coefficient of every contact
it makes, with the hand and with the floor: the object's geom has the higher
contact priority, so MuJoCo uses its value instead of the larger of the two.
The default, 0.5 for PLA, is an estimate - published FDM-PLA values are
0.35-0.55, all measured against steel (0.492 pin-on-disc, Pawlak, Engineering
Mechanics 2018); no PLA-on-PLA measurement was found. To measure the real
pair: put the printed object on a printed PLA surface, tilt the surface
slowly, and mu = tan(angle at which the object starts to slide).

The object collides with the real hand and the floor, never with the purple
ghost; the real hand does not collide with itself (its CAD links overlap at
the joints). Its contacts are as stiff as MuJoCo allows at the 2 ms timestep
(time constant 2 x timestep): a resting box sits 0.08 mm into the palm, where
MuJoCo's default softness gave 0.89 mm and a slow creep. Bigger overlaps are
not contact error: the real hand is a replay, so a fixed box cannot stop a
finger - any overlap is where the real finger is against where the virtual
box is, which is the calibration measurement. A floating box trapped by
closing fingers gets squeezed out. Fingers collide as their meshes' convex
hulls, which can only make contact slightly early, never late. MuJoCo contacts are soft and would nudge a replayed finger
inside a physics step, so after every step the hands' poses are written back
(`World.hold`): what you see and read is always exactly the replayed pose.

One opaque and one translucent, on purpose. Two translucent hands in the same
place flicker: translucent objects are drawn sorted by distance, overlapping
links tie, and encoder noise flips the order every frame. An opaque hand is
drawn first and depth-tested, so there is nothing to tie. Measured with
Studio's renderer under encoder noise: both translucent 1.4-2.5 % of pixels
flickering; real hand opaque 0.14-0.39 %, most of it the real hand genuinely
moving; command opaque instead 1.6-3.3 %. A faint shimmer can remain where
the two hands coincide exactly. Nothing is moved or faked at render time.

The world is built in one place, `src/world.py`, from settings.json only.
World-section edits are saved and taken into the running model in place
(`World.apply`), with no reload: orientation (each hand's base is a mocap
body, so the mount is simulation state), the object's position, mode, weight
and friction. Showing/hiding or resizing the object rebuilds the model, which the
window shows as a brief reload - Studio's renderer reads geometry only when a
model loads. To add something to the world, add it in `_build`.

## The control loop

```
sliders ──> controller.targets
controller.step(dt) ──> command ──> purple command hand
                               └──> hardware bus (written when armed + sending)
hardware bus encoders ───────────> opaque encoder hand
```

`command` is the targets clamped to the URDF limits and slew-limited to
`control.max_speed`. The purple hand shows exactly what goes to the robot.

What happens after that, and where the feedback is:

```
command --(write Hz, sync write)--> goal position on each servo
  servo velocity profile (profile_velocity / profile_acceleration)
  servo position PID (position_p / i / d) on the servo's own encoder
  current clipped at current_limit_ma      (mode 5: current-based position)
```

The software side is open loop (feedforward): encoder readings draw the white
hand and seed *Send*, nothing more. The closed loop is the PID inside each
servo. On a large move the lag you see is mostly the servo's acceleration
ramp, not the PID.

Some joints do not quite reach their goal, from wire drag and parts rubbing
(measured 2026-10-05): middle_proximal stops 4-5 deg short when flexing
(holding 40-60 mA) and rests bent back; index_adduction ~2 deg short one way;
thumb_adduction ~1 deg. The white hand shows it as it happens.

## Temperature protection

Every encoder read also returns velocity, current, voltage and temperature
(one bus transaction). Above `hardware.temp_derate_c` (45 C) a joint's
current cap falls linearly, in 10 % steps; at `hardware.temp_cutoff_c`
(60 C) torque goes off for the whole hand and arming is refused until every
joint is 5 C below. The servo's own hard limit is 70 C (latches an error that
needs a reboot). Hardware section shows the hottest joint; hover for all
sixteen.

The serial port is opened exclusively: while handlab has it, any program
that tries to open it after (the old app included) is refused.

## Camera and hand tracking

A camera watches the person's hand; MediaPipe's hand landmarker
(`assets/models/hand_landmarker.task`, Apache-2.0, from
storage.googleapis.com/mediapipe-models/hand_landmarker/) finds up to
`vision.max_hands` hands in every frame: 21 landmarks each (wrist, then four
per finger, tip last), in image coordinates and in metres around the hand's
centre, with left/right and a confidence. It follows hands from frame to
frame rather than searching each one afresh. One thread owns the camera and
the tracker (`src/vision/camera.py`); the preview and the assistant only read
its latest frame, so neither can slow the simulator.

The camera is off until switched on (`vision.enabled`). The first time, macOS
asks whether the app python runs under (Terminal, VS Code, ...) may use the
camera, and the preview waits for the answer; if it was no, turn it on in
System Settings > Privacy & Security > Camera and press *Retry*.

Tracking and recording always use the image as the camera took it, never
mirrored, and left/right is the person's own hand - MediaPipe's label as it
comes, checked on a recording of a left hand. (Recordings made before
2026-10-22, format 1, stored the two swapped; they are swapped back when
read.) `vision.mirror` (on by default) only flips the preview, like a
mirror, which is easier to work in front of.

Nothing reaches the robot yet: this is the observation side - what the hand
is doing - for retargeting and context later.

## Recordings

*Record* (Camera section) writes an **episode** of the task named beside it:
the frames, the tracked hands and the robot's command and encoders, all on
the camera frame's clock. Processing comes after, on the saved episode, one
step at a time; each step adds its own file and never rewrites the
recording, so a step can be re-run when a better model comes along.

```
data/tasks/<task>/<YYYYmmdd-HHMMSS>/     (data/ is not committed)
  meta.json     format, task, camera, tracker settings, versions; "complete"
  video.mp4     H.264, as the camera took it (never mirrored)
  frames.npz    time (N,) s from the first frame
  hands.npz     image/world (N, 2, 21, 3), score (N, 2), present (N, 2);
                slot 0 = the person's left hand, 1 = right; NaN where unseen
  robot.npz     command, encoders (N, J) rad (NaN: no reading), joints
  context.json  the describe step: the task context and what made it
  masks.npz     the segment step: each object's mask in every frame (bit-packed),
                area per frame, the prompt used, the model and when
  hands_clean.npz  the hands step: one followed hand per slot, filled, smoothed
  object.npz    the object step: centre, area, angle, contact, in-hand motion
  pose3d.npz    transfer, Fit 3D: palm and object poses in metres; the object in the palm
  sim.npz       transfer, Send to sim: the object's motion in the robot hand's frame
  plan.npz      plan in sim: the grip, the hand's joints and the simulated object, errors
```

`meta.json` is written with `"complete": false` when recording starts and
finished when it stops, so a recording cut short says so. The layout lives
in `src/recording/episode.py`.

Steps on a recording, in order (Recordings section):

1. **Describe** - the task context, from a clip of it (see Assistant).
2. **Segment** - a mask of each object the context names, in every frame
   (`src/processing/segment.py`), by `segment.model`:
   - **SAM 2.1** (`facebook/sam2.1-hiera-*`, Apache-2.0, open): starts from
     each object's box in the context, on the frame the box was given for,
     and tracks it forward to the end and backward to the start. Streamed, so
     memory does not grow with the recording. Small: ~0.25 s a frame on an
     M4 Max (Apple GPU).
   - **SAM 3** (`facebook/sam3`, gated): finds every instance of each
     object's *name* and keeps the one that best fits the context's box.
     Needs `HF_TOKEN=` (a read token of a Hugging Face account with access)
     in `handlab/.env`. Runs on the whole video at once - its clean-up of
     duplicate and false tracks needs the frames after. Measured on an 8 s
     recording: ~1.3 s a frame (5x SAM 2.1 small), 8.8 GB at peak, masks
     matching SAM 2.1's (median IoU 0.99) from the name alone. Memory grows
     ~20 MB a frame, so a recording too long for the machine is refused
     before it starts.

   Weights download on first use into `assets/models/huggingface/` (not
   committed): SAM 2.1 small 184 MB, SAM 3 3.4 GB. Steps run one at a time
   off the GUI thread, with progress and *Cancel*.
3. **Hands** - each of the person's hands followed as *one* hand through
   the recording (`src/processing/hands.py` -> `hands_clean.npz`).
   MediaPipe loses a hand hidden behind the object, flips left/right when
   it turns edge-on, and sees phantom hands on the fingers - surer of the
   phantom than of the real hand. So: hands are followed by where the palm
   goes (waiting up to 2 s while one is hidden), a track sitting inside
   another is dropped as a phantom, the side is the vote of all the labels,
   short gaps are filled and every run is smoothed. On the first recording:
   one right hand, 31 labels corrected, 33 phantom detections dropped.
   Playback shows these hands once the step has run.
4. **Object motion** - each object's centre, size, how far it has *turned*,
   which hand holds it (landmarks touching the mask), and its position and
   turn relative to that hand (`src/processing/objects.py` -> `object.npz`).
   The turn compares each frame's mask with a reference frame's and finds
   the rotation that lays one on the other, so it works for any shape - a
   cube has no long axis (its silhouette is 1.05 : 1). Within 1-3 degrees on
   synthetic turning bars and squares, slow, fast and back and forth. In the
   image plane: a turn toward or away from the camera shows only as a
   change of area. Plotted under the step; the plot's cursor is the
   player's frame, and a click on the plot moves it there.

5. **Transfer to sim** - two parts.
   *Fit 3D* (`src/processing/pose3d.py` -> `pose3d.npz`): the hands and the
   object in metres. The hand: MediaPipe's 3D hand shape fitted to its image
   points with the camera's calibration (OpenCV solvePnP), at one size for
   the whole recording (MediaPipe's per-frame size guess wobbled 85-102 mm,
   which swung the distance by 5 cm); a palm frame from the wrist and
   knuckles (x up the fingers, z out of the palm). The object: a box of the
   World section's `object.size_mm`, posed so its projected outline covers
   the mask - its image size gives the distance, its shape the rotation -
   found by searching 2000 orientations on the first frame (and whenever a
   frame fits worse than 0.95) and refined frame to frame. A box looks the
   same turned onto any like face (a cube: 24 ways), so each frame keeps the
   equal pose nearest the last; near square-on an outline hides up to ~10
   degrees of tilt. Both are smoothed (median of 9 frames). With a still hand
   (palm moving under 3 cm across the image) the palm's median pose is used,
   since its orientation is the least sure thing while the object hides it.
   The object's known size also measures the hand: `hand_scale`. On test
   cubes: within 1 degree and 5 mm (depth up to ~4%). On the red-cube
   recording: outline fit 0.97, the cube's centre 20-32 mm out of the palm
   (a 55 mm cube resting on it: 27.5+), turning 8-46 degrees back and forth,
   the hand 0.87x MediaPipe's average size.
   **ArUco markers (testing)**: with `vision.aruco` on, Fit 3D also looks
   for markers in every frame (`src/vision/aruco.py`); the largest is taken
   to sit in the middle of a face of the object, and where one is seen its
   pose is the object's (the wireframe turns magenta). The outline fit
   still runs alone on every frame, and is measured against the marker:
   the summary and the plot show the outline fit's error in mm and degrees.
   On a synthetic recording of a cube carrying a marker: 2.6 mm, 2.9
   degrees. A marker's own accuracy at 0.3 m with this camera: 1-2 mm, 1-2
   degrees tilted, 4-5 degrees square-on. The live preview draws markers
   with their ids. Print them with `scripts/aruco_markers.py`.
   *Send to sim* (`src/processing/transfer.py` -> `sim.npz`): that motion in
   the robot hand's frame - the person's palm laid on the robot's, fingers to
   fingers, palm-out to palm-out, anchored where the World section places
   the calibration object. With *show in sim*, a translucent cyan ghost box
   in the world (`demo`, touching nothing) follows the player's frame. The
   robot is a left hand; a right-hand demonstration is not mirrored yet.
   A step whose input was made again after it ran says so (amber) and the
   next step will not run on it: Send to sim after a new Fit 3D, Plan after
   a new Send to sim.

6. **Plan in sim** (`src/processing/plan.py` -> `plan.npz`): can this hand,
   mounted as the World section has it, do what the recording shows? No
   training. The task sim (`src/sim/task.py`) is the robot with a position
   servo on every joint - XL330-M288 from its spec sheet: 0.52 N m at
   1.47 A, so `hardware.current_limit_ma` caps the torque (300 mA: 0.11 N m)
   - the object free, gravity, and a distance sensor from each fingertip to
   the object. The hand collides with itself, as the real one does: each
   link as its CAD mesh's convex hull; only parts joined at rest are left
   out (a finger's first link against the base and palm it is mounted in,
   and neighbouring fingers' proximal and middle links, which already
   overlap by up to 1 mm) - found from the CAD when the sim is built. Forced
   into each other, the fingers stop: no pair gets more than ~1 mm deeper
   than at rest. (Until 2026-10-23 the hand did not collide with itself, and
   plans pushed fingers through each other.) The planner (`src/sim/planner.py`) first searches for a
   **grip**: one servo target per joint, tried 512 at a time in batched
   MuJoCo rollouts - close on the object held where the motion starts, let
   go, measure how far it slips - narrowed by the cross-entropy method over
   20 rounds. Then it **follows** the motion with predictive sampling:
   every 40 ms, 128 short finger motions are run 0.4 s ahead from the sim's
   state and the one whose object best matches the recording is used. The
   summary gives the grip's slip, whether the object was ever lost, and the
   error while following next to the error had the object just stayed put
   (following no better than that has not moved it). With *show in sim* the
   plan stage replays it at the player's frame; the plot's cursor is the
   player's frame. On the red-cube recording, hand upright: a grip that
   slips 0.7 mm, 1 degree when let go, held throughout; following 4.6 mm /
   5.9 degrees off (median; worst 8.8 mm, 17 degrees) where staying put
   would be 10 mm / 23 degrees; fingers at most 1.8 mm into each other. About
   24 s. A plan made for one mount says so when the World section's
   orientation changes.

7. **Track with the tracker** (`src/processing/track.py` -> `track.npz`):
   the trained general tracker (see *Learning* below) follows the motion -
   no retraining, only a grip where the motion starts (step 6's when
   current, else searched). This is the one-shot test: a recording is never
   trained on. Shown like step 6; when both exist, *plan stage shows* picks
   which one the stage replays.

   Masks are used as the object's own region: SAM leaves stray specks far
   from the object (on the first SAM 3 recording, in 345 of 411 frames),
   which moved the measured centre by 17-28 px. `src/vision/regions.py`
   keeps the largest piece and the pieces of the same object, and drops the
   rest - when masks are made, measured and drawn.

## Learning: a general tracker

One policy that moves the object along whatever motion it is given - so a
new demonstration needs no retraining, only its motion (`src/learning`).

1. **Library** (`scripts/build_library.py` -> `data/learning/library.npz`,
   about 20 min). Starts: where each recording's motion begins, and 5
   shifted / turned a little around each; the planner searches a grip at
   each. Motions: 16 generated per start - up to 3 moves, each a 10-45
   degree turn about a random axis and a shift within 15 mm, minimum-jerk -
   plus the recordings' own. The planner follows every one; feasible = held
   throughout, median position error within 15 mm, and turn error within 12
   degrees or within half of what staying put would leave. The planner is
   a lower bound on the hand, so this only drops what it could not follow
   at all.
2. **Training** (`scripts/train_tracker.py` -> `data/learning/runs/<stamp>/`),
   by `learning.method`:
   - `ppo` - rsl_rl's PPO over 2048 copies of the task sim stepped together
     on the Mac's cores (`mujoco.rollout`): the policy finds its own way by
     trial and error. On the red-cube recordings it learned to hold the cube
     but not to roll it (it follows about half as well as the planner).
   - `planner_teacher` - the planner labels, the policy copies (DAgger,
     `src/learning/teacher.py`): each round runs every training motion once;
     at every step the planner plans from that copy's exact state and gives
     the servo targets it would set; the copy takes the planner's action or,
     more each round, the policy's own, so the policy also learns to recover
     from its own mistakes; then the policy is fitted to all labels so far.
     About 20 min a round (82 ms of planning per step).

   Either way the policy runs at 20 steps a second.

   **On the GPU via ClearML** (`src/learning/gpu.py`, `remote.py`,
   `scripts/clearml_train.py`, the Learning panel's *Train on ClearML*): the
   job goes to the ClearML agent (the Windows PC with the RTX 3090, queue
   `learning.clearml_queue`, project HandRL) as a pipeline. The Mac uploads a
   bundle - its library, the task sim as MJCF, the hand mount - and the agent
   runs `scripts/clearml_gpu_step.py` from GitHub at this commit with
   `requirements-gpu.txt`: PPO over `learning.gpu_envs` task sims stepped
   together on the GPU by MuJoCo Warp, observations and rewards in PyTorch
   on CUDA. Same observations, actions and rewards as the Mac's environment
   (measured: identical at reset; within 0.2 mm and 0.3 degrees after 200
   contact-rich steps), so the weights come back into Runs and run on the
   Mac's CPU (Test, Recordings step 7). Commit and push first - the panel
   says when not. The policy sees
   what the real hand will have - joint angles now and a step ago, servo
   targets, the object's pose as a camera would give it (noisy) - and the
   goal 0.05, 0.25, 0.5 and 1 s ahead as an offset and a turn from the
   object (the turn to the goal's nearest look-alike: a cube looks the same
   24 ways). The critic also sees the object's velocity, fingertip gaps and
   the physics drawn for that copy. Reward: closeness in position and turn,
   less target changes and fingertips off the object; dropping it ends the
   episode. Each episode starts from a library grip, the object just let
   go. Friction, mass and servo strength are drawn per copy, sensors are
   noisy, and new targets arrive up to 20 ms late. Trained on the generated
   motions only; evaluated on every motion, the recordings being the
   one-shot test. `tensorboard --logdir data/learning/runs` to watch.

## Assistant

A vision-language model says **what task** the frames show - **Claude**
(Anthropic Messages API, `claude-opus-5-5` by default) or **OpenAI**
(Responses API, `gpt-6-astra` by default). Nothing is typed to it: it is
always asked the same thing, and has to answer as JSON in one fixed shape
(structured outputs; `src/assistant/context.py`):

| field | what |
| --- | --- |
| `task`, `goal` | what the person does; the end state that means done, in terms of the objects |
| `objects` | name (a short visible noun phrase - a segmentation prompt), role (moved / tool / target / support / other), a rough box in one of the images (to start tracking) |
| `hands` | which of the person's hands take part, doing what |
| `phases` | the steps in order, as ranges of the images sent |
| `uncertain` | what the images could not show |

The system prompt (`assistant.system_prompt`) tells the model what the
answer is for: object names become segmentation prompts, boxes start
tracking, phases split the recording; describe what happens to the objects,
not the fingers. A recording is described from `assistant.clip_frames`
frames (16) spread evenly over it, each with its time in the message - a
"clip"; neither API takes a video file. Frames are scaled to
`assistant.image_max_side` px (at most 2000: Claude's per-image limit when a
request has more than 20). One question at a time, on its own thread.

API keys are **never settings**, and the app never writes one anywhere.
Each provider's key is taken from, in order: one typed into the panel (kept
in memory for that session only); `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in
the environment; the same names in `handlab/.env`, one `NAME=value` per line
(`src/keys.py`; `HF_TOKEN` for SAM 3 comes the same way):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

`.env` is gitignored, and re-read when it changes (no restart). A line
without `NAME=` is ignored, and the panel says so without showing it.

For Claude: thinking is always on with Opus 5.5, and `assistant.claude_effort`
(low ... max, default medium) sets how hard it thinks. Requests opt into
server-side refusal fallback (`fallbacks="default"`): if Claude declines, the
API re-runs the request on Anthropic's recommended fallback model in the same
call, and the answer says which model answered. Answers are streamed and
collected whole.

## The window

MuJoCo Studio (MuJoCo's ImGui viewer). Our panels only ever join Studio's
side panes, so nothing is drawn over the 3D view. A **Robot** panel is docked
on the right, as a tab beside Studio's Inspector; a **Vision** panel on the
left, as a tab beside Studio's Options, and a **Learning** panel as a tab
beside Vision. The menu bar has a menu for each that shows or hides its
sections.

**Vision** (left):

- **Camera** - *on*, *mirror*, and the camera dropdown: the cameras the Mac
  lists, by name, rescanned when it opens (elsewhere, a camera number); the
  status line (camera, size, fps, hands found) or what went wrong, with
  *Retry*; the live preview with the tracked hands drawn on (skeleton,
  left/right and confidence, fps). *camera settings* (collapsed): capture
  size, how many hands, find/follow confidence. *back camera*: a second
  camera filming the other side of the hand (`vision.back_camera_index`,
  none by default) - fingers the object hides from one camera, the other
  sees. It runs beside the first with the same settings and its own hand
  tracker, its preview and status below; *Record* records both: the back
  camera's video_back.mp4, frames_back.npz (same clock) and hands_back.npz.
  In Recordings the player shows the two side by side.

  A USB camera shows in the dropdown only once macOS has let it connect: on
  Apple silicon, answer the "Allow accessory to connect?" prompt when it is
  plugged in (System Settings > Privacy & Security > Allow accessories to
  connect). If FaceTime or Photo Booth can't see it, neither can this.
  Under the camera's name, its calibration (focal length, error, date) or
  *not calibrated*, and **Calibrate...**: a window with the live view, the
  checkerboard's corners drawn when found, and where pictures were already
  taken; *Take picture* / *Undo* / *Clear* (8 or more needed), *Calibrate*,
  then *Save for <camera>*. Show the board (printed, or on a phone or
  tablet) near and far, tilted every way and in every corner of the view.
  The corner count is set there (`vision.checkerboard`, inner corners: a
  board of 10 x 7 squares has 9 x 6); square size does not matter. Saved in
  `settings.json` under `camera_calibration.<camera name>`, and written
  into every recording made with that camera (`meta.json`). On synthetic
  views it recovers the focal length to 0.1%.

  Below the status: *Record* / *Stop* and the *task* name; while recording,
  its length and frame count, then where it was saved.
- **Recordings** - pick a recording (rescanned when the list opens); it
  plays at recorded speed (mirrored like the preview, if *mirror* is on);
  *Play*/*Pause* and a frame slider; switches for what is drawn on it:
  *skeleton* (tracked hands, cleaned once step 3 has run), *masks* (segment
  step, one colour per object), *boxes* (the context's box, on the frame it
  was given for), *axis* (each object's long axis through its centre).
  Then the steps: **1 describe** (*Describe*), **2 segment** (model
  dropdown, *Segment*, progress, *Cancel*; then how much of the recording
  each object was found in), **3 hands** (*Clean hands*; how many hands,
  labels corrected, phantoms dropped), **4 object motion** (*Measure*; who
  holds each object, and a plot of its angle in the image and in the hand -
  click the plot to move the player), **5 transfer to sim** (*Fit 3D*,
  *Send to sim*, *show in sim*), **6 plan in sim** (*Plan*; the grip, held or
  lost, the error plot). *sent frames*: the frames Describe sent, numbered as the
  model saw them, with the same overlays - click one to show it in the
  player. *task context*: task, goal, objects (hover for the box), hands,
  phases, uncertain.
- **Assistant** - *provider*, *model*, *effort* (Claude), where the *API key*
  comes from (or a field to enter one for this session). *Describe frame*
  sends the live camera frame, as a quick look at what the model makes of
  it (not saved). *prompt and limits* (collapsed): the system prompt, max
  tokens, image size, clip frames.

**Learning** (left, beside Vision; see *Learning: a general tracker*):

- **Library** - what the library holds (when made, starts, feasible
  generated motions by kind, the recordings kept for the test); *Build
  library*, with progress, time left and *Cancel*.
- **Train** - *method* (`ppo` or `planner_teacher`) and its settings (PPO:
  *sim copies*, *rounds*; teacher: *rounds*); *Train*, then
  progress, *Cancel*, and a live plot of the turn and position error and the
  exploration, round by round (measured while it explores, so noisier than
  its test).
- **Runs** - every run, newest first: its rounds, its test (each recording -
  never trained on - and the median of the motions it trained on, next to
  the planner and to holding still; a drop in red), its curve; the radio
  button picks the run Recordings step 7 uses (`learning.run`); *Test* for a
  run not yet tested.

Building and training run as processes of their own
(`src/learning/background.py`), reporting into `data/learning/status.json`:
the window stays responsive while they use every core, and closing the
window does not stop a run - *Cancel* does. The same scripts run from a
terminal show in the panel too.

**Robot** (right):

- **Joints** - one slider per URDF joint, grouped by finger, ranges from the
  URDF limits, with the robot's encoder reading beside each. *Zero all* and
  *Match robot* (targets := where the real hand is).
- **World** - *hand* orientation dropdown, then the calibration object:
  *show* and *Reset object*; *mode* (fixed / floating); typed inputs for
  *width*, *depth*, *height* (mm) and *weight* (g); a *friction* slider with
  a *PLA* button that puts it back to the PLA default; sliders for its
  position in the hand's frame - *out of palm*, *across palm*, *along
  fingers* (mm; ctrl-click a slider to type a value). Sliders apply live as
  you drag and save on release. *Show* and the size inputs reload the scene
  (a brief black frame); everything else is live.
- **Hardware**
  - *temp*: the hottest joint, amber while derating, red at cutoff; hover
    for all sixteen.
  - *device* dropdown: none / mock / every USB serial port found (rescanned
    when the list opens, so a U2D2 plugged in later shows up). Picking one
    saves it and connects.
  - *Torque on* energises the motors where they are (goal := present first,
    so nothing jumps). *Send* starts writing the command, after first
    snapping targets and command to the hand's current pose - so flipping it
    moves nothing until a slider does.
  - *motion* quick settings: **smooth** (the bench defaults: eases in over
    0.64 s), **fast** (7.2 rad/s, 0.24 s ramp), **real time** (no servo
    profile: the 100 Hz command stream is the trajectory, limited only by
    max speed). Each sets `control.max_speed` and the two servo profile
    registers; PID and current limit are untouched. Shows *custom* once a
    field is edited by hand. The preset numbers are in `motion_presets` in
    `settings.json` - edit them or add your own.
  - *Config*: baud, write/read Hz, max speed, current limit, velocity
    profile, PID. *Motor map* (collapsed): id, reversed, zero per joint.

Every edit, in any section, is checked by `settings.py` (out-of-range
numbers are clamped, bad values rejected) and saved to `settings.json`.
World edits apply live (showing/hiding or resizing the object reloads the
scene). Hardware edits are applied live the safe
way: profile and PID are written to the motors at once (RAM registers), rates,
max speed and temperature thresholds take effect next loop, the current limit
at the next Torque on
(EEPROM, needs torque off), and a new device, baud or motor map reconnects -
sending stops, torque is left as it is. Numbers commit on Enter or when the
field loses focus, never per keystroke.

Neither switch is ever restored at startup. Connecting is the only thing that
happens on its own, when `hardware.port` is set. Any bus error turns sending
off and shows why. The device can't be changed while sending.

## Settings

`settings.json` is written from the defaults in `src/settings.py` on first
run. Edit it by hand; a wrong value falls back to its default with a message,
unknown keys are dropped, missing ones are filled in. Code reads it through
one module (the GUI edits through it too):

```python
from src import settings
settings.get("hardware.port")
settings.section("hardware")
settings.set("hardware.port", "/dev/cu.usbserial-XYZ"); settings.save()
```

| key | what |
| --- | --- |
| `robot.urdf` | the robot; sliders, world and motor map all follow it |
| `world.*` | timestep, mount position and orientation, each hand's colour and emission (encoder, command, plan) |
| `object.*` | calibration box: shown, fixed/floating, size, weight, friction, position, colour |
| `learning.*` | the tracker's training: method, sim copies and rounds (PPO), rounds (teacher); the run step 7 uses (`""` = newest); the ClearML queue and where the library comes from |
| `control.max_speed` | command slew limit, rad/s |
| `hardware.port` | `""` none, `"mock"` fake bus that echoes commands, or the U2D2 port |
| `hardware.*` | baud, write/read rates, current limit, velocity profile, PID |
| `hardware.temp_*` | temperature derate start and cutoff, deg C |
| `hardware.motors` | per joint: motor `id`, mounting `sign` (±1), `zero` tick |
| `motion_presets` | named sets of max speed + servo profile for the *motion* dropdown |
| `vision.*` | camera on/off, which camera, capture size, mirror, tracker hands and confidences, model file, checkerboard corners and square size, ArUco on/off, dictionary and printed size |
| `camera_calibration` | per camera name: focal length, centre, distortion, error, at the size it was made |
| `recording.*` | where recordings go (`data/tasks`), the task name being recorded |
| `segment.model` | which SAM segments recordings: SAM 2.1 tiny/small/base+/large (box) or SAM 3 (name) |
| `assistant.*` | provider, model names, Claude effort, max tokens, image size, clip frames, system prompt - no keys |
| `viewer.*` | window size; `max_fps` (60) - Studio has no vsync, so frames are capped |

## Scripts

All take no arguments and read `settings.json`.

| script | does |
| --- | --- |
| `scripts/selftest.py` | headless checks: settings, URDF, world, orientations, calibration object, controller, bus and temperature protection (mock bus), hand tracker, camera thread (fake camera), recording round trip, assistant, task-context schema and its API requests (fake provider, no network), segment step (fake model), hands step (flips, phantoms, gaps), object step (known rotations), camera calibration (synthetic and stretched boards), old-format hand labels, box pose from its outline, palm axes, transfer onto the robot hand, ArUco pose, markers in the preview, Fit 3D with a marker (end to end), overlays, the plan stage, the task sim's weld and servos, a grip that holds, the plan step, out-of-date files, generated motions, the tracker's environment |
| `scripts/render.py` | `renders/world.png` - the world without a window |
| `scripts/build_library.py` | the tracker's trajectory library from the recordings (about 30 min; also the Learning panel's *Build library*) |
| `scripts/clearml_train.py` | trains on the ClearML agent's GPU as a pipeline and fetches the run (also *Train on ClearML*) |
| `scripts/clearml_gpu_step.py` | the GPU training step - run by the agent, not by hand |
| `scripts/train_tracker.py` | trains the general tracker on this Mac by `learning.*`, then tests it on every motion (also the Learning panel's *Train*) |
| `scripts/aruco_markers.py` | `renders/aruco_markers.png`: markers 0-5 at `vision.aruco_mm`, 300 dpi, with a 50 mm bar to check the print |
| `scripts/hw_scan.py` | lists serial ports, pings every motor, prints joint angles. Read-only |
| `scripts/hw_watch.py` | live encoder readout for checking motor signs and IDs. Read-only |

## Layout

```
main.py                 opens the world
settings.json           every tunable
assets/h4nd_left/       URDF + meshes (from world/h4nd_left)
assets/models/          MediaPipe hand landmarker
src/
  settings.py           the one settings reader/writer
  paths.py              where things live
  robot.py              joints, order, limits - from the URDF
  world.py              THE world
  control.py            targets -> clamped, slew-limited command
  hardware/dynamixel.py XL330 bus driver (+ MockLink)
  hardware/bus.py       thread that owns the port; connect/arm/stream states
  hardware/thermal.py   temperature derating and cutoff
  vision/camera.py      thread that owns the camera: capture -> track -> overlay
  vision/tracker.py     MediaPipe hand landmarker, 21 landmarks per hand
  vision/overlay.py     draws the tracked hands on the frame
  vision/permission.py  asks macOS for camera access
  vision/devices.py     camera names in OpenCV's numbering
  vision/calibration.py camera calibration from checkerboard pictures
  vision/regions.py     an object's own region in a mask, without specks
  vision/aruco.py       ArUco markers: detect, pose, the box centre behind one
  assistant/assistant.py  background question runner, API keys
  assistant/question.py   Question / Answer for every provider; frame -> JPEG
  assistant/context.py    the task context: JSON schema, instruction, context.json
  recording/episode.py    the episode layout on disk; Episode reads one back
  recording/recorder.py   Recorder: writes an episode from the camera thread
  processing/job.py       Job: one step at a time off the GUI thread, with progress
  processing/segment.py   the segment step: SAM 2.1 / SAM 3 -> masks.npz
  processing/hands.py     the hands step: follow, de-phantom, vote, fill, smooth
  processing/objects.py   the object step: motion in the image and in the hand
  processing/pose3d.py    Fit 3D: hand pose (PnP) and the box fitted to its outline
  processing/transfer.py  Send to sim: onto the robot hand, for the world's ghost
  processing/plan.py      Plan in sim: grip and follow in the task sim -> plan.npz
  sim/task.py             the task sim: robot with M288 servos, free object, gravity
  sim/planner.py          grip search (cross-entropy) and predictive sampling
  processing/files.py     the later steps' files, and which are out of date
  processing/track.py     Track with the tracker: the trained policy -> track.npz
  learning/trajectories.py  the tracker's library: grips, motions, feasibility
  learning/env.py         TrackEnv: thousands of task sims for rsl_rl's PPO
  learning/train.py       training methods, progress, evaluation, a trained policy's path
  learning/teacher.py     the planner teaches the policy (DAgger)
  learning/background.py  library builds and training as processes, their status
  learning/remote.py      training on the ClearML agent's GPU, as a pipeline
  learning/gpu.py         the environment on the GPU (MuJoCo Warp), PPO, test
  keys.py                 secrets (API keys, HF_TOKEN): environment, then .env
  assistant/claude.py     Anthropic Messages API
  assistant/openai_api.py OpenAI Responses API
  viewer/studio.py      launches MuJoCo Studio; the rest of viewer/ are its plugins
  viewer/panel.py       side panel made of collapsible sections
  viewer/form.py        settings-backed input widgets the sections share
  viewer/joints_panel.py    Joints section
  viewer/world_panel.py     World section
  viewer/hardware_panel.py  Hardware section
  viewer/camera_panel.py    Camera section (preview, Record)
  viewer/recordings_panel.py Recordings section (playback, Describe)
  viewer/learning_panel.py  Learning panel (library, training, runs)
  viewer/image.py           an image in a panel (GUI texture rules)
  viewer/assistant_panel.py Assistant section
  app.py                wires it all together
scripts/                helpers, no arguments
```

## Notes

- Studio is `mujoco.experimental.studio`, so mujoco is pinned (3.12.0). Its
  own launcher opens the window off the main thread, which macOS refuses;
  `viewer/studio.py` runs the viewer on the main thread and the sim loop on a
  worker thread instead. Plain `python`, no `mjpython`.
- That wiring uses two PRIVATE Studio classes
  (`launch_passive._PassiveSnapshotChannel`, `_PassiveEventChannel`): there
  is no public way to run Studio's viewer on the main thread. The pin keeps
  them stable; they are the first thing to check on a MuJoCo upgrade.
- Studio redraws as fast as it can - ~300 frames/s measured, the GPU never
  resting and the fans up - and has no vsync option. `viewer/studio.py`
  caps it at `viewer.max_fps` (60): measured, the app went from 84% to 30%
  of one core. The rest is the ~750 Hz sim loop (~5%) and drawing.
- The window is titled "PyStudio handlab": Studio's compiled viewer puts
  "PyStudio" in front of any title, and has no option to change it.
- Studio 3.12's renderer crashes when geoms it has already drawn are hidden
  by geom group (measured: hiding works only from the first frame). So
  nothing in handlab hides by group, and Studio's own *Visibility Groups*
  panel should be left alone while the window is open.
- One process per serial port. Once handlab has the port it holds it
  exclusively, so anything opened after it is refused. It cannot see a
  program that opened the port BEFORE it without a lock (the old
  `archive/main.py` panel does not lock) - close that first.
- Studio 3.12 GUI images (the camera preview): uploading one before the
  renderer has drawn its first model crashes, and freeing one crashes, so
  `viewer/image.py` waits out the frame a model arrives in and keeps one
  texture per image for good. Check both on a MuJoCo upgrade.
- At startup macOS logs `Class SDL... is implemented in both ... cv2 ...
  and ... native_viewer_cc`: OpenCV (which MediaPipe imports) ships its own
  SDL2, and Studio has one built in. Both are SDL 2.32.8, so the two copies
  of each class are the same; the warning is harmless while they match.
  If a future OpenCV or MuJoCo changes either, recheck.
- `renders/` and `data/` are not committed.
