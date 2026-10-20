"""A grid of robot hands, each replaying a different sequence, in one scene.

A metrics table says 40% of the synthetic frames are touching and the teacher
closes 83% of them. It does not say whether the hands LOOK like hands. Sixteen
of them side by side, each with its operator skeleton floating beside it, do:
a bad sampler shows up as a row of hands nobody would make, and a teacher that
misses a pinch shows up as a red skeleton beside a robot whose thumb is
somewhere else.

WHAT THE COLOURS MEAN. Every robot is the teacher's answer, drawn in the
teacher's green. The skeleton is the hand it was given - blue while the
operator is not touching anything, RED while the operator's thumb tip is
within `TOUCH_MM` of a fingertip on the raw view. So red beside a closed robot
thumb is the objective working; red beside an open one is a miss to look at.
The two-digit plate under each cell is the run's index into the file.

Runs the same for a recording, so the two can be compared at a glance: pass a
session directory instead of a pair file.
"""

from __future__ import annotations

import time

import numpy as np

from ..robot.mjcf import TEACHER_RGBA
from ..robot.layout import Grid
from ..robot.options import Scene
from ..robot.sim import HandSim
from ..robot import glyphs
from . import geometry, objective
from .replay import SKELETON_RGBA, rgba_tuple, solved, unscaled

TOUCHING_RGBA = (1.0, 0.28, 0.22, 0.95)
HAND_HEIGHT = 0.20
# The skeleton stands BESIDE its robot, inside the cell, on the side away from
# the next column. The lane layout's offset put it a full lane over, which in
# a grid is on top of the neighbour.
SKELETON_BESIDE = 0.13
PITCH = 0.36


def pick_runs(bounds, hands, seed):
    """`hands` runs, spread across the file rather than the first N - the
    first N of a synthetic set are the first N random hands, but of a
    recording they are all one block."""
    bounds = [tuple(b) for b in bounds if b[1] - b[0] > 2]
    if hands >= len(bounds):
        return bounds
    rng = np.random.default_rng(seed)
    return [bounds[i] for i in sorted(rng.choice(len(bounds), size=hands, replace=False))]



def play(target, hands=16, pitch=PITCH, speed=1.0, loop=True, robot_hand="left",
         seed=0, png=None, frame=0):
    """Open the gallery. With `png`, render one frame headless to that file
    instead - the same scene, no window, which is how it is checked."""
    data, kind = solved(target)
    q_star, keypoints, bounds = data["q_star"], data["keypoints"], data["run_bounds"]
    raw = data.get("keypoints_raw")
    if raw is None:
        raise SystemExit("this file has no raw view - re-run learn.py teacher / synth on it")
    stored = [str(n) for n in data["joint_names"]]

    runs = pick_runs(bounds, hands, seed)
    # Cell 0 holds the simulated hand; every other cell is a posed copy placed
    # relative to it. `layout.Grid` owns the block - see `robot/layout.py`.
    block = Grid(len(runs), robot_hand, pitch=pitch, height=HAND_HEIGHT)
    side = block.side
    camera_pos, camera_lookat = block.camera()
    sim = HandSim(hand=robot_hand, scene=Scene(
        show_viewer=png is None, ghost=False, main_rgba=TEACHER_RGBA,
        position=block.anchor, copies=block.copies("cell", TEACHER_RGBA),
        camera_pos=camera_pos, camera_lookat=camera_lookat))
    sim.build()
    order = [stored.index(n) for n in sim.joint_names]
    prefixes = {k: f"cell{k}_" for k in range(1, len(runs))}

    # What is drawn beside each robot: the operator's own proportions where
    # the file carries them, so the skeleton is a hand and not a solver input.
    drawn = keypoints
    if "digit_scale" in data and "fingers" in data:
        drawn = unscaled(keypoints, dict(zip((str(f) for f in data["fingers"]),
                                             data["digit_scale"])))
    touching = (geometry.pairwise(raw)[:, geometry.THUMB_ROW] < objective.TOUCH_MM).any(axis=1)
    origins = [np.array([dx, side * (dy - SKELETON_BESIDE), HAND_HEIGHT])
               for dx, dy in block.cells]

    cell = min(glyphs.CELL * 1.5, pitch * 0.9 / (glyphs.cells("00")[1] * 1.4))
    base = HAND_HEIGHT - 0.20 + glyphs.LIFT * cell * 1.4
    plates = [(f"{k:02d}", (dx + 0.10, side * dy, base), rgba_tuple(TEACHER_RGBA), cell)
              for k, (dx, dy) in enumerate(block.cells)]

    hz = 125.0                    # pair files without timestamps were made at --hz
    if "t" in data and len(data["t"]) > 1:
        hz = 1.0 / max(float(np.median(np.diff(data["t"]))), 1e-6)
    rows, cols = block.shape
    print(f"[gallery] {target}: {len(runs)} of {len(bounds)} runs, {rows}x{cols} grid, "
          f"pitch {pitch} m, {hz:.0f} Hz")
    if png is None:
        print("[gallery] blue skeleton = not touching, red = operator's thumb on a "
              "fingertip. Close the window to stop.")

    def pose(step, scene=None):
        skeleton, colours = {}, {}
        for k, (start, end) in enumerate(runs):
            index = start + (step % (end - start))
            q = q_star[index][order]
            if k == 0:
                sim.command(q)
                sim.reset(q)
            else:
                sim.set_copy(prefixes[k], q)
            points = drawn[index]
            colour = TOUCHING_RGBA if touching[index] else rgba_tuple(SKELETON_RGBA)
            for slot, digit in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                chain = [points[0]] + [points[4 * slot + 1 + j] for j in range(4)]
                key = f"{k}:{digit}"
                skeleton[key] = [(np.asarray(p, dtype=float) + origins[k]).tolist()
                                 for p in chain]
                colours[key] = colour
        sim.step()
        sim.draw_debug(skeleton=skeleton, colors=colours, extra=plates,
                       draw_joints=len(runs) <= 9, scene=scene)

    if png is not None:
        import mujoco
        renderer = mujoco.Renderer(sim.mj, height=720, width=1280, max_geom=20000)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = sim.camera_lookat
        offset = np.asarray(sim.camera_pos) - np.asarray(sim.camera_lookat)
        camera.distance = float(np.linalg.norm(offset))
        camera.azimuth = float(np.degrees(np.arctan2(-offset[1], -offset[0])))
        camera.elevation = -float(np.degrees(np.arcsin(offset[2] / camera.distance)))
        pose(frame)
        renderer.update_scene(sim.data, camera=camera)
        pose(frame, scene=renderer.scene)
        image = renderer.render()
        try:
            from PIL import Image
            Image.fromarray(image).save(png)
        except ImportError:
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-s", "1280x720", "-i", "-", png],
                           input=image.tobytes(), check=True)
        print(f"[gallery] wrote {png}")
        return 0

    started, step = time.perf_counter(), 0
    longest = max(end - start for start, end in runs)
    while sim.viewer_alive():
        pose(step)
        step += 1
        if step >= longest and not loop:
            break
        if speed > 0:
            due = started + step / (hz * speed)
            while time.perf_counter() < due and sim.viewer_alive():
                time.sleep(0.001)
    sim.close()
    return 0
