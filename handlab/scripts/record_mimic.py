"""Record a Mimic session in a window of its own, without the app.

    python scripts/record_mimic.py

Left, the camera with your tracked hand; right, the robot hand in the pose
to copy. Copy each pose with your hand (left is best) and hold it while the
bar is green - when the robot's fingertips touch, touch yours. `learning.
mimic_prompts` poses (3.5 s each). q or Esc stops and keeps what was
recorded; d discards it. Saved to data/learning/mimic/<stamp>.npz like a
session from the Vision panel (src/learning/mimic.py); then, with enough
poses, the camera model is fitted to every session so far and its held-out
fit printed (src/learning/camera_model.py).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src import settings  # noqa: E402
from src.learning import camera_model, mimic  # noqa: E402
from src.learning.handmotion import followed  # noqa: E402
from src.sim.task import hand_spec  # noqa: E402
from src.vision import Camera  # noqa: E402

SIZE = (640, 480)                       # each half of the window
WINDOW = "Mimic - copy the robot (q: stop and keep, d: discard)"
SAY = {mimic.MOVE: "watch the robot", mimic.READY: "copy it", mimic.HOLD: "HOLD"}
GREEN, AMBER, RED = (80, 220, 80), (60, 190, 245), (60, 60, 230)      # BGR


def main():
    count = settings.get("learning.mimic_prompts")
    print(f"{count} poses, {count * (mimic.MOVE_S + mimic.READY_S + mimic.HOLD_S):.0f} s")
    session = mimic.MimicSession(mimic.prompts(count))
    model = hand_spec()[0].compile()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, SIZE[1], SIZE[0])
    view = mujoco.MjvCamera()
    mujoco.mj_forward(model, data)
    view.lookat[:] = data.xpos[1:].mean(axis=0)
    view.distance, view.azimuth, view.elevation = 0.38, 200.0, -15.0

    camera = Camera()
    camera.reconfigure(dict(settings.section("vision"), enabled=True))
    keep = True
    try:
        while camera.latest() is None:                 # the first frame (or the error)
            if camera.error:
                raise SystemExit(f"camera: {camera.error}")
            print(f"\r{camera.status:60s}", end="", flush=True)
            time.sleep(0.1)
        print()
        start = time.monotonic()
        while True:
            now = time.monotonic() - start
            shown = session.target(now)
            if shown is None:
                break
            frame = camera.latest()
            if frame is None:                          # the camera stopped
                raise SystemExit(f"camera: {camera.error or 'stopped'}")
            session.add(frame, frame.time - start)
            q, phase, number = shown
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            renderer.update_scene(data, view)
            robot = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            preview = cv2.resize(cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR), SIZE)
            screen = np.hstack([preview, robot])
            _label(screen, session, now, phase, number, followed(frame.hands) is not None)
            cv2.imshow(WINDOW, screen)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                keep = False
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
        renderer.close()
    if not keep:
        print("discarded")
        return
    path = session.save(f"camera {settings.get('vision.camera_index')}")
    if path is None:
        print("nothing recorded")
        return
    held = [row for row in session.rows if row[5] == mimic.HOLD]
    seen = sum(row[3] for row in held) / max(1, len(held))
    print(f"saved {path}: {len(session.rows)} frames, your hand seen in {seen:.0%} of the holds")
    try:
        report = camera_model.fit()
    except RuntimeError as failure:
        print(f"camera model not fitted: {failure}")
        return
    print(f"camera model, {report['poses']} poses so far; on {report['held_out_poses']} held "
          f"out: pinches {report['pinch_error']:.3f} palm lengths off, against "
          f"{report['identity_pinch_error']:.3f} taking the camera's hand as the robot's")


def _label(screen, session, now, phase, number, seen):
    """The pose number, what to do, a bar through the session, and whether
    a hand is seen."""
    kind = str(session.kinds[number])
    colour = GREEN if phase == mimic.HOLD else AMBER
    text = (f"pose {number + 1}/{len(session.joints)}  {SAY[phase]} "
            f"{session.left(now):.1f}s  ({kind})")
    cv2.putText(screen, text, (SIZE[0] + 16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(screen, text, (SIZE[0] + 16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2,
                cv2.LINE_AA)
    done = int(screen.shape[1] * now / session.length_s)
    cv2.rectangle(screen, (0, screen.shape[0] - 10), (done, screen.shape[0]), colour, -1)
    if not seen:
        cv2.putText(screen, "no hand in view", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2,
                    cv2.LINE_AA)


if __name__ == "__main__":
    main()
