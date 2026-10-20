"""The window: MuJoCo Studio, run the way macOS needs it.

Studio (`mujoco.experimental.studio`) is MuJoCo's ImGui viewer. It is what
lets us add our own panels to the window. Its own `launch_passive` opens the
window on a background thread, which macOS refuses ("NSWindow should only be
instantiated on the main thread"), so here it is the other way round: the
viewer owns the main thread and the simulation loop runs on a worker thread.
Plain `python main.py` works; no mjpython.

This file launches and wires Studio. The other viewer/ files are Studio
plugins (its message handlers and Dear ImGui bindings). Studio is
experimental upstream: a MuJoCo upgrade that moves it touches viewer/ only.

NOT PUBLIC API: `run` builds Studio's sim<->viewer channels from
`launch_passive._PassiveSnapshotChannel` / `_PassiveEventChannel`, private
classes, because the public `launch_passive()` is the call macOS refuses.
mujoco is pinned (requirements.txt) so they cannot move under us; check them
first on any upgrade.
"""

from __future__ import annotations

import threading
import time
import traceback

import mujoco
from mujoco.experimental.studio import (endpoints, launch_passive, messages,
                                        sim, viewer_app, viewer_handle,
                                        viewer_protocol)

from .. import settings
from ..world import hand_center

# Palm-side three-quarter view of the mounted hand.
CAMERA = {"distance": 0.55, "azimuth": -160.0, "elevation": -20.0}
LOOP_SLEEP = 0.001       # s; the sim loop runs at ~1 kHz without spinning a core


def run(world, tick, plugins=(), title="handlab"):
    """Open the window on `world` and block until it is closed.

    `tick(dt)` runs on the sim thread before every physics advance, with dt
    the wall-clock seconds since the last call; `world.hold()` runs after it.
    If tick rebuilds the world (`world.build()`), the new model goes to the
    viewer before its first state does. `plugins` are Studio viewer plugins:
    panels (see `panel.py`) and anything else that runs per frame.
    """
    size = settings.section("viewer")
    config = viewer_protocol.ViewerConfig(title=title, width=size["width"],
                                          height=size["height"])
    viewer_end, sim_end = endpoints.make_endpoints(
        s2v_snapshot=launch_passive._PassiveSnapshotChannel(),
        s2v_events=launch_passive._PassiveEventChannel(),
        v2s_events=launch_passive._PassiveEventChannel(),
        v2s_snapshot=launch_passive._PassiveSnapshotChannel())

    closed = threading.Event()
    handle = viewer_handle.ViewerHandle(sim_end, plugins=[_KeepWorldSim()],
                                        is_alive_fn=lambda: not closed.is_set())
    loop = threading.Thread(target=_sim_loop, args=(handle, world, tick),
                            name="handlab-sim", daemon=True)
    loop.start()
    plugins = [viewer_app.ViewerApp(), _KeepWorldViewer(world),
               _FrameCap(size["max_fps"]), *plugins]
    try:
        launch_passive.run_viewer_target(config, viewer_end, plugins)
    finally:
        closed.set()
        loop.join(timeout=2.0)


def _sim_loop(handle, world, tick):
    shown = world.model
    handle.send_to_viewer(messages.ModelEvent(model=shown))
    step_control = sim.StepControl()
    last = time.monotonic()
    try:
        while handle.is_running():
            now = time.monotonic()
            tick(now - last)
            last = now
            if world.model is not shown:                 # rebuilt by tick
                shown = world.model
                handle.send_to_viewer(messages.ModelEvent(model=shown))
            step_control.advance(world.model, world.data)
            world.hold()
            handle.sync(world.model, world.data, step_control)
            time.sleep(LOOP_SLEEP)
    except Exception:
        # A dead sim thread under a live window looks like a frozen robot.
        # Say what happened and take the window down with it.
        traceback.print_exc()
        handle.close()


class _FrameCap:
    """Hold the window to `fps` frames a second at most.

    Studio's viewer loop draws the next frame as soon as the last one is
    presented - about 300 a second on this Mac - and has no vsync option,
    so the GPU never rests. This waits out the rest of each frame's time at
    the start of the next one (Studio's per-frame UpdateEvent): the input
    gathered for that frame is at most one frame old."""

    def __init__(self, fps):
        self.period = 1.0 / fps
        self._next = 0.0

    @messages.handler(priority=messages.Priority.CRITICAL + 1)
    def _on_update(self, _: messages.UpdateEvent) -> None:
        now = time.monotonic()
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(self._next, now) + self.period


class _KeepWorldSim:
    """Sim side: the viewer never swaps the world (Studio would, on a dropped
    file). The world rebuilds itself from settings; that is the only swap."""

    @messages.handler(priority=messages.Priority.CRITICAL + 1)
    def _on_model(self, _: messages.ModelEvent) -> bool:
        return True


class _KeepWorldViewer:
    """Viewer side: show our world, frame the camera on it, and refuse any
    other model - one world, however the simulator is opened."""

    def __init__(self, world):
        self.world = world
        self.viewer = None
        self.framed = False

    @messages.handler(priority=messages.Priority.CRITICAL + 1)
    def _on_init(self, event: viewer_protocol.ViewerInitEvent) -> None:
        self.viewer = event.viewer

    @messages.handler(priority=messages.Priority.CRITICAL + 1)
    def _guard(self, event: messages.ModelEvent) -> bool:
        if event.model is not self.world.model:
            print("[viewer] ignored a model swap - the world is fixed")
            return True
        return False

    @messages.handler(priority=messages.Priority.USER)
    def _frame(self, event: messages.ModelEvent) -> bool:
        """Frame the hand on the first model. After a rebuild, only follow
        the hand (a new orientation moves it) and keep the user's view."""
        camera = self.viewer.camera
        camera.lookat[:] = hand_center(self.viewer.model, self.viewer.data)
        if not self.framed:
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = CAMERA["distance"]
            camera.azimuth = CAMERA["azimuth"]
            camera.elevation = CAMERA["elevation"]
            self.framed = True
        return False
