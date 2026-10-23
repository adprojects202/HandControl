"""Headless checks of everything below the window. No hardware, no window.

    python scripts/selftest.py

Exits non-zero if any check fails.
"""

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src import keys, robot as robot_module, settings  # noqa: E402
from src.assistant import (Answer, Assistant, AssistantError, Question,  # noqa: E402
                           claude, context, jpeg, openai_api)
from src.control import JointController  # noqa: E402
from src.hardware import HardwareBus  # noqa: E402
from src.recording import Episode, Recorder  # noqa: E402
from src.vision import Camera, Frame, Hand, HandTracker, overlay, permission  # noqa: E402
from src.vision.camera import CameraError  # noqa: E402
from src.world import HANDS, World  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def settings_validate_and_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "settings.json"
        first = settings.Settings(path)                   # writes defaults
        assert path.exists() and first.values == settings.DEFAULTS
        stored = json.loads(path.read_text())
        stored["control"]["max_speed"] = "fast"            # wrong type
        stored["world"]["bogus"] = 1                       # unknown key
        stored["hardware"]["port"] = "/dev/cu.test"        # a real edit
        path.write_text(json.dumps(stored))
        loaded = settings.Settings(path)
        assert loaded.get("control.max_speed") == settings.DEFAULTS["control"]["max_speed"]
        assert "bogus" not in loaded.get("world")
        assert loaded.get("hardware.port") == "/dev/cu.test"
        assert "bogus" not in json.loads(path.read_text())["world"], "repair not saved"
        loaded.set("hardware.motors.index_distal_joint.sign", 1)
        try:
            loaded.set("world.timestep", "small")
            raise AssertionError("bad type accepted")
        except ValueError:
            pass
        loaded.save()
        again = settings.Settings(path)
        assert again.get("hardware.motors.index_distal_joint.sign") == 1


@check
def settings_ranges_and_choices():
    with tempfile.TemporaryDirectory() as tmp:
        s = settings.Settings(Path(tmp) / "settings.json")
        assert s.set("hardware.position_p", 99999) == 16383, "range not clamped"
        assert s.set("control.max_speed", 0) == 0.1
        for key, bad in (("hardware.baudrate", 12345),
                         ("hardware.motors.index_distal_joint.sign", 0),
                         ("hardware.port", 5)):
            try:
                s.set(key, bad)
                raise AssertionError(f"{key}={bad!r} accepted")
            except ValueError:
                pass
        # a motor-map key that is not in the defaults is still checked
        entry = s.set("hardware.motors.pinky_joint", {"id": 999, "sign": -1})
        assert entry == {"id": 252, "sign": -1, "zero": 2048}, entry
        assert settings.limits("hardware.motors.anything.zero") == (0, 4095)
        assert settings.choices("hardware.motors.anything.sign") == (1, -1)


@check
def motion_presets():
    with tempfile.TemporaryDirectory() as tmp:
        s = settings.Settings(Path(tmp) / "settings.json")
        assert settings.motion_preset(s) == "smooth", "defaults should be smooth"
        for name in s.get("motion_presets"):
            settings.apply_motion_preset(name, s)
            assert settings.motion_preset(s) == name
        assert s.get("hardware.profile_velocity") == 0       # real time
        s.set("hardware.position_p", 400)                     # PID is not motion
        assert settings.motion_preset(s) == "real time"
        s.set("control.max_speed", 3.3)                       # hand-tuned
        assert settings.motion_preset(s) is None              # "custom"


@check
def robot_from_urdf():
    robot = robot_module.load()
    assert len(robot) == 16, len(robot)
    assert list(robot.groups()) == ["index", "middle", "ring", "thumb"]
    assert np.all(robot.lower < robot.upper)
    missing = set(robot.names) - set(settings.get("hardware.motors"))
    assert not missing, f"no motor for {missing}"


@check
def world_is_built_the_same_and_holds_poses():
    """On default settings (not the user's live ones), stepped the way the
    sim loop steps it: physics, then `hold()`."""
    original = settings._current
    try:
        world = _world_with()
        again = World()
        assert world.model.nq == again.model.nq
        assert all(len(world.pose(hand)) == len(world.robot) for hand in HANDS)
        # Everything must sit in a geom group the viewer draws by default:
        # MuJoCo hides groups 3-5, and hiding/showing by group at runtime
        # crashes Studio 3.12, so nothing can switch a hidden group back on.
        default = mujoco.MjvOption().geomgroup
        assert all(default[group] for group in np.unique(world.model.geom_group)), \
            "a geom is in a group the viewer hides by default"
        pose = world.robot.clamp(np.full(len(world.robot), 0.7))
        world.set_pose("command", pose)
        world.set_pose("encoder", np.zeros(len(world.robot)))
        for _ in range(1000):                       # two simulated seconds
            mujoco.mj_step(world.model, world.data)
            world.hold()
        assert np.allclose(world.pose("command"), pose, atol=1e-6), "command hand drifted"
        assert np.allclose(world.pose("encoder"), 0.0, atol=1e-6), "encoder hand drifted"
    finally:
        settings._current = original


def _world_with(**changes):
    """A World built from a throwaway settings file with `changes` applied."""
    tmp = tempfile.mkdtemp()
    settings._current = settings.Settings(Path(tmp) / "settings.json")
    for key, value in changes.items():
        settings.set(key.replace("__", "."), value)
    return World()


def _touching(world):
    pairs = set()
    for c in world.data.contact[:world.data.ncon]:
        a, b = (world.model.body(world.model.geom_bodyid[g]).name.split("/")[0]
                for g in (c.geom1, c.geom2))
        pairs.add(tuple(sorted((a, b))))
    return pairs


@check
def orientations_turn_gravity_against_the_hand():
    from src.world import mount_quat
    # the palm faces the hand's +x: palm up must point it at the sky
    for name, palm_world in (("upright", (1, 0, 0)), ("palm up", (0, 0, 1)),
                             ("palm down", (0, 0, -1)), ("fingers down", (1, 0, 0))):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, mount_quat(name))
        assert np.allclose(rotation.reshape(3, 3) @ [1, 0, 0], palm_world, atol=1e-9), name
    finger = np.zeros(9)
    mujoco.mju_quat2Mat(finger, mount_quat("fingers down"))
    assert np.allclose(finger.reshape(3, 3) @ [0, 0, 1], [0, 0, -1], atol=1e-9)


@check
def fixed_object_stops_the_real_hand_not_the_ghost():
    original = settings._current
    try:
        world = _world_with(object__enabled=True, object__mode="fixed")
        curl = np.zeros(len(world.robot))
        for i, name in enumerate(world.robot.names):
            if not name.startswith("thumb") and "adduction" not in name:
                curl[i] = 1.2
        start = world.object_position()
        for _ in range(200):
            world.set_pose("encoder", curl)
            world.set_pose("command", curl)
            mujoco.mj_step(world.model, world.data)
            world.hold()
        assert _touching(world) == {("encoder", "object")}, _touching(world)
        assert np.allclose(world.object_position(), start), "a fixed object moved"
        assert np.allclose(world.pose("encoder"), world.robot.clamp(curl)), \
            "contact moved the replayed hand"
    finally:
        settings._current = original


@check
def floating_object_falls_and_rests():
    original = settings._current
    try:
        world = _world_with(object__enabled=True, object__mode="floating")
        for _ in range(1000):                               # 2 s
            mujoco.mj_step(world.model, world.data)
            world.hold()
        half = settings.get("object.size_mm")[2] / 2000.0
        assert abs(world.object_position()[2] - half) < 0.002, world.object_position()
        world = _world_with(object__enabled=True, object__mode="floating",
                            world__hand_orientation="palm up")
        start = world.object_position()
        for _ in range(1000):
            world.set_pose("encoder", np.zeros(len(world.robot)))
            mujoco.mj_step(world.model, world.data)
            world.hold()
        assert ("encoder", "object") in _touching(world), "should rest on the palm"
        assert start[2] - world.object_position()[2] < 0.03, "fell past the palm"
        # stiff contacts: it rests ON the palm, barely in it (0.89 mm at
        # MuJoCo's default stiffness, 0.08 mm at ours)
        overlap = -min(c.dist for c in world.data.contact[:world.data.ncon])
        assert overlap < 0.0002, f"rests {overlap * 1000:.2f} mm into the palm"
    finally:
        settings._current = original


@check
def world_edits_apply_in_place_except_size_and_show():
    """Only resizing or showing/hiding the object may build a new model:
    anything else would make the viewer reload (a black flash) on every edit."""
    original = settings._current
    try:
        world = _world_with(object__enabled=True)
        model = world.model
        for key, value in (("world.hand_orientation", "thumb up"),
                           ("object.mode", "floating"), ("object.mass_g", 120.0),
                           ("object.position_mm", [50.0, 10.0, 80.0])):
            settings.set(key, value)
            world.apply()
            assert world.model is model, f"{key} rebuilt the model"
        assert world.model.body_mass[world.model.body("object").id] == 0.12
        for key, value in (("object.size_mm", [40.0, 40.0, 40.0]),
                           ("object.enabled", False)):
            settings.set(key, value)
            world.apply()
            assert world.model is not model, f"{key} must rebuild"
            model = world.model
        assert world.object_position() is None
    finally:
        settings._current = original


@check
def object_friction_governs_its_contacts_live():
    """The object's friction - not MuJoCo's larger-of-the-two default - is
    what its contacts use, and a change applies without a rebuild."""
    original = settings._current
    try:
        world = _world_with(object__enabled=True, object__mode="fixed")
        model = world.model
        curl = np.zeros(len(world.robot))
        for i, name in enumerate(world.robot.names):
            if not name.startswith("thumb") and "adduction" not in name:
                curl[i] = 1.2

        def contact_friction():
            for _ in range(5):
                world.set_pose("encoder", curl)
                mujoco.mj_step(world.model, world.data)
                world.hold()
            return {round(float(c.friction[0]), 6) for c in world.data.contact[:world.data.ncon]}

        assert contact_friction() == {settings.DEFAULTS["object"]["friction"]}
        settings.set("object.friction", 0.2)
        world.apply()
        assert world.model is model, "friction rebuilt the model"
        assert contact_friction() == {0.2}
    finally:
        settings._current = original


@check
def encoder_hand_is_held_at_zero_without_a_reading():
    from src.app import App
    app = App()                          # the bus is never started: no hardware
    world, names = app.world, app.robot.names
    try:
        world.data.qpos[world._qadr["encoder"]] = 0.3        # knocked about
        world.data.qvel[world._vadr["encoder"]] = 2.0
        for _ in range(50):
            app.tick(world.model.opt.timestep)
            mujoco.mj_step(world.model, world.data)
            world.hold()
        assert np.allclose(world.pose("encoder"), 0.0), world.pose("encoder")
        reading = app.robot.clamp(np.full(len(names), 0.2))
        app.bus.encoders = reading                           # connected
        app.tick(0.002)
        assert np.allclose(world.pose("encoder"), reading)
        app.bus.encoders = None                              # disconnected
        app.tick(0.002)
        assert np.allclose(world.pose("encoder"), 0.0), "stale reading kept"
    finally:
        app.camera.close()


@check
def controller_slews_and_clamps():
    robot = robot_module.load()
    control = JointController(robot, max_speed=2.0)
    control.set_targets(np.full(len(robot), 10.0))              # far past limits
    assert np.allclose(control.targets(), robot.upper)
    step = control.step(0.01)
    assert np.allclose(np.abs(step), np.minimum(0.02, np.abs(robot.upper))), step
    control.step(10.0)                                           # capped dt
    assert np.max(np.abs(control.command() - step)) <= 2.0 * 0.05 + 1e-9
    control.prime(np.full(len(robot), 0.3))
    assert np.allclose(control.command(), control.targets())


@check
def bus_interlocks_with_mock_link():
    robot = robot_module.load()
    config = {**settings.section("hardware"), "port": "mock"}
    bus = HardwareBus(robot.names, config).start()
    try:
        _wait(lambda: bus.connected)
        bus.request("stream_on")                                # not armed: refused
        _wait(lambda: bus.error)
        assert not bus.streaming
        bus.request("arm")
        _wait(lambda: bus.armed)
        target = robot.clamp(np.full(len(robot), 0.5))
        bus.send(target)
        time.sleep(0.1)
        assert not np.allclose(bus.encoders, target), "wrote while not streaming"
        bus.request("stream_on")
        _wait(lambda: bus.encoders is not None and np.allclose(bus.encoders, target))
        bus.request("disarm")
        _wait(lambda: not bus.armed)
        assert not bus.streaming, "disarm left streaming on"
        # runtime settings: registers are written in place, a new motor map
        # reconnects, no port disconnects
        bus.reconfigure({**config, "position_p": 300})
        time.sleep(0.05)
        assert bus.connected and not bus.error, bus.error
        motors = {**config["motors"]}
        motors[robot.names[0]] = {**motors[robot.names[0]], "sign": -1}
        bus.reconfigure({**config, "motors": motors})
        _wait(lambda: bus.connected
              and bus.link.config["motors"][robot.names[0]]["sign"] == -1)
        bus.reconfigure({**config, "port": ""})
        _wait(lambda: not bus.connected)
    finally:
        bus.close()


@check
def thermal_guard_derates_then_trips():
    from src.hardware.thermal import ThermalGuard
    guard = ThermalGuard(["a", "b"], derate_c=45.0, cutoff_c=60.0)
    changed, trip = guard.update([30.0, 30.0])
    assert not changed and not trip and guard.status()["level"] == "ok"
    changed, trip = guard.update([52.5, np.nan])          # halfway: 50 % cap
    assert changed == [(0, 0.5)] and not trip, changed
    assert guard.temperature[1] == 30.0, "NaN must keep the last reading"
    changed, trip = guard.update([52.0, 30.0])             # same 10 % step: no rewrite
    assert not changed
    _, trip = guard.update([60.0, 30.0])
    assert trip and guard.tripped
    _, trip = guard.update([61.0, 30.0])
    assert not trip, "trips once, not every frame"
    guard.update([56.0, 30.0])
    assert guard.tripped, "must cool COOL_MARGIN below the cutoff"
    guard.update([54.0, 30.0])
    assert not guard.tripped


@check
def bus_thermal_protection_with_mock_link():
    robot = robot_module.load()
    config = {**settings.section("hardware"), "port": "mock"}
    bus = HardwareBus(robot.names, config).start()
    try:
        _wait(lambda: bus.connected)
        bus.request("arm")
        _wait(lambda: bus.armed)
        bus.send(np.zeros(len(robot)))
        bus.request("stream_on")
        _wait(lambda: bus.streaming)
        link, limit = bus.link, config["current_limit_ma"]
        link.temperature[3] = 52.5                              # derate one joint
        _wait(lambda: link.goal_current.get(robot.names[3]) == int(0.5 * limit))
        assert bus.armed and bus.streaming, "derating must not stop anything"
        link.temperature[3] = 61.0                              # cutoff
        _wait(lambda: not bus.armed)
        assert not bus.streaming and "torque off" in bus.error, bus.error
        bus.request("arm")                                      # still hot
        _wait(lambda: "too hot" in bus.error)
        assert not bus.armed
        link.temperature[3] = 40.0                              # cooled down
        _wait(lambda: not bus.thermal.tripped)
        bus.request("arm")
        _wait(lambda: bus.armed)
        assert bus.telemetry is not None and "temperature" in bus.telemetry
    finally:
        bus.close()


@check
def hand_tracker_runs_and_overlay_draws():
    tracker = HandTracker(settings.DEFAULTS["vision"])
    blank = np.zeros((240, 320, 3), np.uint8)
    try:
        assert tracker.track(blank, 5) == []
        assert tracker.track(blank, 5) == [], "a repeated timestamp must not fail"
    finally:
        tracker.close()
    line = np.linspace(0.2, 0.8, 21)
    hand = Hand("Right", 0.9, image=np.column_stack([line, line, 0 * line]),
                world=np.zeros((21, 3)))
    drawn = overlay.draw(blank, [hand], fps=30.0)
    assert drawn.any() and not blank.any(), "draws on a copy"
    left = np.linspace(0.05, 0.25, 21)                  # a hand on the image's left
    hand = Hand("Left", 0.9, image=np.column_stack([left, 0 * left + 0.5, 0 * left]),
                world=np.zeros((21, 3)))
    plain, mirrored = (overlay.draw(blank, [hand], mirror=m) for m in (False, True))
    assert plain[:, :160].any() and not plain[:, 200:].any()
    assert mirrored[:, 160:].any() and not mirrored[:, :120].any(), "mirror moves it"


class _FakeCamera:
    """Frames whose right half is white; `ok=False` sends nothing."""

    def __init__(self, ok=True):
        self.ok = ok
        self.released = False

    def read(self):
        time.sleep(0.01)
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:, 80:] = 255
        return self.ok, frame if self.ok else None

    def release(self):
        self.released = True


@check
def camera_thread_tracks_mirrors_reopens_and_stops():
    sources = []

    def open_source(config):
        sources.append(_FakeCamera())
        return sources[-1]

    config = dict(settings.DEFAULTS["vision"], enabled=True, mirror=False)
    camera = Camera(open_source)
    try:
        camera.reconfigure(config)
        _wait(lambda: camera.latest() is not None and camera.latest().number >= 3)
        frame = camera.latest()
        assert frame.raw[:, -1].all() and not frame.raw[:, 0].any()
        assert frame.hands == [] and camera.fps > 0 and "fps" in camera.status
        assert frame.image[:, -1].all(), "unmirrored preview"
        camera.reconfigure(dict(config, mirror=True))            # live: no reopen
        _wait(lambda: not camera.latest().image[:, -1].any())   # the preview flips
        assert camera.latest().raw[:, -1].all(), "the true image never does"
        assert len(sources) == 1
        camera.reconfigure(dict(config, camera_index=1))         # reopens
        _wait(lambda: len(sources) == 2 and camera.latest() is not None)
        assert sources[0].released
        camera.reconfigure(dict(config, enabled=False))
        assert not camera.running and camera.latest() is None and sources[1].released
        assert camera.status == "off"
    finally:
        camera.close()


@check
def camera_failures_are_reported_not_raised():
    def refuses(config):
        raise CameraError("camera 0 did not open")

    config = dict(settings.DEFAULTS["vision"], enabled=True)
    for camera, said in ((Camera(refuses), "did not open"),
                         (Camera(lambda config: _FakeCamera(ok=False)), "stopped sending")):
        try:
            camera.reconfigure(config)
            _wait(lambda: not camera.running, timeout=5.0)    # loads the tracker first
            assert said in camera.error and camera.latest() is None, camera.error
            camera.reconfigure(dict(config, enabled=False))
            assert camera.error == "" and camera.status == "off"
        finally:
            camera.close()


@check
def camera_waits_for_permission_and_reports_a_refusal():
    config = dict(settings.DEFAULTS["vision"], enabled=True)
    answers = iter([permission.WAITING] * 3)
    asked = permission.camera_access
    camera = Camera(lambda config: _FakeCamera())
    camera.needs_permission = True                  # as the real camera does
    try:
        permission.camera_access = lambda: next(answers, permission.GRANTED)
        camera.reconfigure(config)
        _wait(lambda: "waiting for camera permission" in camera.status)
        _wait(lambda: camera.latest() is not None)
        camera.stop()
        permission.camera_access = lambda: permission.REFUSED
        camera.reconfigure(config)
        _wait(lambda: not camera.running)
        assert "Privacy & Security > Camera" in camera.error, camera.error
    finally:
        permission.camera_access = asked
        camera.close()


@check
def camera_names_follow_the_system_list():
    from src.vision.devices import camera_names
    names = camera_names()
    if sys.platform == "darwin":
        assert isinstance(names, list) and all(isinstance(n, str) for n in names), names
    else:
        assert names is None


CONTEXT = {"task": "turn the cube", "goal": "cube rests on a new face",
           "objects": [{"name": "orange cube", "role": "moved", "image": 0,
                        "box": {"left": 0.4, "top": 0.4, "right": 0.6, "bottom": 0.6}}],
           "hands": [{"side": "right", "does": "turns the cube"}],
           "phases": [{"name": "turn", "first_image": 0, "last_image": 1,
                       "what_happens": "the cube rolls over"}],
           "uncertain": ""}


class _FakeProvider:
    NAME, KEY_ENV = "Fake", "FAKE_API_KEY"
    asked = []

    @staticmethod
    def ask(question, config, key):
        _FakeProvider.asked.append((question, key))
        time.sleep(0.05)
        if question.images == (b"fail",):
            raise AssistantError("the fake API said no")
        text = "not json" if question.images == (b"junk",) else json.dumps(CONTEXT)
        return Answer(text=text, model="fake-1", input_tokens=10, output_tokens=2)


@check
def assistant_describes_in_the_background_with_keys_never_saved():
    assert not any("key" in name for name in settings.DEFAULTS["assistant"])
    environ = {}
    assistant = Assistant(providers={"claude": _FakeProvider}, secrets=keys.Secrets(
        environ, Path(tempfile.gettempdir()) / "handlab-no-such.env"))
    config = dict(settings.DEFAULTS["assistant"])
    assert not assistant.describe([b"jpeg"], config=config)
    assert "FAKE_API_KEY" in assistant.error and assistant.key_source("claude") == ""
    environ["FAKE_API_KEY"] = "env-key"
    assert assistant.key_source("claude") == "$FAKE_API_KEY"
    assistant.set_session_key("claude", "  session-key ")
    assert assistant.key_source("claude") == "this session"
    assert assistant.describe([b"jpeg"], config=config, label="live") and assistant.busy
    assert not assistant.describe([b"again"], config=config), "one question at a time"
    _wait(lambda: not assistant.busy)
    answer = assistant.answer
    assert answer.data == CONTEXT and answer.seconds > 0 and assistant.label == "live"
    question, key = _FakeProvider.asked[-1]
    assert key == "session-key" and question.images == (b"jpeg",)
    assert question.schema is context.SCHEMA and "one frame" in question.text
    done = []                                   # a clip, loaded on the thread
    assert assistant.describe(lambda: [b"a", b"b"], [0.0, 1.5], config=config,
                              on_done=lambda *args: done.append(args))
    _wait(lambda: not assistant.busy)
    question = _FakeProvider.asked[-1][0]
    assert question.images == (b"a", b"b") and "image 1 at 1.50 s" in question.text
    assert len(done) == 1 and done[0][0].data == CONTEXT
    assistant.set_session_key("claude", "")
    assert assistant.key_source("claude") == "$FAKE_API_KEY"
    assert assistant.describe([b"junk"], config=config, on_done=done.append)
    _wait(lambda: not assistant.busy)
    assert assistant.answer.data is None and assistant.answer.note and len(done) == 1
    assert assistant.describe([b"fail"], config=config)
    _wait(lambda: not assistant.busy)
    assert assistant.answer is None and assistant.error == "the fake API said no"


@check
def assistant_reads_keys_from_a_dotenv_file_without_showing_them():
    with tempfile.TemporaryDirectory() as tmp:
        key_file = Path(tmp) / ".env"
        secrets = keys.Secrets({}, key_file)
        assistant = Assistant(providers={"claude": _FakeProvider}, secrets=secrets)
        assert assistant.key_source("claude") == "" and not secrets.problem
        key_file.write_text("sk-bare-secret\n")                   # no NAME=
        assert assistant.key_source("claude") == ""
        problem = secrets.problem
        assert "NAME=" in problem and "sk-bare-secret" not in problem, problem
        key_file.write_text("# keys\nFAKE_API_KEY=file-key\n")
        assert assistant.key_source("claude") == ".env" and not secrets.problem
        assert keys.Secrets({}, key_file).find("FAKE_API_KEY") == ("file-key", ".env")
        secrets.environ["FAKE_API_KEY"] = "env-key"             # environment wins
        assert assistant.key_source("claude") == "$FAKE_API_KEY"


@check
def provider_requests_carry_frames_and_settings():
    config = dict(settings.DEFAULTS["assistant"])
    image = jpeg(np.full((900, 1600, 3), 128, np.uint8), config["image_max_side"])
    shape = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape
    assert shape[:2] == (576, 1024), shape
    question = Question("what is the hand doing?", (image, image), context.SCHEMA)
    sent = claude.request(question, config)
    content = sent["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image", "image", "text"]
    assert base64.b64decode(content[0]["source"]["data"]) == image
    assert sent["model"] == config["claude_model"] and sent["system"] == config["system_prompt"]
    assert sent["fallbacks"] == "default" and sent["betas"] == [claude.FALLBACK_BETA]
    assert sent["output_config"] == {"effort": config["claude_effort"],
                                     "format": {"type": "json_schema",
                                                "schema": context.SCHEMA}}
    assert "format" not in claude.request(Question("hi"), config)["output_config"]
    sent = openai_api.request(question, config)
    content = sent["input"][0]["content"]
    assert [part["type"] for part in content] == ["input_image", "input_image", "input_text"]
    assert content[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert sent["model"] == config["openai_model"]
    assert sent["max_output_tokens"] == config["max_tokens"]
    assert sent["text"]["format"]["schema"] is context.SCHEMA
    assert sent["text"]["format"]["strict"] is True
    assert "text" not in openai_api.request(Question("hi"), config)


@check
def context_schema_keeps_to_what_both_apis_accept():
    def visit(node, path="$"):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path
            assert set(node["required"]) == set(node["properties"]), path
            for name, child in node["properties"].items():
                visit(child, f"{path}.{name}")
        if node.get("type") == "array":
            visit(node["items"], f"{path}[]")
        for banned in ("minimum", "maximum", "minLength", "maxLength", "maxItems"):
            assert banned not in node, f"{path}: {banned}"
    visit(context.SCHEMA)
    assert context.parse(json.dumps(CONTEXT)) == CONTEXT
    assert context.parse("not json") is None and context.parse('{"task": "x"}') is None


def _frame(number, time_s, hands=()):
    rgb = np.zeros((120, 160, 3), np.uint8)
    rgb[:, :40] = (200, 30, 30)                          # red on the true left
    rgb[:, 40:] = min(255, 20 + 4 * number)
    return Frame(raw=rgb, image=rgb, hands=list(hands), time=time_s, number=number)


@check
def recording_round_trip():
    stored = settings.get("recording.folder")
    line = np.linspace(0.2, 0.8, 21)
    right = Hand("Right", 0.8, np.column_stack([line, line, 0 * line]), np.ones((21, 3)))
    worse = Hand("Right", 0.3, np.zeros((21, 3)), np.zeros((21, 3)))
    names = ["a_joint", "b_joint"]
    encoders = [None, np.array([0.1, 0.2])]
    with tempfile.TemporaryDirectory() as tmp:
        settings.set("recording.folder", tmp)              # in memory only, not saved
        try:
            recorder = Recorder(names, lambda: (np.zeros(2), encoders[0]))
            recorder.add(_frame(0, 0.0))                     # not recording: ignored
            folder = recorder.start("turn the cube!", "Fake Cam", settings.DEFAULTS["vision"])
            assert folder.parent.name == "turn_the_cube" and (folder / "meta.json").exists()
            assert not Episode(folder).complete, "unfinished until stopped"
            for n in range(12):
                if n == 6:
                    encoders[0] = encoders[1]
                recorder.add(_frame(n, 10.0 + n / 30, [right, worse] if n % 2 else []))
            assert recorder.stop() == folder and not recorder.active
            episode = Episode(folder)
            assert episode.complete and episode.frames == 12
            assert episode.meta["camera"]["name"] == "Fake Cam"
            assert episode.meta["camera"]["mirrored"] is False
            assert np.allclose(episode.times, np.arange(12) / 30)
            hands = episode.hands
            assert hands["image"].shape == (12, 2, 21, 3)
            assert not hands["present"][0].any() and hands["present"][1].tolist() == [False, True]
            assert np.isclose(hands["score"][1, 1], 0.8), "the surer duplicate is kept"
            assert episode.hands_at(1)[0].side == "Right" and episode.hands_at(0) == []
            assert np.isnan(episode.robot["encoders"][0]).all()
            assert np.allclose(episode.robot["encoders"][6], [0.1, 0.2])
            first = episode.image(0)
            assert first.shape == (120, 160, 3)
            assert first[:, :30, 0].mean() > 150 > first[:, 60:, 0].mean(), "not mirrored"
            assert episode.sample(4) == [0, 4, 7, 11] and episode.sample(99) == list(range(12))
            assert len(episode.read_frames([0, 5, 11])) == 3
            episode.save_context({"context": CONTEXT})
            assert Episode(folder).context == {"context": CONTEXT}
            episode.close()
        finally:
            settings.set("recording.folder", stored)


@check
def recording_a_back_camera_beside_the_front():
    stored = settings.get("recording.folder")
    line = np.linspace(0.2, 0.8, 21)
    left = Hand("Left", 0.9, np.column_stack([line, line, 0 * line]), np.ones((21, 3)))
    with tempfile.TemporaryDirectory() as tmp:
        settings.set("recording.folder", tmp)              # in memory only, not saved
        try:
            recorder = Recorder(["a_joint"], lambda: (np.zeros(1), None))
            vision = dict(settings.DEFAULTS["vision"], back_camera_index=1)
            folder = recorder.start("grip", "Front Cam", vision, "Back Cam")
            recorder.add(_frame(0, 9.99), "back")                # before the front's first: dropped
            for n in range(10):
                recorder.add(_frame(n, 10.0 + n / 30))
                recorder.add(_frame(n, 10.0 + n / 30 + 0.012, [left] if n > 4 else []), "back")
            recorder.stop()
            episode = Episode(folder)
            assert episode.has_back and episode.meta["back_camera"]["name"] == "Back Cam"
            assert episode.meta["back_camera"]["frames"] == 10
            assert np.allclose(episode.back_times, np.arange(10) / 30 + 0.012), "the same clock"
            assert episode.back_index(7) == 7 and episode.back_image(7).shape == (120, 160, 3)
            assert episode.back_hands_at(6)[0].side == "Left" and episode.back_hands_at(2) == []
            assert episode.frames == 10 and not episode.hands["present"].any(), "streams apart"
            episode.close()
            # Without a back camera nothing changes.
            folder = recorder.start("grip alone", "Front Cam", settings.DEFAULTS["vision"])
            recorder.add(_frame(0, 20.0))
            recorder.add(_frame(0, 20.01), "back")               # no back camera: ignored
            recorder.stop()
            assert not Episode(folder).has_back and "back_camera" not in Episode(folder).meta
        finally:
            settings.set("recording.folder", stored)


@check
def overlay_draws_masks_and_boxes_mirrored_with_the_frame():
    rgb = np.zeros((120, 160, 3), np.uint8)
    mask = np.zeros((120, 160), bool)
    mask[40:80, 10:50] = True                                # on the true left
    box = ((0.05, 0.3, 0.35, 0.7), (255, 0, 0), "box")
    plain = overlay.draw(rgb, [], masks=[(mask, (0, 255, 0), "m")], boxes=[box])
    mirrored = overlay.draw(rgb, [], masks=[(mask, (0, 255, 0), "m")], boxes=[box],
                            mirror=True)
    assert plain[60, 30, 1] > 0 and plain[60, 130].sum() == 0
    assert mirrored[60, 129, 1] > 0 and mirrored[60, 30].sum() == 0, "mask mirrored"


@check
def segment_step_prompts_from_the_context_and_saves_masks():
    from src.processing import Job
    from src.processing.segment import SegmentError, prompts, segment
    stored = settings.get("recording.folder")
    with tempfile.TemporaryDirectory() as tmp:
        settings.set("recording.folder", tmp)              # in memory only, not saved
        try:
            recorder = Recorder(["j"], lambda: (np.zeros(1), None))
            folder = recorder.start("seg", "Fake Cam", settings.DEFAULTS["vision"])
            for n in range(6):
                recorder.add(_frame(n, n / 30))
            recorder.stop()
            episode = Episode(folder)
            try:
                prompts(episode)
                raise AssertionError("segmented with no context")
            except SegmentError:
                pass
            objects = [dict(CONTEXT["objects"][0], image=1)]
            episode.save_context({"context": dict(CONTEXT, objects=objects),
                                  "frames": [{"index": 0}, {"index": 4}]})
            (asked,) = prompts(episode)
            assert asked.frame == 4 and np.allclose(asked.box, (64, 48, 96, 72))
            seen = []

            def fake(model, frames, asked, progress):        # a model's output
                seen.append((model, len(frames)))
                packed = np.zeros((len(asked), len(frames), 120, 20), np.uint8)
                packed[0, :, 48:72, 8:12] = 0xFF             # x 64..95
                return packed, ["note"]

            job, finished = Job(), []
            job.start("seg", lambda p: segment(episode, p, model="m", track=fake),
                      finished.append)
            _wait(lambda: not job.busy)
            assert not job.error and finished, job.error
            episode = Episode(folder)
            assert seen == [("m", 6)] and str(episode.masks["model"]) == "m"
            mask = episode.mask(0, 3)
            assert mask.shape == (120, 160) and mask[60, 64] and not mask[60, 63]
            assert np.isclose(episode.masks["area"][0, 0], 24 * 32 / (120 * 160))
            assert episode.masks["names"].tolist() == [CONTEXT["objects"][0]["name"]]
            from src.processing import segment as segment_module
            assert abs(segment_module.sam3_memory_gb(252) - 8.8) < 0.5, "measured: 8.8 GB"
            machine = segment_module._memory_gb
            segment_module._memory_gb = lambda: 1.0          # a machine too small for SAM 3
            try:
                segment(episode, model=segment_module.SAM3)
                raise AssertionError("SAM 3 started without the memory for it")
            except SegmentError as refusal:
                assert "SAM 2.1" in str(refusal)
            finally:
                segment_module._memory_gb = machine
            cancelled = Job()                                # cancel stops at the next report
            cancelled.start("slow", lambda p: [p(i, 99) or time.sleep(0.01)
                                               for i in range(99)])
            cancelled.cancel()
            _wait(lambda: not cancelled.busy)
            assert cancelled.text == "cancelled" and not cancelled.error
        finally:
            settings.set("recording.folder", stored)


def _bare_episode(folder, frames):
    """An episode folder with no video: meta and frame times only."""
    folder.mkdir(parents=True)
    from src.recording.episode import FORMAT
    (folder / "meta.json").write_text(json.dumps(
        {"format": FORMAT, "complete": True, "camera": {"width": 160, "height": 120}}))
    np.savez(folder / "frames.npz", time=np.arange(frames) / 30.0)


def _hand_at(x, y, scale=0.1):
    """21 landmarks of a flat hand, wrist at (x, y), fingers up."""
    points = np.zeros((21, 3), np.float32)
    for finger in range(5):
        for joint in range(4):
            points[1 + finger * 4 + joint] = (x + (finger - 2) * scale * 0.25,
                                              y - scale * (0.4 + 0.2 * joint), 0)
    points[0] = (x, y, 0)
    return points


@check
def format_1_recordings_have_their_hands_swapped_back():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "task" / "20261006-000000"
        _bare_episode(folder, 2)
        meta = json.loads((folder / "meta.json").read_text())
        (folder / "meta.json").write_text(json.dumps(dict(meta, format=1)))
        present = np.array([[True, False], [True, False]])   # stored in the left slot
        np.savez(folder / "hands.npz", present=present, score=np.ones((2, 2)),
                 image=np.zeros((2, 2, 21, 3)), world=np.zeros((2, 2, 21, 3)))
        assert [h.side for h in Episode(folder).hands_at(0)] == ["Right"]


@check
def hands_step_follows_one_hand_through_flips_phantoms_and_gaps():
    from src.processing.hands import clean_hands
    count = 120
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "task" / "20260101-000000"
        _bare_episode(folder, count)
        shape = (count, 2, 21, 3)
        raw = {"image": np.full(shape, np.nan, np.float32),
               "world": np.zeros(shape, np.float32),
               "score": np.full((count, 2), np.nan, np.float32),
               "present": np.zeros((count, 2), bool)}
        for i in range(count):
            if 40 <= i < 70 or i in (100, 101):                # hidden; a short gap
                continue
            slot = 0 if 80 <= i < 95 else 1                    # labelled left for 15
            raw["image"][i, slot] = _hand_at(0.6 + 0.001 * i, 0.7)
            raw["score"][i, slot] = 0.95
            raw["present"][i, slot] = True
            if i >= 105:                              # a phantom on its fingers
                raw["image"][i, 0] = _hand_at(0.6 + 0.001 * i, 0.66, scale=0.05)
                raw["score"][i, 0] = 0.99
                raw["present"][i, 0] = True
        np.savez(folder / "hands.npz", **raw)
        episode = Episode(folder)
        summary = clean_hands(episode)
        clean = Episode(folder).clean_hands
        assert summary["tracks"] == 1, summary
        assert clean["track_side"].tolist() == ["Right"], clean["track_side"]
        assert summary["relabelled"] == 15 and summary["phantoms"] == 15, summary
        assert not clean["present"][:, 0].any(), "no left hand"
        assert clean["present"][90, 1] and not clean["present"][50, 1], "flip kept, gap left"
        assert clean["filled"][100, 1] and clean["filled"].sum() == 2, "short gap filled"
        assert np.allclose(clean["image"][110, 1, 0, :2], (0.71, 0.7), atol=1e-3), \
            "the real hand kept, not the surer phantom"


@check
def object_step_measures_angle_contact_and_motion_in_hand():
    from src.processing.objects import measure_objects
    count, width, height = 60, 160, 120
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "task" / "20260101-000000"
        _bare_episode(folder, count)
        hand = np.full((count, 2, 21, 3), np.nan, np.float32)
        present = np.zeros((count, 2), bool)
        for i in range(count):
            hand[i, 1] = _hand_at(0.5, 0.75, scale=0.2)
            present[i, 1] = True
        np.savez(folder / "hands.npz", image=hand, world=np.zeros_like(hand),
                 score=np.ones((count, 2)), present=present)
        packed = np.zeros((1, count, height, (width + 7) // 8), np.uint8)
        for i in range(count):                       # a bar turning 1 degree a frame
            mask = np.zeros((height, width), np.uint8)
            box = ((80, 50), (60, 14), float(i))
            cv2.fillPoly(mask, [cv2.boxPoints(box).astype(np.int32)], 1)
            packed[0, i] = np.packbits(mask.astype(bool), axis=-1)
        episode = Episode(folder)
        episode.save_arrays("masks.npz", {
            "masks": packed, "width": width, "height": height,
            "names": np.array(["bar"]), "area": np.ones((1, count))})
        held = measure_objects(episode)
        motion = Episode(folder).arrays("object.npz")
        assert held == {"bar": 1.0} and motion["hand"].tolist() == [1], held
        turned = np.degrees(motion["angle"][0, -1] - motion["angle"][0, 0])
        assert abs(turned - (count - 1)) < 3, turned          # boxPoints: clockwise +
        turn = np.degrees(motion["turn"][0, -1])               # turn: counter-clockwise +
        assert abs(turn + (count - 1)) < 3, turn
        heading = np.degrees(motion["heading"][0])
        assert np.allclose(heading, np.degrees(motion["angle"][0]), atol=3), \
            "heading follows the bar"
        assert np.allclose(motion["centroid"][0, 30], (80, 50), atol=1.0)
        assert abs(motion["area"][0, 0] - 60 * 14 / (width * height)) < 0.01
        assert np.isfinite(motion["in_hand"][0]).all()
        square = np.zeros((1, count, height, (width + 7) // 8), np.uint8)
        for i in range(count):                  # a square turning half a degree a frame
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [cv2.boxPoints(((80, 60), (40, 40), 0.5 * i)).astype(np.int32)], 1)
            square[0, i] = np.packbits(mask.astype(bool), axis=-1)
        episode.save_arrays("masks.npz", {
            "masks": square, "width": width, "height": height,
            "names": np.array(["cube"]), "area": np.ones((1, count))})
        measure_objects(episode)
        turn = np.degrees(Episode(folder).arrays("object.npz")["turn"][0, -1])
        assert abs(turn + 0.5 * (count - 1)) < 3, turn


def _board_views(count=10, focal=520.0, size=(640, 480), stretch=1.0):
    """Pictures of a 9 x 6 checkerboard seen by a known camera. Its squares
    are 40 mm wide and 40 * `stretch` mm high."""
    width, height = size
    matrix = np.array([[focal, 0, 330], [0, focal, 236], [0, 0, 1]])
    square = 40
    board = np.full((9 * square, 12 * square), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                board[(r + 1) * square:(r + 2) * square, (c + 1) * square:(c + 2) * square] = 0
    board = cv2.resize(board, (board.shape[1], round(board.shape[0] * stretch)),
                       interpolation=cv2.INTER_NEAREST)
    rng = np.random.default_rng(0)
    scale = np.array([[0.001, 0, -board.shape[1] * 0.0005],
                      [0, 0.001, -board.shape[0] * 0.0005], [0, 0, 1]])
    pictures = []
    for _ in range(count):
        rvec = rng.uniform(-0.5, 0.5, 3)
        rotation, _ = cv2.Rodrigues(rvec)
        move = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04),
                         rng.uniform(0.4, 0.6)])
        homography = matrix @ np.column_stack([rotation[:, 0], rotation[:, 1], move]) @ scale
        gray = cv2.warpPerspective(board, homography, size, borderValue=200)
        pictures.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    return pictures


@check
def camera_calibration_from_checkerboard_pictures():
    from src.vision import calibration
    pictures = _board_views()
    corners = [c for c in (calibration.find_corners(p, (9, 6)) for p in pictures)
               if c is not None]
    assert len(corners) >= calibration.MIN_VIEWS, len(corners)
    result = calibration.calibrate(corners, (9, 6), (640, 480))
    assert abs(result.fx - 520) < 2 and abs(result.cy - 236) < 2, result
    assert result.rms_px < 0.3 and result.views == len(corners), result
    stretched = [c for c in (calibration.find_corners(p, (9, 6))
                             for p in _board_views(count=12, stretch=1.3)) if c is not None]
    told = calibration.calibrate(stretched, (9, 6), (640, 480), (40, 52))
    assert abs(told.fx - 520) < 5 and abs(told.fy - 520) < 5, told   # squares given
    half = result.scaled(320, 240)
    assert abs(half.fx - result.fx / 2) < 1e-6 and result.scaled(640, 360) is None
    session = calibration.CalibrationSession((9, 6))
    try:
        session.offer(pictures[0])
        _wait(lambda: session.latest[1] is not None)
        assert session.take() == "" and "same view" in session.take()
    finally:
        session.close()
    stored = settings.get("camera_calibration")
    folder_was = settings.get("recording.folder")
    with tempfile.TemporaryDirectory() as tmp:
        saving = settings.save
        settings.save = lambda: None                         # never touch settings.json
        try:
            calibration.save("Test Cam", result)
            assert calibration.for_camera("Test Cam", 320, 240).fx == half.fx
            assert calibration.for_camera("Other Cam", 640, 480) is None
            settings.set("recording.folder", tmp)
            recorder = Recorder(["j"], lambda: (np.zeros(1), None))
            folder = recorder.start("cal", "Test Cam", settings.DEFAULTS["vision"])
            recorder.add(Frame(raw=np.zeros((480, 640, 3), np.uint8), image=None, hands=[],
                               time=0.0, number=1))
            recorder.stop()
            assert Episode(folder).calibration.fx == result.fx, "recorded with the episode"
        finally:
            settings.set("camera_calibration", stored)
            settings.set("recording.folder", folder_was)
            settings.save = saving


@check
def box_pose_is_recovered_from_its_outline():
    from scipy.spatial.transform import Rotation
    from src.processing import pose3d
    matrix = np.array([[520.0, 0, 320], [0, 520.0, 240], [0, 0, 1]])
    camera, size = (matrix, np.zeros(5)), np.array([0.055] * 3)
    from src.sim.symmetry import box_symmetries
    corners, symmetries = pose3d._box_corners(size), box_symmetries(size * 1000)
    assert len(symmetries) == 24 and len(box_symmetries([55, 55, 30])) == 8
    assert len(box_symmetries([40, 55, 30])) == 4
    axis = np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2])
    start = Rotation.from_euler("xyz", [20, 30, 10], degrees=True)
    pose, worst_position, worst_turn = None, 0.0, 0.0
    for frame in range(30):
        truth = Rotation.from_rotvec(axis * np.radians(1.5 * frame)) * start
        place = np.array([0.01 + 0.001 * frame, -0.02, 0.30])
        points = corners @ truth.as_matrix().T + place
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, None)
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pixels[:, 0]).astype(np.int32)), 1)
        pose = pose3d._fit_box(mask.astype(bool), corners, camera, pose, 0.055, symmetries)
        rotation, position, overlap = pose
        worst_position = max(worst_position, np.linalg.norm(position - place))
        worst_turn = max(worst_turn, pose3d._turn_between(truth.as_matrix(), rotation,
                                                          symmetries))
        assert overlap > 0.95, (frame, overlap)
    assert worst_position < 0.015, worst_position          # m: depth, up to ~4% at 0.3 m
    # degrees, up to symmetry; near square-on an outline hides up to ~10 deg of tilt
    assert worst_turn < 8.0, worst_turn


@check
def palm_axes_point_out_of_the_palm():
    from src.processing import pose3d
    # a left hand facing the camera, fingers up: thumb and index on the image's left
    points = np.zeros((21, 3))
    points[0] = (0, 0.05, 0.3)
    for knuckle, x in zip((5, 9, 13, 17), (-0.03, -0.01, 0.01, 0.03)):
        points[knuckle] = (x, -0.03, 0.3)
    axes = pose3d._palm_axes(points, "Left")
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-9) and np.linalg.det(axes) > 0
    assert axes[1, 0] < -0.9, "x runs up the fingers (image y is down)"
    assert axes[2, 2] < -0.9, "z points out of the palm, at the camera"


@check
def transfer_lays_the_palm_on_the_robot_hand():
    from src.processing.transfer import TO_MOUNT, to_mount
    assert np.allclose(TO_MOUNT @ TO_MOUNT.T, np.eye(3)) and np.linalg.det(TO_MOUNT) > 0
    position, rotation = to_mount(np.array([[0.0, 0.0, 0.03], [0.05, 0.0, 0.0]]),
                                  np.stack([np.eye(3)] * 2))
    assert np.allclose(position[0], [0.03, 0, 0]), "out of the palm -> mount x"
    assert np.allclose(position[1], [0, 0, 0.05]), "up the fingers -> mount z"
    world = _world_with()
    from src.world import _compose
    world.set_demo((np.array([0.03, 0.0, 0.07]), np.array([1.0, 0, 0, 0])))
    placed, _ = _compose(world.mount, np.array([0.03, 0.0, 0.07]), np.array([1.0, 0, 0, 0]))
    assert np.allclose(world.data.mocap_pos[world._demo], placed)
    world.set_demo(None)
    assert world.data.mocap_pos[world._demo][2] < -1, "parked out of sight"


SYNTH_K = np.array([[326.0, 0, 320], [0, 326.0, 240], [0, 0, 1]])


def _marker_texture(marker=3, side=400, border=75):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.copyMakeBorder(cv2.aruco.generateImageMarker(dictionary, marker, side),
                              border, border, border, border, cv2.BORDER_CONSTANT, value=255)


def _cube_with_marker(rotation, centre, edge=0.055, marker=0.040):
    """A 640 x 480 view of a light cube whose +z face carries a marker of side
    `marker`, and its mask."""
    from src.processing import pose3d
    image = np.full((480, 640), 60, np.uint8)
    corners = pose3d._box_corners([edge] * 3) @ rotation.T + centre
    pixels, _ = cv2.projectPoints(corners, np.zeros(3), np.zeros(3), SYNTH_K, None)
    hull = cv2.convexHull(np.round(pixels[:, 0]).astype(np.int32))
    cv2.fillConvexPoly(image, hull, 230)
    texture = _marker_texture()
    per_m = 400 / marker                                   # texture px per metre
    to_face = np.array([[1 / per_m, 0, -texture.shape[1] / 2 / per_m],
                        [0, -1 / per_m, texture.shape[0] / 2 / per_m], [0, 0, 1]])
    face_centre = centre + rotation[:, 2] * edge / 2
    homography = SYNTH_K @ np.column_stack([rotation[:, 0], rotation[:, 1], face_centre]) @ to_face
    warped = cv2.warpPerspective(texture, homography, (640, 480), borderValue=0)
    inside = cv2.warpPerspective(np.full_like(texture, 255), homography, (640, 480)) > 0
    image[inside] = warped[inside]
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), mask.astype(bool)


@check
def aruco_markers_give_their_pose():
    from scipy.spatial.transform import Rotation
    from src.vision import aruco
    facing = Rotation.from_euler("x", 180, degrees=True)  # marker z towards the camera
    for euler in ((0, 0, 0), (25, -20, 30)):               # square-on used to flip
        rotation = (Rotation.from_euler("xyz", euler, degrees=True) * facing).as_matrix()
        rgb, _ = _cube_with_marker(rotation, np.array([0.0, 0.0, 0.30]))
        found = aruco.detect(rgb, "DICT_4X4_50")
        assert found and found[0][0] == 3, found
        turn, position, error = aruco.marker_pose(found[0][1], 0.040, SYNTH_K, np.zeros(5))
        centre = aruco.box_centre(turn, position, 0.055)
        assert error < 1.0 and np.linalg.norm(centre - [0, 0, 0.30]) < 0.005, (euler, centre)
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation.T @ turn) - 1) / 2, -1, 1)))
        assert angle < 10, (euler, angle)


@check
def camera_preview_shows_aruco_markers():
    from scipy.spatial.transform import Rotation
    rotation = (Rotation.from_euler("xyz", (20, -15, 10), degrees=True)
                * Rotation.from_euler("x", 180, degrees=True)).as_matrix()
    rgb, _ = _cube_with_marker(rotation, np.array([0.0, 0.0, 0.25]))

    class Still:
        def read(self):
            time.sleep(0.01)
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        def release(self):
            pass

    from src.vision import camera as camera_module
    drawn = camera_module.draw

    def covering(rgb, *args, **kwargs):               # a preview that hides everything
        return np.zeros_like(drawn(rgb, *args, **kwargs))

    camera_module.draw = covering
    camera = Camera(lambda config: Still())
    try:
        camera.reconfigure(dict(settings.DEFAULTS["vision"], enabled=True, aruco=True))
        _wait(lambda: camera.latest() is not None, timeout=5.0)
        frame = camera.latest()
        assert [m[0] for m in frame.markers] == [3], "found on the raw image, not the preview"
        assert np.array_equal(frame.raw, rgb), "the raw image is the camera's, untouched"
        assert frame.image.any(), "markers drawn onto the preview"
    finally:
        camera_module.draw = drawn
        camera.close()


@check
def fit_3d_takes_the_marker_pose_and_measures_the_outline_fit():
    from scipy.spatial.transform import Rotation
    from src.processing.pose3d import POSE3D, fit_3d
    from src.vision.calibration import Calibration
    count = 24
    calibration = Calibration(640, 480, 326.0, 326.0, 320.0, 240.0, (0.0,) * 5, 0.1, 10, "test")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "cube" / "20260101-000000"
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps({
            "format": 2, "complete": True,
            "camera": {"name": "Synth", "width": 640, "height": 480,
                       "calibration": calibration.entry()}}))
        np.savez(folder / "frames.npz", time=np.arange(count) / 30.0)
        writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.CAP_FFMPEG,
                                 cv2.VideoWriter_fourcc(*"avc1"), 30.0, (640, 480))
        packed = np.zeros((1, count, 480, 80), np.uint8)
        truths = []
        for i in range(count):
            rotation = (Rotation.from_euler("xyz", (15 + i, -20, 10 + 0.5 * i), degrees=True)
                        * Rotation.from_euler("x", 180, degrees=True)).as_matrix()
            centre = np.array([0.01 + 0.0005 * i, -0.01, 0.30])
            rgb, mask = _cube_with_marker(rotation, centre)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            packed[0, i] = np.packbits(mask, axis=-1)
            truths.append(centre)
        writer.release()
        none = np.zeros((count, 2), bool)
        np.savez(folder / "hands_clean.npz", image=np.full((count, 2, 21, 3), np.nan),
                 world=np.full((count, 2, 21, 3), np.nan), score=np.full((count, 2), np.nan),
                 present=none, observed=none, filled=none)
        np.savez(folder / "masks.npz", masks=packed, width=640, height=480,
                 names=np.array(["cube"]), area=np.ones((1, count)))
        np.savez(folder / "object.npz", names=np.array(["cube"]), hand=np.array([-1]),
                 visible=np.ones((1, count), bool), contact=np.zeros((1, count, 2), bool))
        size = settings.get("object.size_mm")
        settings.set("object.size_mm", [55.0, 55.0, 55.0])
        try:
            summary = fit_3d(Episode(folder))
        finally:
            settings.set("object.size_mm", size)
        pose = Episode(folder).arrays(POSE3D)
        assert summary["marker_frames"] >= 0.8 * count, summary
        assert (pose["object_source"][0][pose["marker_id"] >= 0] == 2).all()
        error = np.linalg.norm(pose["object_position"][0] - np.array(truths), axis=1)
        assert np.nanmedian(error) < 0.004, np.nanmedian(error)
        assert summary["outline_vs_marker_mm"] < 15, summary


@check
def fit_3d_boxes_only_the_object_being_moved():
    from types import SimpleNamespace
    from src.processing.pose3d import box_tracks
    motion = {"names": np.array(["cube", "table", "cup"]), "hand": np.array([1, 1, -1])}
    told = SimpleNamespace(masks={"roles": np.array(["moved", "support", "other"])})
    assert box_tracks(told, motion) == [0], "the table is held-looking but a support"
    older = SimpleNamespace(masks={})                    # a context without roles
    assert box_tracks(older, motion) == [0, 1], "then what a hand holds"
    assert box_tracks(older, {"names": motion["names"], "hand": np.array([-1, -1, -1])}) == [0]


@check
def pose3d_files_from_before_markers_still_open():
    from src.processing.pose3d import load
    from src.vision.calibration import Calibration
    from src.viewer.recordings_panel import _pose_lines
    count = 3
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "task" / "20261022-000000"
        _bare_episode(folder, count)
        rotation = np.broadcast_to(np.eye(3), (1, count, 3, 3)).copy()
        position = np.tile([0.0, 0.0, 0.3], (1, count, 1))
        position[0, 1] = np.nan                               # one frame unposed
        np.savez(folder / "pose3d.npz",                       # the fields it had then
                 object_position=position, object_rotation=rotation,
                 palm_position=np.full((count, 2, 3), np.nan),
                 palm_rotation=np.full((count, 2, 3, 3), np.nan),
                 box_mm=np.array([55.0] * 3), names=np.array(["cube"]))
        pose = load(Episode(folder))
        assert pose["object_source"][0].tolist() == [1, 0, 1]
        assert pose["boxed"].tolist() == [True], "older files: every object was fitted"
        assert (pose["marker_id"] == -1).all()
        camera = Calibration(160, 120, 100.0, 100.0, 80.0, 60.0, (0.0,) * 5, 0.1, 8, "t")
        assert len(_pose_lines(pose, 0, camera)) == 12        # the box's edges


@check
def world_plan_stage_stands_beside_the_hand_or_parks():
    from src.world import PLAN_OFFSET
    original = settings._current
    try:
        world = _world_with()
        model, data = world.model, world.data
        q = world.robot.clamp(np.full(len(world.robot), 0.5))
        object_pose = (np.array([0.03, 0.0, 0.07]), np.array([1.0, 0, 0, 0]))
        world.set_plan((q, object_pose, object_pose))
        world.hold()
        palm = {hand: data.xpos[model.body(f"{hand}/palm_base_link").id] for hand in HANDS}
        assert np.allclose(palm["plan"] - palm["encoder"], PLAN_OFFSET, atol=1e-9)
        assert np.allclose(world.pose("plan"), q)
        placed = data.xpos[model.body("plan_object").id] - palm["plan"]
        assert np.allclose(placed, object_pose[0], atol=1e-9), placed
        assert all(model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
                   for g in world.geom_ids("plan")), "the plan hand touches nothing"
        world.set_plan(None)
        world.hold()
        assert data.xpos[model.body("plan/palm_base_link").id][2] < -1, "parked"
        assert data.xpos[model.body("plan_object").id][2] < -1, "parked"
    finally:
        settings._current = original


def _task_with_defaults(start=None):
    """The task sim on default settings, the object where the World section
    places it unless `start` says otherwise."""
    from src.sim.task import Task
    tmp = tempfile.mkdtemp()
    settings._current = settings.Settings(Path(tmp) / "settings.json")
    placed = np.array(settings.get("object.position_mm")) / 1000.0
    return Task.build(start or (placed, np.array([1.0, 0, 0, 0])))


@check
def task_sim_holds_the_object_until_released():
    from src.sim.task import STALL_CURRENT_A, STALL_TORQUE_NM
    original = settings._current
    try:
        task = _task_with_defaults()
        model, data = task.model, task.data
        cap = STALL_TORQUE_NM * settings.get("hardware.current_limit_ma") / 1000 / STALL_CURRENT_A
        assert np.allclose(model.actuator_forcerange, [[-cap, cap]] * model.nu), "servo torque"
        assert len(task.tips) == len(robot_module.load().groups()) == len(data.sensordata)
        start = task.object_pose(data.qpos)[0]
        while data.time < 0.5:
            mujoco.mj_step(model, data)
        assert np.linalg.norm(task.object_pose(data.qpos)[0] - start) < 0.002, "held"
        task.release(data)
        while data.time < 1.0:
            mujoco.mj_step(model, data)
        assert task.object_pose(data.qpos)[0][2] < start[2] - 0.02, "falls once let go"
        assert (data.sensordata <= 0.1 + 1e-9).all() and (data.sensordata > -0.01).all(), \
            data.sensordata
    finally:
        settings._current = original


@check
def task_sim_hand_cannot_pass_through_itself():
    original = settings._current
    try:
        task = _task_with_defaults()
        model, data = task.model, task.data
        names = robot_module.load().names
        hand = [c for c in data.contact[:data.ncon] if c.efc_address >= 0
                and "object" not in (model.geom(c.geom1).name, model.geom(c.geom2).name)]
        assert not hand, "links push on each other at rest"
        assert task.self_overlap(data)[0] < 1e-6

        def drive(**degrees):
            mujoco.mj_resetData(model, data)
            data.eq_active[:] = 0
            data.qpos[task.object_qpos:task.object_qpos + 3] += [0.5, 0.0, 0.0]   # out of the way
            for joint, angle in degrees.items():
                data.ctrl[names.index(f"{joint}_joint")] = np.radians(angle)
            worst = 0.0
            for step in range(1000):
                mujoco.mj_step(model, data)
                if step % 100 == 0:
                    worst = max(worst, task.self_overlap(data)[0])
            return worst

        # The middle finger swung at its neighbour stops against it.
        worst = drive(middle_adduction=20)
        reached = np.degrees(data.qpos[names.index("middle_adduction_joint")])
        assert reached < 15 and worst < 0.0015, (reached, worst)
        # The thumb swept across the palm meets the fingers, not passes them.
        assert drive(thumb_adduction=110, thumb_proximal=100, thumb_middle=60) < 0.001
    finally:
        settings._current = original


@check
def planner_finds_a_grip_that_holds_without_the_weld():
    from src.sim import planner as planning
    original = settings._current
    saved = planning.GRASP_SAMPLES, planning.GRASP_ROUNDS
    try:
        planning.GRASP_SAMPLES, planning.GRASP_ROUNDS = 256, 10
        task = _task_with_defaults()
        planner = planning.Planner(task, seed=0)
        weld = task.model.equality("hold").id
        grip = planner.grasp()
        assert task.model.eq_active0[weld] == 0, "rollouts leave the weld as they found it"
        assert grip["slip_m"] < 0.01 and np.degrees(grip["turn_rad"]) < 10, grip
        # In the real sim: close held, let go, and it stays.
        data = task.data
        data.ctrl[:] = grip["targets"]
        while data.time < planning.CLOSE:
            mujoco.mj_step(task.model, data)
        task.release(data)
        before = task.object_pose(data.qpos)[0]
        while data.time < planning.CLOSE + planning.LET_GO:
            mujoco.mj_step(task.model, data)
        assert np.linalg.norm(task.object_pose(data.qpos)[0] - before) < 0.01
    finally:
        planning.GRASP_SAMPLES, planning.GRASP_ROUNDS = saved
        settings._current = original


@check
def plan_step_saves_its_file_and_out_of_date_files_are_flagged():
    from src.processing import plan as plan_step
    from src.processing.files import PLAN, SIM, stale
    from src.sim import planner as planning
    original = settings._current
    saved = planning.GRASP_SAMPLES, planning.GRASP_ROUNDS, planning.SAMPLES
    count = 10
    try:
        planning.GRASP_SAMPLES, planning.GRASP_ROUNDS, planning.SAMPLES = 128, 4, 32
        tmp = tempfile.mkdtemp()
        settings._current = settings.Settings(Path(tmp) / "settings.json")
        folder = Path(tmp) / "task" / "20261023-000000"
        _bare_episode(folder, count)
        placed = np.array(settings.get("object.position_mm")) / 1000.0
        np.savez(folder / "pose3d.npz", made=np.array("2026-10-23T10:00:00"))
        np.savez(folder / "sim.npz", position=np.tile(placed, (count, 1)),
                 quat=np.tile([1.0, 0, 0, 0], (count, 1)), valid=np.ones(count, bool),
                 time=np.arange(count) / 30.0, hand=np.array("Left"),
                 made=np.array("2026-10-23T10:00:01"))
        episode = Episode(folder)
        assert stale(episode) == {}
        result = plan_step.plan(episode)
        saved_plan = episode.arrays(plan_step.PLAN)
        rows = len(saved_plan["time"])
        assert saved_plan["qpos"].shape == (rows, len(robot_module.load()))
        assert saved_plan["quat"].shape == (rows, 4) and saved_plan["gaps"].shape[1] == 4
        assert set(result) >= {"grip_slip_mm", "dropped_at", "position_mm", "still_turn_deg"}
        assert result["still_position_mm"] == (0.0, 0.0), "a goal that never moves"
        np.savez(folder / "pose3d.npz", made=np.array("2099-01-01T00:00:00"))
        episode.reload()
        assert set(stale(episode)) == {SIM, PLAN}
        try:
            plan_step.plan(episode)
            raise AssertionError("planned from an out-of-date sim.npz")
        except ValueError as refused:
            assert "Send to sim again" in str(refused)
    finally:
        planning.GRASP_SAMPLES, planning.GRASP_ROUNDS, planning.SAMPLES = saved
        settings._current = original


@check
def out_of_date_files_follow_what_they_were_made_from():
    from src.processing.files import PLAN, POSE3D, SIM, TRACK, stale
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "task" / "20261023-000000"
        _bare_episode(folder, 3)
        for name, made in ((POSE3D, "10:00"), (SIM, "10:01"), (PLAN, "10:02"), (TRACK, "10:03")):
            np.savez(folder / name, made=np.array(f"2026-10-23T{made}:00"))
        episode = Episode(folder)
        assert stale(episode) == {}
        np.savez(folder / SIM, made=np.array("2026-10-23T10:05:00"))      # sent to sim again
        episode.reload()
        assert set(stale(episode)) == {PLAN, TRACK}
        np.savez(folder / POSE3D, made=np.array("2026-10-23T10:09:00"))   # and fitted again
        episode.reload()
        assert set(stale(episode)) == {SIM, PLAN, TRACK}


@check
def generated_motions_start_where_asked_and_move_smoothly():
    from scipy.spatial.transform import Rotation
    from src.learning import trajectories as library
    rng = np.random.default_rng(0)
    start, quat = np.array([0.03, 0.01, 0.07]), np.array([1.0, 0.0, 0.0, 0.0])
    # minimum-jerk peaks at 1.875x the mean speed; the shortest move is SEGMENT_S[0]
    frames = library.SEGMENT_S[0] * library.RATE
    for _ in range(20):
        positions, quats = library._generate(start, quat, rng)
        assert np.allclose(positions[0], start) and abs(quats[0] @ quat) > 1 - 1e-9
        assert (np.linalg.norm(positions - start, axis=1) <= library.SHIFT_M + 1e-9).all()
        step = np.linalg.norm(np.diff(positions, axis=0), axis=1).max()
        assert step <= 2 * library.SHIFT_M * 1.875 / frames + 1e-9, step
        turns = (Rotation.from_quat(quats[1:, [1, 2, 3, 0]])
                 * Rotation.from_quat(quats[:-1, [1, 2, 3, 0]]).inv()).magnitude()
        assert turns.max() <= library.TURN_RANGE[1] * 1.875 / frames + 1e-9
    # Wobbles: within reach, and no faster than their waves and easing allow.
    fastest = (2 * np.pi * library.WOBBLE_HZ[1] + 1.875 / library.RAMP_S) / library.RATE
    for _ in range(20):
        positions, quats = library._wobble(start, quat, rng)
        assert np.allclose(positions[0], start) and abs(quats[0] @ quat) > 1 - 1e-9
        turned = Rotation.from_quat(quats[:, [1, 2, 3, 0]]) * Rotation.from_quat(quat[[1, 2, 3, 0]]).inv()
        assert (turned.magnitude() <= library.WOBBLE_TURN[1] + 1e-9).all()
        assert (np.linalg.norm(positions - start, axis=1) <= library.SHIFT_M + 1e-9).all()
        assert np.linalg.norm(np.diff(positions, axis=0), axis=1).max() \
            <= fastest * library.SHIFT_M + 1e-9
        steps = (Rotation.from_quat(quats[1:, [1, 2, 3, 0]])
                 * Rotation.from_quat(quats[:-1, [1, 2, 3, 0]]).inv()).magnitude()
        assert steps.max() <= fastest * library.WOBBLE_TURN[1] + 1e-9


@check
def tracker_env_holds_with_the_grip_and_sees_goals_up_to_symmetry():
    import torch
    from scipy.spatial.transform import Rotation
    from src.learning.env import TrackEnv
    from src.learning.trajectories import let_go
    from src.sim import planner as planning
    original = settings._current
    saved = planning.GRASP_SAMPLES, planning.GRASP_ROUNDS
    try:
        planning.GRASP_SAMPLES, planning.GRASP_ROUNDS = 256, 10
        task = _task_with_defaults()
        grip = planning.Planner(task, seed=0).grasp()["targets"]
        state = let_go(task, grip)
        start_position, start_rotation = task.object_pose(state[1:1 + task.model.nq])
        length = 40
        quarter = start_rotation @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
        tilted = start_rotation @ Rotation.from_euler("x", 30, degrees=True).as_matrix()
        motions = [np.repeat(r[None], length, 0) for r in (quarter, tilted)]
        quat = lambda r: Rotation.from_matrix(r).as_quat()[:, [3, 0, 1, 2]]   # noqa: E731
        library = {"start_position": start_position[None], "start_quat": quat(start_rotation[None]),
                   "start_grip": grip[None], "start_state": state[None],
                   "motion_start": np.array([0, 0]), "motion_length": np.array([length] * 2),
                   "motion_position": np.repeat(start_position[None, None], length, 1).repeat(2, 0),
                   "motion_quat": np.stack([quat(m) for m in motions]),
                   "motion_recorded": np.array([False, False]),
                   "motion_feasible": np.array([True, True])}
        env = TrackEnv(library, envs=2, randomize=False, cycle=True)
        policy = env.get_observations()["policy"].numpy()
        assert policy.shape == (2, 3 * len(grip) + 9 + 9 * len(env.future))
        turns = policy[:, -6 * len(env.future):].reshape(2, len(env.future), 6)
        # A cube turned a quarter about its own axis looks the same: no turn
        # left to make. Tilted 30 degrees: the turn the policy is shown.
        assert np.allclose(turns[0], [1, 0, 0, 0, 1, 0], atol=1e-6), turns[0]
        angle = np.degrees(np.arccos((np.trace(_six_back(turns[1, 0])) - 1) / 2))
        assert abs(angle - 30) < 1e-3, angle
        drops = 0
        for _ in range(length - 2):                     # zero action: keep the grip
            _, _, done, extras = env.step(torch.zeros(2, env.num_actions))
            drops += int((done & ~extras["time_outs"]).sum())
        assert drops == 0, "the grip let go"
    finally:
        planning.GRASP_SAMPLES, planning.GRASP_ROUNDS = saved
        settings._current = original


@check
def learning_runs_in_the_background_reports_and_can_be_cancelled():
    from src.learning import background
    from src.paths import ROOT
    saved = background.FOLDER, background.STATUS, background.SCRIPTS
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        status = tmp / "status.json"

        def fake(name, body):
            script = tmp / f"{name}.py"
            script.write_text(
                f"import sys, time, os\nsys.path.insert(0, {str(ROOT)!r})\n"
                "from pathlib import Path\nfrom src.learning import background\n"
                f"background.STATUS = Path({str(status)!r})\n"
                f"background.begin({name!r})\n" + body)
            return script

        try:
            background.FOLDER, background.STATUS = tmp, status
            background.SCRIPTS = {
                "library": fake("library", "background.report(done=1, total=2, text='half')\n"
                                           "background.finish(state='done', text='built')\n"),
                "train": fake("train", "time.sleep(60)\n")}
            task = background.Background()
            assert task.start("library")
            _wait(lambda: task.status()["state"] not in background.ACTIVE, timeout=20)
            done = task.status()
            assert done["state"] == "done" and done["text"] == "built", done
            assert done["log"] == str(tmp / "library.log"), "the panel's log path is kept"
            assert task.start("train")
            _wait(lambda: task.status()["state"] == "running", timeout=20)
            assert not task.start("library"), "one task at a time"
            pid = task.status()["pid"]
            task.cancel()
            assert task.status()["state"] == "cancelled"
            try:
                os.kill(pid, 0)
                raise AssertionError("cancelled, but the process is still there")
            except ProcessLookupError:
                pass
            background.SCRIPTS = {"library": fake("library", "os._exit(3)\n")}
            task.start("library")
            _wait(lambda: task.status()["state"] not in background.ACTIVE, timeout=20)
            assert task.status()["state"] == "stopped", "ended without saying so"
        finally:
            background.FOLDER, background.STATUS, background.SCRIPTS = saved


@check
def training_progress_is_written_each_round():
    from src.learning import background, train
    saved = background.STATUS
    with tempfile.TemporaryDirectory() as tmp:
        try:
            background.STATUS = Path(tmp) / "status.json"
            writer = train.ProgressWriter(tmp, iterations=3)
            for it in range(3):
                writer.add_scalar("/track/turn_deg", 10.0 - it, it)
                writer.add_scalar("/track/position_mm", 5.0, it)
                if it == 1:
                    status = background.read()
                    assert status["done"] == 1 and status["total"] == 3, status
            writer.stop()
            rows = train.progress(Path(tmp))
            assert [r["iteration"] for r in rows] == [0, 1, 2]
            assert rows[2]["/track/turn_deg"] == 8.0
            assert background.read()["done"] == 3
        finally:
            background.STATUS = saved


@check
def the_planner_teaches_the_policy_one_round():
    import io
    import contextlib
    from scipy.spatial.transform import Rotation
    from src.learning import background, teacher, train
    from src.learning.trajectories import let_go
    from src.sim import planner as planning
    original, status = settings._current, background.STATUS
    saved = planning.GRASP_SAMPLES, planning.GRASP_ROUNDS, planning.SAMPLES
    try:
        planning.GRASP_SAMPLES, planning.GRASP_ROUNDS, planning.SAMPLES = 256, 10, 16
        task = _task_with_defaults()
        grip = planning.Planner(task, seed=0).grasp()["targets"]
        state = let_go(task, grip)
        position, rotation = task.object_pose(state[1:1 + task.model.nq])
        length = 12
        turned = rotation @ Rotation.from_euler("x", 10, degrees=True).as_matrix()
        quat = Rotation.from_matrix(turned).as_quat()[[3, 0, 1, 2]]
        library = {"start_position": position[None],
                   "start_quat": Rotation.from_matrix(rotation).as_quat()[None][:, [3, 0, 1, 2]],
                   "start_grip": grip[None], "start_state": state[None],
                   "motion_start": np.array([0]), "motion_length": np.array([length]),
                   "motion_position": np.repeat(position[None, None], length, 1),
                   "motion_quat": np.repeat(quat[None, None], length, 1),
                   "motion_recorded": np.array([False]), "motion_feasible": np.array([True])}
        with tempfile.TemporaryDirectory() as tmp:
            background.STATUS = Path(tmp) / "status.json"
            with contextlib.redirect_stdout(io.StringIO()):
                teacher.teach(library, Path(tmp), 1, train.CONFIG,
                              train.ProgressWriter(tmp, 1))
            rows = train.progress(Path(tmp))
            assert len(rows) == 1 and rows[0]["Teacher/labels"] >= length - 2, rows
            assert np.isfinite(rows[0]["Teacher/loss"]) and rows[0]["Teacher/beta"] == 1.0
            assert (Path(tmp) / "model_0.pt").exists()
            with contextlib.redirect_stdout(io.StringIO()):
                runs = train.follow(Path(tmp), library, [0])      # the checkpoint loads
            assert len(runs[0]["joints"]) >= 1
    finally:
        planning.GRASP_SAMPLES, planning.GRASP_ROUNDS, planning.SAMPLES = saved
        settings._current, background.STATUS = original, status


@check
def clearml_training_needs_pushed_code_and_brings_the_run_back():
    import shutil
    import subprocess
    from src.learning import remote
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def git(*args, where=tmp / "work"):
            subprocess.run(["git", *args], cwd=where, check=True, capture_output=True)

        (tmp / "work").mkdir()
        git("init", "-q", "-b", "main")
        git("config", "user.email", "test@example.com")
        git("config", "user.name", "test")
        (tmp / "work" / "a.txt").write_text("one")
        assert "uncommitted" in remote.blocked(tmp / "work")
        git("add", "a.txt")
        git("commit", "-qm", "one")
        assert "remote" in remote.blocked(tmp / "work")
        git("init", "-q", "--bare", str(tmp / "origin.git"), where=tmp)
        git("remote", "add", "origin", str(tmp / "origin.git"))
        assert "push" in remote.blocked(tmp / "work"), "the branch is not on the remote"
        git("push", "-q", "-u", "origin", "main")
        assert remote.blocked(tmp / "work") == ""
        (tmp / "work" / "a.txt").write_text("two")
        git("commit", "-qam", "two")
        assert "not pushed" in remote.blocked(tmp / "work")
        # A run comes back as a zipped folder and lands among the runs.
        run = tmp / "made" / "20261023-150000"
        run.mkdir(parents=True)
        (run / "run.json").write_text("{}")
        archive = shutil.make_archive(str(run), "zip", root_dir=run.parent, base_dir=run.name)
        landed = remote.unpack(Path(archive), runs=tmp / "runs")
        assert landed == tmp / "runs" / "20261023-150000" and (landed / "run.json").exists()


def _six_back(six):
    """A rotation matrix from its first two columns (6,)."""
    x, y = six[:3], six[3:]
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _wait(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.01)


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as failure:
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(failure).__name__}: {failure}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
