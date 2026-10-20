"""Headless checks of everything below the window. No hardware, no window.

    python scripts/selftest.py

Exits non-zero if any check fails.
"""

import base64
import json
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
