"""Assistant: which vision-language model describes the task, and a quick
look at what it makes of the live camera frame.

Nothing is typed to it: it is always asked the same thing - what task the
frames show - and answers in a fixed JSON shape (src/assistant/context.py).
Recordings are described from the Recordings section, which keeps the
answer with the recording. The question runs on its own thread; the window
keeps drawing while it does.

API keys are never saved by the app: each provider's comes from one entered
here (kept in memory for this session only), its environment variable
(ANTHROPIC_API_KEY, OPENAI_API_KEY), or the same name in handlab/.env.
"""

from __future__ import annotations

import time

from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings
from ..assistant import PROVIDERS, jpeg
from .form import RED, SettingsForm, row, table

AMBER = imgui.Vec4(0.95, 0.75, 0.30, 1.0)
LIVE = "live frame"                         # Assistant.label of a Describe frame
EFFORT_TIP = ("how hard Claude thinks before answering: low is fastest and "
              "cheapest, max the deepest")
LIMITS = (
    ("assistant.max_tokens", "max tokens",
     "longest answer, in tokens (thinking included, for Claude)"),
    ("assistant.image_max_side", "image (px)",
     "frames are scaled down to this on their longer side"),
    ("assistant.clip_frames", "clip frames",
     "how many frames of a recording are sent, spread evenly over it"),
)


class AssistantSection:
    title = "Assistant"

    def __init__(self, assistant, camera):
        self.assistant = assistant
        self.camera = camera
        # Assistant settings are read when a question is asked: nothing to apply.
        self.form = SettingsForm(lambda: None)
        self.message = ""                  # why the last Describe did not go out
        self._key = ""                     # a key being typed, until it is used

    def draw(self):
        name = settings.get("assistant.provider")
        provider = PROVIDERS[name]
        if table("assistant"):
            row("provider", "which company's model to ask")
            self.form.choice("assistant.provider", show=lambda p: PROVIDERS[p].NAME)
            row("model", f"{provider.NAME} model name, as its API spells it")
            self.form.text(f"assistant.{name}_model")
            if name == "claude":
                row("effort", EFFORT_TIP)
                self.form.choice("assistant.claude_effort")
            row("API key", f"entered here for this session only, or "
                           f"{provider.KEY_ENV} from the environment or handlab/.env"
                           " - never saved by the app", fill=False)
            self._key_cell(name, provider)
            imgui.EndTable()

        imgui.BeginDisabled(self.assistant.busy)
        if imgui.Button("Describe frame"):
            self._describe_frame()
        imgui.EndDisabled()
        imgui.SetItemTooltip("send the camera's current frame and show what task "
                             "the model makes of it (not saved)")
        if self.assistant.label == LIVE:
            status(self.assistant, self.message)
            if self.assistant.answer is not None:
                draw_answer(self.assistant.answer)
        elif self.message:
            status(self.assistant, self.message)

        if imgui.TreeNode("prompt and limits"):
            imgui.TextDisabled("system prompt")
            self.form.text("assistant.system_prompt", lines=8)
            if table("limits"):
                for key, label, tip in LIMITS:
                    low, high = settings.limits(key)
                    row(label, f"{tip}\n{low:g} - {high:g}")
                    self.form.field(key)
                imgui.EndTable()
            imgui.TreePop()
        self.form.draw_message()

    def _key_cell(self, name, provider):
        source = self.assistant.key_source(name)
        if source:
            imgui.AlignTextToFramePadding()
            imgui.Text(f"from {source}")
            if source == "this session":
                imgui.SameLine()
                if imgui.SmallButton("forget"):
                    self.assistant.set_session_key(name, "")
            return
        imgui.SetNextItemWidth(-1)
        _, self._key = imgui.InputTextWithHint(
            "##key", f"none - {provider.KEY_ENV}, or paste one",
            self._key, int(imgui.InputTextFlags.Password))
        if imgui.IsItemDeactivatedAfterEdit():
            self.assistant.set_session_key(name, self._key)
            self._key = ""
        if self.assistant.secrets.problem:
            imgui.PushTextWrapPos(0.0)
            imgui.TextColored(AMBER, self.assistant.secrets.problem)
            imgui.PopTextWrapPos()

    def _describe_frame(self):
        self.message = ""
        frame = self.camera.latest()
        if frame is None:
            self.message = "no camera frame: turn the camera on"
            return
        image = jpeg(frame.raw, settings.get("assistant.image_max_side"))
        self.assistant.describe([image], label=LIVE)


def status(assistant, message=""):
    """The assistant's progress or problem, one line."""
    if assistant.busy:
        name = PROVIDERS[settings.get("assistant.provider")].NAME
        imgui.TextDisabled(f"asking {name} ... "
                           f"{time.monotonic() - assistant.asked_at:.0f} s")
    problem = message or assistant.error
    if problem:
        imgui.PushTextWrapPos(0.0)
        imgui.TextColored(RED, problem)
        imgui.PopTextWrapPos()


def draw_answer(answer):
    """An Answer: who answered and how, then its context."""
    imgui.TextDisabled(f"{answer.model}, {answer.seconds:.1f} s, "
                       f"{answer.input_tokens} in / {answer.output_tokens} out tokens")
    if answer.note:
        imgui.TextColored(AMBER, answer.note)
    if answer.data is not None:
        draw_context(answer.data)
    elif answer.text:
        imgui.PushTextWrapPos(0.0)
        imgui.TextUnformatted(answer.text)
        imgui.PopTextWrapPos()


def draw_context(context):
    """A task context (context.SCHEMA), laid out to read."""
    imgui.PushTextWrapPos(0.0)
    imgui.SeparatorText("task")
    imgui.TextUnformatted(context.get("task", ""))
    imgui.TextDisabled("goal")
    imgui.TextUnformatted(context.get("goal", ""))
    if context.get("objects"):
        imgui.SeparatorText("objects")
        for item in context["objects"]:
            box = item.get("box", {})
            imgui.BulletText(f"{item.get('name', '?')}  ({item.get('role', '?')})")
            imgui.SetItemTooltip(
                f"box in image {item.get('image', '?')}: "
                + ", ".join(f"{edge} {box.get(edge, float('nan')):.2f}"
                            for edge in ("left", "top", "right", "bottom")))
    if context.get("hands"):
        imgui.SeparatorText("hands")
        for hand in context["hands"]:
            imgui.BulletText(f"{hand.get('side', '?')}: {hand.get('does', '')}")
    if context.get("phases"):
        imgui.SeparatorText("phases")
        for phase in context["phases"]:
            imgui.BulletText(f"{phase.get('name', '?')}  (images {phase.get('first_image')}"
                             f"-{phase.get('last_image')}): {phase.get('what_happens', '')}")
    if context.get("uncertain"):
        imgui.SeparatorText("uncertain")
        imgui.TextColored(AMBER, context["uncertain"])
    imgui.PopTextWrapPos()
