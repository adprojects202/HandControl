"""Asks a vision-language model for the task context of camera frames, off
the GUI thread.

    assistant = Assistant()
    assistant.describe([jpeg(frame.raw, 1024)])            # one live frame
    assistant.describe(load_jpegs, times, on_done=save)   # a recording's clip
    assistant.busy, assistant.answer, assistant.error     # polled by the GUI

The answer is structured: JSON in the shape of context.SCHEMA, parsed into
`answer.data`. `describe` returns at once; the question runs on its own
thread and one runs at a time. Provider, model and system prompt come from
the assistant settings section, read when the question is asked.

API keys are never settings, and nothing here ever writes one. Each
provider's key is a key given for this session (`set_session_key`, kept in
memory only), or else its environment variable (ANTHROPIC_API_KEY,
OPENAI_API_KEY) or the same name in handlab/.env - see src/keys.py.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from .. import keys, settings
from . import claude, openai_api
from . import context
from .question import AssistantError, Question

# settings: assistant.provider -> the module that asks it.
PROVIDERS = {"claude": claude, "openai": openai_api}


class Assistant:
    def __init__(self, providers=None, secrets=None):
        """`secrets` is a keys.Secrets; the app's own by default."""
        self.providers = providers or PROVIDERS
        self.secrets = secrets or keys.default()
        self.busy = False
        self.answer = None           # the last Answer
        self.error = ""              # or why the last question got none
        self.asked_at = 0.0          # time.monotonic() of the last question
        self.label = ""              # who asked it (describe's `label`)
        self._session_keys = {}

    def key_source(self, provider):
        """Where `provider`'s key comes from: "this session", "$ENV_NAME",
        ".env", or "" when there is none."""
        return self._key(provider)[1]

    def _key(self, provider):
        """(key, where it came from), or ("", "")."""
        session = self._session_keys.get(provider, "").strip()
        if session:
            return session, "this session"
        return self.secrets.find(self.providers[provider].KEY_ENV)

    def set_session_key(self, provider, key):
        """Use `key` for `provider` until the app closes (blank: forget it)."""
        key = key.strip()
        if key:
            self._session_keys[provider] = key
        else:
            self._session_keys.pop(provider, None)

    def describe(self, images, times=None, on_done=None, config=None, label=""):
        """Start asking for the task context of `images` - JPEG bytes, oldest
        first, or a function returning them (run on the assistant's thread:
        for slow sources, such as decoding a recording). `times` are their
        seconds from the recording's start; None for one live frame.
        `on_done(answer, question, config)` runs on that thread once a
        context has come back. `label` is kept in `self.label`, so a caller
        can tell its answer from another's. False (with `error` set) if it
        cannot start."""
        if self.busy:
            return False
        config = dict(config or settings.section("assistant"))
        name = config["provider"]
        provider = self.providers[name]
        key, _ = self._key(name)
        if not key:
            self.error = (f"no {provider.NAME} API key: set {provider.KEY_ENV} in the "
                          f"environment or in {self.secrets.key_file.name}, or enter "
                          "one for this session")
            return False
        self.busy, self.answer, self.error, self.label = True, None, "", label
        self.asked_at = time.monotonic()
        threading.Thread(target=self._run,
                         args=(provider, images, times, on_done, config, key),
                         name="handlab-assistant", daemon=True).start()
        return True

    def _run(self, provider, images, times, on_done, config, key):
        try:
            images = tuple(images() if callable(images) else images)
            question = Question(context.instruction(times), images, context.SCHEMA)
            answer = provider.ask(question, config, key)
            answer = replace(answer, seconds=time.monotonic() - self.asked_at,
                             data=context.parse(answer.text))
            if answer.data is None and not answer.note:
                answer = replace(answer, note="the answer was not a task context")
            self.answer = answer
            if answer.data is not None and on_done is not None:
                on_done(answer, question, config)
        except AssistantError as failure:
            self.error = str(failure)
        except Exception as failure:         # a bad answer never takes the app down
            self.error = f"{type(failure).__name__}: {failure}"
        finally:
            self.busy = False
