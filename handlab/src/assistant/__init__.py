"""Asking a vision-language model (Claude or OpenAI) what task camera frames
show, as structured data.

    assistant.py   the background question runner and API keys
    context.py     the task context's JSON schema, the instruction, context.json
    question.py    Question / Answer, the same for every provider; jpeg()
    claude.py      Anthropic's Messages API
    openai_api.py  OpenAI's Responses API

A provider is a module with NAME, KEY_ENV, request(question, config) (the API
arguments, testable offline) and ask(question, config, key) -> Answer; a
question with a schema must come back as JSON that follows it.
"""

from .assistant import PROVIDERS, Assistant
from .question import Answer, AssistantError, Question, jpeg

__all__ = ["PROVIDERS", "Answer", "Assistant", "AssistantError", "Question", "jpeg"]
