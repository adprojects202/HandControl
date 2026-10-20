"""Claude, through Anthropic's Messages API.

Thinking is always on for Claude Opus 5.5; `claude_effort` (low ... max)
sets how deep it goes. A question with a schema is answered as JSON that
follows it (`output_config.format`, structured outputs).

Requests carry `fallbacks="default"` (beta server-side-fallback-2026-07-01):
if the model declines a request, the API re-runs it on Anthropic's
recommended fallback model within the same call, and the answer's note says
so. Answers are streamed and collected whole, so a long one never runs into
the SDK's timeout for a single request.
"""

from __future__ import annotations

import base64

import anthropic

from .question import Answer, AssistantError

NAME = "Claude"
KEY_ENV = "ANTHROPIC_API_KEY"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def request(question, config):
    """The Messages API arguments for `question` under the assistant settings."""
    content = [{"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": base64.standard_b64encode(image).decode("ascii")}}
               for image in question.images]
    content.append({"type": "text", "text": question.text})
    output = {"effort": config["claude_effort"]}
    if question.schema is not None:
        output["format"] = {"type": "json_schema", "schema": question.schema}
    return {
        "model": config["claude_model"],
        "max_tokens": config["max_tokens"],
        "system": config["system_prompt"],
        "output_config": output,
        "messages": [{"role": "user", "content": content}],
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }


def ask(question, config, key):
    client = anthropic.Anthropic(api_key=key)
    try:
        with client.beta.messages.stream(**request(question, config)) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as failure:
        raise AssistantError(f"Claude rejected the API key ({KEY_ENV})") from failure
    except anthropic.APIStatusError as failure:
        raise AssistantError(f"Claude API error {failure.status_code}: "
                             f"{failure.message}") from failure
    except anthropic.APIConnectionError as failure:
        raise AssistantError("could not reach the Claude API - check the network") from failure

    notes = []
    if any(step.type == "fallback_message" for step in message.usage.iterations or []):
        notes.append(f"answered by the fallback model, {message.model}")
    if message.stop_reason == "refusal":
        notes.append("declined")
    elif message.stop_reason == "max_tokens":
        notes.append("cut off at max_tokens")
    return Answer(
        text="".join(block.text for block in message.content if block.type == "text"),
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        note="; ".join(notes))
