"""OpenAI, through the Responses API. (Named openai_api so it is never
mistaken for the `openai` package it uses.) A question with a schema is
answered as JSON that follows it (`text.format`, strict structured outputs)."""

from __future__ import annotations

import base64

import openai

from .question import Answer, AssistantError

NAME = "OpenAI"
KEY_ENV = "OPENAI_API_KEY"
SCHEMA_NAME = "answer"


def request(question, config):
    """The Responses API arguments for `question` under the assistant settings."""
    content = [{"type": "input_image",
                "image_url": "data:image/jpeg;base64,"
                             + base64.standard_b64encode(image).decode("ascii"),
                "detail": "auto"}
               for image in question.images]
    content.append({"type": "input_text", "text": question.text})
    arguments = {
        "model": config["openai_model"],
        "instructions": config["system_prompt"],
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": config["max_tokens"],
    }
    if question.schema is not None:
        arguments["text"] = {"format": {"type": "json_schema", "name": SCHEMA_NAME,
                                        "schema": question.schema, "strict": True}}
    return arguments


def ask(question, config, key):
    client = openai.OpenAI(api_key=key)
    try:
        response = client.responses.create(**request(question, config))
    except openai.AuthenticationError as failure:
        raise AssistantError(f"OpenAI rejected the API key ({KEY_ENV})") from failure
    except openai.APIStatusError as failure:
        raise AssistantError(f"OpenAI API error {failure.status_code}: "
                             f"{failure.message}") from failure
    except openai.APIConnectionError as failure:
        raise AssistantError("could not reach the OpenAI API - check the network") from failure

    note = ""
    if response.status == "incomplete" and response.incomplete_details is not None:
        note = f"incomplete: {response.incomplete_details.reason}"
    usage = response.usage
    return Answer(
        text=response.output_text,
        model=response.model,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        note=note)
