"""The task context: what a vision-language model is asked to give back.

Not free text - a fixed JSON shape (SCHEMA), which both providers are made to
follow (structured outputs), so the rest of the pipeline can read it:

    task, goal   what the person does, and the end state that means done
    objects      each with a short visible name (a segmentation prompt), a
                 role, and a rough box in one of the images (to start tracking)
    hands        which of the person's hands take part, doing what
    phases       the steps in order, as ranges of the images sent
    uncertain    what the images could not show

The schema keeps to what both APIs accept: every field required, no extra
fields, no numeric limits (boxes are 0..1 by description, not by rule).
"""

from __future__ import annotations

import json
from datetime import datetime

FORMAT = 1


def _object(required, properties, description=""):
    schema = {"type": "object", "additionalProperties": False,
              "required": list(required), "properties": properties}
    if description:
        schema["description"] = description
    return schema


def _text(description):
    return {"type": "string", "description": description}


def _index(description):
    return {"type": "integer", "description": description}


_BOX = _object(
    ("left", "top", "right", "bottom"),
    {edge: {"type": "number"} for edge in ("left", "top", "right", "bottom")},
    "The object's bounding box in that image, as fractions of the image width "
    "and height measured from its top-left corner, each 0 to 1.")

SCHEMA = _object(
    ("task", "goal", "objects", "hands", "phases", "uncertain"),
    {
        "task": _text("What the person is doing, in one sentence."),
        "goal": _text("The end state that means the task is done, said as where "
                      "the objects are, how they are turned or arranged."),
        "objects": {
            "type": "array",
            "description": "Every object the hands touch or the task is about - "
                           "not the hands, arms or body.",
            "items": _object(("name", "role", "image", "box"), {
                "name": _text("A short noun phrase that picks out this object and "
                              "no other, from what can be seen: colour, shape, "
                              "material - e.g. 'orange cube'. It is used as a "
                              "segmentation prompt."),
                "role": {"type": "string",
                         "enum": ["moved", "tool", "target", "support", "other"],
                         "description": "moved: the hands move it; tool: used on "
                                        "another object; target: where something "
                                        "is put; support: holds things up."},
                "image": _index("The index of an image (0 = first) where the "
                                "object is clearly visible; the box is in it."),
                "box": _BOX,
            }),
        },
        "hands": {
            "type": "array",
            "description": "Each of the person's hands that takes part.",
            "items": _object(("side", "does"), {
                "side": {"type": "string", "enum": ["left", "right"],
                         "description": "The person's own left or right hand. The "
                                        "images are the camera's true view, not "
                                        "mirrored."},
                "does": _text("What this hand does in the task, in a few words."),
            }),
        },
        "phases": {
            "type": "array",
            "description": "The task's steps in order - e.g. reach, grasp, lift, "
                           "turn, place, release.",
            "items": _object(("name", "first_image", "last_image", "what_happens"), {
                "name": _text("One or two words."),
                "first_image": _index("Index of the first image in this step."),
                "last_image": _index("Index of the last image in this step."),
                "what_happens": _text("What happens to the objects in this step."),
            }),
        },
        "uncertain": _text("What could not be told from the images. Empty if "
                           "nothing."),
    })


def instruction(times=None):
    """The user message that goes with the images: what they are and when
    they were taken. `times` (s from the recording's start) for a clip;
    None for one live frame."""
    view = "The images are the camera's true view, not mirrored."
    if times is None:
        return ("This is one frame from a live camera watching a person's hands. "
                f"{view} Describe the task as far as this one frame shows it.")
    stamps = ", ".join(f"image {i} at {t:.2f} s" for i, t in enumerate(times))
    return (f"These are {len(times)} frames from one recording of a person doing a "
            f"task, oldest first: {stamps}. {view} Describe the task.")


def parse(text):
    """The context in an answer's text, or None if it is not one."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not set(SCHEMA["required"]) <= data.keys():
        return None
    return data


def record(answer, question, config, frames=None):
    """What context.json holds: the context and everything that made it."""
    return {
        "format": FORMAT,
        "context": answer.data,
        "asked": datetime.now().isoformat(timespec="seconds"),
        "provider": config["provider"],
        "model": answer.model,
        "note": answer.note,
        "frames": frames,
        "system_prompt": config["system_prompt"],
        "instruction": question.text,
        "tokens": {"input": answer.input_tokens, "output": answer.output_tokens},
        "seconds": round(answer.seconds, 2),
    }
