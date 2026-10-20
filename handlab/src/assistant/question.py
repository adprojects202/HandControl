"""What goes to a vision-language model and what comes back, the same for
every provider."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

JPEG_QUALITY = 85


class AssistantError(RuntimeError):
    """A failure worth showing as it is: no key, key rejected, API error."""


@dataclass(frozen=True)
class Question:
    text: str
    # JPEG images, in order, shown to the model before the text. Several are
    # read as a sequence - frames of a clip, oldest first.
    images: tuple[bytes, ...] = ()
    # A JSON schema the answer must follow (structured outputs), or None.
    schema: dict | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    model: str               # the model that actually answered
    input_tokens: int
    output_tokens: int
    note: str = ""           # e.g. declined, cut off, answered by a fallback
    seconds: float = 0.0     # wall clock, question to answer
    data: dict | None = None # the answer parsed, when a schema was asked for


def jpeg(rgb, max_side):
    """An RGB frame as JPEG bytes, scaled down so its longer side is at most
    `max_side` px (never up)."""
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise AssistantError("could not encode the frame as JPEG")
    return data.tobytes()
