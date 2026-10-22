"""The segment step: a mask of each object the task context names, in every
frame of a recording -> masks.npz (layout in recording/episode.py).

    segment(episode, progress)          # progress(done, total, text)

The objects come from the context (the describe step), so describe first.
Two kinds of model, by `segment.model`:

  SAM 2.1  facebook/sam2.1-hiera-{tiny,small,base-plus,large}: open
           (Apache-2.0). Starts from each object's box in the context, on the
           frame the box was given for, and tracks it forward to the end and
           backward to the start. Streamed - one frame at a time - so memory
           does not grow with the recording. Measured on an M4 Max (Apple GPU),
           small: ~0.25 s a frame; the same masks as whole-video mode.
  SAM 3    facebook/sam3: gated on Hugging Face - needs HF_TOKEN (a read
           token of an account with access) in the environment or
           handlab/.env. Finds every instance of each object's name, and
           keeps the one that best fits the context's box. Whole-video mode,
           since its clean-up of duplicate and false tracks needs the frames
           after; the processed video is held in CPU memory. Measured on the
           same 252-frame recording: ~1.3 s a frame, 8.8 GB at peak, masks
           matching SAM 2.1's (median IoU 0.99). Memory grows ~20 MB a frame,
           so a recording too long for this machine is refused up front.

Each mask is saved as the object's own region, without the stray specks
the models leave (vision/regions.py).

Weights download once, into assets/models/huggingface/ (not committed).
Runs on the Apple GPU (MPS) when there is one, else the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .. import keys, paths, settings
from ..recording.episode import MASKS
from ..vision.regions import main_region

CACHE = paths.ROOT / "assets/models/huggingface"
SAM3 = "facebook/sam3"
TOKEN = "HF_TOKEN"
# SAM 3's memory, measured: the model plus each frame held at 1008 x 1008.
SAM3_BASE_GB, SAM3_FRAME_GB = 3.6, 0.021
MEMORY_SHARE = 0.6            # most of this machine's memory a step may plan on


class SegmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    track: int               # place in the context's object list
    name: str
    role: str
    frame: int               # recording frame the box is in
    box: tuple               # (x0, y0, x1, y1) px


def prompts(episode):
    """One Prompt per object of the episode's context."""
    record = episode.context or {}
    objects = (record.get("context") or {}).get("objects") or []
    sent = record.get("frames") or []
    if not objects or not sent:
        raise SegmentError("no objects to segment: describe the recording first")
    height, width = episode.meta["camera"]["height"], episode.meta["camera"]["width"]
    found = []
    for track, item in enumerate(objects):
        image = min(max(int(item.get("image", 0)), 0), len(sent) - 1)
        box = item["box"]
        x0, x1 = sorted((box["left"], box["right"]))
        y0, y1 = sorted((box["top"], box["bottom"]))
        found.append(Prompt(track, item["name"], item.get("role", ""), int(sent[image]["index"]),
                            (np.clip(x0, 0, 1) * width, np.clip(y0, 0, 1) * height,
                             np.clip(x1, 0, 1) * width, np.clip(y1, 0, 1) * height)))
    return found


def segment(episode, progress=lambda done, total, text="": None, model=None, track=None):
    """Segment `episode` and save masks.npz. `model` defaults to the
    setting; `track(model, frames, prompts, progress)` -> packed masks
    (O, N, H, ceil(W/8)) and notes (O,), the model's own by default."""
    model = model or settings.get("segment.model")
    asked = prompts(episode)
    progress(0, episode.frames, "reading the video")
    frames = episode.read_frames(list(range(episode.frames)))
    if len(frames) != episode.frames:
        raise SegmentError(f"read {len(frames)} of {episode.frames} frames from the video")
    if model == SAM3 and track is None:
        need, have = sam3_memory_gb(len(frames)), _memory_gb() * MEMORY_SHARE
        if need > have:
            raise SegmentError(
                f"SAM 3 would need ~{need:.0f} GB for {len(frames)} frames "
                f"(this machine: {have:.0f} GB to spare) - use SAM 2.1, which "
                "streams, or a shorter recording")
    track = track or (_sam3 if model == SAM3 else _sam2)
    packed, notes = track(model, frames, asked, progress)
    height, width = frames[0].shape[:2]
    area = np.unpackbits(packed, axis=-1, count=width).sum(axis=(2, 3)) / (height * width)
    episode.save_arrays(MASKS, {
        "masks": packed, "area": area.astype(np.float32),
        "width": width, "height": height,
        "names": np.array([p.name for p in asked]),
        "roles": np.array([p.role for p in asked]),
        "prompt_frame": np.array([p.frame for p in asked]),
        "prompt_box": np.array([p.box for p in asked], np.float32),
        "notes": np.array(notes),
        "model": np.array(model),
        "made": np.array(datetime.now().isoformat(timespec="seconds")),
    })
    return area


def sam3_memory_gb(frames):
    """Roughly how much memory SAM 3 needs for a recording of `frames`."""
    return SAM3_BASE_GB + SAM3_FRAME_GB * frames


def _memory_gb():
    import os
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9


# ------------------------------------------------------------------ models
_loaded = {}


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load(name):
    """(model, processor) for `name`, downloaded once and kept loaded."""
    if name not in _loaded:
        import torch  # noqa: F401 - imported here: slow, and only this step needs it
        from transformers import (Sam2VideoModel, Sam2VideoProcessor, Sam3VideoModel,
                                  Sam3VideoProcessor)
        model_class, processor_class = ((Sam3VideoModel, Sam3VideoProcessor) if name == SAM3
                                        else (Sam2VideoModel, Sam2VideoProcessor))
        token = keys.find(TOKEN)[0] or None
        try:
            model = model_class.from_pretrained(name, cache_dir=CACHE, token=token)
            processor = processor_class.from_pretrained(name, cache_dir=CACHE, token=token)
        except OSError as failure:              # transformers wraps the hub's errors
            if not _gated(failure):
                raise
            raise SegmentError(
                f"{name} is gated on Hugging Face. " + (
                    f"Put {TOKEN}=<a read token> in handlab/.env." if token is None else
                    f"The {TOKEN} found was refused: its account needs access to "
                    f"{name} (approval can take a while to arrive), and a "
                    "fine-grained token needs read access to gated repos.")
            ) from failure
        _loaded[name] = (model.to(_device()).eval(), processor)
    return _loaded[name]


def _gated(failure):
    """Whether a failed download was refused for want of access."""
    from huggingface_hub.errors import GatedRepoError
    while failure is not None:
        if isinstance(failure, GatedRepoError) or "gated repo" in str(failure):
            return True
        failure = failure.__cause__ or failure.__context__
    return False


def _sam2(name, frames, asked, progress):
    """Box prompts; streamed forward and backward from each prompt frame.
    Objects whose boxes are on the same frame are tracked together."""
    import torch
    model, processor = _load(name)
    device = _device()
    count = len(frames)
    height, width = frames[0].shape[:2]
    packed = np.zeros((len(asked), count, height, (width + 7) // 8), np.uint8)
    groups = {}
    for prompt in asked:
        groups.setdefault(prompt.frame, []).append(prompt)
    total, done = sum(count + 1 for _ in groups), 0
    with torch.inference_mode():
        for start, group in groups.items():
            for order in (range(start, count), range(start, -1, -1)):
                session = processor.init_video_session(inference_device=device,
                                                       dtype=torch.float32)
                for step, index in enumerate(order):
                    inputs = processor(images=frames[index], device=device,
                                       return_tensors="pt")
                    if step == 0:
                        processor.add_inputs_to_inference_session(
                            inference_session=session, frame_idx=0,
                            obj_ids=[p.track + 1 for p in group],
                            input_boxes=[[list(p.box) for p in group]],
                            original_size=inputs.original_sizes[0])
                    output = model(inference_session=session, frame=inputs.pixel_values[0])
                    masks = processor.post_process_masks(
                        [output.pred_masks], original_sizes=inputs.original_sizes,
                        binarize=True)[0]
                    for obj_id, mask in zip(output.object_ids, masks):
                        packed[obj_id - 1, index] = np.packbits(
                            main_region(mask[0].cpu().numpy().astype(bool)), axis=-1)
                    done += 1
                    progress(done, total, f"{name.split('/')[-1]}: frame {index}")
    return packed, [""] * len(asked)


def _sam3(name, frames, asked, progress):
    """Text prompts (the object names); every instance tracked through the
    whole video, then per object the instance that best fits its box."""
    import torch
    model, processor = _load(name)
    count = len(frames)
    height, width = frames[0].shape[:2]
    candidates = {}                  # SAM 3 object id -> what it found
    with torch.inference_mode():
        session = processor.init_video_session(
            video=frames, inference_device=_device(), processing_device="cpu",
            video_storage_device="cpu", dtype=torch.float32)
        processor.add_text_prompt(session, sorted({p.name for p in asked}))
        for done, output in enumerate(model.propagate_in_video_iterator(
                inference_session=session), start=1):
            result = processor.postprocess_outputs(session, output)
            prompt_of = {obj: text for text, ids in result["prompt_to_obj_ids"].items()
                         for obj in ids}
            for obj, score, box, mask in zip(result["object_ids"].tolist(),
                                             result["scores"].tolist(),
                                             result["boxes"].tolist(), result["masks"]):
                found = candidates.setdefault(obj, {"name": prompt_of.get(obj),
                                                    "masks": {}, "boxes": {}, "score": 0.0})
                found["masks"][output.frame_idx] = np.packbits(
                    main_region(mask.cpu().numpy().astype(bool)), axis=-1)
                found["boxes"][output.frame_idx] = box
                found["score"] += score
            progress(done, count, f"sam3: frame {output.frame_idx}")
    packed = np.zeros((len(asked), count, height, (width + 7) // 8), np.uint8)
    notes = []
    for prompt in asked:
        pool = [c for c in candidates.values() if c["name"] == prompt.name]
        if not pool:
            notes.append(f"SAM 3 found no '{prompt.name}'")
            continue
        best = max(pool, key=lambda c: (_iou(c["boxes"].get(prompt.frame), prompt.box),
                                        c["score"]))
        for index, mask in best["masks"].items():
            packed[prompt.track, index] = mask
        notes.append("" if len(pool) == 1 else
                     f"{len(pool)} '{prompt.name}' found; kept the one at its box")
    return packed, notes


def _iou(a, b):
    """Overlap of two (x0, y0, x1, y1) boxes, 0..1; -1 if `a` is missing."""
    if a is None:
        return -1.0
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
