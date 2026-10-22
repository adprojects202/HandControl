"""The hands step: each of the person's hands followed as one hand through the
whole recording -> hands_clean.npz (layout in recording/episode.py).

    clean_hands(episode, progress)

MediaPipe works frame by frame and gets three things wrong that matter
later (all seen in the first recording):

  - it loses a hand hidden behind the object it holds - for 23 and 34
    frames there - and finds it again where it was;
  - it flips left/right when a hand turns edge-on: the right hand was
    labelled left for 33 frames in a row;
  - it finds a phantom second "hand" on the fingers of the first, and is
    surer of it (0.99) than of the real one (0.86-0.97).

So here a label is only a vote, a score proves nothing, and a hand is what
moves continuously:

  1. Each hand is followed by where its palm goes, nearest first (TRACK_GATE
     a frame, growing by GATE_GROWTH for each frame it is unseen, for up to
     MAX_LOST frames - a hand rarely goes far while hidden).
  2. A track that mostly sits inside a longer one (their landmark boxes
     overlap by MERGE_OVERLAP in most shared frames) is a phantom: dropped.
     The real hand is the one that continues its track, whatever the score.
  3. A followed hand's side is the score-weighted vote of all its labels.
     Two at once that vote the same side: the one with the weaker vote
     takes the other side.
  4. Tracks shorter than MIN_TRACK frames are dropped (flickers).
  5. Gaps up to MAX_GAP frames are filled by interpolation (flagged
     `filled`); each run is smoothed (Savitzky-Golay, SMOOTH_WINDOW frames).

hands.npz stays as recorded.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.signal import savgol_filter

from ..vision.tracker import SIDES

PALM = [0, 5, 9, 13, 17]          # wrist and the four knuckles
MERGE_OVERLAP = 0.6               # share of the smaller box inside the larger
TRACK_GATE = 0.15                 # palm moves at most this much (of the image) a frame
GATE_GROWTH = 0.01                # ... plus this per frame unseen
MAX_LOST = 60                     # frames a track waits for its hand (2 s)
MIN_TRACK = 5                     # frames; shorter tracks are flickers
MAX_GAP = 5                       # frames filled by interpolation
SMOOTH_WINDOW, SMOOTH_ORDER = 7, 2


def clean_hands(episode, progress=lambda done, total, text="": None):
    """Clean the episode's hands and save hands_clean.npz; returns a summary."""
    raw = episode.hands
    if not raw:
        raise ValueError("the recording has no hands.npz")
    count = len(raw["present"])
    progress(0, 3, "following hands")
    tracks = _follow(_detections(raw))
    progress(1, 3, "dropping phantoms")
    tracks, phantoms = _drop_phantoms(tracks)
    tracks = [t for t in tracks if len(t["frames"]) >= MIN_TRACK]
    sides = _sides(tracks)
    progress(2, 3, "filling and smoothing")
    out = _assemble(tracks, sides, count)
    relabelled = int(sum(label != sides[n] for n, t in enumerate(tracks)
                         for label in t["labels"]))
    summary = {"tracks": len(tracks), "relabelled": relabelled, "phantoms": phantoms,
               "filled_frames": int(out["filled"].sum())}
    episode.save_arrays("hands_clean.npz", dict(
        out, track_side=np.array([sides[n] for n in range(len(tracks))]),
        track_frames=np.array([len(t["frames"]) for t in tracks]),
        made=np.array(datetime.now().isoformat(timespec="seconds")),
        **{key: np.array(value) for key, value in summary.items()}))
    progress(3, 3, "done")
    return summary


def _detections(raw):
    """Per frame, the detections as dicts."""
    return [[{"image": raw["image"][index, slot], "world": raw["world"][index, slot],
              "score": float(raw["score"][index, slot]), "label": SIDES[slot]}
             for slot in range(len(SIDES)) if raw["present"][index, slot]]
            for index in range(len(raw["present"]))]


def _drop_phantoms(tracks):
    """Tracks that mostly sit inside a longer one, dropped; and how many
    detections went with them."""
    kept, dropped = [], 0
    for track in sorted(tracks, key=lambda t: -len(t["frames"])):
        frames = dict(zip(track["frames"], track["found"]))
        phantom = False
        for other in kept:
            shared = [(frames[i], d) for i, d in zip(other["frames"], other["found"])
                      if i in frames]
            inside = sum(_overlap(a["image"], b["image"]) > MERGE_OVERLAP
                         for a, b in shared)
            if inside > len(shared) / 2 and len(shared) > len(frames) / 2:
                phantom = True
                break
        if phantom:
            dropped += len(track["frames"])
        else:
            kept.append(track)
    kept.sort(key=lambda t: t["frames"][0])
    return kept, dropped


def _overlap(a, b):
    """Share of the smaller landmark box (x, y) that lies inside the other."""
    (ax0, ay0), (ax1, ay1) = a[:, :2].min(0), a[:, :2].max(0)
    (bx0, by0), (bx1, by1) = b[:, :2].min(0), b[:, :2].max(0)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return inter / smaller if smaller > 0 else 0.0


def _palm(detection):
    return detection["image"][PALM, :2].mean(0)


def _follow(frames):
    """Tracks: each a dict of frames, detections and labels, in time order."""
    tracks, active = [], []
    for index, found in enumerate(frames):
        active = [t for t in active if index - t["frames"][-1] <= MAX_LOST]
        pairs = sorted(
            (float(np.linalg.norm(_palm(d) - _palm(t["found"][-1]))), n, m)
            for n, d in enumerate(found) for m, t in enumerate(active))
        taken_d, taken_t = set(), set()
        for distance, n, m in pairs:
            track = active[m]
            gate = TRACK_GATE + GATE_GROWTH * (index - track["frames"][-1] - 1)
            if n in taken_d or m in taken_t or distance > gate:
                continue
            taken_d.add(n)
            taken_t.add(m)
            _extend(track, index, found[n])
        for n, detection in enumerate(found):
            if n not in taken_d:
                track = {"frames": [], "found": [], "labels": [], "scores": []}
                _extend(track, index, detection)
                tracks.append(track)
                active.append(track)
    return tracks


def _extend(track, index, detection):
    track["frames"].append(index)
    track["found"].append(detection)
    track["labels"].append(detection["label"])
    track["scores"].append(detection["score"])


def _sides(tracks):
    """Each track's side by weighted vote; overlapping same-side tracks split."""
    votes = []
    for track in tracks:
        weight = {side: sum(s for label, s in zip(track["labels"], track["scores"])
                            if label == side) for side in SIDES}
        total = sum(weight.values()) or 1.0
        best = max(SIDES, key=weight.get)
        votes.append((best, weight[best] / total))
    sides = {n: side for n, (side, _) in enumerate(votes)}
    order = sorted(range(len(tracks)), key=lambda n: -votes[n][1])
    for position, n in enumerate(order):
        span = set(tracks[n]["frames"])
        for m in order[:position]:
            if sides[m] == sides[n] and span & set(tracks[m]["frames"]):
                sides[n] = SIDES[1 - SIDES.index(sides[n])]
                break
    return sides


def _assemble(tracks, sides, count):
    shape = (count, len(SIDES), 21, 3)
    out = {"image": np.full(shape, np.nan, np.float32),
           "world": np.full(shape, np.nan, np.float32),
           "score": np.full((count, len(SIDES)), np.nan, np.float32),
           "observed": np.zeros((count, len(SIDES)), bool),
           "filled": np.zeros((count, len(SIDES)), bool),
           "track": np.full((count, len(SIDES)), -1, np.int32)}
    for n, track in enumerate(tracks):
        slot = SIDES.index(sides[n])
        frames = np.array(track["frames"])
        for key in ("image", "world"):
            values = np.stack([d[key] for d in track["found"]])
            out[key][frames, slot] = values
        out["score"][frames, slot] = track["scores"]
        out["observed"][frames, slot] = True
        out["track"][frames, slot] = n
        for start, end in zip(frames[:-1], frames[1:]):         # fill short gaps
            if 1 < end - start <= MAX_GAP + 1:
                fill = np.arange(start + 1, end)
                weight = ((fill - start) / (end - start))[:, None, None]
                for key in ("image", "world"):
                    out[key][fill, slot] = ((1 - weight) * out[key][start, slot]
                                            + weight * out[key][end, slot])
                out["score"][fill, slot] = np.nan
                out["filled"][fill, slot] = True
                out["track"][fill, slot] = n
    out["present"] = out["observed"] | out["filled"]
    for slot in range(len(SIDES)):                              # smooth each run
        for start, end in _runs(out["present"][:, slot]):
            if end - start >= SMOOTH_WINDOW:
                for key in ("image", "world"):
                    out[key][start:end, slot] = savgol_filter(
                        out[key][start:end, slot], SMOOTH_WINDOW, SMOOTH_ORDER, axis=0)
    return out


def _runs(mask):
    """(start, end) of each run of True."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
