"""GRAB's hands -> their 21 points over time, as WiLoR gives a hand: people's
own motions handling objects, for src/learning/grab.py to put onto the
robot. Runs in .venv-wilor, which has MANO's loader (smplx, chumpy):

    ../.venv-wilor/bin/python wilor/grab.py

GRAB (Taheri et al., ECCV 2020; grab.is.tue.mpg.de, research use only): 10
people, 51 objects, 1,335 takes - lifting, passing, using, eating, drinking
- captured at 120 fps and fitted with SMPL-X and MANO. Its files in
data/external/grab (as GRAB's unzip script lays them out):

    grab/s<1-10>/<object>_<intent>[_n].npz   one take: each hand's MANO
                 parameters (wrist turn and place, the pose as 24 PCA
                 components - the full 45 equal them), the person's own hand
                 shape, the object's pose, contact labels
    tools/subject_meshes/<gender>/s<n>_<rhand|lhand>.ply   those shapes
    tools/object_meshes/contact_meshes/<object>.ply        the objects and
                 the table, in their own frames

Each hand through smplx's MANO as GRAB's own code does (the person's shape
as its template, a flat mean pose, 24 components), its fingertips from
smplx's MANO vertices and its points put in OpenPose order - WiLoR's
convention (its mano_wrapper), so a GRAB hand and a tracked one are one
thing. Every STEP-th frame is kept (RATE). Contact: GRAB labels every object
vertex with the body part touching it (its tools/utils.py contact_ids);
each hand's PARTS - palm, then each finger's three segments - touching the
object or not. And where the object and the table are, and which way the
person faces (SMPL-X's forward, +z of the body), with the meshes of every
object handled and the table - what a camera's view of the hands has in
front of them (src/learning/grab.py views()).

The takes are pickled dicts; they are read with an unpickler that lets in
only NumPy's array reconstruction (all they hold).

    data/learning/intent/grab_hands.npz
        joints (N, 2, 21, 3) m: right then left hand, GRAB's world axes
        (z up); contact (N, 2, 16) bool, PARTS; object_pose, table_pose
        (N, 6): turn (axis-angle) and place (m), a mesh's frame in the
        world; take (N,) - the take each frame is from (frames of a take
        in a row); subject, object, intent, name, facing (takes, 3): the
        person's forward at the take's first frame; rate; made
    data/learning/intent/grab_meshes.npz
        <object>_vertices (V, 3) m, <object>_faces (F, 3): each object
        handled, and the table
"""

import io
import pickle
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
import trimesh
from numpy.lib import format as npy
from smplx.vertex_ids import vertex_ids

ROOT = Path(__file__).resolve().parents[1]
GRAB = ROOT / "data" / "external" / "grab"
MANO = ROOT / "data" / "external" / "mano_v1_2" / "models"
OUT = ROOT / "data" / "learning" / "intent" / "grab_hands.npz"
MESHES = OUT.with_name("grab_meshes.npz")
SOURCE_RATE = 120
STEP = 6
RATE = SOURCE_RATE // STEP     # Hz
COMPONENTS = 24
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
HANDS = ("rhand", "lhand")
# GRAB's contact ids: hand 22 (right) / 21 (left); index, middle, pinky, ring,
# thumb segments 41-55 (right) / 26-40 (left), three each in that order.
PARTS = ("palm", "thumb1", "thumb2", "thumb3", "index1", "index2", "index3", "middle1", "middle2",
         "middle3", "ring1", "ring2", "ring3", "pinky1", "pinky2", "pinky3")
_SEGMENTS = {"index": 0, "middle": 3, "pinky": 6, "ring": 9, "thumb": 12}
CONTACT_IDS = {hand: np.array([palm] + [first + _SEGMENTS[p[:-1]] + int(p[-1]) - 1 for p in PARTS[1:]])
               for hand, palm, first in (("rhand", 22, 41), ("lhand", 21, 26))}
LABELS = 56                    # contact ids 0 (none) to 55
_ARRAYS = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}


def main():
    takes = sorted(GRAB.glob("grab/s*/*.npz"), key=lambda p: (int(p.parent.name[1:]), p.stem))
    models = {}
    out = {"joints": [], "contact": [], "object_pose": [], "table_pose": [], "take": []}
    meta = {"subject": [], "object": [], "intent": [], "name": [], "facing": []}
    meshes = {}
    started = time.monotonic()
    for index, path in enumerate(takes):
        take = read(path)
        keep = slice(0, None, STEP)
        joints = np.stack([_points(models, take[hand], keep) for hand in HANDS], 1)
        labels = take["contact"]["object"][keep]
        touched = np.zeros((len(labels), LABELS), bool)
        touched[np.arange(len(labels))[:, None], labels] = True       # body parts on the object
        contact = np.stack([touched[:, CONTACT_IDS[hand]] for hand in HANDS], 1)
        out["joints"].append(joints.astype(np.float32))
        out["contact"].append(contact)
        for key, thing, mesh in (("object_pose", take["object"], "object_mesh"),
                                 ("table_pose", take["table"], "table_mesh")):
            params = thing["params"]
            out[key].append(np.concatenate([params["global_orient"][keep], params["transl"][keep]],
                                           1).astype(np.float32))
            name = Path(thing[mesh]).stem
            if name not in meshes:
                shape = trimesh.load(GRAB / thing[mesh], process=False)
                meshes[name] = (shape.vertices.astype(np.float32), shape.faces.astype(np.int32))
        out["take"].append(np.full(len(joints), index, np.int16))
        turn, _ = cv2.Rodrigues(np.asarray(take["body"]["params"]["global_orient"][0], float))
        meta["facing"].append(turn[:, 2])
        for key, value in (("subject", take["sbj_id"]), ("object", take["obj_name"]),
                           ("intent", take["motion_intent"]), ("name", path.stem)):
            meta[key].append(str(value))
        if (index + 1) % 100 == 0:
            print(f"{index + 1} of {len(takes)} takes, {time.monotonic() - started:.0f} s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(OUT, **arrays, **{k: np.array(v) for k, v in meta.items()}, rate=np.array(RATE),
             made=np.array(datetime.now().isoformat(timespec="seconds")))
    np.savez_compressed(MESHES, **{f"{name}_{part}": array for name, (vertices, faces) in meshes.items()
                                   for part, array in (("vertices", vertices), ("faces", faces))})
    print(f"wrote {OUT}: {len(takes)} takes, {len(arrays['take'])} frames at {RATE} Hz; "
          f"{len(meshes)} meshes", flush=True)


def read(path):
    """A GRAB take: its arrays, the pickled ones as dicts."""
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            stream = io.BytesIO(archive.read(name))
            major, _ = npy.read_magic(stream)
            header = npy.read_array_header_1_0 if major == 1 else npy.read_array_header_2_0
            shape, fortran, dtype = header(stream)
            if dtype.hasobject:
                value = _Arrays(stream, encoding="latin1").load()
            else:
                value = np.frombuffer(stream.read(), dtype).reshape(shape, order="F" if fortran else "C")
            out[name.removesuffix(".npy")] = value.item() if value.shape == () else value
    return out


class _Arrays(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ARRAYS:
            raise pickle.UnpicklingError(f"GRAB file wants {module}.{name}")
        return super().find_class(module, name)


@torch.no_grad()
def _points(models, hand, keep):
    """One hand's 21 points (T, 21, 3) m, OpenPose order."""
    params = {k: torch.as_tensor(hand["params"][k][keep], dtype=torch.float32)
              for k in ("global_orient", "hand_pose", "transl")}
    model = _model(models, hand["vtemp"])
    out = model(betas=torch.zeros(len(params["transl"]), model.num_betas), **params)
    tips = out.vertices[:, list(vertex_ids["mano"].values())]
    return torch.cat([out.joints, tips], 1)[:, MANO_TO_OPENPOSE].numpy()


def _model(models, vtemp):
    """smplx's MANO with a person's own hand shape (the shape is in it, so
    no shape coefficients)."""
    if vtemp not in models:
        shape = trimesh.load(GRAB / vtemp, process=False).vertices
        models[vtemp] = smplx.MANO(str(MANO), is_rhand="rhand" in Path(vtemp).stem, v_template=shape,
                                   num_pca_comps=COMPONENTS, flat_hand_mean=True)
    return models[vtemp]


if __name__ == "__main__":
    main()
