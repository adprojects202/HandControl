"""WiLoR (CVPR 2025, Potamias et al.) - a 3D hand from one RGB image - set up
for this Mac. It runs in its own environment, HandControl/.venv-wilor (its
MANO loader, chumpy, needs numpy < 1.24, which the app's cannot have), from
the official code and weights in data/external/wilor_code, with MANO (from
mano.is.tue.mpg.de, after registering) in data/external/mano_v1_2.
Licences: WiLoR CC-BY-NC-ND, MANO non-commercial - research use.

    from model import load, detect, hand
    wilor, cfg, detector = load()               # backbone on the Apple GPU
    found = detect(detector, bgr)               # [(confidence, is_right, box)]
    joints, uv = hand(wilor, cfg, bgr, box, is_right)   # (21, 3) m, (21, 2) px

Set up as WiLoR's demo does, but without its renderer (it draws training
visuals with an OpenGL macOS lacks: PYOPENGL_PLATFORM=darwin is set before
WiLoR is imported) and with these fixes, each measured on FreiHAND:

  - PyTorch 2.5.1: 2.6 and later refuse the detector's checkpoint
    (ultralytics 8.1.34 predates their safer loading default)
  - the refinement head on the CPU: it samples the feature map at the hand
    mesh's projected points with F.grid_sample, and those points sit at the
    map's edge, where MPS interpolates wrongly - the hand came out ~100 px
    from where it is in the image (4.5 px on the CPU), its shape still
    right. Forward hooks move the head's inputs to the CPU and its outputs
    back; the backbone stays on the GPU: 53 ms a hand, against 255 ms all
    on the CPU
  - the hand detector (a YOLOv8 pose model) on the CPU: ultralytics warns of
    a known MPS bug in pose models; ~60 ms a frame
  - inputs cast to float32 (MPS has no float64)

Joints come in OpenPose order (MediaPipe's: wrist, then thumb, index,
middle, ring, little, base to tip), camera axes, metres, a left hand as it
is (WiLoR mirrors it to reconstruct).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "data" / "external" / "wilor_code"
os.environ.setdefault("PYOPENGL_PLATFORM", "darwin")
sys.path.insert(0, str(CODE))

from wilor.configs import get_config  # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from wilor.models import WiLoR  # noqa: E402
from wilor.utils.renderer import cam_crop_to_full  # noqa: E402

DETECT_CONFIDENCE = 0.3
RESCALE = 2.0                  # the crop around a detected box, as WiLoR's demo


def load(device="mps"):
    """(WiLoR on `device`, its config, the hand detector on the CPU)."""
    from ultralytics import YOLO

    cfg = get_config(str(CODE / "pretrained_models" / "model_config.yaml"), update_cachedir=True)
    cfg.defrost()
    if "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    mano = str(CODE / "mano_data") + "/"
    cfg.MANO.DATA_DIR = cfg.MANO.MODEL_PATH = mano
    cfg.MANO.MEAN_PARAMS = mano + "mano_mean_params.npz"
    cfg.freeze()
    model = WiLoR.load_from_checkpoint(str(CODE / "pretrained_models" / "wilor_final.ckpt"),
                                       strict=False, cfg=cfg, init_renderer=False,
                                       map_location="cpu").to(device).eval()
    if torch.device(device).type == "mps":
        head = model.refine_net.to("cpu")
        head.register_forward_pre_hook(
            lambda m, args, kwargs: (_moved(args, "cpu"), _moved(kwargs, "cpu")), with_kwargs=True)
        head.register_forward_hook(lambda m, args, out: _moved(out, device))
    return model, cfg, YOLO(str(CODE / "pretrained_models" / "detector.pt"))


def detect(detector, bgr):
    """The hands in a BGR image: [(confidence, is_right, box (4,) x0 y0 x1 y1)]."""
    found = detector(bgr, conf=DETECT_CONFIDENCE, verbose=False, device="cpu")[0]
    return [(float(d.boxes.conf.item()), float(d.boxes.cls.item()), d.boxes.xyxy[0].cpu().numpy())
            for d in found]


def hand(model, cfg, bgr, box, is_right):
    """The hand in `box`: (joints (21, 3) m in camera axes, their pixels (21, 2))."""
    device = next(model.backbone.parameters()).device
    dataset = ViTDetDataset(cfg, bgr, np.asarray(box, float)[None], np.array([float(is_right)]),
                            rescale_factor=RESCALE)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    batch = {k: (v.float() if v.is_floating_point() else v).to(device) if torch.is_tensor(v) else v
             for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch)
    flip = 2 * float(is_right) - 1
    joints = out["pred_keypoints_3d"][0].cpu().numpy()
    joints[:, 0] *= flip
    cam = out["pred_cam"].clone()
    cam[:, 1] *= flip
    size = batch["img_size"].float()
    focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * size.max()
    translation = cam_crop_to_full(cam, batch["box_center"].float(), batch["box_size"].float(),
                                   size, focal)[0].cpu().numpy()
    points = joints + translation
    height, width = bgr.shape[:2]
    uv = points[:, :2] / points[:, 2:] * float(focal) + np.array([width / 2, height / 2])
    return points, uv


def _moved(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _moved(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_moved(v, device) for v in value)
    return value
