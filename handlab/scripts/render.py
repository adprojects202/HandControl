"""A picture of the world without opening a window.

    python scripts/render.py        -> renders/world.png

The same World main.py opens, with the command hand in a sample pose (index
curled, thumb across) and the encoder hand at zero, from the viewer's camera.
Drawn with MuJoCo's classic renderer, not Studio's, so colours and lighting
differ a little from the window.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.paths import ROOT  # noqa: E402
from src.viewer.studio import CAMERA  # noqa: E402
from src.world import World  # noqa: E402


def main():
    world = World()
    pose = np.zeros(len(world.robot))
    for name, angle in (("index_proximal_joint", 0.9), ("index_middle_joint", 0.9),
                        ("index_distal_joint", 0.9), ("thumb_proximal_joint", 1.0)):
        if name in world.robot.names:
            pose[world.robot.names.index(name)] = angle
    world.set_pose("command", pose)
    mujoco.mj_forward(world.model, world.data)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = world.hand_center()
    camera.distance, camera.azimuth, camera.elevation = (
        CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"])
    with mujoco.Renderer(world.model, 720, 1080) as renderer:
        renderer.update_scene(world.data, camera)
        image = renderer.render()

    out = ROOT / "renders" / "world.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(image).save(out)
    print(out)


if __name__ == "__main__":
    main()
