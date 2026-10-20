"""An RGB image in a panel, as wide as the panel: the camera preview, a
recording's frame.

GUI textures belong to Studio's renderer, which is (re)made for each new
model in `viewer.sync()` - after the GUI pass, in the same frame. An upload
before the first one exists crashes the process, and a texture from before a
remake is not relied on after it. So in the frame a model arrives (the first
one included) the image waits, and the frame after starts a new texture.
An image of another size is uploaded into the same texture, which takes the
new size. (Freeing a texture - an upload of no bytes - crashes in MuJoCo
3.12, so it is never done; each GuiImage keeps its one.)
"""

from __future__ import annotations

import cv2
from mujoco.experimental.dear_imgui import dear_imgui as imgui

BYTES_PER_PIXEL = 4      # the GUI texture is RGBA


class GuiImage:
    def __init__(self):
        self._model = None       # the viewer's model the texture was made under
        self._texture = 0
        self._shown = None       # the array on the texture

    def draw(self, viewer, rgb):
        """Show `rgb` (H x W x 3 uint8). Pass the same array again to show it
        again without another upload."""
        if viewer.model is not self._model:       # see the module docstring
            self._model = viewer.model
            self._texture, self._shown = 0, None
            return
        height, width = rgb.shape[:2]
        if rgb is not self._shown:
            rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
            self._texture = viewer.upload_image(self._texture, rgba.tobytes(),
                                                width, height, BYTES_PER_PIXEL)
            self._shown = rgb
        shown = imgui.GetContentRegionAvail().x
        imgui.Image(self._texture, imgui.Vec2(shown, shown * height / width))
