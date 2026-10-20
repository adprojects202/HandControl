"""Which cameras there are, by name, in OpenCV's numbering.

OpenCV opens a camera by number. On macOS its numbers are positions in
AVFoundation's `devicesWithMediaType:` list (video devices, then muxed ones;
see OpenCV's cap_avfoundation_mac.mm), so the same query gives each number
its name. Listing needs no camera permission and shows no prompt.

Cameras missing from that list - an iPhone as Continuity Camera, Desk View -
cannot be opened by number at all. A USB camera only appears once macOS has
let it connect (on Apple silicon: the "Allow accessory to connect?" prompt).
Other systems have no names: `camera_names()` returns None there.
"""

from __future__ import annotations

import sys

if sys.platform == "darwin":
    import AVFoundation as av


def camera_names():
    """Camera names in OpenCV's order (list index = camera number), or None
    where names cannot be known."""
    if sys.platform != "darwin":
        return None
    devices = (list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo))
               + list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeMuxed)))
    return [str(device.localizedName()) for device in devices]
