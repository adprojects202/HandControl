"""Permission to use the camera.

macOS asks the user once whether an app may use the camera. For `python
main.py` that app is the one python runs under (Terminal, VS Code, ...),
and the prompt names it. OpenCV can only ask from the main thread, which the
window owns, so the asking is done here, through AVFoundation, which can ask
from any thread. Other systems have nothing to ask.
"""

from __future__ import annotations

import sys

GRANTED, WAITING, REFUSED = "granted", "waiting", "refused"
HOW_TO_ALLOW = ("camera access is off for the app that runs python - turn it on "
                "in System Settings > Privacy & Security > Camera, then Retry")

if sys.platform == "darwin":
    import AVFoundation as av

_asked = False


def camera_access():
    """GRANTED, REFUSED, or WAITING while the user has not answered. The first
    call that finds no answer shows the macOS prompt."""
    global _asked
    if sys.platform != "darwin":
        return GRANTED
    status = av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeVideo)
    if status == av.AVAuthorizationStatusAuthorized:
        return GRANTED
    if status == av.AVAuthorizationStatusNotDetermined:
        if not _asked:
            # The answer is read back with authorizationStatus; the callback
            # only has to exist.
            av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                av.AVMediaTypeVideo, lambda granted: None)
            _asked = True
        return WAITING
    return REFUSED       # denied, or restricted by a device policy
