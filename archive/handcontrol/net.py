"""A UDP line between the pygame panel and the simulator process.

They are separate processes because they have to be: on macOS neither MuJoCo's
viewer nor pygame's window will run off the main thread, and there is only one
main thread. UDP rather than a pipe because dropping a stale slider position is
the correct behaviour - only the newest one matters.
"""

from __future__ import annotations

import json
import socket

HOST = "127.0.0.1"
COMMAND_PORT = 9870          # panel -> sim
STATE_PORT = 9871            # sim -> panel
KEYPOINT_PORT = 9872         # sim -> keyvector bridge
SOLUTION_PORT = 9873         # bridge -> sim


class Channel:
    """Send JSON one way, receive JSON the other. Never blocks."""

    def __init__(self, bind=None, send_to=None, host=HOST):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        if bind is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.socket.bind((host, bind))
            except OSError as error:
                # Almost always a previous run that is still up. Two of them
                # sharing the port would be worse than refusing to start, so
                # say which port and move the decision to the operator.
                raise SystemExit(
                    f"UDP port {bind} is already in use - another handcontrol "
                    f"is probably still running.\n"
                    f"  lsof -i :{bind}      to see what holds it\n"
                    f"  --command-port / --state-port    to use different ones"
                ) from error
        self.target = (host, send_to) if send_to is not None else None

    def send(self, message):
        if self.target is None:
            return
        try:
            self.socket.sendto(json.dumps(message).encode(), self.target)
        except OSError:
            pass          # the other end is not up yet; nothing worth doing

    def drain(self):
        """Every message waiting, oldest first."""
        messages = []
        while True:
            try:
                data, _ = self.socket.recvfrom(65535)
            except (BlockingIOError, OSError):
                return messages
            try:
                messages.append(json.loads(data.decode()))
            except (ValueError, UnicodeDecodeError):
                continue

    def latest(self, kind=None):
        """The newest message, or the newest of one `type`. Older ones are dropped."""
        found = None
        for message in self.drain():
            if kind is None or message.get("type") == kind:
                found = message
        return found

    def close(self):
        self.socket.close()
