"""Secrets: API keys and tokens. Never settings, never written by the app.

    keys.find("HF_TOKEN")            # -> (value, where it came from) or ("", "")

A secret is looked for in the environment python started with, then in
handlab/.env - one NAME=value per line, gitignored. The file is re-read when
it changes, so a key added there needs no restart. A line with no NAME= in
front is reported (`Secrets.problem`) without ever showing its text, which
may be a bare key.
"""

from __future__ import annotations

import os

from dotenv import dotenv_values

from . import paths

KEY_FILE = paths.ROOT / ".env"


class Secrets:
    def __init__(self, environ=None, key_file=KEY_FILE):
        self.environ = os.environ if environ is None else environ
        self.key_file = key_file
        self.problem = ""            # what is wrong with the file - never its contents
        self._values = {}
        self._stamp = None

    def find(self, name):
        """(value, source) - source "$NAME" or the file's name - or ("", "")."""
        for value, source in ((self.environ.get(name), f"${name}"),
                              (self._read().get(name), self.key_file.name)):
            if value and value.strip():
                return value.strip(), source
        return "", ""

    def _read(self):
        try:
            stat = self.key_file.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            stamp = None
        if stamp != self._stamp:
            self._stamp = stamp
            values = dotenv_values(self.key_file) if stamp else {}
            self._values = {name: value for name, value in values.items() if value}
            # A line with no "=" parses as a name with no value. Its text may
            # be a bare key, so say what is wrong without showing it.
            self.problem = (
                f"{self.key_file.name}: a line has no NAME= in front, so it is "
                "ignored - write it as NAME=value, e.g. ANTHROPIC_API_KEY=..."
                if any(value is None for value in values.values()) else "")
        return self._values


_default = None


def default():
    """The app's Secrets: the real environment and handlab/.env."""
    global _default
    if _default is None:
        _default = Secrets()
    return _default


def find(name):
    return default().find(name)
