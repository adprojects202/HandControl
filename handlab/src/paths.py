"""Where things live. Every path is relative to the project folder, so the
folder can be copied anywhere and still run."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "settings.json"


def resolve(path):
    """A path from settings.json: relative ones are relative to ROOT."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path
