"""World: how the hand is mounted, and the calibration object.

Edits are saved to settings.json through the settings form and applied to the
running world in place (`World.apply`) - instantly, no reload - except
showing, hiding or resizing the object, which rebuild the world and show as a
brief reload (see LIVE EDITS in world.py for why).
"""

from __future__ import annotations

from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings
from .form import SettingsForm, row, table

# Typed in: (setting, list index, label, unit, tooltip).
SIZE_AND_WEIGHT = (
    ("object.size_mm", 1, "width", "mm", "across the palm (hand y)"),
    ("object.size_mm", 0, "depth", "mm", "out of the palm (hand x)"),
    ("object.size_mm", 2, "height", "mm", "along the fingers (hand z)"),
    ("object.mass_g", None, "weight", "g", "matters when floating"),
)
FRICTION_TIP = ("sliding friction of the object against the hand and the floor\n"
                "PLA = 0.5, an estimate: published FDM-PLA values are 0.35-0.55,\n"
                "measured against steel. Tilt test on the real parts:\n"
                "mu = tan(angle at which the object starts to slide)")
# Dragged: (list index, label, tooltip) of object.position_mm, in the hand frame.
POSITION = (
    (0, "out of palm", "centre's distance out from the palm (hand x)"),
    (1, "across palm", "centre's offset across the palm (hand y)"),
    (2, "along fingers", "centre's height along the fingers (hand z)"),
)
MODES = {"fixed": "held still where it is placed",
         "floating": "free: falls, can be pushed, rests on the palm or the floor"}


class WorldSection:
    title = "World"

    def __init__(self, apply_world, reset_object):
        """`apply_world()` takes the saved settings into the running world;
        `reset_object()` puts the object back where it is placed."""
        self.reset_object = reset_object
        self.form = SettingsForm(apply_world)

    def draw(self):
        if not table("world"):
            return
        row("hand", "which way gravity pulls on the hand - set it to "
                            "match the real one on the bench")
        self.form.choice("world.hand_orientation")
        imgui.EndTable()

        imgui.SeparatorText("calibration object")
        changed, shown = imgui.Checkbox("show", settings.get("object.enabled"))
        imgui.SetItemTooltip("adding or removing it reloads the scene")
        if changed:
            self.form.save("object.enabled", shown)
        if shown:
            imgui.SameLine()
            if imgui.Button("Reset object"):
                self.reset_object()
            imgui.SetItemTooltip("put it back where it is placed, at rest")
            self._object()
        self.form.draw_message()

    def _object(self):
        if not table("object"):
            return
        row("mode", "\n".join(f"{k}: {v}" for k, v in MODES.items()))
        self.form.choice("object.mode")
        row("training", "where Send to sim (Recordings, step 5) puts a recording's object:\n"
                        "fixed pose - its motion, centred on the position below\n"
                        "from the hand - where the person held it relative to their palm, "
                        "matched to the robot's palm (along the palm scaled to the robot's "
                        "size, out of the palm palm-surface to palm-surface)")
        self.form.choice("object.training")
        for key, index, label, unit, tip in SIZE_AND_WEIGHT:
            low, high = settings.limits(key)
            row(f"{label} ({unit})", f"{tip}\n{low:g} - {high:g} {unit}"
                        + ("\nchanging the size reloads the scene" if unit == "mm" else ""))
            self.form.field(key, index, fmt="%.1f")
        row("friction", FRICTION_TIP, fill=False)
        if imgui.Button("PLA"):
            self.form.save("object.friction", settings.DEFAULTS["object"]["friction"])
        imgui.SetItemTooltip("back to the PLA default, 0.5")
        imgui.SameLine()
        imgui.SetNextItemWidth(-1)
        low, high = settings.limits("object.friction")
        self.form.slider("object.friction", low, high, fmt="mu %.2f")
        imgui.EndTable()

        imgui.TextDisabled("position (mm, in the hand frame)")
        if not table("position"):
            return
        low, high = settings.limits("object.position_mm")
        for index, label, tip in POSITION:
            row(label, f"{tip}\n{low:g} - {high:g} mm; ctrl-click to type")
            self.form.slider("object.position_mm", low, high, index=index, fmt="%.0f mm")
        imgui.EndTable()
