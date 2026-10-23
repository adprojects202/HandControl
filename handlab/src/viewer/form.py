"""Settings-backed input widgets for the side panel's sections.

Every edit goes through `src/settings.py` - checked against its type, range
and choices, saved to settings.json - and then `on_saved()` applies it. A
rejected value leaves the settings untouched and is shown in red.

A typed number is committed when its field loses focus or Enter is pressed,
never per keystroke: typing 250 must not send P=2 and then P=25 to sixteen
motors. A slider applies live while it is dragged (checked, in memory) and is
written to settings.json once, on release.

`field` and `slider` are bound to a setting by its key, and `index` picks one
element of a list setting such as object.position_mm. `text` is a typed
string (a model name, a prompt), committed the same way as a number.
"""

from __future__ import annotations

from mujoco.experimental.dear_imgui import dear_imgui as imgui

from .. import settings

RED = imgui.Vec4(0.95, 0.35, 0.35, 1.0)
GREEN = imgui.Vec4(0.45, 0.85, 0.45, 1.0)


class SettingsForm:
    def __init__(self, on_saved):
        self.on_saved = on_saved
        self.message = ""            # the last rejection, until the next good save
        self._pending = {}           # field id -> value being typed

    def save(self, key, value):
        """Check, save, apply. Returns whether it was accepted."""
        try:
            settings.set(key, value)
        except (KeyError, ValueError) as failure:
            self.message = str(failure)
            return False
        settings.save()
        self.message = ""
        self.on_saved()
        return True

    def number(self, field_id, value, commit, fmt="%.4g"):
        """A number field showing `value`; `commit(new)` on Enter or focus loss.
        Integer or float by the type of `value`."""
        shown = self._pending.get(field_id, value)
        if isinstance(value, int):
            changed, typed = imgui.InputInt(f"##{field_id}", shown, 0, 0)
        else:
            changed, typed = imgui.InputFloat(f"##{field_id}", shown, 0.0, 0.0, fmt)
        if changed:
            self._pending[field_id] = typed
        if imgui.IsItemDeactivatedAfterEdit():
            final = self._pending.pop(field_id, typed)
            if final != value:
                commit(final)

    def field(self, key, index=None, fmt="%.4g"):
        """A typed number bound to a setting, saved on Enter or focus loss."""
        value = _get(key, index)
        self.number(_id(key, index), value,
                    lambda v: self.save(key, _with(key, index, v)), fmt=fmt)

    def text(self, key, lines=1):
        """A text setting, saved on focus loss (or Enter, for one line).
        With `lines` > 1, a wrapped box that tall; Enter is a new line."""
        value = settings.get(key)
        shown = self._pending.get(key, value)
        if lines > 1:
            size = imgui.Vec2(-1, imgui.GetTextLineHeight() * lines
                              + 2 * imgui.GetStyle().FramePadding.y)
            changed, typed = imgui.InputTextMultiline(
                f"##{key}", shown, size, int(imgui.InputTextFlags.WordWrap))
        else:
            changed, typed = imgui.InputText(f"##{key}", shown)
        if changed:
            self._pending[key] = typed
        if imgui.IsItemDeactivatedAfterEdit():
            final = self._pending.pop(key, typed)
            if final != value:
                self.save(key, final)

    def slider(self, key, low, high, index=None, fmt="%.0f"):
        """A slider bound to a setting: applied live while dragged, saved to
        settings.json on release."""
        value = float(_get(key, index))
        changed, new = imgui.SliderFloat(f"##{_id(key, index)}", value, low, high, fmt)
        if changed:
            try:
                settings.set(key, _with(key, index, new))
            except (KeyError, ValueError) as failure:
                self.message = str(failure)
            else:
                self.on_saved()
        if imgui.IsItemDeactivatedAfterEdit():
            settings.save()

    def choice(self, key, show=str):
        """A dropdown over the setting's CHOICES; saves on pick."""
        value = settings.get(key)
        if imgui.BeginCombo(f"##{key}", show(value)):
            for option in settings.choices(key):
                if imgui.Selectable(show(option), option == value) and option != value:
                    self.save(key, option)
            imgui.EndCombo()

    def draw_message(self):
        if self.message:
            imgui.PushTextWrapPos(0.0)
            imgui.TextColored(RED, self.message)
            imgui.PopTextWrapPos()


def table(name):
    """Begin a two-column table of labelled rows (see `row`); End it with
    imgui.EndTable() when this returns True."""
    if not imgui.BeginTable(name, 2, int(imgui.TableFlags.SizingStretchProp)):
        return False
    imgui.TableSetupColumn("name", int(imgui.TableColumnFlags.WidthFixed))
    imgui.TableSetupColumn("value", int(imgui.TableColumnFlags.WidthStretch))
    return True


def row(text, tip, fill=True):
    """A table row: the label (with its tooltip) in the first column, then
    the value cell - with `fill`, ready for one widget as wide as the cell."""
    imgui.TableNextRow()
    imgui.TableNextColumn()
    imgui.AlignTextToFramePadding()
    imgui.Text(text)
    imgui.SetItemTooltip(tip)
    imgui.TableNextColumn()
    if fill:
        imgui.SetNextItemWidth(-1)


def _get(key, index):
    value = settings.get(key)
    return value if index is None else value[index]


def _with(key, index, element):
    """The setting's whole value with one element replaced (or the value)."""
    if index is None:
        return element
    value = settings.get(key)
    value[index] = element
    return value


def _id(key, index):
    return key if index is None else f"{key}[{index}]"
