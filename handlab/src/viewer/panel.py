"""A side panel in the Studio window, made of collapsible sections.

    SidePanel("Robot", [JointsSection(...), HardwareSection(...)])

docks a "Robot" window on the right of the viewer (as a tab beside Studio's
Inspector; `beside="Options"` puts one on the left). Panels only ever join
Studio's side panes, so nothing is drawn over the 3D view. Each section is
any object with a `title` and a `draw()` that makes ImGui calls; it shows as
a collapsible sub-menu, in order. A section that needs the viewer itself (to
upload an image, say) also has `attach(viewer)`, called once at startup. The
menu bar gets a menu named after the panel with one entry per section to show
or hide it.
"""

from __future__ import annotations

from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.studio import messages, viewer_protocol

# Below Studio's own GUI handlers (priority 1), so its dock layout exists
# by the time this window asks to join it.
AFTER_STUDIO = 0


class SidePanel:
    def __init__(self, title, sections, beside="Inspector"):
        self.title = title
        self.sections = list(sections)
        self.beside = beside
        self.shown = {section.title: True for section in self.sections}
        self._docked = False

    @messages.handler(priority=AFTER_STUDIO)
    def _on_init(self, event: viewer_protocol.ViewerInitEvent) -> None:
        for section in self.sections:
            if hasattr(section, "attach"):
                section.attach(event.viewer)

    @messages.handler(priority=AFTER_STUDIO)
    def _on_build_gui(self, _: messages.BuildGuiEvent) -> None:
        self._menu()
        if not any(self.shown.values()):
            return
        if not self._docked:
            self._dock_beside(self.beside)
        visible, keep_open = imgui.Begin(self.title, True)
        if visible:
            for section in self.sections:
                if not self.shown[section.title]:
                    continue
                if imgui.CollapsingHeader(section.title,
                                          int(imgui.TreeNodeFlags.DefaultOpen)):
                    imgui.PushID(section.title)
                    section.draw()
                    imgui.PopID()
        imgui.End()
        if not keep_open:
            self.shown = dict.fromkeys(self.shown, False)

    def _menu(self):
        if imgui.BeginMainMenuBar():
            if imgui.BeginMenu(self.title):
                for title, shown in self.shown.items():
                    if imgui.MenuItem(title, "", shown):
                        self.shown[title] = not shown
                imgui.EndMenu()
            imgui.EndMainMenuBar()

    def _dock_beside(self, window):
        """Join `window`'s dock node as a tab, once, and take focus."""
        imgui.Begin(window)             # appends nothing; just asks its dock id
        dock = imgui.GetWindowDockID()
        imgui.End()
        if dock:
            imgui.SetNextWindowDockID(dock, int(imgui.Cond.FirstUseEver))
            imgui.SetNextWindowFocus()
            self._docked = True
