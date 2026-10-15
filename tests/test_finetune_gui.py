"""The quick fine-tune window should build with the dark theme."""

from __future__ import annotations

import pytest


def test_finetune_gui_builds_dark_theme():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display is not available")

    from finetuning.finetune_gui import APP_BG, LABEL_MODES, FinetuneGUI

    try:
        root.withdraw()
        gui = FinetuneGUI(root)
        root.update_idletasks()
        assert root.cget("bg") == APP_BG
        assert str(gui.start_btn.cget("style")) == "Accent.TButton"
        assert gui.stage_text.get() == "Idle"

        binary = next(name for name, mode in LABEL_MODES.items() if mode == "binary")
        layers = next(name for name, mode in LABEL_MODES.items() if mode == "rgt_layers")
        gui.label_mode.set(binary)
        root.update_idletasks()
        assert str(gui.layers_spin.cget("state")) == "disabled"
        gui.label_mode.set(layers)
        root.update_idletasks()
        assert str(gui.layers_spin.cget("state")) == "normal"
    finally:
        root.destroy()


def _inside(widget, shell, slack: int = 8) -> bool:
    x, y = widget.winfo_rootx(), widget.winfo_rooty()
    w, h = widget.winfo_width(), widget.winfo_height()
    sx, sy = shell.winfo_rootx(), shell.winfo_rooty()
    sw, sh = shell.winfo_width(), shell.winfo_height()
    return x >= sx - slack and y >= sy - slack and x + w <= sx + sw + slack and y + h <= sy + sh + slack


def test_data_and_output_follow_the_log_sash():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display is not available")

    from finetuning.finetune_gui import FinetuneGUI

    try:
        root.geometry("1500x900+40+40")
        gui = FinetuneGUI(root)
        root.update()
        gui._place_sash()
        root.update()
        data, output = gui._data_shell, gui._output_shell
        full_data, full_output = data.winfo_height(), output.winfo_height()
        assert full_data > 80 and full_output > 80
        assert _inside(gui.preview_btn, data)
        assert _inside(gui.target_label, output)
        assert gui.start_btn.winfo_viewable()

        sash = gui._panes.sashpos(0)
        gui._panes.sashpos(0, max(180, sash - 140))
        root.update()
        assert data.winfo_height() < full_data
        assert output.winfo_height() < full_output
        assert _inside(gui.preview_btn, data)
        assert _inside(gui.target_label, output)
        assert gui.start_btn.winfo_viewable()
        # The two cards still share the column, including any scrolled overflow.
        assert output.winfo_rooty() >= data.winfo_rooty() + data.winfo_height() - 8

        gui._panes.sashpos(0, sash)
        root.update()
        assert abs(data.winfo_height() - full_data) < 40
        assert abs(output.winfo_height() - full_output) < 40
        gui.epochs.set(4)
        assert gui.epochs.get() == 4
    finally:
        root.destroy()
