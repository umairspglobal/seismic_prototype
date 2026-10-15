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
