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


def test_csd_preset_follows_the_generator_suffix(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    import time

    root = None
    error = None
    for _ in range(3):
        try:
            root = tk.Tk()
            break
        except tk.TclError as exc:
            error = exc
            time.sleep(0.3)
    if root is None:
        pytest.skip(f"Tk display is not available: {error}")

    import json

    import finetuning.finetune_gui as ft

    data = tmp_path / "data"
    data.mkdir()
    (data / "generator_config.json").write_text(json.dumps({"output": {"suffix": "80"}}), encoding="utf-8")
    (data / "Seis75").mkdir()
    monkeypatch.setattr(ft, "SETTINGS_FILE", tmp_path / "ft.json")
    try:
        root.withdraw()
        gui = ft.FinetuneGUI(root)
        gui.data_root.set(str(data))
        gui.preset.set(next(k for k, v in ft.PRESETS.items() if v and v["task"] == "csd"))
        gui._apply_preset()
        assert gui.image_dir.get() == str(data / "Seis80")
        assert gui.label_dir.get() == str(data / "CsdHorizon80")
        assert gui.prompt.get() == "horizon"
        assert ft.LABEL_MODES[gui.label_mode.get()] == "binary"
        assert str(gui.layers_spin.cget("state")) == "disabled"

        gui.preset.set(next(k for k, v in ft.PRESETS.items() if v and v["task"] == "facies"))
        gui._apply_preset()
        assert gui.label_dir.get() == str(data / "TauAtT80")
        assert str(gui.layers_spin.cget("state")) == "normal"
    finally:
        root.destroy()
