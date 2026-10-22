"""Desktop GUI for the synthetic seismic dataset generator.

    python -m synthetic_generator          (or double-click start_synthetic_generator.cmd)

Left: every generator setting, grouped in tabs, plus the dataset card (output
folder, size estimate, Generate / Stop, time estimate, progress). The Data Viewer
tab opens a folder of sections and draws it on the right. Right: a live preview of the
block that will be written - seismic, fault mask, tau, all labels, the six
slices, the four bandpass replicas with spectra, and exactly what the Quick
Fine-Tuner feeds SAM 3. Generation runs in a child process
(``synthetic_generator.cli generate --events``), like the fine-tuner's trainer.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from finetuning.presets import PRESETS as FINETUNE_PRESETS, folders_for  # noqa: E402
from synthetic_generator import __version__  # noqa: E402
from synthetic_generator.block import Block, block_csd, block_streams, generate_block, products  # noqa: E402
from synthetic_generator.config import (ARRAY_KINDS, DEFAULT_PRESET, PRESETS, GeneratorConfig,  # noqa: E402
                                        default_config, preset)
from synthetic_generator.dataset import (CONFIG_NAME, DEFAULT_WRITE_BPS, auto_workers, block_bytes,  # noqa: E402
                                         block_cpu_weight, check_disk, dataset_bytes, estimate_run_seconds,
                                         file_bytes, measure_write_bps, next_start_id, plan_blocks)
from synthetic_generator.params import TABS, Param, all_params  # noqa: E402
from synthetic_generator.preview import SEISMIC_CMAPS, VIEWS, TileInfo, fit_tile, render_view  # noqa: E402
from synthetic_generator.viewer import (catalog_summary, default_keys, default_overlay_color, describe_id,  # noqa: E402
                                        fit_sample_tile, load_sample, render_sample, scan_dataset, step_id)

SETTINGS_FILE = REPO_ROOT / "outputs" / "synthetic_generator_gui.json"
RUN_DIR = REPO_ROOT / "outputs" / "synthetic_generator"
EVENT_PREFIX = "@@"
_DATA_PARENT = REPO_ROOT.parent / "Data"
DEFAULT_OUT_DIR = (_DATA_PARENT if _DATA_PARENT.is_dir() else REPO_ROOT / "outputs") / "synthetic_75"

APP_BG = "#101418"
CARD = "#171c22"
FIELD = "#0f1317"
BORDER = "#2c343d"
FG = "#e6e9ec"
MUTED = "#8a949e"
LABEL = "#b8c0c8"
ACCENT = "#2563eb"
ACCENT_HOVER = "#3b82f6"
OK = "#19c37d"
WARN = "#f0b429"
ERR = "#ef5350"
CONSOLE_BG = "#0c1014"
DISABLED = "#6b7683"

KIND_LABELS = {
    "seis": "Seismic", "fault": "Fault mask", "tau": "Tau (RGT)", "dip": "Dip",
    "weight": "Scale weight", "straddle": "Straddle", "halo": "Halo",
}


def _console_python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def _windowed_python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "python.exe" and exe.with_name("pythonw.exe").exists():
        return str(exe.with_name("pythonw.exe"))
    return str(exe)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return "<1s" if seconds > 0 else "0s"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget, self.text, self.tip, self.job = widget, text, None, None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self.job = self.widget.after(450, self._show)

    def _cancel(self) -> None:
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None

    def _show(self) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.configure(bg=BORDER)
        tk.Label(self.tip, text=self.text, justify="left", wraplength=380, bg="#1f262e", fg=FG,
                 font=("Segoe UI", 9), padx=10, pady=7).pack(padx=1, pady=1)
        self.tip.wm_geometry(f"+{x}+{y}")

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    """Vertically scrolling frame; ``GeneratorGUI`` routes the mouse wheel to the one under the pointer."""

    def __init__(self, parent):
        super().__init__(parent, style="App.TFrame")
        self.canvas = tk.Canvas(self, bg=APP_BG, highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, style="App.TFrame")
        self.window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window, width=e.width))
        self.canvas._scroll_owner = self  # type: ignore[attr-defined]

    def scroll(self, delta: int) -> None:
        top, bottom = self.canvas.yview()
        if top <= 0.0 and bottom >= 1.0:
            return
        self.canvas.yview_scroll(int(-delta / 120) or (-1 if delta > 0 else 1), "units")


def _spin_format(step: float) -> str:
    if step >= 1 and float(step).is_integer():
        return "%.0f"
    decimals = max(1, min(6, int(np.ceil(-np.log10(step))) if step > 0 else 2))
    return f"%.{decimals}f"


@dataclass
class PreviewJob:
    cfg: GeneratorConfig
    key: str
    block_index: int
    view: str
    replica: int
    slice_index: int
    scale: int
    cmap: str
    faults: bool
    tile: int
    layers: int


class GeneratorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.vars: dict[str, tk.Variable | tuple[tk.StringVar, tk.StringVar]] = {}
        self.array_vars: dict[str, tk.BooleanVar] = {}
        self.param_widgets: dict[str, list[tk.Widget]] = {}
        self._loading = False
        self._change_job = None
        self.preview_q: queue.Queue = queue.Queue()
        self.gen_q: queue.Queue = queue.Queue()
        self.probe_q: queue.Queue = queue.Queue()
        self.preview_busy = False
        self.preview_again = False
        self.cached_key: str | None = None
        self.cached_block: Block | None = None
        self.preview_photo = None
        self.preview_infos: list[TileInfo] = []
        self.preview_offset = (0, 0)
        # Stand-in until the preview times a block on this computer. The weight
        # is the config that time belongs to, so later setting changes scale it.
        self.last_gen_seconds = 0.65
        self._timed_weight = block_cpu_weight(default_config())
        self.write_bps = float(DEFAULT_WRITE_BPS)
        self._probed_drive: str | None = None
        self._probing_drive: str | None = None
        self._probe_token = 0
        self._probe_thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.stop_file = RUN_DIR / "stop.flag"
        self.stop_sent = False
        self.run_total = 0
        self.run_started = 0.0
        self.zoom_windows: list[ZoomWindow] = []
        self._start_cache: tuple[tuple, int, float] | None = None
        self.preview_thread: threading.Thread | None = None
        self._closing = False
        self.catalog = None
        self.viewer_checks: dict[str, tk.BooleanVar] = {}
        self._disk_mode = False
        self._disk_after = None
        self._viewer_loading = False
        self._saved_viewer_arrays: list[str] | None = None

        root.title(f"Seismic Synthetic Generator {__version__} - SAM 3 datasets")
        root.configure(bg=APP_BG)
        scale = root.winfo_fpixels("1i") / 96.0
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        width = min(int(1820 * scale), int(sw * 0.95))
        height = min(int(1000 * scale), int(sh * 0.92))
        root.geometry(f"{width}x{height}+{(sw - width) // 2}+{int(16 * scale)}")
        root.minsize(min(int(1200 * scale), int(sw * 0.9)), min(int(720 * scale), int(sh * 0.8)))
        self._apply_theme()
        self._build_vars()
        self._build_ui()
        self._load_settings()
        root.bind_all("<MouseWheel>", self._wheel, add="+")
        for cls in ("TSpinbox", "TCombobox"):
            root.bind_class(cls, "<MouseWheel>", self._wheel_no_change)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(60, self._drain)
        root.after(250, lambda: self.request_preview(regenerate=True))

    def _scroll_owner(self, event) -> ScrollFrame | None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            owner = getattr(widget, "_scroll_owner", None)
            if owner is not None:
                return owner
            if isinstance(widget, ScrollFrame):
                return widget
            widget = widget.master
        return None

    def _wheel(self, event) -> None:
        if isinstance(event.widget, tk.Text):
            return
        owner = self._scroll_owner(event)
        if owner is not None:
            owner.scroll(event.delta)

    def _wheel_no_change(self, event) -> str:
        """Spinboxes and comboboxes must not change value while the form is scrolled."""
        owner = self._scroll_owner(event)
        if owner is not None:
            owner.scroll(event.delta)
        return "break"

    # --- Theme ---------------------------------------------------------------

    def _apply_theme(self) -> None:
        import tkinter.font as tkfont

        for name, family, size in (("TkDefaultFont", "Segoe UI", 10), ("TkTextFont", "Segoe UI", 10),
                                   ("TkHeadingFont", "Segoe UI", 11), ("TkFixedFont", "Consolas", 10)):
            try:
                tkfont.nametofont(name).configure(family=family, size=size)
            except tk.TclError:
                pass
        r = self.root
        r.option_add("*TCombobox*Listbox.background", FIELD)
        r.option_add("*TCombobox*Listbox.foreground", FG)
        r.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        r.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        r.option_add("*TCombobox*Listbox.font", "{Segoe UI} 10")
        style = ttk.Style(r)
        style.theme_use("clam")
        style.configure(".", background=CARD, foreground=FG, bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, troughcolor=FIELD, font=("Segoe UI", 10), focuscolor=ACCENT)
        style.configure("TFrame", background=APP_BG)
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=CARD, foreground=FG)
        style.configure("App.TLabel", background=APP_BG, foreground=FG)
        style.configure("Title.TLabel", background=APP_BG, foreground=FG, font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", background=APP_BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Field.TLabel", background=CARD, foreground=LABEL)
        style.configure("Hint.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Estimate.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 8))
        style.configure("AppHint.TLabel", background=APP_BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=CARD, foreground="#93c5fd", font=("Segoe UI", 10, "bold"))
        style.configure("Warn.TLabel", background=CARD, foreground=WARN, font=("Segoe UI", 9))
        style.configure("Err.TLabel", background=CARD, foreground=ERR, font=("Segoe UI", 9))
        style.configure("Ok.TLabel", background=CARD, foreground=OK, font=("Segoe UI", 9))
        style.configure("Stage.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 12, "bold"))
        style.configure("TPanedwindow", background=APP_BG)
        style.configure("TNotebook", background=APP_BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
        style.configure("TNotebook.Tab", background=FIELD, foreground=LABEL, padding=(12, 6), borderwidth=0,
                        font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", CARD), ("active", "#1c242c")],
                  foreground=[("selected", FG)])
        for name in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(name, fieldbackground=FIELD, background=FIELD, foreground=FG, insertcolor=FG,
                            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, arrowcolor=LABEL,
                            padding=(6, 3), arrowsize=12)
            style.map(name, bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)],
                      darkcolor=[("focus", ACCENT)], fieldbackground=[("readonly", FIELD), ("disabled", "#12171c")],
                      foreground=[("readonly", FG), ("disabled", DISABLED)],
                      background=[("readonly", FIELD), ("active", "#1c242c")])
        style.configure("TCheckbutton", background=CARD, foreground=LABEL, indicatorcolor=FIELD, focuscolor=CARD)
        style.map("TCheckbutton", background=[("active", CARD)], foreground=[("active", FG), ("disabled", DISABLED)],
                  indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT), ("!selected", FIELD)])
        flat = [("Button.border", {"sticky": "nswe", "children": [
            ("Button.padding", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]})]})]
        style.layout("TButton", flat)
        style.configure("TButton", background="#1c242c", foreground=FG, bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, padding=(12, 6), focusthickness=0, relief="flat")
        style.map("TButton", background=[("pressed", FIELD), ("active", "#24303c"), ("disabled", CARD)],
                  foreground=[("disabled", DISABLED)], bordercolor=[("active", "#3a4652")])
        for name, bg, hover, fg in (("Accent.TButton", ACCENT, ACCENT_HOVER, "#ffffff"),
                                    ("Danger.TButton", "#3f1d1d", "#5c2424", "#ffb4b4")):
            style.configure(name, background=bg, foreground=fg, bordercolor=bg, lightcolor=bg, darkcolor=bg,
                            padding=(14, 7), font=("Segoe UI", 10, "bold"), focusthickness=0, relief="flat")
            style.map(name, background=[("pressed", bg), ("active", hover), ("disabled", "#24303f")],
                      foreground=[("disabled", DISABLED)], bordercolor=[("active", hover), ("disabled", "#24303f")])
        style.configure("Tool.TButton", background=FIELD, foreground=LABEL, bordercolor=BORDER, padding=(9, 3),
                        font=("Segoe UI", 9), focusthickness=0, relief="flat")
        style.map("Tool.TButton", background=[("active", "#1c242c"), ("disabled", CARD)],
                  foreground=[("active", FG), ("disabled", DISABLED)])
        style.configure("Tool.TMenubutton", background=FIELD, foreground=LABEL, bordercolor=BORDER,
                        arrowcolor=LABEL, padding=(9, 3), font=("Segoe UI", 9), relief="flat")
        style.map("Tool.TMenubutton", background=[("active", "#1c242c")], foreground=[("active", FG)])
        style.configure("View.Toolbutton", background=FIELD, foreground=LABEL, bordercolor=BORDER, padding=(10, 4),
                        font=("Segoe UI", 9), relief="flat", anchor="center")
        style.map("View.Toolbutton", background=[("selected", ACCENT), ("active", "#1c242c")],
                  foreground=[("selected", "#ffffff"), ("active", FG)])
        style.layout("Horizontal.TProgressbar", [("Horizontal.Progressbar.trough", {"sticky": "nswe", "children": [
            ("Horizontal.Progressbar.pbar", {"side": "left", "sticky": "ns"})]})])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#232b33", bordercolor="#232b33",
                        lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
        style.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
            ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
        style.configure("Vertical.TScrollbar", background="#2a333c", troughcolor=APP_BG, bordercolor=APP_BG,
                        lightcolor="#2a333c", darkcolor="#2a333c", arrowsize=0, width=10)
        style.map("Vertical.TScrollbar", background=[("active", "#3a4652")])

    def _dark_titlebar(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            value = ctypes.c_int(1)
            for attr in (20, 19):
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value),
                                                              ctypes.sizeof(value)) == 0:
                    break
        except (AttributeError, OSError):
            pass

    # --- Variables -------------------------------------------------------------

    def _build_vars(self) -> None:
        for p in all_params():
            if p.kind in ("range_int", "range_float"):
                pair = (tk.StringVar(), tk.StringVar())
                self.vars[p.path] = pair
                for v in pair:
                    v.trace_add("write", lambda *_: self._on_change())
            elif p.kind == "bool":
                self.vars[p.path] = tk.BooleanVar()
                self.vars[p.path].trace_add("write", lambda *_: self._on_change())
            elif p.kind == "arrays":
                for kind in ARRAY_KINDS:
                    self.array_vars[kind] = tk.BooleanVar(value=True)
                    self.array_vars[kind].trace_add("write", lambda *_: self._on_change())
            else:
                self.vars[p.path] = tk.StringVar()
                self.vars[p.path].trace_add("write", lambda *_: self._on_change())
        self.preset_var = tk.StringVar(value=DEFAULT_PRESET)
        self.view_var = tk.StringVar(value="overview")
        self.slice_var = tk.IntVar(value=1)
        self.replica_var = tk.IntVar(value=0)
        self.scale_var = tk.StringVar()
        self.cmap_var = tk.StringVar(value="gray")
        self.faults_var = tk.BooleanVar(value=True)
        self.auto_var = tk.BooleanVar(value=True)
        self.block_var = tk.IntVar(value=0)
        self.layers_var = tk.IntVar(value=8)
        self.summary_text = tk.StringVar()
        self.eta_text = tk.StringVar(value=" ")
        self.errors_text = tk.StringVar()
        self.preview_status = tk.StringVar(value="")
        self.readout_text = tk.StringVar(value="Hover the preview to read values; double-click a tile to zoom.")
        self.info_text = tk.StringVar()
        self.stage_text = tk.StringVar(value="Idle")
        self.progress_text = tk.StringVar()
        self.viewer_dir = tk.StringVar()
        self.viewer_id = tk.IntVar(value=0)
        self.viewer_cmap = tk.StringVar(value="gray")
        self.viewer_faults = tk.BooleanVar(value=True)
        self.viewer_horizons = tk.BooleanVar(value=True)
        self.viewer_overlay = tk.BooleanVar(value=False)
        self.viewer_opacity = tk.DoubleVar(value=0.4)
        self.viewer_opacity_text = tk.StringVar(value="40%")
        self._overlay_colors: dict[str, tuple[int, int, int]] = {}
        self._overlay_chip_sig: tuple | None = None
        self.viewer_status = tk.StringVar(value="Choose a folder, then press View.")
        self.viewer_place = tk.StringVar(value="")
        self.viewer_banner_text = tk.StringVar(value="")
        for var in (self.view_var, self.slice_var, self.replica_var, self.scale_var, self.cmap_var, self.faults_var,
                    self.layers_var):
            var.trace_add("write", lambda *_: self._schedule_view())
        self.block_var.trace_add("write", lambda *_: self._schedule_view(regenerate=True))

    # --- Layout ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self.root
        header = ttk.Frame(root, style="App.TFrame", padding=(16, 10, 16, 4))
        header.pack(fill="x")
        title = ttk.Frame(header, style="App.TFrame")
        title.pack(side="left")
        ttk.Label(title, text="Seismic Synthetic Generator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title, text="Faulted seismic + fault masks + tau + dip / weight / straddle / halo labels, in the "
                              "training_10_7_26 layout the SAM 3 Quick Fine-Tuner reads.",
                  style="Sub.TLabel").pack(anchor="w")
        tools = ttk.Frame(header, style="App.TFrame")
        tools.pack(side="right")
        ttk.Label(tools, text="Preset", style="App.TLabel").pack(side="left", padx=(0, 6))
        self.preset_box = ttk.Combobox(tools, textvariable=self.preset_var, values=list(PRESETS), state="readonly",
                                       width=36)
        self.preset_box.pack(side="left")
        self.preset_box.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        Tooltip(self.preset_box, "Loads a full set of settings (output folder and seed are kept).")
        ttk.Button(tools, text="Load...", style="Tool.TButton", command=self._load_json).pack(side="left", padx=(8, 0))
        ttk.Button(tools, text="Save...", style="Tool.TButton", command=self._save_json).pack(side="left", padx=4)
        ttk.Button(tools, text="Reset", style="Tool.TButton", command=self._reset).pack(side="left")

        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        left = ttk.Frame(paned, style="App.TFrame")
        right = ttk.Frame(paned, style="App.TFrame")
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        self._build_settings(left)
        self._build_preview(right)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab)
        self.paned = paned
        root.after(100, lambda: self._place_sash())

    def _place_sash(self) -> None:
        try:
            scale = self.root.winfo_fpixels("1i") / 96.0
            self.paned.sashpos(0, int(560 * scale))
        except tk.TclError:
            pass

    def _card(self, parent, title: str | None = None, pad=(14, 10)) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=pad)
        if title:
            ttk.Label(card, text=title, style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w",
                                                                    pady=(0, 6))
        card.columnconfigure(1, weight=1)
        return card

    def _build_settings(self, parent) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        for tab_name, sections in TABS:
            sf = ScrollFrame(self.notebook)
            self.notebook.add(sf, text=tab_name)
            for section, params in sections:
                card = self._card(sf.body, section)
                card.pack(fill="x", pady=(8, 0), padx=(0, 4))
                for row, p in enumerate(params, start=1):
                    self._param_row(card, row, p)
            if tab_name == "Labels":
                self._label_notes(sf.body)
            elif tab_name.startswith("CSD"):
                self._csd_notes(sf.body)
        self._build_viewer_tab()
        self._build_generate_card(parent)

    def _param_row(self, card: ttk.Frame, row: int, p: Param) -> None:
        text = p.label + ("  (i)" if p.help else "")
        lab = ttk.Label(card, text=text, style="Field.TLabel")
        lab.grid(row=row, column=0, sticky="w", pady=3, padx=(0, 10))
        if p.help:
            Tooltip(lab, p.help)
        widgets: list[tk.Widget] = []
        if p.kind in ("range_int", "range_float"):
            box = ttk.Frame(card, style="Card.TFrame")
            box.grid(row=row, column=1, sticky="w")
            lo_var, hi_var = self.vars[p.path]
            for i, var in enumerate((lo_var, hi_var)):
                sp = ttk.Spinbox(box, textvariable=var, from_=p.lo if p.lo is not None else -1e9,
                                 to=p.hi if p.hi is not None else 1e9, increment=p.step, width=8,
                                 format=_spin_format(p.step))
                sp.pack(side="left")
                widgets.append(sp)
                if i == 0:
                    ttk.Label(box, text="to", style="Hint.TLabel").pack(side="left", padx=6)
        elif p.kind in ("int", "float", "prob"):
            lo, hi, step = (0.0, 1.0, 0.05) if p.kind == "prob" else (p.lo, p.hi, p.step)
            sp = ttk.Spinbox(card, textvariable=self.vars[p.path], from_=lo if lo is not None else -1e9,
                             to=hi if hi is not None else 1e9, increment=step, width=10,
                             format=_spin_format(step) if p.kind != "int" else "%.0f")
            sp.grid(row=row, column=1, sticky="w")
            widgets.append(sp)
        elif p.kind == "bool":
            cb = ttk.Checkbutton(card, variable=self.vars[p.path])
            cb.grid(row=row, column=1, sticky="w")
            widgets.append(cb)
        elif p.kind == "choice":
            cb = ttk.Combobox(card, textvariable=self.vars[p.path], values=list(p.choices), state="readonly", width=16)
            cb.grid(row=row, column=1, sticky="w")
            widgets.append(cb)
        elif p.kind in ("text", "list"):
            en = ttk.Entry(card, textvariable=self.vars[p.path], width=26 if p.kind == "list" else 12)
            en.grid(row=row, column=1, sticky="ew" if p.kind == "list" else "w")
            widgets.append(en)
        elif p.kind == "dir":
            box = ttk.Frame(card, style="Card.TFrame")
            box.grid(row=row, column=1, sticky="ew")
            box.columnconfigure(0, weight=1)
            en = ttk.Entry(box, textvariable=self.vars[p.path])
            en.grid(row=0, column=0, sticky="ew")
            ttk.Button(box, text="...", style="Tool.TButton", width=3,
                       command=lambda v=self.vars[p.path]: self._browse_dir(v)).grid(row=0, column=1, padx=(4, 0))
            widgets.append(en)
        elif p.kind == "arrays":
            box = ttk.Frame(card, style="Card.TFrame")
            box.grid(row=row, column=1, sticky="w")
            for i, kind in enumerate(ARRAY_KINDS):
                cb = ttk.Checkbutton(box, text=KIND_LABELS[kind], variable=self.array_vars[kind])
                cb.grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 10))
                widgets.append(cb)
        for w in widgets:
            if p.help:
                Tooltip(w, p.help)
        self.param_widgets[p.path] = widgets

    def _label_notes(self, parent) -> None:
        card = self._card(parent, "What each label folder holds")
        card.pack(fill="x", pady=(8, 0), padx=(0, 4))
        notes = (
            "Fault: 1 on the fault stroke, 0 = background (like 1.dat).\n"
            "TauAtT: relative geologic time; the facies preset cuts it into layers.\n"
            "dipXscaleS: tau[y,x] - tau[y,x+S], the vertical shift to the same reflector S traces right.\n"
            "wtXStraddleScaleS: 1 where x and x+S sit on opposite sides of a fault (both off the stroke).\n"
            "wtXscaleS: loss weight; 0 near faults, near the right edge and where the pair is invalid.\n"
            "wtXHaloScaleS: blur of the fault mask with radius S, peaking at 1.\n"
            "Only Seis differs between the bandpass replicas; the other arrays are repeated."
        )
        ttk.Label(card, text=notes, style="Hint.TLabel", justify="left", wraplength=480).grid(
            row=1, column=0, columnspan=3, sticky="w")

    def _csd_notes(self, parent) -> None:
        card = self._card(parent, "What the CSD folders hold (in progress)")
        card.pack(fill="x", pady=(8, 0), padx=(0, 4))
        notes = (
            "A horizon is one reflector (one tau level). Its track follows the folds, steps down by the throw "
            "where it crosses a fault and then continues.\n"
            "CsdHorizon: 1 on the tracked horizons (the target mask). The Quick Fine-Tuner coarse-scale dips "
            "preset trains on this folder with the text prompt horizon.\n"
            "CsdDipXscaleS: the prompt - dipXscaleS on the horizon pixels, the background value elsewhere.\n"
            "CsdDistance: distance in px to the nearest horizon pixel.\n"
            "Open the CSD preview view to see them laid out per scale."
        )
        ttk.Label(card, text=notes, style="Hint.TLabel", justify="left", wraplength=480).grid(
            row=1, column=0, columnspan=3, sticky="w")

    def _build_viewer_tab(self) -> None:
        sf = ScrollFrame(self.notebook)
        self.notebook.add(sf, text="Data Viewer")
        card = self._card(sf.body, "Folder")
        card.pack(fill="x", pady=(8, 0), padx=(0, 4))
        ttk.Label(card, text="Dataset folder", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=3,
                                                                          padx=(0, 10))
        box = ttk.Frame(card, style="Card.TFrame")
        box.grid(row=1, column=1, sticky="ew")
        box.columnconfigure(0, weight=1)
        entry = ttk.Entry(box, textvariable=self.viewer_dir)
        entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(box, text="...", style="Tool.TButton", width=3, command=self._browse_viewer).grid(
            row=0, column=1, padx=(4, 0))
        Tooltip(entry, "A dataset folder (Seis75, Fault75, ...) or a folder of numbered sections (1.dat, 2.npy).")
        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 2))
        view_btn = ttk.Button(buttons, text="View", style="Accent.TButton", command=self._view_folder)
        view_btn.pack(side="left")
        Tooltip(view_btn, "Read this folder and show the sample on the right.")
        use_out = ttk.Button(buttons, text="Use output folder", style="Tool.TButton", command=self._use_output_folder)
        use_out.pack(side="left", padx=6)
        Tooltip(use_out, "Open the output folder from Generate dataset.")
        self.viewer_status_label = ttk.Label(card, textvariable=self.viewer_status, style="Hint.TLabel",
                                             justify="left", wraplength=480)
        self.viewer_status_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        notes = (
            "Point at the dataset folder that contains Seis75, Fault75 and the other array folders, "
            "or at one folder of numbered sections (1.dat, 2.npy). View draws that sample on the right. "
            "Hover a tile to read a value; double-click to zoom. Other tabs return to the generated-block preview."
        )
        ttk.Label(card, text=notes, style="Hint.TLabel", justify="left", wraplength=480).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        sample = self._card(sf.body, "Sample")
        sample.pack(fill="x", pady=(8, 0), padx=(0, 4))
        nav = ttk.Frame(sample, style="Card.TFrame")
        nav.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Button(nav, text="<", style="Tool.TButton", width=2, command=lambda: self._step_viewer_id(-1)).pack(
            side="left")
        self.viewer_spin = ttk.Spinbox(nav, textvariable=self.viewer_id, from_=0, to=10_000_000, width=8)
        self.viewer_spin.pack(side="left", padx=4)
        ttk.Button(nav, text=">", style="Tool.TButton", width=2, command=lambda: self._step_viewer_id(1)).pack(
            side="left")
        ttk.Label(sample, textvariable=self.viewer_place, style="Hint.TLabel", justify="left", wraplength=480).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        arrays = self._card(sf.body, "Arrays")
        arrays.pack(fill="x", pady=(8, 0), padx=(0, 4))
        self.viewer_checks_frame = ttk.Frame(arrays, style="Card.TFrame")
        self.viewer_checks_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Label(self.viewer_checks_frame, text="Press View to list the arrays in the folder.",
                  style="Hint.TLabel").grid(row=0, column=0, sticky="w")
        tools = ttk.Frame(arrays, style="Card.TFrame")
        tools.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(tools, text="Seismic only", style="Tool.TButton",
                   command=lambda: self._viewer_select("seis")).pack(side="left")
        ttk.Button(tools, text="All arrays", style="Tool.TButton",
                   command=lambda: self._viewer_select("all")).pack(side="left", padx=4)
        row = ttk.Frame(arrays, style="Card.TFrame")
        row.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(row, text="Colormap", style="Field.TLabel").pack(side="left")
        cmap = ttk.Combobox(row, textvariable=self.viewer_cmap, values=list(SEISMIC_CMAPS), state="readonly", width=8)
        cmap.pack(side="left", padx=(4, 12))
        self.viewer_fault_check = ttk.Checkbutton(row, text="Fault overlay", variable=self.viewer_faults)
        self.viewer_fault_check.pack(side="left")
        Tooltip(self.viewer_fault_check, "Draws the fault mask on the seismic section when that sample has a Fault file.")
        self.viewer_horizon_check = ttk.Checkbutton(row, text="Horizon overlay", variable=self.viewer_horizons)
        self.viewer_horizon_check.pack(side="left", padx=(12, 0))
        Tooltip(self.viewer_horizon_check,
                "Draws the CsdHorizon tracks in cyan on the seismic section. Those tracks are the coarse-scale-dip label.")
        overlay_row = ttk.Frame(arrays, style="Card.TFrame")
        overlay_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.viewer_overlay_check = ttk.Checkbutton(overlay_row, text="Overlay checked arrays",
                                                    variable=self.viewer_overlay)
        self.viewer_overlay_check.pack(side="left")
        Tooltip(self.viewer_overlay_check,
                "Paints the checked arrays on the seismic, each in its own color, so the section stays visible "
                "underneath. The slider sets how solid those colors are. Click a color chip to change that array. "
                "Fault overlay and Horizon overlay are included while those boxes are checked. A mask colors pixels "
                "above 0.5. Dip colors the strong values and leaves the flat areas clear.")
        ttk.Label(overlay_row, text="Opacity", style="Field.TLabel").pack(side="left", padx=(12, 4))
        self.viewer_opacity_scale = ttk.Scale(overlay_row, from_=0.1, to=0.85, variable=self.viewer_opacity,
                                              length=120)
        self.viewer_opacity_scale.pack(side="left")
        self.viewer_opacity_label = ttk.Label(overlay_row, textvariable=self.viewer_opacity_text, style="Hint.TLabel",
                                              width=4)
        self.viewer_opacity_label.pack(side="left", padx=(4, 0))
        self.viewer_colors_frame = ttk.Frame(arrays, style="Card.TFrame")
        self.viewer_colors_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        for var in (self.viewer_id, self.viewer_cmap, self.viewer_faults, self.viewer_horizons, self.viewer_overlay):
            var.trace_add("write", lambda *_: self._on_viewer_control())
        self.viewer_opacity.trace_add("write", self._on_overlay_opacity)
        self._refresh_overlay_chips()

    def _build_generate_card(self, parent) -> None:
        card = self._card(parent, "Generate dataset", pad=(14, 10))
        card.pack(fill="x", pady=(10, 0))
        ttk.Label(card, text="Output folder", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        box = ttk.Frame(card, style="Card.TFrame")
        box.grid(row=1, column=1, columnspan=3, sticky="ew")
        box.columnconfigure(0, weight=1)
        ttk.Entry(box, textvariable=self.vars["output.out_dir"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(box, text="...", style="Tool.TButton", width=3,
                   command=lambda: self._browse_dir(self.vars["output.out_dir"])).grid(row=0, column=1, padx=(4, 0))
        ttk.Label(card, text="Blocks", style="Field.TLabel").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Spinbox(card, textvariable=self.vars["output.num_blocks"], from_=1, to=1_000_000, increment=10,
                    width=10).grid(row=2, column=1, sticky="w")
        self.summary_label = ttk.Label(card, textvariable=self.summary_text, style="Hint.TLabel", justify="left",
                                       wraplength=500)
        self.summary_label.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 2))
        self.errors_label = ttk.Label(card, textvariable=self.errors_text, style="Err.TLabel", justify="left",
                                      wraplength=500)
        self.errors_label.grid(row=4, column=0, columnspan=4, sticky="w")
        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 4))
        gen_box = ttk.Frame(buttons, style="Card.TFrame")
        gen_box.pack(side="left", anchor="n")
        self.start_btn = ttk.Button(gen_box, text="Generate dataset", style="Accent.TButton", command=self.start)
        self.start_btn.pack(anchor="w")
        self.eta_label = ttk.Label(gen_box, textvariable=self.eta_text, style="Estimate.TLabel")
        self.eta_label.pack(anchor="w", pady=(2, 0))
        self.eta_tip = Tooltip(self.eta_label, "How long these blocks are expected to take on this computer.")
        self.stop_btn = ttk.Button(buttons, text="Stop", style="Danger.TButton", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", anchor="n", padx=6)
        resume = ttk.Button(buttons, text="Resume last run", style="Tool.TButton", command=self.resume)
        resume.pack(side="left", anchor="n")
        self.resume_btn = resume
        Tooltip(resume, "Reloads the settings of the last run in the output folder (same seed and first id) and "
                        "writes only the blocks that are missing or incomplete.")
        tools = ttk.Frame(card, style="Card.TFrame")
        tools.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Button(tools, text="Open folder", style="Tool.TButton", command=self._open_output).pack(side="left")
        ttk.Button(tools, text="Inspect", style="Tool.TButton", command=self.inspect).pack(side="left", padx=4)
        self.ft_btn = ttk.Menubutton(tools, text="Send to Quick Fine-Tuner", style="Tool.TMenubutton")
        menu = tk.Menu(self.ft_btn, tearoff=0, bg=FIELD, fg=FG, activebackground=ACCENT, activeforeground="#fff")
        for spec in FINETUNE_PRESETS.values():
            if not spec:
                continue
            task = spec["task"]
            menu.add_command(label=spec["menu"], command=lambda t=task: self.send_to_finetuner(t))
        self.ft_btn["menu"] = menu
        self.ft_btn.pack(side="left")
        Tooltip(self.ft_btn, "Points the Quick Fine-Tuner at this folder (faults, facies, or coarse-scale dips) "
                             "and opens it.")
        self.button_rows = (buttons, tools)
        status = ttk.Frame(card, style="Card.TFrame")
        status.grid(row=7, column=0, columnspan=4, sticky="ew")
        status.columnconfigure(1, weight=1)
        self.stage_label = ttk.Label(status, textvariable=self.stage_text, style="Stage.TLabel")
        self.stage_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=1.0)
        self.progress.grid(row=0, column=1, sticky="ew")
        ttk.Label(status, textvariable=self.progress_text, style="Hint.TLabel").grid(row=1, column=0, columnspan=2,
                                                                                   sticky="w")

    def _build_preview(self, parent) -> None:
        bar = self._card(parent, None, pad=(12, 8))
        bar.pack(fill="x")
        self.preview_bar = bar
        row1 = ttk.Frame(bar, style="Card.TFrame")
        row1.grid(row=0, column=0, sticky="ew")
        ttk.Label(row1, text="Preview block", style="Field.TLabel").pack(side="left")
        ttk.Button(row1, text="<", style="Tool.TButton", width=2,
                   command=lambda: self.block_var.set(max(0, self._int(self.block_var, 0) - 1))).pack(side="left",
                                                                                                    padx=(6, 2))
        ttk.Spinbox(row1, textvariable=self.block_var, from_=0, to=10_000_000, width=7).pack(side="left")
        ttk.Button(row1, text=">", style="Tool.TButton", width=2,
                   command=lambda: self.block_var.set(self._int(self.block_var, 0) + 1)).pack(side="left", padx=2)
        rnd = ttk.Button(row1, text="Random", style="Tool.TButton",
                         command=lambda: self.block_var.set(int(np.random.default_rng().integers(0, 100_000))))
        rnd.pack(side="left", padx=(4, 0))
        Tooltip(rnd, "Preview block k is exactly dataset block k (same seed), so what you see is what gets written.")
        ttk.Button(row1, text="Refresh", style="Tool.TButton",
                   command=lambda: self.request_preview(regenerate=True)).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(row1, text="Auto refresh", variable=self.auto_var).pack(side="left", padx=(10, 0))
        ttk.Label(row1, textvariable=self.preview_status, style="Hint.TLabel").pack(side="right")

        row2 = ttk.Frame(bar, style="Card.TFrame")
        row2.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for key, name in VIEWS.items():
            ttk.Radiobutton(row2, text=name, value=key, variable=self.view_var, style="View.Toolbutton").pack(
                side="left", padx=(0, 3))

        row3 = ttk.Frame(bar, style="Card.TFrame")
        row3.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(row3, text="Slice", style="Field.TLabel").pack(side="left")
        self.slice_spin = ttk.Spinbox(row3, textvariable=self.slice_var, from_=1, to=6, width=4)
        self.slice_spin.pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Replica", style="Field.TLabel").pack(side="left")
        self.replica_spin = ttk.Spinbox(row3, textvariable=self.replica_var, from_=0, to=3, width=4)
        self.replica_spin.pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Scale S", style="Field.TLabel").pack(side="left")
        self.scale_box = ttk.Combobox(row3, textvariable=self.scale_var, state="readonly", width=5)
        self.scale_box.pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Colormap", style="Field.TLabel").pack(side="left")
        ttk.Combobox(row3, textvariable=self.cmap_var, values=list(SEISMIC_CMAPS), state="readonly",
                     width=8).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(row3, text="Fault overlay", variable=self.faults_var).pack(side="left", padx=(0, 12))
        ttk.Label(row3, text="Facies layers", style="Field.TLabel").pack(side="left")
        ttk.Spinbox(row3, textvariable=self.layers_var, from_=2, to=40, width=4).pack(side="left", padx=(4, 0))
        bar.columnconfigure(0, weight=1)

        self.viewer_banner = self._card(parent, None, pad=(12, 8))
        ttk.Label(self.viewer_banner, textvariable=self.viewer_banner_text, style="Field.TLabel", justify="left",
                  wraplength=980).grid(row=0, column=0, sticky="ew")
        self.viewer_banner.columnconfigure(0, weight=1)

        holder = ttk.Frame(parent, style="App.TFrame")
        holder.pack(fill="both", expand=True, pady=(8, 0))
        self.canvas_holder = holder
        self.canvas = tk.Canvas(holder, bg=APP_BG, highlightthickness=1, highlightbackground=BORDER, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._schedule_view(delay=250))
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Double-Button-1>", self._on_double)

        info = self._card(parent, None, pad=(12, 6))
        info.pack(fill="x", pady=(8, 0))
        ttk.Label(info, textvariable=self.readout_text, style="Field.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.info_text, style="Hint.TLabel", justify="left", wraplength=1100).grid(
            row=1, column=0, sticky="w")

        console = ttk.Frame(parent, style="App.TFrame")
        console.pack(fill="x", pady=(8, 0))
        self.log = tk.Text(console, height=5, bg=CONSOLE_BG, fg="#d5dbe1", insertbackground=FG, relief="flat",
                           font=("Consolas", 9), wrap="none", padx=8, pady=6, highlightthickness=1,
                           highlightbackground=BORDER)
        self.log.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(console, orient="vertical", command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set, state="disabled")
        for tag, color in (("ok", OK), ("warn", WARN), ("err", ERR), ("muted", MUTED), ("info", "#93c5fd")):
            self.log.tag_configure(tag, foreground=color)

    # --- Config <-> widgets ---------------------------------------------------------

    def _write_config(self, cfg: GeneratorConfig) -> None:
        self._loading = True
        try:
            for p in all_params():
                value = cfg.get(p.path)
                if p.kind in ("range_int", "range_float"):
                    lo, hi = self.vars[p.path]
                    lo.set(_fmt(value[0]))
                    hi.set(_fmt(value[1]))
                elif p.kind == "bool":
                    self.vars[p.path].set(bool(value))
                elif p.kind == "arrays":
                    for kind in ARRAY_KINDS:
                        self.array_vars[kind].set(kind in value)
                elif p.kind == "list":
                    self.vars[p.path].set(", ".join(_fmt(v) for v in value))
                else:
                    self.vars[p.path].set(_fmt(value) if not isinstance(value, str) else value)
        finally:
            self._loading = False
        self._on_change(immediate=True)

    def read_config(self) -> tuple[GeneratorConfig, list[str]]:
        cfg = GeneratorConfig()
        errors: list[str] = []
        for p in all_params():
            try:
                if p.kind in ("range_int", "range_float"):
                    lo, hi = self.vars[p.path]
                    cfg.set(p.path, (float(lo.get()), float(hi.get())))
                elif p.kind == "bool":
                    cfg.set(p.path, bool(self.vars[p.path].get()))
                elif p.kind == "arrays":
                    cfg.set(p.path, [k for k in ARRAY_KINDS if self.array_vars[k].get()])
                elif p.kind == "list":
                    cfg.set(p.path, self.vars[p.path].get())
                elif p.kind in ("int", "float", "prob"):
                    cfg.set(p.path, float(self.vars[p.path].get()))
                else:
                    cfg.set(p.path, self.vars[p.path].get().strip())
            except (ValueError, TypeError, tk.TclError):
                errors.append(f"{p.label}: not a valid value.")
        cfg.grid.width = cfg.grid.height
        if not errors:
            errors = cfg.validate()
        return cfg, errors

    def _on_change(self, immediate: bool = False) -> None:
        if self._loading:
            return
        if self._change_job:
            self.root.after_cancel(self._change_job)
        self._change_job = self.root.after(10 if immediate else 450, self._apply_change)

    def _apply_change(self) -> None:
        self._change_job = None
        cfg, errors = self.read_config()
        self.errors_text.set("\n".join(errors[:4]) + ("\n..." if len(errors) > 4 else ""))
        running = self.proc is not None
        self.start_btn.configure(state="disabled" if errors or running else "normal")
        self.resume_btn.configure(state="disabled" if running else "normal")
        self.slice_spin.configure(to=max(1, cfg.grid.n_slices))
        self.replica_spin.configure(to=max(0, cfg.grid.n_replicas - 1))
        scales = [str(s) for s in cfg.grid.scales] or ["12"]
        self.scale_box.configure(values=scales)
        if self.scale_var.get() not in scales:
            self.scale_var.set(scales[0])
        if self._int(self.slice_var, 1) > cfg.grid.n_slices:
            self.slice_var.set(cfg.grid.n_slices)
        if self._int(self.replica_var, 0) >= cfg.grid.n_replicas:
            self.replica_var.set(cfg.grid.n_replicas - 1)
        self._update_summary(cfg, errors)
        if not errors and self.auto_var.get():
            self.request_preview(regenerate=True)

    def _scaled_compute(self, cfg: GeneratorConfig) -> float:
        """Preview time of one block, scaled when the settings have moved on since that preview."""
        weight = block_cpu_weight(cfg, csd=cfg.csd.enabled)
        base = self._timed_weight or weight
        return max(0.0, self.last_gen_seconds) * weight / base

    def _update_eta(self, cfg: GeneratorConfig) -> None:
        seconds = estimate_run_seconds(cfg, self._scaled_compute(cfg), write_bps=self.write_bps)
        workers = auto_workers(cfg)
        self.eta_text.set(f"Estimated time: about {fmt_duration(seconds)}")
        compute = self._scaled_compute(cfg)
        write_s = block_bytes(cfg) / max(self.write_bps, 1.0)
        noun = "worker" if workers == 1 else "workers"
        self.eta_tip.text = (
            f"About {fmt_duration(seconds)} for {cfg.output.num_blocks:,} blocks on {workers} {noun}. "
            f"A block like the preview takes {compute:.2f}s to build and {write_s:.1f}s to write; "
            f"the workers split that. Blocks already in the folder are skipped, so a resumed run can be shorter."
        )

    def _drive_key(self, path: Path) -> str | None:
        try:
            anchor = path.resolve().anchor or path.anchor
        except OSError:
            anchor = path.anchor
        return anchor.lower() if anchor else None

    def _schedule_write_probe(self, out: Path | None) -> None:
        """Time a short write on the output drive so the estimate is not stuck at the default rate."""
        if out is None or self._closing:
            return
        probe = out
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        if not probe.exists():
            return
        key = self._drive_key(probe)
        if key is None or key == self._probed_drive or key == self._probing_drive:
            return
        # Prefer the system temp folder when it is on the same drive, so the
        # sample write does not land in the dataset folder.
        target = probe
        temp = Path(tempfile.gettempdir())
        if temp.exists() and self._drive_key(temp) == key:
            target = temp
        self._probing_drive = key
        self._probe_token += 1
        token = self._probe_token
        self._probe_thread = threading.Thread(target=self._probe_write, args=(target, key, token), daemon=True)
        self._probe_thread.start()

    def _probe_write(self, directory: Path, key: str, token: int) -> None:
        try:
            bps = measure_write_bps(directory)
        except OSError:
            bps = 0.0
        self.probe_q.put((token, key, bps))

    def _drain_probe(self) -> None:
        while True:
            try:
                token, key, bps = self.probe_q.get_nowait()
            except queue.Empty:
                return
            if self._closing or token != self._probe_token:
                continue
            self._probing_drive = None
            # A few megabytes can land in the write cache and look faster than a real run.
            if 20_000_000 <= bps <= 2_000_000_000:
                self.write_bps = bps
            self._probed_drive = key
            try:
                cfg, errors = self.read_config()
            except (tk.TclError, ValueError):
                return
            if not errors:
                self._update_eta(cfg)

    def _update_summary(self, cfg: GeneratorConfig, errors: list[str]) -> None:
        if errors:
            self.summary_text.set("Fix the settings below to see the dataset size.")
            self.eta_text.set(" ")
            return
        out = Path(cfg.output.out_dir) if cfg.output.out_dir else None
        ids = cfg.output.num_blocks * cfg.ids_per_block
        n_arrays = len(products(cfg))
        size = dataset_bytes(cfg)
        lines = [f"{cfg.output.num_blocks:,} blocks x {cfg.ids_per_block} ids = {ids:,} ids  |  {n_arrays} arrays per id "
                 f"= {ids * n_arrays:,} files of {human_bytes(file_bytes(cfg))}  |  {human_bytes(size)}"]
        if out is not None:
            try:
                start = cfg.output.start_id if cfg.output.start_id >= 0 else self._next_start(out, cfg)
                ok, free = check_disk(out, size)
                lines.append(f"ids {start:,}-{start + ids - 1:,}"
                             + (" (continuing after existing ids)" if cfg.output.start_id < 0 and start else "")
                             + f"  |  free {human_bytes(free)}"
                             + ("" if ok else "  - less than the full size (complete blocks are skipped)"))
            except OSError:
                pass
        self.summary_text.set("\n".join(lines))
        self._update_eta(cfg)
        self._schedule_write_probe(Path(cfg.output.out_dir) if cfg.output.out_dir.strip() else None)

    # --- Preview ---------------------------------------------------------------------

    def _next_start(self, out: Path, cfg: GeneratorConfig) -> int:
        """``next_start_id`` scans folders, so it is cached for a few seconds while settings change."""
        key = (str(out), cfg.output.suffix, cfg.ids_per_block, tuple(cfg.output.arrays), tuple(cfg.grid.scales))
        now = time.time()
        if self._start_cache and self._start_cache[0] == key and now - self._start_cache[2] < 8.0:
            return self._start_cache[1]
        value = next_start_id(out, cfg)
        self._start_cache = (key, value, now)
        return value

    def _int(self, var: tk.Variable, default: int) -> int:
        try:
            return int(float(var.get()))
        except (tk.TclError, ValueError):
            return default

    def _schedule_view(self, regenerate: bool = False, delay: int = 60) -> None:
        if getattr(self, "_view_job", None):
            self.root.after_cancel(self._view_job)
            self._view_job = None
        if self._disk_mode:
            self._schedule_disk(delay)
            return
        self._view_job = self.root.after(delay, lambda: self.request_preview(regenerate=regenerate))

    def _preview_key(self, cfg: GeneratorConfig, block_index: int) -> str:
        data = cfg.to_dict()
        data.pop("output", None)
        return json.dumps([data, cfg.output.seed, block_index], sort_keys=True)

    def request_preview(self, regenerate: bool = False) -> None:
        self._view_job = None
        if self._disk_mode:
            return
        cfg, errors = self.read_config()
        if errors:
            self.preview_status.set("Preview paused: fix the settings first.")
            return
        cw, ch = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        view = self.view_var.get()
        try:
            scale = int(self.scale_var.get())
        except ValueError:
            scale = cfg.grid.scales[0]
        block_index = max(0, self._int(self.block_var, 0))
        slice_index = min(max(self._int(self.slice_var, 1) - 1, 0), cfg.grid.n_slices - 1)
        replica = min(max(self._int(self.replica_var, 0), 0), cfg.grid.n_replicas - 1)
        job = PreviewJob(
            cfg=cfg, key=self._preview_key(cfg, block_index), block_index=block_index,
            view=view, replica=replica, slice_index=slice_index,
            scale=scale, cmap=self.cmap_var.get(), faults=bool(self.faults_var.get()),
            tile=fit_tile(view, cfg, cw - 8, ch - 8), layers=max(2, self._int(self.layers_var, 8)),
        )
        self._pending_job = job
        if self.preview_busy:
            self.preview_again = True
            return
        self._start_preview(job)

    def _start_preview(self, job: PreviewJob) -> None:
        if self._closing:
            return
        self.preview_busy = True
        self.preview_again = False
        cached = self.cached_block if job.key == self.cached_key else None
        self.preview_status.set("Generating block..." if cached is None else "Rendering...")
        self.preview_thread = threading.Thread(target=self._preview_worker, args=(job, cached), daemon=True)
        self.preview_thread.start()

    def _preview_worker(self, job: PreviewJob, cached: Block | None) -> None:
        try:
            t0 = time.perf_counter()
            if cached is None:
                # Time only the work a dataset run does. The preview still builds
                # CSD afterwards so that view can open when CSD writing is off.
                block = generate_block(job.cfg, job.block_index, csd=job.cfg.csd.enabled)
                gen_s = time.perf_counter() - t0
                if block.csd is None:
                    csd_rng = block_streams(job.cfg.output.seed, job.block_index)[3]
                    arrays, counts = block_csd(csd_rng, job.cfg, block.tau, block.fault, block.labels)
                    block.csd = arrays
                    block.meta["csd_horizons"] = counts
            else:
                block = cached
                gen_s = None
            t1 = time.perf_counter()
            image, infos = render_view(block, job.cfg, job.view, replica=job.replica, slice_index=job.slice_index,
                                       scale=job.scale, tile=job.tile, cmap=job.cmap, show_faults=job.faults,
                                       facies_layers=job.layers)
            self.preview_q.put(("ok", job, block, image, infos, gen_s, time.perf_counter() - t1))
        except Exception:  # noqa: BLE001 - shown in the console
            self.preview_q.put(("error", job, traceback.format_exc()))

    def _show_preview(self, job: PreviewJob, block: Block, image: Image.Image, infos: list[TileInfo],
                      gen_s: float | None, render_s: float) -> None:
        self.cached_key, self.cached_block = job.key, block
        if gen_s is not None:
            self.last_gen_seconds = gen_s
            self._timed_weight = block_cpu_weight(job.cfg, csd=job.cfg.csd.enabled)
            current, errors = self.read_config()
            self._update_summary(current, errors)
        timing = f"generated in {gen_s:.2f}s, " if gen_s is not None else ""
        m = block.meta
        faults = m["geology"]["faults"]
        seis = m["seismic"]
        fault_txt = ", ".join(f"{f['dip_deg']:.0f}deg {f['dips']} throw {f['throw']:.1f}" for f in faults) or "none"
        unc = m["geology"]["unconformities"]
        clips = ", ".join(f"{c:.2f}{'*' if k else ''}" for c, k in zip(seis["clips"], seis["clipped"]))
        self._show_image(
            image, infos, f"{timing}rendered in {render_s:.2f}s",
            f"Block {block.index}: {len(faults)} faults ({fault_txt})  |  unconformities {unc}  |  "
            f"chaotic zones {seis['chaotic_zones']}  |  centers {', '.join(f'{h:.0f}' for h in seis['centers_hz'])} Hz  |  "
            f"max |amp| {clips} (* clipped)  |  fault pixels {m['fault_fraction'][job.slice_index] * 100:.2f}%")

    def _locate(self, event) -> tuple[TileInfo, int, int] | None:
        ox, oy = self.preview_offset
        scale = getattr(self, "preview_scale", 1.0) or 1.0
        px, py = (event.x - ox) / scale, (event.y - oy) / scale
        for info in self.preview_infos:
            if info.x <= px < info.x + info.w and info.y <= py < info.y + info.h and info.array is not None:
                H, W = info.array.shape[:2]
                col = int((px - info.x) / info.w * W)
                row = int((py - info.y) / info.h * H)
                return info, min(row, H - 1), min(col, W - 1)
        return None

    def _on_motion(self, event) -> None:
        hit = self._locate(event)
        if hit is None:
            return
        info, row, col = hit
        self.readout_text.set(f"{info.label}   x={col}  y={row}   value = {info.array[row, col]:.4f}")

    def _on_double(self, event) -> None:
        hit = self._locate(event)
        if hit is None:
            return
        zoom = ZoomWindow(self.root, hit[0])
        self.zoom_windows.append(zoom)
        zoom.top.bind("<Destroy>", lambda e, z=zoom: self.zoom_windows.remove(z)
                      if e.widget is z.top and z in self.zoom_windows else None)

    def _show_image(self, image: Image.Image, infos: list[TileInfo], status: str, info: str) -> None:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        box_w, box_h = max(cw, 8), max(ch, 8)
        scale = min(1.0, (box_w - 4) / max(image.width, 1), (box_h - 4) / max(image.height, 1))
        if scale < 1.0:
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
        self.preview_scale = scale
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        ox, oy = max(0, (cw - image.width) // 2), max(0, (ch - image.height) // 2)
        self.canvas.create_image(ox, oy, image=self.preview_photo, anchor="nw")
        self.preview_offset = (ox, oy)
        self.preview_infos = infos
        self.preview_status.set(status)
        self.info_text.set(info)

    def _show_canvas_message(self, text: str) -> None:
        self.preview_photo = None
        self.preview_infos = []
        self.preview_offset = (0, 0)
        self.preview_scale = 1.0
        self.canvas.delete("all")
        self.canvas.create_text(24, 24, text=text, anchor="nw", fill=MUTED, font=("Segoe UI", 12), width=720)
        self.preview_status.set("")
        self.info_text.set("")

    # --- Data viewer -----------------------------------------------------------------

    def _on_tab(self, _event=None) -> None:
        try:
            name = self.notebook.tab(self.notebook.select(), "text")
        except tk.TclError:
            return
        if name == "Data Viewer":
            if not self._disk_mode:
                self._enter_disk_mode()
        elif self._disk_mode:
            self._leave_disk_mode()

    def _enter_disk_mode(self) -> None:
        if getattr(self, "_view_job", None):
            try:
                self.root.after_cancel(self._view_job)
            except tk.TclError:
                pass
            self._view_job = None
        self._disk_mode = True
        self.preview_bar.pack_forget()
        self.viewer_banner.pack(fill="x", before=self.canvas_holder)
        self._render_disk()

    def _leave_disk_mode(self) -> None:
        self._disk_mode = False
        if self._disk_after is not None:
            try:
                self.root.after_cancel(self._disk_after)
            except tk.TclError:
                pass
            self._disk_after = None
        self.viewer_banner.pack_forget()
        self.preview_bar.pack(fill="x", before=self.canvas_holder)
        self.request_preview()

    def _schedule_disk(self, delay: int = 60) -> None:
        if self._viewer_loading or not self._disk_mode or self._closing:
            return
        if self._disk_after is not None:
            try:
                self.root.after_cancel(self._disk_after)
            except tk.TclError:
                pass
        self._disk_after = self.root.after(delay, self._render_disk)

    def _on_overlay_opacity(self, *_args) -> None:
        try:
            pct = int(round(float(self.viewer_opacity.get()) * 100))
        except (tk.TclError, ValueError):
            return
        self.viewer_opacity_text.set(f"{pct}%")
        self._schedule_disk(40)

    def _color_for_key(self, key: str) -> tuple[int, int, int]:
        saved = self._overlay_colors.get(key)
        if saved is not None:
            return saved
        src = next((folder for folder in (self.catalog.arrays if self.catalog else []) if folder.key == key), None)
        if src is None:
            return default_overlay_color("fault")
        return default_overlay_color(src.kind, src.scale)

    def _overlay_targets(self) -> list:
        """Arrays painted on top of the base image while Overlay is on."""
        cat = self.catalog
        if cat is None or not self.viewer_overlay.get():
            return []
        checked = [src for src in cat.arrays if self.viewer_checks.get(src.key) and self.viewer_checks[src.key].get()]
        if not checked:
            return []
        base = next((src for src in checked if src.kind == "seis"), checked[0])
        layers = [src for src in checked if src.key != base.key]
        kinds = {src.kind for src in checked}
        if self.viewer_faults.get() and "fault" not in kinds:
            fault = next((src for src in cat.arrays if src.kind == "fault"), None)
            if fault is not None:
                layers.append(fault)
        if self.viewer_horizons.get() and "csd_mask" not in kinds:
            horizon = next((src for src in cat.arrays if src.kind == "csd_mask"), None)
            if horizon is not None:
                layers.append(horizon)
        return layers

    def _overlay_chip_signature(self) -> tuple:
        if not self.viewer_overlay.get():
            return ("off",)
        return tuple((src.key, _rgb_hex(self._color_for_key(src.key))) for src in self._overlay_targets())

    def _refresh_overlay_chips(self) -> None:
        signature = self._overlay_chip_signature()
        if signature == self._overlay_chip_sig:
            return
        self._overlay_chip_sig = signature
        frame = self.viewer_colors_frame
        for child in frame.winfo_children():
            child.destroy()
        on = bool(self.viewer_overlay.get())
        self.viewer_opacity_scale.state(["!disabled"] if on else ["disabled"])
        if not on:
            ttk.Label(frame, text="Turn this on to paint the checked arrays on the seismic.",
                      style="Hint.TLabel").grid(row=0, column=0, sticky="w")
            return
        layers = self._overlay_targets()
        if not layers:
            ttk.Label(frame, text="Check another array to lay it on the seismic.",
                      style="Hint.TLabel").grid(row=0, column=0, sticky="w")
            return
        for i, src in enumerate(layers):
            color = _rgb_hex(self._color_for_key(src.key))
            btn = tk.Button(frame, bg=color, activebackground=color, width=2, relief="flat", cursor="hand2",
                            command=lambda key=src.key: self._pick_overlay_color(key))
            btn.grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=(0, 4), pady=1)
            ttk.Label(frame, text=src.name, style="Hint.TLabel").grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=(0, 12))

    def _pick_overlay_color(self, key: str) -> None:
        from tkinter import colorchooser

        chosen = colorchooser.askcolor(color=_rgb_hex(self._color_for_key(key)), parent=self.root,
                                       title="Overlay color")
        if not chosen or not chosen[1]:
            return
        rgb = _hex_rgb(chosen[1])
        if rgb is None:
            return
        self._overlay_colors[key] = rgb
        self._overlay_chip_sig = None
        self._refresh_overlay_chips()
        self._schedule_disk(1)

    def _on_viewer_control(self, *_args) -> None:
        if self._viewer_loading or not self._disk_mode or self._closing:
            return
        self._refresh_overlay_chips()
        self._render_disk()

    def _set_viewer_status(self, text: str, error: bool = False) -> None:
        self.viewer_status.set(text)
        self.viewer_status_label.configure(style="Err.TLabel" if error else "Hint.TLabel")

    def _viewer_chosen(self, catalog) -> set[str]:
        available = {folder.key for folder in catalog.arrays}
        previous = {key for key, var in self.viewer_checks.items() if var.get()}
        if self.viewer_checks and previous & available:
            return previous & available
        saved = self._saved_viewer_arrays
        if not self.viewer_checks and isinstance(saved, list) and set(saved) & available:
            return set(saved) & available
        return default_keys(catalog.arrays)

    def _fill_viewer_checks(self, catalog) -> None:
        chosen = self._viewer_chosen(catalog)
        for child in self.viewer_checks_frame.winfo_children():
            child.destroy()
        self.viewer_checks.clear()
        if not catalog.arrays:
            ttk.Label(self.viewer_checks_frame, text="No arrays to show.", style="Hint.TLabel").grid(
                row=0, column=0, sticky="w")
            return
        for i, src in enumerate(catalog.arrays):
            var = tk.BooleanVar(value=src.key in chosen)
            var.trace_add("write", lambda *_: self._on_viewer_control())
            ttk.Checkbutton(self.viewer_checks_frame, text=src.name, variable=var).grid(
                row=i // 2, column=i % 2, sticky="w", padx=(0, 12), pady=1)
            self.viewer_checks[src.key] = var

    def _apply_catalog(self, catalog) -> None:
        self._viewer_loading = True
        try:
            self.catalog = catalog
            self._fill_viewer_checks(catalog)
            self._set_viewer_status(catalog_summary(catalog), error=not catalog.arrays)
            has_fault = any(folder.kind == "fault" for folder in catalog.arrays)
            has_horizon = any(folder.kind == "csd_mask" for folder in catalog.arrays)
            self.viewer_fault_check.configure(state="normal" if has_fault else "disabled")
            self.viewer_horizon_check.configure(state="normal" if has_horizon else "disabled")
            self._overlay_chip_sig = None
            self._refresh_overlay_chips()
            if catalog.ids:
                lo, hi = catalog.ids[0], catalog.ids[-1]
                self.viewer_spin.configure(from_=lo, to=hi)
                cur = self._int(self.viewer_id, lo)
                if cur < lo or cur > hi:
                    self.viewer_id.set(lo)
            self.viewer_place.set(describe_id(catalog.layout, self._int(self.viewer_id, 0)) if catalog.ids else "")
        finally:
            self._viewer_loading = False

    def _render_disk(self) -> None:
        self._disk_after = None
        if self._closing or not self._disk_mode:
            return
        try:
            self._draw_folder_sample()
        except Exception:  # noqa: BLE001 - shown in the console
            self.preview_status.set("Could not draw this sample.")
            self._log(traceback.format_exc(), "err")

    def _draw_folder_sample(self) -> None:
        cat = self.catalog
        if cat is None:
            self.viewer_banner_text.set("Choose a folder and press View.")
            self.viewer_place.set("")
            self._show_canvas_message("Choose a folder in Data Viewer and press View.")
            return
        if not cat.arrays:
            self.viewer_banner_text.set(str(cat.root))
            self.viewer_place.set("")
            self._show_canvas_message(catalog_summary(cat))
            return
        keys = {key for key, var in self.viewer_checks.items() if var.get()}
        if not keys:
            self.viewer_banner_text.set(str(cat.root))
            self._show_canvas_message("Select at least one array.")
            self.preview_status.set("Select at least one array.")
            return
        sid = self._int(self.viewer_id, cat.ids[0])
        place = describe_id(cat.layout, sid)
        self.viewer_place.set(place)
        self.viewer_banner_text.set(f"{place}    {cat.root}")
        overlay_fault = bool(self.viewer_faults.get()) and any(folder.kind == "fault" for folder in cat.arrays)
        overlay_horizon = bool(self.viewer_horizons.get()) and any(folder.kind == "csd_mask" for folder in cat.arrays)
        sample = load_sample(cat, sid, keys, overlay_fault, overlay_horizon)
        if not sample.sections:
            detail = sample.errors[0] if sample.errors else f"No section files for id {sid} in the selected arrays."
            self._show_canvas_message(detail)
            self.preview_status.set(detail)
            return
        cw, ch = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        stack = bool(self.viewer_overlay.get())
        tile = fit_sample_tile(1 if stack else len(sample.sections), cw - 8, ch - 8)
        cmap = self.viewer_cmap.get()
        if cmap not in SEISMIC_CMAPS:
            cmap = "gray"
        colors = {src.key: self._color_for_key(src.key) for src in cat.arrays} if stack else None
        try:
            opacity = float(self.viewer_opacity.get())
        except (tk.TclError, ValueError):
            opacity = 0.4
        image, infos = render_sample(cat, sample, sid, cmap, bool(self.viewer_faults.get()), tile,
                                      show_horizons=bool(self.viewer_horizons.get()), stack=stack,
                                      stack_opacity=opacity, stack_colors=colors)
        parts = []
        seis = next((arr for src, arr in sample.sections if src.kind == "seis"), None)
        if seis is not None and seis.size:
            parts.append(f"seismic  min {seis.min():.4g}  max {seis.max():.4g}  mean {seis.mean():.4g}")
        if sample.fault is not None and self.viewer_faults.get():
            parts.append(f"fault pixels {float((sample.fault > 0.5).mean()) * 100:.2f}%")
        if sample.horizon is not None and self.viewer_horizons.get():
            parts.append(f"horizon pixels {float((sample.horizon > 0.5).mean()) * 100:.2f}%")
        if stack:
            names = [src.name for src in self._overlay_targets()]
            listed = ", ".join(names) if names else "nothing else checked"
            parts.append(f"overlay {int(round(opacity * 100))}%: {listed}")
        if cat.shape:
            parts.append(f"{cat.shape[0]} x {cat.shape[1]}")
        if sample.errors:
            parts.append(sample.errors[0])
        n = len(sample.sections)
        status = "overlay" if stack else f"{n} array" + ("s" if n != 1 else "")
        self._show_image(image, infos, status, "   |   ".join(parts))

    def _view_folder(self) -> None:
        path_text = self.viewer_dir.get().strip()
        if not path_text:
            self._set_viewer_status("Choose a folder first.", error=True)
            return
        path = Path(path_text).expanduser()
        self._set_viewer_status("Reading folder...")
        self.root.update_idletasks()
        try:
            catalog = scan_dataset(path)
        except (OSError, ValueError) as exc:
            self._set_viewer_status(str(exc), error=True)
            return
        self._apply_catalog(catalog)
        if not self._disk_mode:
            self._enter_disk_mode()
        else:
            self._render_disk()

    def _browse_viewer(self) -> None:
        before = self.viewer_dir.get()
        self._browse_dir(self.viewer_dir)
        if self.viewer_dir.get() != before and Path(self.viewer_dir.get()).is_dir():
            self._view_folder()

    def _use_output_folder(self) -> None:
        path = self.vars["output.out_dir"].get().strip()
        if not path:
            self._set_viewer_status("Set an output folder on the Dataset tab first.", error=True)
            return
        self.viewer_dir.set(path)
        self._view_folder()

    def _step_viewer_id(self, delta: int) -> None:
        cat = self.catalog
        if cat is None or not cat.ids:
            return
        cur = self._int(self.viewer_id, cat.ids[0])
        nxt = step_id(cat.ids, cur, delta)
        if nxt != cur:
            self.viewer_id.set(nxt)

    def _viewer_select(self, mode: str) -> None:
        if not self.viewer_checks or self.catalog is None:
            return
        self._viewer_loading = True
        try:
            for src in self.catalog.arrays:
                self.viewer_checks[src.key].set(True if mode == "all" else src.kind == "seis")
            if mode == "seis" and not any(var.get() for var in self.viewer_checks.values()):
                self.viewer_checks[self.catalog.arrays[0].key].set(True)
        finally:
            self._viewer_loading = False
        self._on_viewer_control()

    # --- Generation ----------------------------------------------------------------------

    def start(self) -> None:
        if self.proc is not None:
            messagebox.showinfo("Generate dataset", "A run is already in progress.", parent=self.root)
            return
        cfg, errors = self.read_config()
        if errors:
            messagebox.showerror("Invalid settings", "\n".join(errors), parent=self.root)
            return
        if not cfg.output.out_dir.strip():
            messagebox.showerror("No output folder", "Choose an output folder first.", parent=self.root)
            return
        out = Path(cfg.output.out_dir)
        start = cfg.output.start_id if cfg.output.start_id >= 0 else next_start_id(out, cfg)
        try:
            pending, skipped, size = plan_blocks(out, cfg, start)
            ok, free = check_disk(out, size)
        except OSError as exc:
            messagebox.showerror("Output folder", str(exc), parent=self.root)
            return
        if not ok:
            messagebox.showerror("Not enough disk space",
                                 f"The missing files need {human_bytes(size)} but only {human_bytes(free)} is free.",
                                 parent=self.root)
            return
        if not pending:
            messagebox.showinfo("Generate dataset", f"All {skipped:,} blocks from id {start:,} are already complete "
                                                    "in this folder. Tick 'Rewrite existing blocks' to redo them.",
                                parent=self.root)
            return
        ids = cfg.output.num_blocks * cfg.ids_per_block
        done_note = f"\n{skipped:,} of them are already complete and will be skipped." if skipped else ""
        if not messagebox.askokcancel(
                "Generate dataset",
                f"Write {cfg.output.num_blocks:,} blocks ({ids:,} ids {start:,}-{start + ids - 1:,}, "
                f"{human_bytes(size)} of new files) to\n{out}?{done_note}", parent=self.root):
            return
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path = RUN_DIR / "last_run_config.json"
        cfg.save(cfg_path)
        self.stop_file.unlink(missing_ok=True)
        self.stop_sent = False
        cmd = [_console_python(), "-u", "-m", "synthetic_generator.cli", "generate", "--config", str(cfg_path),
               "--events", "--stop-file", str(self.stop_file)]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        except OSError as exc:
            messagebox.showerror("Could not start", str(exc), parent=self.root)
            self.proc = None
            return
        self.run_started = time.time()
        self.progress.configure(value=0.0)
        self._set_stage("Starting", WARN)
        self.progress_text.set("")
        self.start_btn.configure(state="disabled")
        self.resume_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal", text="Stop")
        self._log(f"Generating into {out}", "info")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def resume(self) -> None:
        if self.proc is not None:
            messagebox.showinfo("Resume", "Wait for the current run to finish first.", parent=self.root)
            return
        out = Path(self.vars["output.out_dir"].get().strip() or ".")
        record = out / CONFIG_NAME
        if not record.is_file():
            messagebox.showinfo("Resume", f"No previous run found in\n{out}", parent=self.root)
            return
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Resume", str(exc), parent=self.root)
            return
        cfg = GeneratorConfig.from_dict(data)
        cfg.output.out_dir = str(out)
        cfg.output.start_id = int(data.get("run", {}).get("start_id", cfg.output.start_id))
        cfg.output.overwrite = False
        self._write_config(cfg)
        self._log(f"Loaded the last run of {out}: first id {cfg.output.start_id}, {cfg.output.num_blocks} blocks, "
                  f"seed {cfg.output.seed}. Complete blocks will be skipped.", "info")
        self.root.after(600, self.start)

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.gen_q.put(("line", line.rstrip("\n")))
        self.gen_q.put(("exit", proc.wait()))

    def stop(self) -> None:
        if not self.proc:
            return
        if not self.stop_sent:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("stop", encoding="utf-8")
            self.stop_sent = True
            self._set_stage("Stopping", WARN)
            self.stop_btn.configure(text="Force stop")
            self._log("Stop requested: finishing the blocks in progress...", "warn")
            return
        self._kill()

    def _kill(self) -> None:
        if not self.proc:
            return
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self.proc.kill()
        self._log("Generator process killed; the last blocks may be incomplete (they are redone on resume).", "err")

    def _handle_event(self, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "start":
            self.run_total = max(1, ev["total"])
            self._set_stage("Running", ACCENT_HOVER)
            self._log(f"{ev['total']} blocks to write ({ev['skipped']} already complete), first id {ev['start_id']}, "
                      f"{ev['workers']} workers, {human_bytes(ev['bytes'])}.", "info")
            self._log("Folders: " + ", ".join(ev["folders"]), "muted")
        elif kind == "block":
            self.progress.configure(value=ev["done"] / self.run_total)
            eta = f", ETA {fmt_duration(ev['eta'])}" if ev.get("eta") is not None else ""
            rate = ev["done"] / ev["elapsed"] if ev.get("elapsed") else 0
            self.progress_text.set(f"Block {ev['done']}/{ev['total']}  |  ids {ev['ids'][0]}-{ev['ids'][1]}  |  "
                                   f"{rate:.1f} blocks/s{eta}")
            self._log(f"block {ev['block']}: ids {ev['ids'][0]}-{ev['ids'][1]}, {ev['faults']} faults, "
                      f"bandpass {', '.join(f'{h:.0f}' for h in ev['centers_hz'])} Hz ({ev['seconds']:.2f}s)", "muted")
        elif kind == "stopping":
            self._set_stage("Stopping", WARN)
        elif kind in ("done", "stopped"):
            tone = OK if kind == "done" else WARN
            self._set_stage("Done" if kind == "done" else "Stopped", tone)
            self._log(f"{kind.capitalize()}: {ev['written']} blocks written, {ev['skipped']} skipped, "
                      f"{fmt_duration(ev['seconds'])}. Next free id {ev['next_id']}.", "ok" if kind == "done" else "warn")
            self.progress.configure(value=1.0 if kind == "done" else self.progress["value"])
        elif kind == "error":
            self._set_stage("Error", ERR)
            self._log(ev.get("message", "error"), "err")
            messagebox.showerror("Generation failed", ev.get("message", "error"), parent=self.root)

    def _finished(self, code: int) -> None:
        self.proc = None
        self._start_cache = None
        self.stop_btn.configure(state="disabled", text="Stop")
        self.stop_file.unlink(missing_ok=True)
        self._apply_change()
        if code != 0 and self.stage_text.get() not in ("Error", "Stopped"):
            self._set_stage("Error", ERR)
            self._log(f"Generator exited with code {code}.", "err")
        elif code == 0:
            self.inspect(quiet=True)

    # --- Inspect / fine-tuner ---------------------------------------------------------------

    def inspect(self, quiet: bool = False) -> None:
        out = self.vars["output.out_dir"].get().strip()
        if not out or not Path(out).is_dir():
            if not quiet:
                messagebox.showinfo("Inspect", "The output folder does not exist yet.", parent=self.root)
            return
        self._log(f"Inspecting {out} ...", "info")

        def work() -> None:
            try:
                from synthetic_generator.checks import inspect_dataset

                rep = inspect_dataset(out, sample=60)
                self.gen_q.put(("inspect", rep))
            except Exception:  # noqa: BLE001
                self.gen_q.put(("line", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _show_inspect(self, rep: dict) -> None:
        folders = rep.get("folders", {})
        self._log(f"{len(folders)} folders, {rep.get('common_ids', 0):,} ids in every folder"
                  + (f", image {rep['image_size']}x{rep['image_size']}" if "image_size" in rep else ""), "info")
        if "finetuner_pairs" in rep:
            self._log(f"Quick Fine-Tuner sees {rep['finetuner_pairs']:,} Seis/Fault pairs.", "ok")
        if "finetuner_csd_pairs" in rep:
            self._log(f"Quick Fine-Tuner sees {rep['finetuner_csd_pairs']:,} Seis/CsdHorizon pairs.", "ok")
        if "replicas_share_labels" in rep:
            self._log(f"Labels repeated across replicas: {rep['replicas_share_labels']}", "muted")
        st = rep.get("stats", {})
        if st:
            keys = ("seis_std", "seis_absmax", "seis_centroid", "fault_fraction", "fault_angle", "fault_row_width")
            self._log("  ".join(f"{k}={st[k]:.3f}" for k in keys if k in st), "muted")
        for p in rep.get("problems", []):
            self._log("Problem: " + p, "err")

    def send_to_finetuner(self, mode: str) -> None:
        cfg, _ = self.read_config()
        spec = next((v for v in FINETUNE_PRESETS.values() if v and v["task"] == mode), None)
        if spec is None:
            messagebox.showerror("Quick Fine-Tuner", f"Unknown preset {mode!r}.", parent=self.root)
            return

        root = Path(cfg.output.out_dir)
        image_name, label_name = folders_for(spec, cfg.output.suffix)
        image, label = root / image_name, root / label_name
        if not image.is_dir() or not label.is_dir():
            messagebox.showinfo("Quick Fine-Tuner", f"Generate a dataset first: {image.name} and {label.name} are "
                                                    f"needed in\n{root}", parent=self.root)
            return
        try:
            from finetuning.finetune_gui import LABEL_MODES, SETTINGS_FILE as FT_SETTINGS
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Quick Fine-Tuner", f"Cannot load the fine-tuner: {exc}", parent=self.root)
            return
        settings = {}
        if FT_SETTINGS.exists():
            try:
                settings = json.loads(FT_SETTINGS.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                settings = {}
        preset_name = next(k for k, v in FINETUNE_PRESETS.items() if v and v["task"] == mode)
        settings.update({
            "preset": preset_name, "data_root": str(root), "image_dir": str(image), "label_dir": str(label),
            "label_mode": next(k for k, v in LABEL_MODES.items() if v == spec["mode"]),
            "prompt": spec["prompt"],
        })
        FT_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        FT_SETTINGS.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.Popen([_windowed_python(), "-m", "finetuning.finetune_gui"], cwd=str(REPO_ROOT), creationflags=flags)
        self._log(f"Quick Fine-Tuner opened with {image.name} + {label.name} from {root}.", "ok")

    # --- Misc ---------------------------------------------------------------------------------

    def _apply_preset(self) -> None:
        current, _ = self.read_config()
        name = self.preset_var.get()
        cfg = preset(name, current)
        if name.startswith("Faults only"):
            cfg.output.arrays = ["seis", "fault"]
        self._write_config(cfg)
        if name.startswith("CSD") or self.view_var.get() == "csd":
            self.view_var.set("overview")
        self._log(f"Preset: {name}", "info")

    def _reset(self) -> None:
        cfg = default_config()
        cfg.output.out_dir = str(DEFAULT_OUT_DIR)
        self.preset_var.set(DEFAULT_PRESET)
        self._write_config(cfg)

    def _load_json(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Settings", "*.json")],
                                          initialdir=str(REPO_ROOT / "outputs"))
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._write_config(GeneratorConfig.from_dict(data.get("config", data)))
            self._log(f"Loaded settings from {path}", "info")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            messagebox.showerror("Load settings", str(exc), parent=self.root)

    def _save_json(self) -> None:
        cfg, errors = self.read_config()
        if errors:
            messagebox.showerror("Save settings", "\n".join(errors), parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                            filetypes=[("Settings", "*.json")], initialdir=str(REPO_ROOT / "outputs"),
                                            initialfile="synthetic_settings.json")
        if path:
            cfg.save(path)
            self._log(f"Saved settings to {path}", "info")

    def _browse_dir(self, var: tk.StringVar) -> None:
        start = var.get() if Path(var.get() or ".").is_dir() else str(DEFAULT_OUT_DIR.parent)
        chosen = filedialog.askdirectory(initialdir=start, parent=self.root)
        if chosen:
            var.set(str(Path(chosen)))

    def _open_output(self) -> None:
        out = Path(self.vars["output.out_dir"].get() or ".")
        target = out if out.is_dir() else out.parent
        if target.is_dir():
            if sys.platform == "win32":
                os.startfile(str(target))  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", str(target)])

    def _set_stage(self, text: str, color: str) -> None:
        self.stage_text.set(text)
        ttk.Style(self.root).configure("Stage.TLabel", foreground=color)

    def _log(self, text: str, tag: str = "") -> None:
        self.log.configure(state="normal")
        self.log.insert("end", time.strftime("%H:%M:%S  "), ("muted",))
        self.log.insert("end", text + "\n", (tag,) if tag else ())
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 3000:
            self.log.delete("1.0", f"{lines - 3000}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain(self) -> None:
        try:
            self._drain_preview()
            self._drain_generation()
            self._drain_probe()
        except Exception:  # noqa: BLE001 - keep the event loop alive, show the error
            self._log(traceback.format_exc(), "err")
        finally:
            if not self._closing:
                self.root.after(60, self._drain)

    def _drain_preview(self) -> None:
        while True:
            try:
                item = self.preview_q.get_nowait()
            except queue.Empty:
                return
            self.preview_busy = False
            try:
                if item[0] == "ok":
                    if not self._disk_mode:
                        _, job, block, image, infos, gen_s, render_s = item
                        self._show_preview(job, block, image, infos, gen_s, render_s)
                elif self._disk_mode:
                    self._log(item[2], "err")
                else:
                    self.preview_status.set("Preview failed - see the console.")
                    self._log(item[2], "err")
            finally:
                if self.preview_again and not self._disk_mode:
                    self._start_preview(self._pending_job)
                else:
                    self.preview_again = False

    def _drain_generation(self) -> None:
        while True:
            try:
                kind, payload = self.gen_q.get_nowait()
            except queue.Empty:
                return
            if kind == "line":
                if payload.startswith(EVENT_PREFIX):
                    try:
                        self._handle_event(json.loads(payload[len(EVENT_PREFIX):]))
                    except json.JSONDecodeError:
                        self._log(payload)
                elif payload.strip():
                    self._log(payload, "warn")
            elif kind == "exit":
                self._finished(payload)
            elif kind == "inspect":
                self._show_inspect(payload)

    def _save_settings(self) -> None:
        cfg, _ = self.read_config()
        data = {
            "config": cfg.to_dict(), "preset": self.preset_var.get(), "view": self.view_var.get(),
            "block": self._int(self.block_var, 0), "auto": bool(self.auto_var.get()), "cmap": self.cmap_var.get(),
            "faults": bool(self.faults_var.get()), "layers": self._int(self.layers_var, 8),
            "viewer_dir": self.viewer_dir.get(), "viewer_id": self._int(self.viewer_id, 0),
            "viewer_cmap": self.viewer_cmap.get(), "viewer_faults": bool(self.viewer_faults.get()),
            "viewer_horizons": bool(self.viewer_horizons.get()),
            "viewer_overlay": bool(self.viewer_overlay.get()),
            "viewer_overlay_opacity": float(self.viewer_opacity.get()),
            "viewer_overlay_colors": {key: _rgb_hex(color) for key, color in self._overlay_colors.items()},
            "viewer_arrays": ([key for key, var in self.viewer_checks.items() if var.get()]
                              if self.viewer_checks else self._saved_viewer_arrays),
        }
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_settings(self) -> None:
        data = {}
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        if isinstance(data.get("config"), dict):
            cfg = GeneratorConfig.from_dict(data["config"])
            if data.get("preset") not in PRESETS:
                cfg = preset(DEFAULT_PRESET, cfg)
        else:
            cfg = default_config()
        if not cfg.output.out_dir:
            cfg.output.out_dir = str(DEFAULT_OUT_DIR)
        if data.get("preset") in PRESETS:
            self.preset_var.set(data["preset"])
        if data.get("view") in VIEWS:
            self.view_var.set(data["view"])
        if data.get("cmap") in SEISMIC_CMAPS:
            self.cmap_var.set(data["cmap"])
        for key, var in (("block", self.block_var), ("auto", self.auto_var), ("faults", self.faults_var),
                         ("layers", self.layers_var)):
            if key in data:
                var.set(data[key])
        saved_arrays = data.get("viewer_arrays")
        if isinstance(saved_arrays, list):
            self._saved_viewer_arrays = [str(item) for item in saved_arrays]
        if isinstance(data.get("viewer_dir"), str):
            self.viewer_dir.set(data["viewer_dir"])
        viewer_id = data.get("viewer_id")
        if isinstance(viewer_id, (int, float)) and not isinstance(viewer_id, bool):
            self.viewer_id.set(int(viewer_id))
        if data.get("viewer_cmap") in SEISMIC_CMAPS:
            self.viewer_cmap.set(data["viewer_cmap"])
        if isinstance(data.get("viewer_faults"), bool):
            self.viewer_faults.set(data["viewer_faults"])
        if isinstance(data.get("viewer_horizons"), bool):
            self.viewer_horizons.set(data["viewer_horizons"])
        if isinstance(data.get("viewer_overlay"), bool):
            self.viewer_overlay.set(data["viewer_overlay"])
        opacity = data.get("viewer_overlay_opacity")
        if isinstance(opacity, (int, float)) and not isinstance(opacity, bool):
            self.viewer_opacity.set(float(min(0.85, max(0.1, opacity))))
        saved_colors = data.get("viewer_overlay_colors")
        if isinstance(saved_colors, dict):
            parsed: dict[str, tuple[int, int, int]] = {}
            for key, value in saved_colors.items():
                rgb = _hex_rgb(str(value))
                if rgb is not None:
                    parsed[str(key)] = rgb
            self._overlay_colors = parsed
            self._overlay_chip_sig = None
        self._refresh_overlay_chips()
        self._write_config(cfg)
        self._dark_titlebar()
        self._log("Ready. Preview block k is exactly dataset block k for the same seed. The Data Viewer tab opens a "
                  "folder of sections. Hover a tile to read values, double-click to zoom; hover a setting's (i) for help.",
                  "info")

    def shutdown(self) -> None:
        """Stop scheduling work and wait for a running preview, before the window is destroyed."""
        self._closing = True
        if self._disk_after is not None:
            try:
                self.root.after_cancel(self._disk_after)
            except tk.TclError:
                pass
            self._disk_after = None
        thread = self.preview_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=15)
        probe = self._probe_thread
        if probe is not None and probe.is_alive():
            probe.join(timeout=15)

    def _on_close(self) -> None:
        if self.proc and not messagebox.askyesno(
                "Generation running", "A dataset is still being written. Stop it and close?", parent=self.root):
            return
        if self.proc:
            self._kill()
        self._save_settings()
        self.shutdown()
        self.root.destroy()


class ZoomWindow:
    """One array at a large size with a value readout."""

    def __init__(self, master: tk.Misc, info: TileInfo):
        from synthetic_generator.preview import colorize, dip_range, seismic_rgb

        a = info.array
        self.array = a
        top = tk.Toplevel(master)
        self.top = top
        top.title(info.label)
        top.configure(bg=APP_BG)
        lower = info.label.lower()
        if lower.startswith("seis"):
            rgb = seismic_rgb(a, "gray")
        elif lower.startswith("csddip"):
            vals = a[a != 0]
            rgb = colorize(a, "csd", *dip_range(vals if vals.size else a))
        elif lower.startswith("csddistance"):
            rgb = colorize(a, "inferno", 0, float(a.max()) or 1.0)
        elif lower.startswith("dip"):
            m = float(np.percentile(np.abs(a), 99)) or 1.0
            rgb = colorize(a, "diverging", -m, m)
        elif lower.startswith("halo"):
            rgb = colorize(a, "inferno", 0, 1)
        elif lower.startswith("tau"):
            rgb = colorize(a, "viridis", float(a.min()), float(a.max()))
        else:
            rgb = colorize(a, "gray", float(a.min()), float(a.max()) if a.max() > a.min() else a.min() + 1)
        side = int(min(top.winfo_screenheight() * 0.75, 900))
        H, W = a.shape
        scale = side / max(H, W)
        self.size = (int(W * scale), int(H * scale))
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize(self.size, Image.NEAREST))
        canvas = tk.Canvas(top, width=self.size[0], height=self.size[1], bg=APP_BG, highlightthickness=0)
        canvas.pack(padx=10, pady=(10, 4))
        canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.text = tk.StringVar(value=f"{info.label}  {W}x{H}  min {a.min():.4g}  max {a.max():.4g}")
        tk.Label(top, textvariable=self.text, bg=APP_BG, fg=FG, font=("Segoe UI", 10)).pack(pady=(0, 10))
        canvas.bind("<Motion>", self._motion)
        self.label = info.label

    def _motion(self, event) -> None:
        H, W = self.array.shape
        col = min(W - 1, max(0, int(event.x / self.size[0] * W)))
        row = min(H - 1, max(0, int(event.y / self.size[1] * H)))
        self.text.set(f"{self.label}   x={col}  y={row}   value = {self.array[row, col]:.5f}")


def _rgb_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(channel) for channel in color))


def _hex_rgb(text: str) -> tuple[int, int, int] | None:
    text = text.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text
    return str(value)


def main() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    GeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
