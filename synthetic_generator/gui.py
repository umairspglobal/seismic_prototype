"""Desktop GUI for the synthetic seismic dataset generator.

    python -m synthetic_generator          (or double-click start_synthetic_generator.cmd)

Left: every generator setting, grouped in tabs, plus the dataset card (output
folder, size estimate, Generate / Stop, progress). Right: a live preview of the
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

from synthetic_generator import __version__  # noqa: E402
from synthetic_generator.block import Block, generate_block, products  # noqa: E402
from synthetic_generator.config import (ARRAY_KINDS, DEFAULT_PRESET, PRESETS, GeneratorConfig,  # noqa: E402
                                        default_config, preset)
from synthetic_generator.dataset import (CONFIG_NAME, check_disk, dataset_bytes, file_bytes,  # noqa: E402
                                         next_start_id, plan_blocks)
from synthetic_generator.params import TABS, Param, all_params  # noqa: E402
from synthetic_generator.preview import SEISMIC_CMAPS, VIEWS, TileInfo, fit_tile, render_view  # noqa: E402

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
    seconds = int(max(0, seconds))
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
        self.preview_busy = False
        self.preview_again = False
        self.cached_key: str | None = None
        self.cached_block: Block | None = None
        self.preview_photo = None
        self.preview_infos: list[TileInfo] = []
        self.preview_offset = (0, 0)
        self.last_gen_seconds = 0.5
        self.proc: subprocess.Popen | None = None
        self.stop_file = RUN_DIR / "stop.flag"
        self.stop_sent = False
        self.run_total = 0
        self.run_started = 0.0
        self.zoom_windows: list[ZoomWindow] = []
        self._start_cache: tuple[tuple, int, float] | None = None
        self.preview_thread: threading.Thread | None = None
        self._closing = False

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
        self.errors_text = tk.StringVar()
        self.preview_status = tk.StringVar(value="")
        self.readout_text = tk.StringVar(value="Hover the preview to read values; double-click a tile to zoom.")
        self.info_text = tk.StringVar()
        self.stage_text = tk.StringVar(value="Idle")
        self.progress_text = tk.StringVar()
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
            "CsdHorizon: 1 on the tracked horizons (the target mask).\n"
            "CsdDipXscaleS: the prompt - dipXscaleS on the horizon pixels, the background value elsewhere.\n"
            "CsdDistance: distance in px to the nearest horizon pixel.\n"
            "Open the CSD preview view to see them laid out per scale."
        )
        ttk.Label(card, text=notes, style="Hint.TLabel", justify="left", wraplength=480).grid(
            row=1, column=0, columnspan=3, sticky="w")

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
        self.start_btn = ttk.Button(buttons, text="Generate dataset", style="Accent.TButton", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop", style="Danger.TButton", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        resume = ttk.Button(buttons, text="Resume last run", style="Tool.TButton", command=self.resume)
        resume.pack(side="left")
        self.resume_btn = resume
        Tooltip(resume, "Reloads the settings of the last run in the output folder (same seed and first id) and "
                        "writes only the blocks that are missing or incomplete.")
        tools = ttk.Frame(card, style="Card.TFrame")
        tools.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Button(tools, text="Open folder", style="Tool.TButton", command=self._open_output).pack(side="left")
        ttk.Button(tools, text="Inspect", style="Tool.TButton", command=self.inspect).pack(side="left", padx=4)
        self.ft_btn = ttk.Menubutton(tools, text="Send to Quick Fine-Tuner", style="Tool.TMenubutton")
        menu = tk.Menu(self.ft_btn, tearoff=0, bg=FIELD, fg=FG, activebackground=ACCENT, activeforeground="#fff")
        menu.add_command(label="Faults model (Seis + Fault)", command=lambda: self.send_to_finetuner("faults"))
        menu.add_command(label="Facies model (Seis + layers from TauAtT)",
                         command=lambda: self.send_to_finetuner("facies"))
        self.ft_btn["menu"] = menu
        self.ft_btn.pack(side="left")
        Tooltip(self.ft_btn, "Points the Quick Fine-Tuner at this folder (Seis/Fault or Seis/TauAtT) and opens it.")
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

        holder = ttk.Frame(parent, style="App.TFrame")
        holder.pack(fill="both", expand=True, pady=(8, 0))
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

    def _update_summary(self, cfg: GeneratorConfig, errors: list[str]) -> None:
        if errors:
            self.summary_text.set("Fix the settings below to see the dataset size.")
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
        workers = cfg.output.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
        est = max(cfg.output.num_blocks * self.last_gen_seconds * 1.4 / workers, size / 300e6)
        lines.append(f"about {fmt_duration(est)} with {workers} workers (assuming ~300 MB/s disk writes)")
        self.summary_text.set("\n".join(lines))

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
        self._view_job = self.root.after(delay, lambda: self.request_preview(regenerate=regenerate))

    def _preview_key(self, cfg: GeneratorConfig, block_index: int) -> str:
        data = cfg.to_dict()
        data.pop("output", None)
        return json.dumps([data, cfg.output.seed, block_index], sort_keys=True)

    def request_preview(self, regenerate: bool = False) -> None:
        self._view_job = None
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
            block = cached or generate_block(job.cfg, job.block_index, csd=True)
            gen_s = time.perf_counter() - t0 if cached is None else None
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
            self._update_summary(job.cfg, [])
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min(1.0, (cw - 4) / image.width, (ch - 4) / image.height)
        if scale < 1.0:
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
        self.preview_scale = scale
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        ox, oy = max(0, (cw - image.width) // 2), max(0, (ch - image.height) // 2)
        self.canvas.create_image(ox, oy, image=self.preview_photo, anchor="nw")
        self.preview_offset = (ox, oy)
        self.preview_infos = infos
        timing = f"generated in {gen_s:.2f}s, " if gen_s is not None else ""
        self.preview_status.set(f"{timing}rendered in {render_s:.2f}s")
        m = block.meta
        faults = m["geology"]["faults"]
        seis = m["seismic"]
        fault_txt = ", ".join(f"{f['dip_deg']:.0f}deg {f['dips']} throw {f['throw']:.1f}" for f in faults) or "none"
        unc = m["geology"]["unconformities"]
        clips = ", ".join(f"{c:.2f}{'*' if k else ''}" for c, k in zip(seis["clips"], seis["clipped"]))
        self.info_text.set(
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
        root = Path(cfg.output.out_dir)
        sfx = cfg.output.suffix
        image = root / f"Seis{sfx}"
        label = root / (f"Fault{sfx}" if mode == "faults" else f"TauAtT{sfx}")
        if not image.is_dir() or not label.is_dir():
            messagebox.showinfo("Quick Fine-Tuner", f"Generate a dataset first: {image.name} and {label.name} are "
                                                    f"needed in\n{root}", parent=self.root)
            return
        try:
            from finetuning.finetune_gui import LABEL_MODES, PRESETS as FT_PRESETS, SETTINGS_FILE as FT_SETTINGS
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Quick Fine-Tuner", f"Cannot load the fine-tuner: {exc}", parent=self.root)
            return
        settings = {}
        if FT_SETTINGS.exists():
            try:
                settings = json.loads(FT_SETTINGS.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                settings = {}
        label_mode = "binary" if mode == "faults" else "rgt_layers"
        preset_name = next((k for k, v in FT_PRESETS.items()
                            if v and v["image"] == image.name and v["label"] == label.name), "Custom")
        settings.update({
            "preset": preset_name, "data_root": str(root), "image_dir": str(image), "label_dir": str(label),
            "label_mode": next(k for k, v in LABEL_MODES.items() if v == label_mode),
            "prompt": "fault" if mode == "faults" else "seismic facies",
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
                    _, job, block, image, infos, gen_s, render_s = item
                    self._show_preview(job, block, image, infos, gen_s, render_s)
                else:
                    self.preview_status.set("Preview failed - see the console.")
                    self._log(item[2], "err")
            finally:
                if self.preview_again:
                    self._start_preview(self._pending_job)

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
        self._write_config(cfg)
        self._dark_titlebar()
        self._log("Ready. Preview block k is exactly dataset block k for the same seed. Hover a tile to read values, "
                  "double-click to zoom; hover a setting's (i) for help.", "info")

    def shutdown(self) -> None:
        """Stop scheduling work and wait for a running preview, before the window is destroyed."""
        self._closing = True
        thread = self.preview_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=15)

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
