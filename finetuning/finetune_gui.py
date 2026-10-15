"""Small desktop GUI for ``finetuning.quick_finetune``.

    python -m finetuning.finetune_gui        (or double-click start_finetune_gui.cmd)

Pick a preset (faults / facies), check the folders, name the model and press
Start. Training runs in a child process; its log is shown underneath with
progress, metrics, a loss chart and validation previews alongside.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.quick_data import list_pairs  # noqa: E402

DEFAULT_DATA_ROOT = Path(r"C:\Users\umair_tariq\Downloads\Downloads\Data\training_10_7_26")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "finetuned_checkpoints"
SETTINGS_FILE = REPO_ROOT / "outputs" / "finetune_gui_settings.json"
EVENT_PREFIX = "@@"

PRESETS = {
    "Faults  (Seis75 + Fault75)": {
        "image": "Seis75", "label": "Fault75", "mode": "binary", "prompt": "fault", "name": "faults",
    },
    "Facies  (Seis75 + layers from TauAtT75)": {
        "image": "Seis75", "label": "TauAtT75", "mode": "rgt_layers", "prompt": "seismic facies", "name": "facies",
    },
    "Custom": None,
}
LABEL_MODES = {
    "Binary mask (values 0/1)": "binary",
    "Layers from relative geologic time": "rgt_layers",
}
MOSAICS = {
    "4 x 4 mosaic (matches app pixel scale)": 4,
    "3 x 3 mosaic": 3,
    "2 x 2 mosaic": 2,
    "Single section (zoomed 4.5x)": 1,
}
SCOPES = {
    "Mask decoder + DETR encoder (recommended)": "heads_encoder",
    "Mask decoder only (fastest)": "heads",
    "Mask decoder + DETR encoder + FPN neck": "heads_encoder_neck",
}
PROMPTS = ["fault", "seismic facies", "channel", "salt body", "horizon"]

LOG_RE = re.compile(r"^(\d\d:\d\d:\d\d) \| (\w+)\s*\| (.*)$")

# Same palette as the seismic viewer, so this window feels like part of the app.
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

COLORS = {
    "time": "#6b7683",
    "INFO": "#d5dbe1",
    "WARNING": WARN,
    "ERROR": ERR,
    "metric": OK,
    "banner": "#93c5fd",
    "other": MUTED,
    "gui": "#93c5fd",
}


def _console_python() -> str:
    """python.exe next to pythonw.exe, so the trainer can print a log."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        sibling = exe.with_name("python.exe")
        if sibling.exists():
            return str(sibling)
    return str(exe)


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


class FinetuneGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.events: queue.Queue = queue.Queue()
        self.stop_file: Path | None = None
        self.stop_sent = False
        self.output_path: Path | None = None
        self.loss_points: list[tuple[int, float]] = []
        self.val_points: list[tuple[int, float]] = []
        self.total_steps = 1
        self.last_step = 0
        self.base_iou: float | None = None
        self.best_iou: float | None = None
        self.preview_photo = None
        self.preview_path: str | None = None
        self.pair_count = 0
        self._count_job = None
        self.run_started = 0.0

        root.title("SAM 3 Quick Fine-Tuner")
        root.configure(bg=APP_BG)
        scale = root.winfo_fpixels("1i") / 96.0
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        # Wide enough that the status column is a real panel, not a sliver beside the form.
        width = min(int(1760 * scale), int(screen_w * 0.94))
        height = min(int(960 * scale), int(screen_h * 0.92))
        root.geometry(f"{width}x{height}+{(screen_w - width) // 2}+{int(20 * scale)}")
        root.minsize(min(int(1280 * scale), int(screen_w * 0.92)), min(int(760 * scale), int(screen_h * 0.85)))
        self._build_vars()
        self._build_ui()
        self._load_settings()
        self._refresh_counts()
        self._update_estimate()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(50, lambda: self._apply_dark_titlebar(root))
        root.after(100, self._drain_events)
        self._gui_log("Ready. Pick a preset, check the folders and press Start.")

    # --- Variables -----------------------------------------------------------

    def _build_vars(self) -> None:
        v = tk.StringVar
        self.preset = v(value=next(iter(PRESETS)))
        self.data_root = v(value=str(DEFAULT_DATA_ROOT))
        self.image_dir = v()
        self.label_dir = v()
        self.label_mode = v(value=next(iter(LABEL_MODES)))
        self.num_layers = tk.IntVar(value=8)
        self.prompt = v(value="fault")
        self.model_name = v()
        self.output_root = v(value=str(DEFAULT_OUTPUT_ROOT))
        self.base_model = v(value="facebook/sam3")
        self.max_samples = tk.IntVar(value=2000)
        self.val_percent = tk.IntVar(value=10)
        self.epochs = tk.IntVar(value=3)
        self.batch_size = tk.IntVar(value=2)
        self.mosaic = v(value=next(iter(MOSAICS)))
        self.lr = v(value="1e-4")
        self.scope = v(value=next(iter(SCOPES)))
        self.vit_blocks = tk.IntVar(value=0)
        self.eval_base = tk.BooleanVar(value=True)
        self.hflip = tk.BooleanVar(value=True)
        self.keep_best = tk.BooleanVar(value=True)
        self.device = v(value="cuda")
        self.autoscroll = tk.BooleanVar(value=True)

        self.count_text = v(value="")
        self.target_text = v(value="")
        self.estimate_text = v(value="")
        self.stage_text = v(value="Idle")
        self.progress_text = v(value="")
        self.metric_text = {k: v(value="-") for k in ("base", "best", "iou", "f1", "precision", "recall", "loss")}

        for var in (self.image_dir, self.label_dir):
            var.trace_add("write", lambda *_: self._schedule_count())
        for var in (self.model_name, self.output_root):
            var.trace_add("write", lambda *_: self._update_target())
        for var in (self.max_samples, self.val_percent, self.epochs, self.batch_size, self.mosaic,
                    self.vit_blocks, self.scope, self.device):
            var.trace_add("write", lambda *_: self._update_estimate())
        self.label_mode.trace_add("write", lambda *_: self._update_layer_state())

    # --- Layout --------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Flat dark theme. Controls stay in the same rows; only the chrome changes."""
        import tkinter.font as tkfont

        for name, family, size in (
            ("TkDefaultFont", "Segoe UI", 10),
            ("TkTextFont", "Segoe UI", 10),
            ("TkHeadingFont", "Segoe UI", 11),
            ("TkFixedFont", "Consolas", 10),
        ):
            try:
                tkfont.nametofont(name).configure(family=family, size=size)
            except tk.TclError:
                pass

        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.font", "{Segoe UI} 10")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=CARD, foreground=FG, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, troughcolor=FIELD,
                        font=("Segoe UI", 10), focuscolor=ACCENT)
        style.configure("TFrame", background=APP_BG)
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=CARD, foreground=FG, font=("Segoe UI", 10))
        style.configure("Field.TLabel", background=CARD, foreground=LABEL, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=CARD, foreground=FG, font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Warn.TLabel", background=CARD, foreground=WARN, font=("Segoe UI", 9))
        style.configure("Stage.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 14, "bold"))
        style.configure("TPanedwindow", background=APP_BG, borderwidth=0)
        style.configure("TEntry", fieldbackground=FIELD, foreground=FG, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, insertcolor=FG, padding=(8, 5))
        style.map("TEntry",
                  bordercolor=[("focus", ACCENT)],
                  lightcolor=[("focus", ACCENT)],
                  darkcolor=[("focus", ACCENT)])
        style.configure("TCombobox", fieldbackground=FIELD, background=FIELD, foreground=FG,
                        arrowcolor=LABEL, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                        insertcolor=FG, padding=(8, 5), arrowsize=14)
        style.map("TCombobox",
                  fieldbackground=[("readonly", FIELD), ("disabled", "#12171c")],
                  foreground=[("readonly", FG), ("disabled", DISABLED)],
                  background=[("readonly", FIELD), ("active", "#1c242c"), ("disabled", "#12171c")],
                  arrowcolor=[("disabled", "#4b5563"), ("active", FG)],
                  bordercolor=[("focus", ACCENT)],
                  lightcolor=[("focus", ACCENT)],
                  darkcolor=[("focus", ACCENT)])
        style.configure("TSpinbox", fieldbackground=FIELD, background=FIELD, foreground=FG,
                        arrowcolor=LABEL, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                        insertcolor=FG, padding=(6, 4), arrowsize=12)
        style.map("TSpinbox",
                  fieldbackground=[("disabled", "#12171c")],
                  foreground=[("disabled", DISABLED)],
                  background=[("disabled", "#12171c")],
                  arrowcolor=[("disabled", "#4b5563")],
                  bordercolor=[("focus", ACCENT)],
                  lightcolor=[("focus", ACCENT)],
                  darkcolor=[("focus", ACCENT)])
        style.configure("TCheckbutton", background=CARD, foreground=LABEL, font=("Segoe UI", 10),
                        indicatorcolor=FIELD, focuscolor=CARD)
        style.map("TCheckbutton",
                  background=[("active", CARD), ("disabled", CARD)],
                  foreground=[("disabled", DISABLED), ("active", FG)],
                  indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT), ("!selected", FIELD)])
        flat_button = [
            ("Button.border", {"sticky": "nswe", "children": [
                ("Button.padding", {"sticky": "nswe", "children": [
                    ("Button.label", {"sticky": "nswe"}),
                ]}),
            ]}),
        ]
        style.layout("TButton", flat_button)
        style.configure("TButton", background="#1c242c", foreground=FG, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, padding=(12, 7), font=("Segoe UI", 10),
                        focusthickness=0, focuscolor=ACCENT, relief="flat")
        style.map("TButton",
                  background=[("pressed", FIELD), ("active", "#24303c"), ("disabled", CARD)],
                  foreground=[("disabled", DISABLED)],
                  bordercolor=[("disabled", "#232b33"), ("active", "#3a4652")],
                  lightcolor=[("disabled", "#232b33"), ("active", "#3a4652")],
                  darkcolor=[("disabled", "#232b33"), ("active", "#3a4652")])
        for name, bg, hover, fg, font in (
            ("Accent.TButton", ACCENT, ACCENT_HOVER, "#ffffff", ("Segoe UI", 10, "bold")),
            ("Warn.TButton", "#3a2e14", "#4a3b1a", WARN, ("Segoe UI", 10)),
            ("Danger.TButton", "#3f1d1d", "#5c2424", "#ffb4b4", ("Segoe UI", 10)),
        ):
            style.configure(name, background=bg, foreground=fg, bordercolor=bg, lightcolor=bg,
                            darkcolor=bg, padding=(14, 8), font=font, focusthickness=0, relief="flat")
            style.map(name,
                      background=[("pressed", bg), ("active", hover), ("disabled", "#24303f")],
                      foreground=[("disabled", DISABLED)],
                      bordercolor=[("disabled", "#24303f"), ("active", hover)],
                      lightcolor=[("disabled", "#24303f"), ("active", hover)],
                      darkcolor=[("disabled", "#24303f"), ("active", hover)])
        style.configure("Tool.TButton", background=FIELD, foreground=LABEL, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, padding=(10, 4), font=("Segoe UI", 9),
                        focusthickness=0, relief="flat")
        style.map("Tool.TButton",
                  background=[("pressed", "#0c1014"), ("active", "#1c242c"), ("disabled", CARD)],
                  foreground=[("disabled", DISABLED), ("active", FG)],
                  bordercolor=[("active", "#3a4652"), ("disabled", "#232b33")],
                  lightcolor=[("active", "#3a4652"), ("disabled", "#232b33")],
                  darkcolor=[("active", "#3a4652"), ("disabled", "#232b33")])
        style.layout("Horizontal.TProgressbar", [
            ("Horizontal.Progressbar.trough", {"sticky": "nswe", "children": [
                ("Horizontal.Progressbar.pbar", {"side": "left", "sticky": "ns"}),
            ]}),
        ])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#232b33",
                        bordercolor="#232b33", lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
        style.layout("Vertical.TScrollbar", [
            ("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
                ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"}),
            ]}),
        ])
        style.configure("Vertical.TScrollbar", background="#2a333c", troughcolor=CONSOLE_BG,
                        bordercolor=CONSOLE_BG, lightcolor="#2a333c", darkcolor="#2a333c",
                        arrowsize=0, width=10)
        style.map("Vertical.TScrollbar", background=[("active", "#3a4652"), ("pressed", "#4b5a68")])

    def _apply_dark_titlebar(self, window: tk.Misc) -> None:
        """Match the Windows caption to the window so the chrome is not a light strip."""
        if sys.platform != "win32":
            return
        try:
            import ctypes

            window.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()

            def bgr(hex_color: str) -> int:
                raw = hex_color.lstrip("#")
                red, green, blue = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
                return red | (green << 8) | (blue << 16)

            def set_attr(attr: int, value: int) -> None:
                data = ctypes.c_int(value)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(data), ctypes.sizeof(data))

            set_attr(20, 1)
            set_attr(34, bgr(BORDER))
            set_attr(35, bgr(APP_BG))
            set_attr(36, bgr(FG))
        except (AttributeError, OSError, tk.TclError, ValueError):
            pass

    def _section(self, parent, title: str, *, expand: bool = False, fill_pane: bool = False, placed: bool = False):
        """Card with a title. Returns the body (for fields) and the header (for side actions)."""
        shell = tk.Frame(parent, bg=BORDER, bd=0, highlightthickness=0)
        if placed:
            pass
        elif fill_pane:
            shell.pack(fill=tk.BOTH, expand=True)
        else:
            shell.pack(fill=tk.BOTH if expand else tk.X, expand=expand, pady=(0, 10))
        card = tk.Frame(shell, bg=CARD, bd=0, highlightthickness=0)
        card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        accent = tk.Frame(card, bg=ACCENT, height=2, bd=0, highlightthickness=0)
        accent.pack(fill=tk.X)
        accent.pack_propagate(False)
        header = tk.Frame(card, bg=CARD, bd=0, highlightthickness=0)
        header.pack(fill=tk.X, padx=14, pady=(10, 0))
        tk.Label(header, text=title, bg=CARD, fg=FG, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        body = ttk.Frame(card, style="Card.TFrame", padding=(14, 8, 14, 12))
        body.pack(fill=tk.BOTH, expand=True)
        body.card_shell = shell
        return body, header

    def _wrap_label(self, parent, row, columnspan, textvariable, style, pady=(2, 0)):
        """A label that wraps to the card width without stretching the card."""
        holder = tk.Frame(parent, bg=CARD, bd=0, highlightthickness=0, height=18)
        holder.grid(row=row, column=0, columnspan=columnspan, sticky="ew", pady=pady)
        holder.grid_propagate(False)
        label = ttk.Label(holder, textvariable=textvariable, style=style, anchor="nw", justify=tk.LEFT,
                          wraplength=200)
        label.place(x=0, y=0, relwidth=1.0, relheight=1.0)
        self._wrap_labels.append(label)

        def fit(_event=None, target=label, box=holder):
            width = max(40, box.winfo_width() - 2)
            if int(target.cget("wraplength")) != width:
                target.configure(wraplength=width)
            needed = max(target.winfo_reqheight(), 16)
            if int(box.cget("height")) != needed:
                box.configure(height=needed)
                self._schedule_column_layout()

        holder.bind("<Configure>", fit)
        textvariable.trace_add("write", lambda *_a: holder.after_idle(fit))
        return label

    def _build_ui(self) -> None:
        self._apply_theme()

        header = tk.Frame(self.root, bg=APP_BG, bd=0, highlightthickness=0)
        header.pack(fill=tk.X, padx=16, pady=(14, 10))
        tk.Label(header, text="SAM 3 Quick Fine-Tuner", bg=APP_BG, fg=FG,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(
            header,
            text="Pick a preset, check the folders, name the model, then start.",
            bg=APP_BG, fg=MUTED, font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        panes = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 14))
        self._panes = panes

        top = tk.Frame(panes, bg=APP_BG, bd=0, highlightthickness=0)
        panes.add(top, weight=0)
        self._top = top
        # Requested control widths stay modest so a narrow window can still
        # give the status panel room; the grid then stretches fields to fit.
        top.columnconfigure(0, weight=3, minsize=self._px(430))
        top.columnconfigure(1, weight=3, minsize=self._px(470))
        top.columnconfigure(2, weight=2, minsize=self._px(320))
        top.rowconfigure(0, weight=1)
        col_a = tk.Frame(top, bg=APP_BG, bd=0, highlightthickness=0)
        col_b = tk.Frame(top, bg=APP_BG, bd=0, highlightthickness=0)
        col_c = tk.Frame(top, bg=APP_BG, bd=0, highlightthickness=0)
        col_a.grid(row=0, column=0, sticky="nsew")
        col_b.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        col_c.grid(row=0, column=2, sticky="nsew", padx=(12, 0))

        self._col_c = col_c
        self._base_pady = {}
        self._factor = {}
        self._fit_key = {}
        self._compact = {}
        self._layout_sig = {}
        self._layout_pending = False
        self._layout_again = False
        self._laying_out = False
        self._scroll_canvases = set()
        self._wrap_labels = []
        self._data_inner, self._data_canvas = self._scroll_column(col_a)
        self._build_buttons(col_b)
        self._training_inner, self._training_canvas = self._scroll_column(col_b)
        self._build_data_frame(self._data_inner)
        self._build_output_frame(self._data_inner)
        self._build_training_frame(self._training_inner)
        self._build_status(col_c)

        bottom = ttk.PanedWindow(panes, orient=tk.HORIZONTAL)
        panes.add(bottom, weight=1)
        console_frame = tk.Frame(bottom, bg=APP_BG, bd=0, highlightthickness=0)
        bottom.add(console_frame, weight=3)
        self._build_console(console_frame)
        preview_frame = tk.Frame(bottom, bg=APP_BG, bd=0, highlightthickness=0)
        bottom.add(preview_frame, weight=1)
        self._build_preview(preview_frame)
        self.root.after(80, self._place_sash)

    def _px(self, points: float) -> int:
        """Scale a 96-DPI size up to this screen, so the layout matches the font."""
        return max(1, int(points * self.root.winfo_fpixels("1i") / 96.0))

    def _place_sash(self) -> None:
        """Keep the form at its own height and give the remaining space to the log."""
        if not self.root.winfo_exists():
            return
        self.root.update_idletasks()
        self._remember_form_height()
        height = self._panes.winfo_height()
        want = getattr(self, "_form_natural", 0) + self._px(8)
        if want < 40:
            want = self._top.winfo_reqheight() + self._px(8)
        if height > self._px(200) and want > 40:
            # Keep the whole form on screen, and still leave a usable log strip.
            log_min = self._px(160)
            self._panes.sashpos(0, min(want, max(self._px(240), height - log_min)))
        for entry in getattr(self, "_path_entries", ()):
            entry.xview_moveto(1.0)
        self._layout_columns()

    def _scroll_column(self, parent) -> tuple[tk.Frame, tk.Canvas]:
        """A column whose cards fill the visible area and scroll once they cannot shrink further."""
        canvas = tk.Canvas(parent, bg=APP_BG, highlightthickness=0, bd=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(canvas, bg=APP_BG, bd=0, highlightthickness=0)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        bar = ttk.Scrollbar(parent, command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.scroll_bar = bar
        canvas.scroll_window = window
        canvas.scroll_inner = inner
        self._scroll_canvases.add(canvas)
        canvas.bind("<Configure>", self._schedule_column_layout)
        if not getattr(self, "_wheel_bound", False):
            self.root.bind_all("<MouseWheel>", self._on_mousewheel)
            self._wheel_bound = True
        return inner, canvas

    def _on_mousewheel(self, event) -> str | None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        canvas = None
        while widget is not None:
            if widget in self._scroll_canvases:
                canvas = widget
                break
            widget = getattr(widget, "master", None)
        if canvas is None:
            return None
        bbox = canvas.bbox("all")
        if not bbox or bbox[3] <= canvas.winfo_height() + 2:
            return None
        canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _schedule_column_layout(self, _event=None) -> None:
        if self._laying_out:
            self._layout_again = True
            return
        if self._layout_pending:
            return
        self._layout_pending = True
        self.root.after_idle(self._layout_columns)

    def _remember_form_height(self) -> None:
        """Record the unscaled card heights so later sash moves scale from that size."""
        if getattr(self, "_form_ready", False):
            return
        self._settle_wraps()
        self.root.update_idletasks()
        data_h = self._data_shell.winfo_reqheight()
        output_h = self._output_shell.winfo_reqheight()
        training_h = self._training_shell.winfo_reqheight()
        if min(data_h, output_h, training_h) < 40:
            return
        self._natural_h = {"data": data_h, "output": output_h, "training": training_h}
        self._shell_chrome = {
            "data": data_h - self._data_body.winfo_reqheight(),
            "output": output_h - self._output_body.winfo_reqheight(),
            "training": training_h - self._training_body.winfo_reqheight(),
        }
        pad_save = self._px(16)

        def minimum(body, natural: int) -> int:
            _fixed, pad = self._content_parts(body)
            return max(self._px(120), natural - pad - pad_save)

        self._min_h = {
            "data": minimum(self._data_body, data_h),
            "output": minimum(self._output_body, output_h),
            "training": minimum(self._training_body, training_h),
        }
        gap = self._px(6)
        left = data_h + gap + output_h
        middle = training_h + self._px(8) + max(self._button_bar.winfo_reqheight(), self._px(36))
        right = self._col_c.winfo_reqheight()
        self._form_natural = max(left, middle, right)
        self._form_ready = True

    def _settle_wraps(self) -> None:
        for label in self._wrap_labels:
            holder = label.master
            width = holder.winfo_width()
            if width < 80:
                width = self._data_canvas.winfo_width() - self._px(48)
            if width < 80:
                width = 420
            label.configure(wraplength=max(40, width))
            holder.configure(height=max(label.winfo_reqheight(), 16))

    def _adopt_grown_cards(self) -> None:
        """A wrapped path can get taller after the first measure. Keep that as the card's full size."""
        if not getattr(self, "_form_ready", False):
            return
        changed = False
        for key, shell, body in (
            ("data", self._data_shell, self._data_body),
            ("output", self._output_shell, self._output_body),
            ("training", self._training_shell, self._training_body),
        ):
            if self._compact.get(body) or self._factor.get(body, 1.0) < 0.99:
                continue
            req = shell.winfo_reqheight()
            if req > self._natural_h[key] + 2:
                self._natural_h[key] = req
                _fixed, pad = self._content_parts(body)
                self._min_h[key] = max(self._px(120), req - pad - self._px(16))
                self._shell_chrome[key] = max(0, req - body.winfo_reqheight())
                changed = True
        if changed:
            gap = self._px(6)
            left = self._natural_h["data"] + gap + self._natural_h["output"]
            middle = self._natural_h["training"] + self._px(8) + max(self._button_bar.winfo_reqheight(), self._px(36))
            self._form_natural = max(left, middle, self._form_natural)
            self._layout_sig.clear()

    def _split_pad(self, raw) -> tuple[int, int]:
        if isinstance(raw, (tuple, list)):
            if len(raw) >= 2:
                return int(float(raw[0])), int(float(raw[1]))
            if len(raw) == 1:
                val = int(float(raw[0]))
                return val, val
            return 0, 0
        parts = str(raw).split()
        if len(parts) >= 2:
            return int(float(parts[0])), int(float(parts[1]))
        if parts and parts[0]:
            val = int(float(parts[0]))
            return val, val
        return 0, 0

    def _remember_pads(self, body) -> None:
        for widget in body.grid_slaves():
            if widget in self._base_pady:
                continue
            self._base_pady[widget] = self._split_pad(widget.grid_info().get("pady", 0))

    def _content_parts(self, body) -> tuple[int, int]:
        """Fixed row height and the vertical padding those rows have at scale 1."""
        rows: dict[int, tuple[int, int]] = {}
        for widget in body.grid_slaves():
            info = widget.grid_info()
            if not info:
                continue
            row = int(info["row"])
            top, bottom = self._base_pady.get(widget, (0, 0))
            fixed, pad = rows.get(row, (0, 0))
            rows[row] = (max(fixed, widget.winfo_reqheight()), max(pad, top + bottom))
        return sum(fixed for fixed, _pad in rows.values()), sum(pad for _fixed, pad in rows.values())

    def _sticky(self, widget) -> str:
        raw = widget.grid_info().get("sticky", "")
        if isinstance(raw, (tuple, list)):
            return "".join(str(part) for part in raw)
        return str(raw)

    def _fit_shell(self, body, key: str, slot: int) -> None:
        """Tighten a card that lost height, or spread its rows when the column got taller."""
        if slot < 20 or not body.grid_slaves():
            return
        allowed = max(20, slot - self._shell_chrome.get(key, 0))
        compact = slot < self._natural_h[key] - self._px(8)
        if self._compact.get(body) != compact:
            self._compact[body] = compact
            body.configure(padding=(8, 1, 8, 2) if compact else (14, 8, 14, 12))
        fixed, pad = self._content_parts(body)
        last = self._factor.get(body, 1.0)
        chrome = max(0, body.winfo_reqheight() - (fixed + pad * last))
        if pad <= 0:
            factor = 1.0
        else:
            factor = (allowed - chrome - fixed) / pad
        factor = max(0.0, min(1.0, factor))
        need = chrome + fixed + pad * factor
        weight = 1 if factor >= 0.99 and allowed > need + 8 else 0
        fit_key = (round(factor, 2), weight, compact)
        if self._fit_key.get(body) == fit_key:
            return
        self._fit_key[body] = fit_key
        self._factor[body] = factor
        for widget in body.grid_slaves():
            top, bottom = self._base_pady.get(widget, (0, 0))
            sticky = self._sticky(widget)
            if weight and "n" not in sticky:
                sticky = (sticky + "n") if sticky else "nw"
            widget.grid_configure(pady=(round(top * factor), round(bottom * factor)), sticky=sticky)
        seen = {int(widget.grid_info()["row"]) for widget in body.grid_slaves() if widget.grid_info()}
        for row in seen:
            body.rowconfigure(row, weight=weight)

    def _layout_stack(self, canvas: tk.Canvas, items: list[tuple[tk.Frame, ttk.Frame, str]]) -> None:
        available = canvas.winfo_height()
        width = canvas.winfo_width()
        if available < 40 or width < 40 or not getattr(self, "_form_ready", False):
            return
        gap = self._px(6)
        gaps = gap * (len(items) - 1)
        naturals = [self._natural_h[key] for _shell, _body, key in items]
        minimums = [self._min_h[key] for _shell, _body, key in items]
        natural = sum(naturals) + gaps
        minimum = sum(minimums) + gaps
        # Fill the column when there is room. Below the compact size, keep the fields
        # intact and let the column scroll instead of clipping them.
        total = available if available >= minimum else minimum
        if total >= natural and sum(naturals) > 0:
            space = max(1, total - gaps)
            heights = [max(1, int(space * height / sum(naturals))) for height in naturals]
        else:
            span = max(1, natural - minimum)
            blend = min(1.0, max(0.0, (total - minimum) / span))
            heights = [max(1, int(low + (high - low) * blend)) for low, high in zip(minimums, naturals)]
        heights[-1] = max(1, total - gaps - sum(heights[:-1]))
        signature = (available, width, tuple(heights))
        if self._layout_sig.get(canvas) == signature:
            return
        self._layout_sig[canvas] = signature
        y = 0
        inner = canvas.scroll_inner
        for (shell, _body, _key), height in zip(items, heights):
            shell.place(x=0, y=y, relwidth=1, height=height)
            y += height + gap
        inner_h = max(1, y - gap)
        canvas.itemconfigure(canvas.scroll_window, width=width, height=inner_h)
        inner.configure(width=width, height=inner_h)
        canvas.configure(scrollregion=(0, 0, width, inner_h))
        bar = canvas.scroll_bar
        if inner_h > available + 2:
            if not bar.winfo_ismapped():
                bar.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            if bar.winfo_ismapped():
                bar.pack_forget()
            canvas.yview_moveto(0)
        for (shell, body, key), height in zip(items, heights):
            self._fit_shell(body, key, height)

    def _layout_columns(self) -> None:
        if self._laying_out or not self.root.winfo_exists():
            return
        self._layout_pending = False
        self._remember_form_height()
        self._adopt_grown_cards()
        self._laying_out = True
        try:
            self._layout_stack(self._data_canvas, [
                (self._data_shell, self._data_body, "data"),
                (self._output_shell, self._output_body, "output"),
            ])
            self._layout_stack(self._training_canvas, [
                (self._training_shell, self._training_body, "training"),
            ])
        finally:
            self._laying_out = False
            if self._layout_again:
                self._layout_again = False
                self._schedule_column_layout()

    def _path_entry(self, parent, var: tk.StringVar) -> ttk.Entry:
        """Entry that keeps the end of a long path (the folder name) in view."""
        entry = ttk.Entry(parent, textvariable=var)
        self._path_entries = getattr(self, "_path_entries", [])
        self._path_entries.append(entry)

        def show_end(widget=entry):
            if widget.focus_get() is not widget:
                widget.xview_moveto(1.0)

        var.trace_add("write", lambda *_e: entry.after_idle(show_end))
        entry.bind("<Configure>", lambda _e: entry.after_idle(show_end))
        entry.bind("<FocusOut>", lambda _e: entry.xview_moveto(1.0))
        return entry

    def _row(self, parent, r: int, label: str, widget, browse=None) -> None:
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=r, column=0, sticky="w", padx=(0, 10), pady=3)
        if browse:
            widget.grid(row=r, column=1, sticky="ew", pady=3, padx=(0, 8))
            ttk.Button(parent, text="Browse...", width=9, style="Tool.TButton", cursor="hand2",
                       command=browse).grid(row=r, column=2, pady=3)
            return
        widget.grid(row=r, column=1, columnspan=2, sticky="ew", pady=3)

    def _spin_row(self, parent, r, label, variable, *, from_, to, increment=1, width=10, hint=None):
        """Number field kept next to its label. Created inside the row so it is not covered."""
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=r, column=0, sticky="w", padx=(0, 10), pady=3)
        box = ttk.Frame(parent, style="Card.TFrame")
        box.grid(row=r, column=1, columnspan=2, sticky="w", pady=3)
        spin = ttk.Spinbox(box, from_=from_, to=to, increment=increment, textvariable=variable, width=width)
        spin.pack(side=tk.LEFT)
        if hint:
            ttk.Label(box, text=hint, style="Hint.TLabel").pack(side=tk.LEFT, padx=(8, 0))
        return spin

    def _build_data_frame(self, parent) -> None:
        f, _header = self._section(parent, "1. Data", placed=True)
        f.columnconfigure(1, weight=1)
        preset = ttk.Combobox(f, textvariable=self.preset, values=list(PRESETS), state="readonly")
        preset.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())
        self._row(f, 0, "Preset", preset)
        self._row(f, 1, "Data folder", self._path_entry(f, self.data_root),
                  browse=lambda: self._browse_dir(self.data_root, then=self._apply_preset))
        self._row(f, 2, "Seismic images", self._path_entry(f, self.image_dir),
                  browse=lambda: self._browse_dir(self.image_dir))
        self._row(f, 3, "Labels", self._path_entry(f, self.label_dir),
                  browse=lambda: self._browse_dir(self.label_dir))
        self._row(f, 4, "Label type",
                  ttk.Combobox(f, textvariable=self.label_mode, values=list(LABEL_MODES), state="readonly"))
        self.layers_spin = ttk.Spinbox(f, from_=2, to=40, textvariable=self.num_layers, width=6)
        ttk.Label(f, text="Layers per section", style="Field.TLabel").grid(
            row=5, column=0, sticky="w", padx=(0, 10), pady=3)
        self.layers_spin.grid(row=5, column=1, sticky="w", pady=3)
        self._row(f, 6, "Text prompt", ttk.Combobox(f, textvariable=self.prompt, values=PROMPTS))
        info = ttk.Frame(f, style="Card.TFrame")
        info.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.count_label = ttk.Label(info, textvariable=self.count_text, style="Hint.TLabel")
        self.count_label.pack(side=tk.LEFT)
        self.preview_btn = ttk.Button(info, text="Preview data", style="Tool.TButton", cursor="hand2",
                                      command=self._preview_data)
        self.preview_btn.pack(side=tk.RIGHT)
        self._data_body = f
        self._data_shell = f.card_shell
        self._remember_pads(f)

    def _build_output_frame(self, parent) -> None:
        f, _header = self._section(parent, "2. Output model", placed=True)
        f.columnconfigure(1, weight=1)
        self._row(f, 0, "Model name", ttk.Entry(f, textvariable=self.model_name))
        self._row(f, 1, "Save in folder", self._path_entry(f, self.output_root),
                  browse=lambda: self._browse_dir(self.output_root))
        base = ttk.Combobox(f, textvariable=self.base_model, values=self._base_choices())
        base.bind("<Button-1>", lambda _e: base.configure(values=self._base_choices()))
        self._row(f, 2, "Start from", base, browse=lambda: self._browse_dir(self.base_model))
        self.target_label = self._wrap_label(f, 3, 3, self.target_text, "Hint.TLabel", pady=(4, 0))
        self._output_body = f
        self._output_shell = f.card_shell
        self._remember_pads(f)

    def _build_training_frame(self, parent) -> None:
        f, _header = self._section(parent, "3. Training settings", placed=True)
        f.columnconfigure(1, weight=1)
        self._spin_row(f, 0, "Sections to use", self.max_samples, from_=50, to=100000, increment=500,
                       hint="0 = all")
        self._spin_row(f, 1, "Validation %", self.val_percent, from_=2, to=40)
        self._spin_row(f, 2, "Epochs", self.epochs, from_=1, to=200)
        self._spin_row(f, 3, "Batch size", self.batch_size, from_=1, to=8, hint="2 = ~4.5 GB VRAM")
        self._row(f, 4, "Input layout",
                  ttk.Combobox(f, textvariable=self.mosaic, values=list(MOSAICS), state="readonly"))
        self._row(f, 5, "Learning rate", ttk.Entry(f, textvariable=self.lr, width=12))
        self._row(f, 6, "What to train",
                  ttk.Combobox(f, textvariable=self.scope, values=list(SCOPES), state="readonly"))
        self._spin_row(f, 7, "Unfreeze ViT blocks", self.vit_blocks, from_=0, to=32,
                       hint="last N of 32; slower")
        self._row(f, 8, "Device", ttk.Combobox(f, textvariable=self.device, values=["cuda", "cpu"],
                                               state="readonly", width=10))
        checks = ttk.Frame(f, style="Card.TFrame")
        checks.grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 2))
        ttk.Checkbutton(checks, text="Score base model first", variable=self.eval_base).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(checks, text="Flip augmentation", variable=self.hflip).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(checks, text="Keep best epoch", variable=self.keep_best).pack(side=tk.LEFT)
        self._wrap_label(f, 10, 3, self.estimate_text, "Hint.TLabel", pady=(2, 0))
        self._training_body = f
        self._training_shell = f.card_shell
        self._remember_pads(f)

    def _build_buttons(self, parent) -> None:
        f = tk.Frame(parent, bg=APP_BG, bd=0, highlightthickness=0)
        f.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        self._button_bar = f
        self.start_btn = ttk.Button(f, text="Start fine-tuning", style="Accent.TButton", cursor="hand2",
                                    command=self._start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(f, text="Stop and save", style="TButton", cursor="hand2",
                                   command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(f, text="Open output folder", style="TButton", cursor="hand2",
                   command=self._open_output).pack(side=tk.RIGHT)

    def _metric_tile(self, parent, title: str, key: str, row: int, col: int) -> None:
        tile = tk.Frame(parent, bg=FIELD, highlightthickness=1, highlightbackground=BORDER,
                        highlightcolor=BORDER, bd=0)
        tile.grid(row=row, column=col, sticky="nsew", padx=(0, 6), pady=(0, 6))
        tk.Label(tile, text=title, bg=FIELD, fg=MUTED, font=("Segoe UI", 8), anchor="w").pack(
            fill=tk.X, padx=6, pady=(5, 0))
        tk.Label(tile, textvariable=self.metric_text[key], bg=FIELD, fg=FG, anchor="w",
                 font=("Consolas", 12)).pack(fill=tk.X, padx=6, pady=(0, 5))

    def _build_status(self, parent) -> None:
        f, _header = self._section(parent, "Status")
        f.columnconfigure(0, weight=1)
        self.stage_label = ttk.Label(f, textvariable=self.stage_text, style="Stage.TLabel")
        self.stage_label.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self._set_stage("Idle", "idle")
        self.progress = ttk.Progressbar(f, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 4), ipady=1)
        self._wrap_label(f, 2, 1, self.progress_text, "Card.TLabel", pady=(0, 8))

        metrics = tk.Frame(f, bg=CARD, bd=0, highlightthickness=0)
        metrics.grid(row=3, column=0, sticky="ew")
        for col in range(4):
            metrics.columnconfigure(col, weight=1, uniform="metric")
        cells = [[("Base IoU", "base"), ("Best IoU", "best"), ("Last IoU", "iou"), ("Val loss", "loss")],
                 [("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")]]
        for row, row_cells in enumerate(cells):
            for col, (title, key) in enumerate(row_cells):
                self._metric_tile(metrics, title, key, row, col)

        chart_body, _chart_head = self._section(parent, "Loss", expand=True)
        legend = tk.Frame(chart_body, bg=CARD, bd=0, highlightthickness=0)
        legend.pack(fill=tk.X, pady=(0, 4))
        tk.Label(legend, text="●", bg=CARD, fg=OK, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Label(legend, text=" Training average", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Label(legend, text="    ●", bg=CARD, fg=WARN, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Label(legend, text=" Validation", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.chart = tk.Canvas(chart_body, bg=FIELD, highlightthickness=0, height=self._px(48), bd=0)
        self.chart.pack(fill=tk.BOTH, expand=True)
        self.chart.bind("<Configure>", lambda _e: self._draw_chart())

    def _build_preview(self, parent) -> None:
        # A canvas (not a label) so a large preview is scaled to the pane
        # instead of pushing the pane wider and getting clipped.
        body, header = self._section(parent, "Validation preview", fill_pane=True)
        tk.Label(header, text="Double-click to open", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.RIGHT)
        self.preview_canvas = tk.Canvas(body, bg=FIELD, highlightthickness=0, bd=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", lambda _e: self._show_preview())
        self.preview_canvas.bind("<Double-Button-1>", lambda _e: self._open_preview())
        self._preview_note = "Appears after the first validation"

    def _build_console(self, parent) -> None:
        body, header = self._section(parent, "Training output", fill_pane=True)
        ttk.Button(header, text="Clear", style="Tool.TButton", cursor="hand2",
                   command=self._clear_log).pack(side=tk.RIGHT)
        ttk.Button(header, text="Copy all", style="Tool.TButton", cursor="hand2",
                   command=self._copy_log).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Checkbutton(header, text="Auto-scroll", variable=self.autoscroll).pack(side=tk.RIGHT, padx=(0, 10))

        frame = tk.Frame(body, bg=CONSOLE_BG, bd=0, highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=BORDER)
        frame.pack(fill=tk.BOTH, expand=True)
        self.console = tk.Text(
            frame, bg=CONSOLE_BG, fg=FG, insertbackground=FG, font=("Consolas", 10),
            wrap=tk.WORD, relief=tk.FLAT, bd=0, padx=10, pady=8, state=tk.DISABLED,
            selectbackground=ACCENT, selectforeground="#ffffff", insertwidth=1,
        )
        scroll = ttk.Scrollbar(frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for tag, color in COLORS.items():
            self.console.tag_configure(tag, foreground=color)
        self.console.tag_configure("bold", font=("Consolas", 10, "bold"))

    def _set_stage(self, text: str, tone: str = "idle") -> None:
        self.stage_text.set(text)
        color = {"idle": MUTED, "run": "#93c5fd", "ok": OK, "warn": WARN, "err": ERR}.get(tone, FG)
        label = getattr(self, "stage_label", None)
        if label is not None:
            label.configure(foreground=color)

    # --- Form helpers --------------------------------------------------------

    def _browse_dir(self, var: tk.StringVar, then=None) -> None:
        start = var.get() if Path(var.get() or ".").is_dir() else str(REPO_ROOT)
        chosen = filedialog.askdirectory(initialdir=start, parent=self.root)
        if chosen:
            var.set(str(Path(chosen)))
            if then:
                then()

    def _base_choices(self) -> list[str]:
        choices = ["facebook/sam3"]
        out_root = Path(self.output_root.get() or DEFAULT_OUTPUT_ROOT)
        if out_root.is_dir():
            choices += [str(p) for p in sorted(out_root.iterdir()) if (p / "config.json").is_file()]
        return choices

    def _apply_preset(self) -> None:
        preset = PRESETS.get(self.preset.get())
        if not preset:
            return
        root = Path(self.data_root.get())
        self.image_dir.set(str(root / preset["image"]))
        self.label_dir.set(str(root / preset["label"]))
        self.label_mode.set(next(k for k, v in LABEL_MODES.items() if v == preset["mode"]))
        self.prompt.set(preset["prompt"])
        self.model_name.set(f"{preset['name']}_{datetime.now():%Y%m%d_%H%M}")

    def _update_layer_state(self) -> None:
        rgt = LABEL_MODES.get(self.label_mode.get()) == "rgt_layers"
        self.layers_spin.configure(state=tk.NORMAL if rgt else tk.DISABLED)

    def _schedule_count(self) -> None:
        if self._count_job:
            self.root.after_cancel(self._count_job)
        self._count_job = self.root.after(300, self._refresh_counts)

    def _refresh_counts(self) -> None:
        self._count_job = None
        img, lab = Path(self.image_dir.get()), Path(self.label_dir.get())
        if not img.is_dir() or not lab.is_dir():
            self.pair_count = 0
            missing = "seismic folder" if not img.is_dir() else "label folder"
            self.count_text.set(f"The {missing} does not exist.")
        else:
            try:
                self.pair_count = len(list_pairs(img, lab))
                self.count_text.set(f"{self.pair_count:,} matched image/label files")
            except OSError as exc:
                self.pair_count = 0
                self.count_text.set(f"Cannot read folders: {exc}")
        if getattr(self, "count_label", None) is not None:
            self.count_label.configure(foreground=OK if self.pair_count else WARN)
        self._update_estimate()

    def _update_target(self) -> None:
        name = self.model_name.get().strip()
        if not name:
            self.target_text.set("Enter a model name.")
            return
        target = Path(self.output_root.get()) / name
        exists = (target / "config.json").exists()
        note = "  (exists - will ask before overwriting)" if exists else ""
        in_app = Path(self.output_root.get()).resolve() == DEFAULT_OUTPUT_ROOT.resolve()
        app_note = "\nSaved here, it shows up in the app's text-checkpoint list." if in_app else ""
        self.target_text.set(f"Saves to: {target}{note}{app_note}")
        self.target_label.configure(style="Warn.TLabel" if exists else "Hint.TLabel")

    def _int(self, var: tk.Variable, default: int) -> int:
        try:
            return int(var.get())
        except (tk.TclError, ValueError):
            return default

    def _update_estimate(self) -> None:
        n = self.pair_count
        use = n if self._int(self.max_samples, 0) <= 0 else min(n, self._int(self.max_samples, 0))
        if use < 2:
            self.estimate_text.set("")
            return
        val = min(256, max(1, round(use * self._int(self.val_percent, 10) / 100)))
        grid = MOSAICS.get(self.mosaic.get(), 4)
        bs = max(1, self._int(self.batch_size, 2))
        steps = max(1, (use - val) // (grid * grid * bs)) * max(1, self._int(self.epochs, 1))
        sec = 0.15 + 0.43 * bs
        if SCOPES.get(self.scope.get()) == "heads_encoder_neck":
            sec *= 1.2
        sec *= 1 + 0.12 * self._int(self.vit_blocks, 0)
        if self.device.get() == "cpu":
            sec *= 40
        total = steps * sec + 60
        self.estimate_text.set(
            f"{use - val:,} train / {val} val sections -> {steps:,} steps, roughly {fmt_duration(total)} on this GPU"
        )

    # --- Data preview --------------------------------------------------------

    def _preview_data(self) -> None:
        import random

        import numpy as np
        from PIL import Image, ImageTk

        from finetuning.quick_data import load_sections

        img, lab = Path(self.image_dir.get()), Path(self.label_dir.get())
        if self.pair_count < 1:
            messagebox.showerror("No data", "No matched image/label files in those folders.", parent=self.root)
            return
        try:
            stems = random.sample(list_pairs(img, lab), k=min(4, self.pair_count))
            sec = load_sections(img, lab, stems, LABEL_MODES[self.label_mode.get()], self._int(self.num_layers, 8))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot read data", str(exc), parent=self.root)
            return
        rows = []
        for g, m in zip(sec.images, sec.masks):
            rgb = np.repeat(g[..., None], 3, -1).astype(np.float32)
            over = rgb.copy()
            over[m] = 0.45 * over[m] + 0.55 * np.array((0, 200, 255), np.float32)
            rows.append(np.concatenate([rgb, over], axis=1).astype(np.uint8))
        grid = Image.fromarray(np.concatenate(rows, axis=0))
        scale = max(1, 760 // grid.height)
        grid = grid.resize((grid.width * scale, grid.height * scale), Image.NEAREST)
        win = tk.Toplevel(self.root)
        win.title(f"Data preview - files {', '.join(stems)}  (label in blue)")
        win.configure(bg=APP_BG)
        self._apply_dark_titlebar(win)
        photo = ImageTk.PhotoImage(grid)
        lbl = tk.Label(win, image=photo, bg=APP_BG, bd=0)
        lbl.image = photo
        lbl.pack(padx=16, pady=(16, 8))
        tk.Label(
            win, text=f"Foreground covers {100 * sec.positive_fraction:.1f}% of these sections",
            bg=APP_BG, fg=MUTED, font=("Segoe UI", 10),
        ).pack(pady=(0, 14))

    # --- Run control ---------------------------------------------------------

    def _validate(self) -> list[str] | None:
        errors = []
        img, lab = Path(self.image_dir.get()), Path(self.label_dir.get())
        if not img.is_dir():
            errors.append(f"Seismic folder not found:\n{img}")
        if not lab.is_dir():
            errors.append(f"Label folder not found:\n{lab}")
        if self.pair_count < 2:
            errors.append("Need at least 2 matched image/label files.")
        name = self.model_name.get().strip()
        if not name or re.search(r'[<>:"/\\|?*]', name):
            errors.append("Model name is empty or contains characters not allowed in folder names.")
        if not self.prompt.get().strip():
            errors.append("Text prompt is empty.")
        if self.label_mode.get() not in LABEL_MODES:
            errors.append("Pick a label type.")
        if self.mosaic.get() not in MOSAICS:
            errors.append("Pick an input layout.")
        if self.scope.get() not in SCOPES:
            errors.append("Pick what to train.")
        try:
            lr = float(self.lr.get())
            if not 0 < lr < 1:
                raise ValueError
        except ValueError:
            errors.append("Learning rate must be a number like 1e-4.")
        if errors:
            messagebox.showerror("Check settings", "\n\n".join(errors), parent=self.root)
            return None

        target = Path(self.output_root.get()) / name
        args = [
            "--image-dir", str(img), "--label-dir", str(lab),
            "--label-mode", LABEL_MODES[self.label_mode.get()],
            "--num-layers", str(self._int(self.num_layers, 8)),
            "--prompt", self.prompt.get().strip(),
            "--base-model", self.base_model.get().strip() or "facebook/sam3",
            "--output-dir", self.output_root.get(),
            "--model-name", name,
            "--max-samples", str(self._int(self.max_samples, 2000)),
            "--val-fraction", str(self._int(self.val_percent, 10) / 100),
            "--epochs", str(self._int(self.epochs, 3)),
            "--batch-size", str(self._int(self.batch_size, 2)),
            "--mosaic", str(MOSAICS[self.mosaic.get()]),
            "--lr", self.lr.get().strip(),
            "--train-scope", SCOPES[self.scope.get()],
            "--unfreeze-vit-blocks", str(self._int(self.vit_blocks, 0)),
            "--device", self.device.get(),
        ]
        if not self.eval_base.get():
            args.append("--no-eval-base")
        if not self.hflip.get():
            args.append("--no-hflip")
        if not self.keep_best.get():
            args.append("--keep-last")
        if (target / "config.json").exists():
            if not messagebox.askyesno("Overwrite?", f"A model already exists at\n{target}\n\nOverwrite it?",
                                       parent=self.root):
                return None
            args.append("--overwrite")
        self.output_path = target
        return args

    def _start(self) -> None:
        if self.proc is not None:
            return
        args = self._validate()
        if args is None:
            return
        self._save_settings()
        self.stop_file = Path(tempfile.gettempdir()) / f"sam3_quick_finetune_{os.getpid()}_{int(time.time())}.stop"
        self.stop_file.unlink(missing_ok=True)
        args += ["--stop-file", str(self.stop_file)]
        cmd = [_console_python(), "-u", "-m", "finetuning.quick_finetune", *args]

        self._reset_status()
        self._gui_log("Starting: " + " ".join(f'"{a}"' if " " in a else a for a in cmd[1:]))
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   HF_HUB_DISABLE_PROGRESS_BARS="1", TQDM_DISABLE="1")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, creationflags=flags,
            )
        except OSError as exc:
            messagebox.showerror("Could not start", str(exc), parent=self.root)
            return
        self.run_started = time.time()
        self.stop_sent = False
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL, text="Stop and save", style="Warn.TButton")
        self._set_stage("Starting...", "run")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n").split("\r")[-1]
            if line.strip():
                self.events.put(("line", line))
        self.events.put(("exit", proc.wait()))

    def _stop(self) -> None:
        if self.proc is None:
            return
        if not self.stop_sent:
            self.stop_sent = True
            if self.stop_file:
                self.stop_file.write_text("stop", encoding="utf-8")
            self._gui_log("Stop requested: the current step finishes, then the model is validated and saved.")
            self._set_stage("Stopping - validating and saving...", "warn")
            self.stop_btn.configure(text="Force kill", style="Danger.TButton")
        elif messagebox.askyesno("Force kill?", "Kill training now? Nothing will be saved.", parent=self.root):
            self.proc.kill()
            self._gui_log("Training process killed.", tag="ERROR")

    def _on_close(self) -> None:
        if self.proc is not None:
            if not messagebox.askyesno("Training running", "Training is still running. Kill it and quit?",
                                       parent=self.root):
                return
            self.proc.kill()
        self._save_settings()
        self.root.destroy()

    def _open_output(self) -> None:
        target = self.output_path if self.output_path and self.output_path.exists() else Path(self.output_root.get())
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(str(target))  # noqa: S606 - Windows only, intended

    # --- Event handling ------------------------------------------------------

    def _drain_events(self) -> None:
        try:
            for _ in range(400):
                kind, payload = self.events.get_nowait()
                if kind == "line":
                    self._handle_line(payload)
                else:
                    self._finished(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _handle_line(self, line: str) -> None:
        if line.startswith(EVENT_PREFIX):
            try:
                self._handle_event(json.loads(line[len(EVENT_PREFIX):]))
            except json.JSONDecodeError:
                self._append(line, "other")
            return
        m = LOG_RE.match(line)
        if not m:
            tag = "WARNING" if re.search(r"warn", line, re.I) else "other"
            tag = "ERROR" if re.search(r"error|traceback", line, re.I) else tag
            self._append(line, tag)
            return
        stamp, level, msg = m.groups()
        tag = level if level in COLORS else "INFO"
        if tag == "INFO" and ("IoU" in msg or msg.startswith("Saved") or msg.startswith("Finished")):
            tag = "metric"
        if msg.startswith("===="):
            tag = "banner"
        self._append_parts([(stamp + "  ", "time"), (msg, tag)])

    def _handle_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "stage":
            self._set_stage(ev.get("message", ""), "run")
            if ev.get("stage") in ("model", "validate", "save", "setup"):
                self.progress.configure(mode="indeterminate")
                self.progress.start(15)
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate")
        elif kind == "data":
            self.progress.stop()
            self.progress.configure(mode="determinate", value=100 * ev["done"] / max(1, ev["total"]))
            self.progress_text.set(f"Reading sections {ev['done']:,} / {ev['total']:,}")
        elif kind == "progress":
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.total_steps = ev["total"]
            self.last_step = ev["step"]
            self.progress["value"] = 100 * ev["step"] / max(1, ev["total"])
            self._set_stage(f"Training - epoch {ev['epoch']} of {ev['epochs']}", "run")
            self.progress_text.set(
                f"Step {ev['step']:,} / {ev['total']:,}     loss {ev['avg_loss']:.4f}     "
                f"{ev['sec_per_step']:.2f} s/step\n"
                f"ETA {fmt_duration(ev['eta'])}     elapsed {fmt_duration(time.time() - self.run_started)}"
                f"     GPU memory {ev['gpu_gb']:.1f} GB"
            )
            self.loss_points.append((ev["step"], ev["avg_loss"]))
            self._draw_chart()
        elif kind == "val":
            iou = ev["iou"]
            if ev.get("epoch") == 0:
                self.base_iou = iou
                self.metric_text["base"].set(f"{iou:.3f}")
            else:
                self.best_iou = iou if self.best_iou is None else max(self.best_iou, iou)
                self.metric_text["best"].set(f"{self.best_iou:.3f}")
                self.val_points.append((self.last_step, ev["loss"]))
            for key in ("iou", "f1", "precision", "recall"):
                self.metric_text[key].set(f"{ev[key]:.3f}")
            self.metric_text["loss"].set(f"{ev['loss']:.4f}")
            self._draw_chart()
        elif kind == "preview":
            self.preview_path = ev["path"]
            self._show_preview()
        elif kind == "done":
            if ev.get("saved"):
                base = ev.get("base_iou")
                change = f"IoU {base:.3f} -> {ev['iou']:.3f}" if base is not None else f"IoU {ev['iou']:.3f}"
                self._set_stage(f"Done - {change}", "ok")
                self.progress_text.set(f"Saved to {ev['output']}")
            else:
                self._set_stage("Stopped - nothing saved", "warn")
        elif kind == "error":
            self._set_stage("Failed - see the log below", "err")

    def _finished(self, code: int) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate")
        elapsed = fmt_duration(time.time() - self.run_started)
        if code == 0:
            self.progress["value"] = 100
            self._gui_log(f"Training process finished after {elapsed}.")
        else:
            if not self.stage_text.get().startswith("Failed"):
                self._set_stage(f"Failed (exit code {code}) - see the log below", "err")
            self._gui_log(f"Training process exited with code {code} after {elapsed}.", tag="ERROR")
        self.proc = None
        if self.stop_file:
            self.stop_file.unlink(missing_ok=True)
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED, text="Stop and save", style="TButton")
        self._update_target()

    def _reset_status(self) -> None:
        self.loss_points.clear()
        self.val_points.clear()
        self.base_iou = self.best_iou = None
        self.last_step = 0
        self.total_steps = 1
        for var in self.metric_text.values():
            var.set("-")
        self.progress["value"] = 0
        self.progress_text.set("")
        self.preview_path = None
        self.preview_photo = None
        self._preview_key = None
        self._show_preview()
        self._draw_chart()

    # --- Drawing -------------------------------------------------------------

    def _draw_chart(self) -> None:
        c = self.chart
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or h < 50:
            return
        pad_l, pad_r, pad_t, pad_b = self._px(78), self._px(10), self._px(14), self._px(22)
        axis_font = self._point_font(13)
        pts = self.loss_points + self.val_points
        if not pts:
            c.create_text(w / 2, h / 2, text="Waiting for training steps", fill=MUTED,
                          font=self._point_font(13))
            return
        ys = [p[1] for p in pts]
        lo, hi = min(ys), max(ys)
        if hi - lo < 1e-6:
            hi = lo + 1e-3
        span_x = max(self.total_steps, max(p[0] for p in pts), 1)

        def xy(step: float, val: float) -> tuple[float, float]:
            x = pad_l + (w - pad_l - pad_r) * step / span_x
            y = pad_t + (h - pad_t - pad_b) * (1 - (val - lo) / (hi - lo))
            return x, y

        c.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline=BORDER)
        for frac in (0, 0.5, 1):
            val = lo + (hi - lo) * frac
            _, y = xy(0, val)
            c.create_line(pad_l, y, w - pad_r, y, fill="#1e262e")
            c.create_text(pad_l - self._px(6), y, text=f"{val:.2f}", anchor="e", fill=MUTED, font=axis_font)
        c.create_text(w - pad_r, h - self._px(4), text=f"step {span_x}", anchor="se", fill=MUTED, font=axis_font)
        c.create_text(pad_l, h - self._px(4), text="0", anchor="sw", fill=MUTED, font=axis_font)
        if len(self.loss_points) > 1:
            stride = max(1, len(self.loss_points) // 600)
            coords = [v for p in self.loss_points[::stride] for v in xy(*p)]
            c.create_line(*coords, fill=OK, width=2)
        for step, val in self.val_points:
            x, y = xy(step, val)
            r = self._px(4)
            c.create_oval(x - r, y - r, x + r, y + r, fill=WARN, outline="")

    def _point_font(self, pixels: int, family: str = "Segoe UI") -> tuple[str, int]:
        """A Tk font whose on-screen size stays near ``pixels`` at any DPI."""
        scaling = float(self.root.tk.call("tk", "scaling")) or 1.0
        return (family, max(1, round(pixels / scaling)))

    def _show_preview(self) -> None:
        canvas = getattr(self, "preview_canvas", None)
        if canvas is None:
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("all")
        if w < 20 or h < 20 or not self.preview_path or not Path(self.preview_path).exists():
            canvas.create_text(
                max(w, 20) / 2, max(h, 20) / 2, text=self._preview_note, fill=MUTED,
                font=self._point_font(14),
            )
            return
        from PIL import Image, ImageTk

        key = (self.preview_path, w // 8, h // 8)
        if key != getattr(self, "_preview_key", None) or self.preview_photo is None:
            self._preview_key = key
            try:
                img = Image.open(self.preview_path)
                img.thumbnail((max(40, w - 8), max(40, h - 8)), Image.LANCZOS)
                self.preview_photo = ImageTk.PhotoImage(img)
            except OSError:
                self.preview_photo = None
        if self.preview_photo is not None:
            canvas.create_image(w / 2, h / 2, image=self.preview_photo)

    def _open_preview(self) -> None:
        if self.preview_path and Path(self.preview_path).exists():
            os.startfile(self.preview_path)  # noqa: S606

    # --- Console -------------------------------------------------------------

    def _append(self, text: str, tag: str = "INFO") -> None:
        self._append_parts([(text, tag)])

    def _append_parts(self, parts: list[tuple[str, str]]) -> None:
        self.console.configure(state=tk.NORMAL)
        for text, tag in parts:
            self.console.insert(tk.END, text, tag)
        self.console.insert(tk.END, "\n")
        lines = int(self.console.index("end-1c").split(".")[0])
        if lines > 6000:
            self.console.delete("1.0", f"{lines - 5000}.0")
        self.console.configure(state=tk.DISABLED)
        if self.autoscroll.get():
            self.console.see(tk.END)

    def _gui_log(self, msg: str, tag: str = "gui") -> None:
        self._append_parts([(datetime.now().strftime("%H:%M:%S") + "  ", "time"), (msg, tag)])

    def _clear_log(self) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.configure(state=tk.DISABLED)

    def _copy_log(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.console.get("1.0", tk.END))
        self._gui_log("Log copied to the clipboard.")

    # --- Settings persistence ------------------------------------------------

    _PERSIST = ("preset", "data_root", "image_dir", "label_dir", "label_mode", "num_layers", "prompt",
                "output_root", "base_model", "max_samples", "val_percent", "epochs", "batch_size", "mosaic",
                "lr", "scope", "vit_blocks", "eval_base", "hflip", "keep_best", "device")

    def _save_settings(self) -> None:
        data = {}
        for key in self._PERSIST:
            try:
                data[key] = getattr(self, key).get()
            except tk.TclError:
                pass
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
        if not data:
            self._apply_preset()
        for key, value in data.items():
            if key in self._PERSIST:
                try:
                    getattr(self, key).set(value)
                except tk.TclError:
                    pass
        if data:
            preset = PRESETS.get(self.preset.get())
            stem = preset["name"] if preset else "model"
            self.model_name.set(f"{stem}_{datetime.now():%Y%m%d_%H%M}")
        self._update_layer_state()
        self._update_target()


def main() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    FinetuneGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
