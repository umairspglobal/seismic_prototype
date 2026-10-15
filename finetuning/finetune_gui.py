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

BG = "#1e1f22"
FG = "#d4d4d4"
COLORS = {
    "time": "#6b7280",
    "INFO": FG,
    "WARNING": "#f0b45a",
    "ERROR": "#ff6b6b",
    "metric": "#7ee787",
    "banner": "#79c0ff",
    "other": "#9ca3af",
    "gui": "#c4a7ff",
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
        scale = root.winfo_fpixels("1i") / 96.0
        width = min(int(1460 * scale), int(root.winfo_screenwidth() * 0.95))
        height = min(int(900 * scale), int(root.winfo_screenheight() * 0.9))
        root.geometry(f"{width}x{height}+{(root.winfo_screenwidth() - width) // 2}+{int(20 * scale)}")
        root.minsize(int(1100 * scale), int(680 * scale))
        self._build_vars()
        self._build_ui()
        self._load_settings()
        self._refresh_counts()
        self._update_estimate()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
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

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Stage.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 6))
        style.configure("Hint.TLabel", foreground="#555")
        style.configure("Warn.TLabel", foreground="#b45309")
        style.configure("Metric.TLabel", font=("Consolas", 11))

        panes = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._panes = panes

        top = ttk.Frame(panes)
        panes.add(top, weight=0)
        self._top = top
        col_a = ttk.Frame(top)
        col_a.pack(side=tk.LEFT, fill=tk.Y)
        col_b = ttk.Frame(top)
        col_b.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        col_c = ttk.Frame(top)
        col_c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self._build_data_frame(col_a)
        self._build_output_frame(col_a)
        self._build_training_frame(col_b)
        self._build_buttons(col_b)
        self._build_status(col_c)

        bottom = ttk.PanedWindow(panes, orient=tk.HORIZONTAL)
        panes.add(bottom, weight=1)
        console_frame = ttk.Frame(bottom)
        bottom.add(console_frame, weight=3)
        self._build_console(console_frame)
        preview_frame = ttk.LabelFrame(bottom, text=" Validation preview (double-click to open) ")
        bottom.add(preview_frame, weight=1)
        self._build_preview(preview_frame)
        self.root.after(80, self._place_sash)

    def _px(self, points: float) -> int:
        """Scale a 96-DPI size up to this screen, so the layout matches the font."""
        return max(1, int(points * self.root.winfo_fpixels("1i") / 96.0))

    def _place_sash(self) -> None:
        """Keep the form at its own height and give the remaining space to the log."""
        self.root.update_idletasks()
        height = self._panes.winfo_height()
        want = self._top.winfo_reqheight() + self._px(8)
        if height > self._px(200):
            self._panes.sashpos(0, min(want, int(height * 0.68)))
        for entry in getattr(self, "_path_entries", ()):
            entry.xview_moveto(1.0)

    def _path_entry(self, parent, var: tk.StringVar) -> ttk.Entry:
        """Entry that keeps the end of a long path (the folder name) in view."""
        entry = ttk.Entry(parent, textvariable=var, width=36)
        self._path_entries = getattr(self, "_path_entries", [])
        self._path_entries.append(entry)
        var.trace_add("write", lambda *_e, widget=entry: widget.after_idle(lambda: widget.xview_moveto(1.0)))
        entry.bind("<FocusOut>", lambda _e: entry.xview_moveto(1.0))
        return entry

    def _row(self, parent, r: int, label: str, widget, browse=None, hint: str | None = None) -> None:
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(6, 6), pady=2)
        widget.grid(row=r, column=1, sticky="ew", pady=2)
        if browse:
            ttk.Button(parent, text="Browse...", width=9, command=browse).grid(row=r, column=2, padx=(4, 6))
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel").grid(row=r, column=2, sticky="w", padx=(6, 6))

    def _build_data_frame(self, parent) -> None:
        f = ttk.LabelFrame(parent, text=" 1. Data ")
        f.pack(fill=tk.X, pady=(0, 6))
        f.columnconfigure(1, weight=1)
        preset = ttk.Combobox(f, textvariable=self.preset, values=list(PRESETS), state="readonly", width=44)
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
        ttk.Label(f, text="Layers per section").grid(row=5, column=0, sticky="w", padx=6, pady=2)
        self.layers_spin.grid(row=5, column=1, sticky="w", pady=2)
        self._row(f, 6, "Text prompt", ttk.Combobox(f, textvariable=self.prompt, values=PROMPTS))
        info = ttk.Frame(f)
        info.grid(row=7, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 4))
        ttk.Label(info, textvariable=self.count_text, style="Hint.TLabel").pack(side=tk.LEFT)
        ttk.Button(info, text="Preview data", command=self._preview_data).pack(side=tk.RIGHT)

    def _build_output_frame(self, parent) -> None:
        f = ttk.LabelFrame(parent, text=" 2. Output model ")
        f.pack(fill=tk.X, pady=6)
        f.columnconfigure(1, weight=1)
        self._row(f, 0, "Model name", ttk.Entry(f, textvariable=self.model_name))
        self._row(f, 1, "Save in folder", self._path_entry(f, self.output_root),
                  browse=lambda: self._browse_dir(self.output_root))
        base = ttk.Combobox(f, textvariable=self.base_model, values=self._base_choices())
        base.bind("<Button-1>", lambda _e: base.configure(values=self._base_choices()))
        self._row(f, 2, "Start from", base, browse=lambda: self._browse_dir(self.base_model))
        self.target_label = ttk.Label(f, textvariable=self.target_text, style="Hint.TLabel",
                                      wraplength=self._px(460))
        self.target_label.grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(2, 4))

    def _build_training_frame(self, parent) -> None:
        f = ttk.LabelFrame(parent, text=" 3. Training settings ")
        f.pack(fill=tk.X, pady=6)
        f.columnconfigure(1, weight=1)
        self._row(f, 0, "Sections to use", ttk.Spinbox(f, from_=50, to=100000, increment=500,
                                                        textvariable=self.max_samples, width=10),
                  hint="0 = all")
        self._row(f, 1, "Validation %", ttk.Spinbox(f, from_=2, to=40, textvariable=self.val_percent, width=10))
        self._row(f, 2, "Epochs", ttk.Spinbox(f, from_=1, to=200, textvariable=self.epochs, width=10))
        self._row(f, 3, "Batch size", ttk.Spinbox(f, from_=1, to=8, textvariable=self.batch_size, width=10),
                  hint="2 = ~4.5 GB VRAM")
        self._row(f, 4, "Input layout",
                  ttk.Combobox(f, textvariable=self.mosaic, values=list(MOSAICS), state="readonly", width=42))
        self._row(f, 5, "Learning rate", ttk.Entry(f, textvariable=self.lr, width=12))
        self._row(f, 6, "What to train",
                  ttk.Combobox(f, textvariable=self.scope, values=list(SCOPES), state="readonly", width=42))
        self._row(f, 7, "Unfreeze ViT blocks", ttk.Spinbox(f, from_=0, to=32, textvariable=self.vit_blocks, width=10),
                  hint="last N of 32; slower")
        self._row(f, 8, "Device", ttk.Combobox(f, textvariable=self.device, values=["cuda", "cpu"],
                                               state="readonly", width=10))
        checks = ttk.Frame(f)
        checks.grid(row=9, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 2))
        ttk.Checkbutton(checks, text="Score base model first", variable=self.eval_base).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(checks, text="Flip augmentation", variable=self.hflip).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(checks, text="Keep best epoch", variable=self.keep_best).pack(side=tk.LEFT, padx=2)
        ttk.Label(f, textvariable=self.estimate_text, style="Hint.TLabel", wraplength=self._px(520)).grid(
            row=10, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))

    def _build_buttons(self, parent) -> None:
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=(6, 0))
        self.start_btn = ttk.Button(f, text="Start fine-tuning", style="Big.TButton", command=self._start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(f, text="Stop and save", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(f, text="Open output folder", command=self._open_output).pack(side=tk.RIGHT)

    def _build_status(self, parent) -> None:
        f = ttk.LabelFrame(parent, text=" Status ")
        f.pack(fill=tk.X)
        ttk.Label(f, textvariable=self.stage_text, style="Stage.TLabel").pack(anchor="w", padx=8, pady=(6, 2))
        self.progress = ttk.Progressbar(f, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(f, textvariable=self.progress_text, wraplength=self._px(520)).pack(
            anchor="w", padx=8, pady=(0, 6))

        m = ttk.Frame(f)
        m.pack(fill=tk.X, padx=8, pady=(0, 8))
        cells = [[("Base IoU", "base"), ("Best IoU", "best"), ("Last IoU", "iou"), ("Val loss", "loss")],
                 [("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")]]
        for block, row_cells in enumerate(cells):
            for col, (title, key) in enumerate(row_cells):
                ttk.Label(m, text=title, style="Hint.TLabel").grid(
                    row=2 * block, column=col, padx=(0, 22), sticky="w")
                ttk.Label(m, textvariable=self.metric_text[key], style="Metric.TLabel").grid(
                    row=2 * block + 1, column=col, padx=(0, 22), sticky="w", pady=(0, 4))

        chart_frame = ttk.LabelFrame(parent, text=" Loss  (green = training average, orange = validation) ")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.chart = tk.Canvas(chart_frame, bg="white", highlightthickness=0, height=self._px(150))
        self.chart.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.chart.bind("<Configure>", lambda _e: self._draw_chart())

    def _build_preview(self, parent) -> None:
        # A canvas (not a label) so a large preview is scaled to the pane
        # instead of pushing the pane wider and getting clipped.
        self.preview_canvas = tk.Canvas(parent, bg="#f4f4f4", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.preview_canvas.bind("<Configure>", lambda _e: self._show_preview())
        self.preview_canvas.bind("<Double-Button-1>", lambda _e: self._open_preview())
        self._preview_note = "Appears after the first validation"

    def _build_console(self, parent) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(bar, text="Training output", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(bar, text="Clear", command=self._clear_log).pack(side=tk.RIGHT)
        ttk.Button(bar, text="Copy all", command=self._copy_log).pack(side=tk.RIGHT, padx=4)
        ttk.Checkbutton(bar, text="Auto-scroll", variable=self.autoscroll).pack(side=tk.RIGHT, padx=8)

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        self.console = tk.Text(frame, bg=BG, fg=FG, insertbackground=FG, font=("Consolas", 10),
                               wrap=tk.WORD, relief=tk.FLAT, padx=8, pady=6, state=tk.DISABLED)
        scroll = ttk.Scrollbar(frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for tag, color in COLORS.items():
            self.console.tag_configure(tag, foreground=color)
        self.console.tag_configure("bold", font=("Consolas", 10, "bold"))

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
        photo = ImageTk.PhotoImage(grid)
        lbl = ttk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack(padx=6, pady=6)
        ttk.Label(win, text=f"Foreground covers {100 * sec.positive_fraction:.1f}% of these sections").pack(pady=(0, 6))

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
        self.stop_btn.configure(state=tk.NORMAL, text="Stop and save")
        self.stage_text.set("Starting...")
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
            self.stage_text.set("Stopping - validating and saving...")
            self.stop_btn.configure(text="Force kill")
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
            self.stage_text.set(ev.get("message", ""))
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
            self.stage_text.set(f"Training - epoch {ev['epoch']} of {ev['epochs']}")
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
                self.stage_text.set(f"Done - {change}")
                self.progress_text.set(f"Saved to {ev['output']}")
            else:
                self.stage_text.set("Stopped - nothing saved")
        elif kind == "error":
            self.stage_text.set("Failed - see the log below")

    def _finished(self, code: int) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate")
        elapsed = fmt_duration(time.time() - self.run_started)
        if code == 0:
            self.progress["value"] = 100
            self._gui_log(f"Training process finished after {elapsed}.")
        else:
            if not self.stage_text.get().startswith("Failed"):
                self.stage_text.set(f"Failed (exit code {code}) - see the log below")
            self._gui_log(f"Training process exited with code {code} after {elapsed}.", tag="ERROR")
        self.proc = None
        if self.stop_file:
            self.stop_file.unlink(missing_ok=True)
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED, text="Stop and save")
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
            c.create_text(w / 2, h / 2, text="Waiting for training steps", fill="#999")
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

        c.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline="#ddd")
        for frac in (0, 0.5, 1):
            val = lo + (hi - lo) * frac
            _, y = xy(0, val)
            c.create_line(pad_l, y, w - pad_r, y, fill="#eee")
            c.create_text(pad_l - self._px(6), y, text=f"{val:.2f}", anchor="e", fill="#666", font=axis_font)
        c.create_text(w - pad_r, h - self._px(4), text=f"step {span_x}", anchor="se", fill="#666", font=axis_font)
        c.create_text(pad_l, h - self._px(4), text="0", anchor="sw", fill="#666", font=axis_font)
        if len(self.loss_points) > 1:
            stride = max(1, len(self.loss_points) // 600)
            coords = [v for p in self.loss_points[::stride] for v in xy(*p)]
            c.create_line(*coords, fill="#16a34a", width=2)
        for step, val in self.val_points:
            x, y = xy(step, val)
            r = self._px(4)
            c.create_oval(x - r, y - r, x + r, y + r, fill="#f97316", outline="")

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
                max(w, 20) / 2, max(h, 20) / 2, text=self._preview_note, fill="#888",
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
