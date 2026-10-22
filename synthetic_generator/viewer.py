"""Open a folder of seismic sections and draw one sample.

The folder is either a generator dataset (``Seis75``, ``Fault75``, ... next to
``generator_config.json``) or one directory of numbered sections (``1.dat``,
``2.npy``, ...). Each file is one 2D section, the same layout the Quick
Fine-Tuner reads. SEG-Y volumes are a different viewer.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .block import FOLDER_PREFIX, KIND_TITLES, SCALED_KINDS, Product
from .dataset import CONFIG_NAME
from .preview import Tile, TileInfo, array_rgb, colorize, grid, overlay, seismic_rgb

# One section, not a volume. 64 MB is a square of about 2896 float64 samples.
MAX_SECTION_BYTES = 64 * 1024 * 1024
_DEFAULT_KINDS = ("seis", "fault", "tau")
_KIND_ORDER = {kind: i for i, kind in enumerate(FOLDER_PREFIX)}
_PREFIXES = sorted(FOLDER_PREFIX.items(), key=lambda item: len(item[1]), reverse=True)
_SCALED_REST = re.compile(r"^(\d+)(?:_[A-Za-z0-9]*)?$")
_PLAIN_REST = re.compile(r"^(?:\d+)?(?:_[A-Za-z0-9]*)?$")


@dataclass(frozen=True)
class SampleLayout:
    """Replica-major id layout recorded in ``generator_config.json``."""

    n_slices: int
    n_replicas: int


@dataclass(frozen=True)
class ArrayFolder:
    key: str
    path: Path
    kind: str
    scale: int | None
    ids: tuple[int, ...]
    files: dict[int, Path] = field(repr=False)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def zoom_label(self) -> str:
        """Label the zoom window already knows how to color."""
        if self.kind == "seis":
            return "Seis"
        if self.kind == "tau":
            return "Tau"
        if self.kind == "fault":
            return "Fault"
        if self.kind == "dip":
            return f"dip S={self.scale}"
        if self.kind == "halo":
            return f"halo S={self.scale}"
        if self.kind == "csd_dip":
            return f"CsdDip S={self.scale}"
        if self.kind == "csd_dist":
            return "CsdDistance"
        return KIND_TITLES.get(self.kind, self.name)


@dataclass(frozen=True)
class Catalog:
    root: Path
    arrays: tuple[ArrayFolder, ...]
    ids: tuple[int, ...]
    layout: SampleLayout | None
    shape: tuple[int, int] | None
    hint: str


@dataclass
class LoadedSample:
    sections: list[tuple[ArrayFolder, np.ndarray]]
    fault: np.ndarray | None
    errors: list[str]


def _parse_folder_name(name: str) -> tuple[str, int | None] | None:
    """``Seis75`` -> ``(seis, None)``, ``dipXscale12_75`` -> ``(dip, 12)``."""
    for kind, prefix in _PREFIXES:
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        if kind in SCALED_KINDS:
            match = _SCALED_REST.match(rest)
            if match:
                return kind, int(match.group(1))
            continue
        if _PLAIN_REST.match(rest):
            return kind, None
    return None


def section_index(folder: Path) -> dict[int, Path]:
    """Numeric ``.dat`` / ``.npy`` stems in ``folder``. A ``.dat`` wins over a ``.npy`` with the same id."""
    found: dict[int, Path] = {}
    rank: dict[int, int] = {}
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return found
    for entry in entries:
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        stem, ext = os.path.splitext(entry.name)
        ext = ext.lower()
        if ext not in (".dat", ".npy") or not stem.isdigit():
            continue
        sid = int(stem)
        score = 2 if ext == ".dat" else 1
        if sid not in found or score > rank[sid]:
            found[sid] = Path(entry.path)
            rank[sid] = score
    return found


def _folder(key: str, path: Path, kind: str, scale: int | None, files: dict[int, Path]) -> ArrayFolder:
    return ArrayFolder(key, path, kind, scale, tuple(sorted(files)), files)


def _layout_file(path: Path) -> SampleLayout | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        grid = data.get("grid")
        if not isinstance(grid, dict):
            return None
        n_slices = int(grid["n_slices"])
        n_replicas = int(grid["n_replicas"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if n_slices < 1 or n_replicas < 1:
        return None
    return SampleLayout(n_slices, n_replicas)


def _read_layout(root: Path, single: bool) -> SampleLayout | None:
    layout = _layout_file(root / CONFIG_NAME)
    if layout is None and single:
        layout = _layout_file(root.parent / CONFIG_NAME)
    return layout


def _shape_of(path: Path) -> tuple[int, int] | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > MAX_SECTION_BYTES:
        return None
    if path.suffix.lower() == ".npy":
        try:
            arr = np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            return None
        if getattr(arr, "ndim", 0) == 2:
            return int(arr.shape[0]), int(arr.shape[1])
        return None
    for width in (8, 4):
        if size % width:
            continue
        count = size // width
        side = math.isqrt(count)
        if side * side == count and side > 0:
            return side, side
    return None


def _empty_hint(children: list[Path]) -> str:
    sgy = loose = 0
    dirs = []
    for entry in children:
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if is_dir:
            dirs.append(entry)
            continue
        low = entry.name.lower()
        if low.endswith((".sgy", ".segy")):
            sgy += 1
        elif low.endswith((".dat", ".npy")):
            loose += 1
    if sgy:
        return ("This folder contains SEG-Y files. Point at a section dataset "
                "(the folder that holds Seis75, Fault75, ...) or at a folder of numbered .dat / .npy sections.")
    if loose:
        return "Section files here need numeric names such as 1.dat or 12.npy."
    deeper: list[str] = []
    if len(dirs) <= 40:
        for kid in dirs:
            try:
                inner = list(kid.iterdir())
            except OSError:
                continue
            if any(c.is_dir() and _parse_folder_name(c.name) for c in inner):
                deeper.append(kid.name)
    if len(deeper) == 1:
        return f"No sections in this folder. Open {deeper[0]}, which looks like a dataset."
    if deeper:
        return "No sections in this folder. Dataset folders inside: " + ", ".join(deeper[:6]) + "."
    return ("No seismic sections here. Open the dataset folder that contains Seis75, Fault75 and the "
            "other array folders, or a folder of numbered .dat / .npy sections.")


def scan_dataset(root: str | Path) -> Catalog:
    """List the array folders and sample ids under ``root`` without reading the sections."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        raise OSError(f"Cannot read {root}: {exc}") from exc

    folders: list[ArrayFolder] = []
    seen: set[str] = set()
    named: list[Path] = []
    for child in children:
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if not is_dir:
            continue
        parsed = _parse_folder_name(child.name)
        if parsed is None:
            continue
        named.append(child)
        kind, scale = parsed
        key = kind if scale is None else f"{kind}{scale}"
        if key in seen:
            continue
        files = section_index(child)
        if not files:
            continue
        seen.add(key)
        folders.append(_folder(key, child, kind, scale, files))

    single = False
    if not folders:
        files = section_index(root)
        if files:
            parsed = _parse_folder_name(root.name)
            if parsed is None:
                kind, scale, key = "section", None, "section"
            else:
                kind, scale = parsed
                key = kind if scale is None else f"{kind}{scale}"
            folders.append(_folder(key, root, kind, scale, files))
            single = True

    folders.sort(key=lambda a: (_KIND_ORDER.get(a.kind, 100), -1 if a.scale is None else a.scale, a.name))
    ids = tuple(sorted({sid for folder in folders for sid in folder.ids}))
    shape = _shape_of(folders[0].files[folders[0].ids[0]]) if folders else None
    if folders:
        hint = ""
    elif named:
        shown = ", ".join(p.name for p in named[:6])
        hint = f"Found {shown}, but no numbered sections (1.dat, 2.npy, ...) inside."
    else:
        hint = _empty_hint(children)
    return Catalog(root, tuple(folders), ids, _read_layout(root, single), shape, hint)


def catalog_summary(catalog: Catalog) -> str:
    if not catalog.arrays:
        return catalog.hint
    names = [folder.name for folder in catalog.arrays]
    head = ", ".join(names[:3])
    if len(names) > 3:
        head += f" + {len(names) - 3} more"
    ids = catalog.ids
    bits = [head, f"{len(ids):,} ids ({ids[0]:,}-{ids[-1]:,})"]
    if catalog.shape:
        h, w = catalog.shape
        bits.append(f"{h} x {w}")
    return "  |  ".join(bits)


def describe_id(layout: SampleLayout | None, sid: int) -> str:
    """``id 7  |  block 1  |  slice 2/3  |  replica 0`` when the generator layout is known.

    Slice numbers start at 1, matching the preview. Replica numbers start at 0,
    matching the preview. Block numbers start at 0.
    """
    if layout is None:
        return f"id {sid}"
    per = layout.n_slices * layout.n_replicas
    block, rem = divmod(int(sid), per)
    replica, sl = divmod(rem, layout.n_slices)
    return f"id {sid}  |  block {block}  |  slice {sl + 1}/{layout.n_slices}  |  replica {replica}"


def step_id(ids: tuple[int, ...] | list[int], value: int, delta: int) -> int:
    """Next or previous id in ``ids``. A value that is not in the list steps to the neighbour on that side."""
    if not ids:
        raise ValueError("No sample ids.")
    i = bisect.bisect_left(ids, value)
    if delta > 0:
        if i < len(ids) and ids[i] == value:
            i += 1
        i = min(i, len(ids) - 1)
    else:
        i = max((i - 1) if i < len(ids) and ids[i] == value else i - 1, 0)
    return ids[i]


def default_keys(arrays: tuple[ArrayFolder, ...] | list[ArrayFolder]) -> set[str]:
    """Seismic, fault and tau when those folders exist; otherwise the first few arrays."""
    preferred = {folder.key for folder in arrays if folder.kind in _DEFAULT_KINDS}
    if preferred:
        return preferred
    return {folder.key for folder in list(arrays)[:6]}


def read_section_file(path: Path) -> np.ndarray:
    """One 2D section as float64. ``.dat`` is a square little-endian float; ``.npy`` may be rectangular."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    if size <= 0:
        raise ValueError(f"{path.name} is empty")
    if size > MAX_SECTION_BYTES:
        raise ValueError(f"{path.name} is {size / (1024 * 1024):.0f} MB; the viewer shows one section per file")
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if getattr(data, "ndim", 0) != 2:
            raise ValueError(f"{path.name}: expected a 2D section, got shape {getattr(data, 'shape', None)}")
        return np.asarray(data, dtype=np.float64)
    from .checks import read_square

    return read_square(path)


def load_sample(catalog: Catalog, sid: int, keys: set[str], overlay_fault: bool) -> LoadedSample:
    """Read the selected arrays for one id. The fault array is included when an overlay is requested."""
    sections: list[tuple[ArrayFolder, np.ndarray]] = []
    fault = None
    errors: list[str] = []
    for src in catalog.arrays:
        wanted = src.key in keys
        if not wanted and not (overlay_fault and src.kind == "fault"):
            continue
        path = src.files.get(sid)
        if path is None:
            if wanted:
                errors.append(f"{src.name} has no file for id {sid}")
            continue
        try:
            arr = read_section_file(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{src.name}: {exc}")
            continue
        if src.kind == "fault":
            fault = arr
        if wanted:
            sections.append((src, arr))
    return LoadedSample(sections, fault, errors)


def _finite(values: np.ndarray) -> np.ndarray:
    if values.size and np.isfinite(values).all():
        return values
    return np.nan_to_num(values)


def _looks_seismic(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return abs(float(values.mean())) < 0.35 and 0.4 < float(values.std()) < 2.5


def paint_section(src: ArrayFolder, values: np.ndarray, cmap: str) -> np.ndarray:
    shown = _finite(values)
    if src.kind in FOLDER_PREFIX:
        return array_rgb(Product(src.kind, src.scale), shown, cmap)
    if _looks_seismic(shown):
        return seismic_rgb(shown, cmap)
    lo, hi = float(shown.min()), float(shown.max())
    if lo >= 0.0 and hi <= 1.0:
        return colorize(shown, "gray", 0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return colorize(shown, "gray", lo, hi)


def _range_text(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return "no finite values"
    return f"min {finite.min():.3g}  max {finite.max():.3g}  mean {finite.mean():.3g}"


def _cols(n: int) -> int:
    if n <= 3:
        return max(1, n)
    if n <= 8:
        return 3
    return 5


def fit_sample_tile(n: int, width: int, height: int, gap: int = 10) -> int:
    """Largest tile so ``n`` sections fit in ``width x height`` pixels."""
    n = max(1, n)
    cols = _cols(n)
    rows = (n + cols - 1) // cols
    tw = (width - (cols + 1) * gap) / cols
    th = (height - 34 - rows * (34 + gap) - gap) / rows
    return int(max(48, min(tw, th)))


def render_sample(catalog: Catalog, sample: LoadedSample, sid: int, cmap: str, show_faults: bool,
                  tile: int) -> tuple[Image.Image, list[TileInfo]]:
    """Draw the loaded sections. A fault overlay is painted on seismic tiles; the readout array stays the amplitude."""
    fault = sample.fault
    tiles: list[Tile] = []
    for src, arr in sample.sections:
        rgb = paint_section(src, arr, cmap)
        subtitle = _range_text(arr)
        label = src.zoom_label
        if src.kind == "section" and _looks_seismic(_finite(arr)):
            label = "Seis"
        if show_faults and src.kind == "seis" and fault is not None and fault.shape == arr.shape:
            rgb = overlay(rgb, fault)
            subtitle = "fault overlay  |  " + subtitle
        tiles.append(Tile(src.name, rgb, arr, subtitle, label))
    title = f"{describe_id(catalog.layout, sid)}  |  {catalog.root.name}"
    return grid(tiles, _cols(len(tiles)), tile, title)
