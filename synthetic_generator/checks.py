"""Dataset checks: layout, file sizes, replica grouping, fine-tuner compatibility and statistics."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .block import FOLDER_PREFIX
from .dataset import CONFIG_NAME

FOLDER_RE = re.compile(r"^(Seis|Fault|TauAtT|dipXscale|wtXscale|wtXStraddleScale|wtXHaloScale)(\d*)_?(.*)$")


def discover(root: Path) -> dict[str, Path]:
    """Map ``kind[scale]`` (e.g. ``seis``, ``dip12``) to folders found under ``root``."""
    by_prefix = {v: k for k, v in FOLDER_PREFIX.items()}
    found: dict[str, Path] = {}
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if not child.is_dir():
            continue
        m = FOLDER_RE.match(child.name)
        if not m:
            continue
        prefix, digits, _ = m.groups()
        kind = by_prefix[prefix]
        if kind in ("seis", "fault", "tau"):
            key = kind
        else:
            if not digits:
                continue
            key = f"{kind}{digits}"
        found.setdefault(key, child)
    return found


def _ids(folder: Path) -> list[int]:
    with os.scandir(folder) as it:
        return sorted(int(e.name[:-4]) for e in it if e.name.lower().endswith(".dat") and e.name[:-4].isdigit())


def read_square(path: Path) -> np.ndarray:
    size = path.stat().st_size
    for dtype, width in (("<f8", 8), ("<f4", 4)):
        if size % width:
            continue
        side = math.isqrt(size // width)
        if side * side * width == size:
            return np.fromfile(path, dtype=dtype).reshape(side, side).astype(np.float64)
    raise ValueError(f"{path}: {size} bytes is not a square float64/float32 image")


def spectral_centroid(section: np.ndarray) -> float:
    n = section.shape[0]
    spec = np.abs(np.fft.rfft((section - section.mean(0)) * np.hanning(n)[:, None], axis=0)) ** 2
    p = spec.mean(1)
    f = np.fft.rfftfreq(n)
    return float((f * p).sum() / p.sum()) if p.sum() > 0 else 0.0


def fault_shape(mask: np.ndarray) -> tuple[int, float, list[float]]:
    """Number of strokes, mean row run length and stroke angles (deg from horizontal)."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    runs, angles = [], []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if ys.size < 30:
            continue
        for y in np.unique(ys)[2:-2]:
            row = np.sort(xs[ys == y])
            if row.size and np.all(np.diff(row) == 1):
                runs.append(row.size)
        pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(float)
        _, vecs = np.linalg.eigh(pts.T @ pts)
        d = vecs[:, 1]
        angles.append(float(np.degrees(np.arctan2(abs(d[1]), abs(d[0])))))
    return n, float(np.mean(runs)) if runs else float("nan"), angles


def section_stats(root: Path, folders: dict[str, Path], ids: list[int]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        if value == value:
            acc.setdefault(key, []).append(value)

    for sid in ids:
        name = f"{sid}.dat"
        if "seis" in folders and (folders["seis"] / name).exists():
            s = read_square(folders["seis"] / name)
            add("seis_mean", float(s.mean()))
            add("seis_std", float(s.std()))
            add("seis_absmax", float(np.abs(s).max()))
            add("seis_centroid", spectral_centroid(s))
        if "fault" in folders and (folders["fault"] / name).exists():
            m = read_square(folders["fault"] / name) > 0.5
            add("fault_fraction", float(m.mean()))
            n, run, angles = fault_shape(m)
            add("fault_strokes", float(n))
            add("fault_row_width", run)
            for a in angles:
                add("fault_angle", a)
        if "tau" in folders and (folders["tau"] / name).exists():
            t = read_square(folders["tau"] / name)
            add("tau_min", float(t.min()))
            add("tau_max", float(t.max()))
            add("tau_dy_median", float(np.median(np.diff(t, axis=0))))
        for key, folder in folders.items():
            if key in ("seis", "fault", "tau") or not (folder / name).exists():
                continue
            a = read_square(folder / name)
            add(f"{key}_mean", float(a.mean()))
            if key.startswith("dip"):
                add(f"{key}_absp95", float(np.percentile(np.abs(a), 95)))
    return {k: float(np.median(v)) for k, v in acc.items()}


def inspect_dataset(root: str | Path, sample: int = 120, seed: int = 0) -> dict:
    """Validate a dataset folder; returns a report dict (``problems`` empty when all is well)."""
    root = Path(root)
    report: dict = {"root": str(root), "problems": [], "folders": {}, "notes": []}
    folders = discover(root)
    if not folders:
        report["problems"].append("No Seis/Fault/TauAtT/... folders found.")
        return report
    id_sets = {}
    for key, folder in folders.items():
        ids = _ids(folder)
        id_sets[key] = set(ids)
        report["folders"][key] = {"folder": folder.name, "files": len(ids),
                                  "first": ids[0] if ids else None, "last": ids[-1] if ids else None}
    common = set.intersection(*id_sets.values()) if id_sets else set()
    report["common_ids"] = len(common)
    for key, ids in id_sets.items():
        extra = len(ids - common)
        if extra:
            report["notes"].append(f"{folders[key].name}: {extra} ids missing from at least one other folder.")
    if not common:
        report["problems"].append("No id is present in every folder.")
        return report

    sizes = {(folders[k] / f"{min(common)}.dat").stat().st_size for k in folders}
    if len(sizes) != 1:
        report["problems"].append(f"Files for the same id have different sizes: {sorted(sizes)}.")
    try:
        side = read_square(folders["seis" if "seis" in folders else next(iter(folders))] / f"{min(common)}.dat").shape[0]
        report["image_size"] = side
    except ValueError as exc:
        report["problems"].append(str(exc))

    cfg_path = root / CONFIG_NAME
    if cfg_path.exists():
        run = json.loads(cfg_path.read_text(encoding="utf-8")).get("run", {})
        report["generator_run"] = run
        per = run.get("ids_per_block")
        grid = json.loads(cfg_path.read_text(encoding="utf-8")).get("grid", {})
        R, N = grid.get("n_replicas"), grid.get("n_slices")
        if per and R and N and "fault" in folders:
            start = run.get("start_id", 0)
            if all((folders["fault"] / f"{start + r * N}.dat").exists() for r in range(R)):
                base = read_square(folders["fault"] / f"{start}.dat")
                same = all(np.array_equal(base, read_square(folders["fault"] / f"{start + r * N}.dat")) for r in range(1, R))
                report["replicas_share_labels"] = bool(same)
                if not same:
                    report["problems"].append("Non-seismic arrays differ between replicas of the same slice.")

    if "seis" in folders and "fault" in folders:
        try:
            from finetuning.quick_data import list_pairs, read_section

            pairs = list_pairs(folders["seis"], folders["fault"])
            report["finetuner_pairs"] = len(pairs)
            if pairs:
                read_section(folders["seis"] / f"{pairs[0]}.dat")
                read_section(folders["fault"] / f"{pairs[0]}.dat")
                report["finetuner_ok"] = True
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            report["problems"].append(f"Quick Fine-Tuner cannot read the data: {exc}")

    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(sorted(common), size=min(sample, len(common)), replace=False).tolist())
    report["stats"] = section_stats(root, folders, chosen)
    report["sampled_ids"] = len(chosen)
    return report


def compare(gen_root: str | Path, ref_root: str | Path, sample: int = 120) -> list[tuple[str, float, float]]:
    gen = inspect_dataset(gen_root, sample).get("stats", {})
    ref = inspect_dataset(ref_root, sample).get("stats", {})
    keys = sorted(set(gen) | set(ref))
    return [(k, ref.get(k, float("nan")), gen.get(k, float("nan"))) for k in keys]
