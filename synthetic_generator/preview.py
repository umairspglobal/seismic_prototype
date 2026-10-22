"""Preview images: colormaps, overlays, tile grids, spectra and wiggle traces.

``render_view`` turns one generated block into a PIL image plus a list of
``TileInfo`` (position, source array, label) that the GUI uses for the
value-under-cursor readout.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .block import Block, Product, products
from .config import GeneratorConfig
from .seismic import amplitude_spectrum

VIEWS = {
    "overview": "Overview",
    "arrays": "All arrays",
    "slices": "Slices",
    "replicas": "Replicas and spectra",
    "labels": "Labels at one scale",
    "sam3": "SAM 3 input",
    "csd": "CSD",
}
SEISMIC_CMAPS = ("gray", "seismic", "petrel")

BG = (16, 20, 24)
PANEL = (23, 28, 34)
TEXT = (230, 233, 236)
MUTED = (138, 148, 158)
FAULT_RGB = (255, 64, 64)
REPLICA_COLORS = [(96, 165, 250), (52, 211, 153), (251, 191, 36), (244, 114, 182),
                  (167, 139, 250), (248, 113, 113), (45, 212, 191), (250, 204, 21)]

_CMAP_ANCHORS = {
    "gray": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "seismic": [(0.0, (0, 0, 120)), (0.25, (40, 90, 255)), (0.5, (245, 245, 245)), (0.75, (255, 70, 40)),
                (1.0, (120, 0, 0))],
    "petrel": [(0.0, (160, 255, 255)), (0.2, (0, 60, 200)), (0.4, (120, 120, 140)), (0.5, (235, 235, 235)),
               (0.6, (140, 120, 110)), (0.8, (200, 40, 0)), (1.0, (255, 240, 0))],
    "viridis": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)),
                (1.0, (253, 231, 37))],
    "inferno": [(0.0, (0, 0, 4)), (0.25, (87, 16, 110)), (0.5, (188, 55, 84)), (0.75, (249, 142, 9)),
                (1.0, (252, 255, 164))],
    "diverging": [(0.0, (59, 76, 192)), (0.5, (242, 242, 242)), (1.0, (180, 4, 38))],
    "csd": [(0.0, (250, 196, 110)), (0.3, (200, 120, 60)), (0.5, (60, 25, 75)), (0.7, (60, 110, 180)),
            (1.0, (130, 215, 245))],
}


@lru_cache(maxsize=None)
def lut(name: str) -> np.ndarray:
    anchors = _CMAP_ANCHORS[name]
    xs = np.array([a for a, _ in anchors])
    cols = np.array([c for _, c in anchors], dtype=float)
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, xs, cols[:, k]) for k in range(3)], axis=1).astype(np.uint8)


def colorize(a: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    span = vmax - vmin if vmax > vmin else 1.0
    idx = np.clip((a - vmin) / span * 255.0, 0, 255).astype(np.uint8)
    return lut(cmap)[idx]


def seismic_rgb(s: np.ndarray, cmap: str = "gray", clip: float = 3.0) -> np.ndarray:
    return colorize(s, cmap, -clip, clip)


def overlay(rgb: np.ndarray, mask: np.ndarray, color=FAULT_RGB, alpha: float = 0.75) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask > 0.5
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.astype(np.uint8)


def tau_rgb(tau: np.ndarray, spacing: float = 8.0) -> np.ndarray:
    rgb = colorize(tau, "viridis", float(tau.min()), float(tau.max())).astype(np.float32)
    band = np.floor(tau / spacing)
    edge = np.zeros(tau.shape, dtype=bool)
    edge[:-1] |= band[:-1] != band[1:]
    edge[:, :-1] |= band[:, :-1] != band[:, 1:]
    rgb[edge] *= 0.35
    return rgb.astype(np.uint8)


def dip_range(a: np.ndarray) -> tuple[float, float]:
    lo, hi = (float(v) for v in np.percentile(a, [1, 99]))
    return (lo, hi) if hi > lo else (lo - 1.0, lo + 1.0)


def array_rgb(prod: Product, a: np.ndarray, cmap: str = "gray") -> np.ndarray:
    if prod.kind == "seis":
        return seismic_rgb(a, cmap)
    if prod.kind == "tau":
        return tau_rgb(a)
    if prod.kind == "dip":
        m = float(np.percentile(np.abs(a), 99)) or 1.0
        return colorize(a, "diverging", -m, m)
    if prod.kind == "halo":
        return colorize(a, "inferno", 0.0, 1.0)
    if prod.kind == "csd_dip":
        vals = a[a != 0]
        return colorize(a, "csd", *dip_range(vals if vals.size else a))
    if prod.kind == "csd_dist":
        return colorize(a, "inferno", 0.0, float(a.max()) or 1.0)
    return colorize(a, "gray", 0.0, 1.0)


@lru_cache(maxsize=16)
def font(size: int, bold: bool = False):
    names = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "arial.ttf",
                                                                               "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class TileInfo:
    x: int
    y: int
    w: int
    h: int
    array: np.ndarray | None
    label: str


@dataclass
class Tile:
    title: str
    rgb: np.ndarray
    array: np.ndarray | None = None
    subtitle: str = ""
    label: str = ""


def _resize(rgb: np.ndarray, w: int, h: int) -> Image.Image:
    img = Image.fromarray(rgb)
    if img.size == (w, h):
        return img
    method = Image.NEAREST if w >= 2 * rgb.shape[1] or w <= rgb.shape[1] // 2 else Image.BICUBIC
    return img.resize((w, h), method)


def grid(tiles: list[Tile], cols: int, tile: int, title: str = "", gap: int = 10) -> tuple[Image.Image, list[TileInfo]]:
    rows = (len(tiles) + cols - 1) // cols
    head = 34 if title else 6
    label_h = 34
    width = cols * tile + (cols + 1) * gap
    height = head + rows * (tile + label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((gap, 8), title, fill=TEXT, font=font(16, True))
    infos: list[TileInfo] = []
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = gap + c * (tile + gap)
        y = head + r * (tile + label_h + gap) + label_h
        aspect = t.rgb.shape[0] / t.rgb.shape[1]
        w, h = (tile, int(round(tile * aspect))) if aspect <= 1 else (int(round(tile / aspect)), tile)
        draw.rectangle([x - 1, y - 1, x + w, y + h], outline=(44, 52, 61))
        canvas.paste(_resize(t.rgb, w, h), (x, y))
        draw.text((x, y - label_h + 2), t.title, fill=TEXT, font=font(13, True))
        if t.subtitle:
            draw.text((x, y - label_h + 18), t.subtitle, fill=MUTED, font=font(11))
        infos.append(TileInfo(x, y, w, h, t.array, t.label or t.title))
    return canvas, infos


def _fmt_range(a: np.ndarray) -> str:
    return f"min {a.min():.3g}  max {a.max():.3g}  mean {a.mean():.3g}"


def _labels_composite(block: Block, s: int, scale: int, cmap: str, replica: int) -> np.ndarray:
    """Seismic with straddle (cyan), zero-weight (dark) and fault (red) on top."""
    rgb = seismic_rgb(block.seis[replica, s], cmap).astype(np.float32)
    w = block.labels[s][("weight", scale)]
    st = block.labels[s][("straddle", scale)]
    rgb[w < 0.5] *= 0.45
    rgb[st > 0.5] = 0.5 * rgb[st > 0.5] + 0.5 * np.array([34, 211, 238])
    return overlay(rgb.astype(np.uint8), block.fault[s])


def spectrum_chart(block: Block, cfg: GeneratorConfig, size: tuple[int, int], slice_index: int) -> np.ndarray:
    """Measured amplitude spectra (solid) and wavelet responses (dotted) of every replica."""
    W, H = size
    img = Image.new("RGB", (W, H), PANEL)
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 46, W - 12, 26, H - 34
    fmax = 0.5
    d.rectangle([left, top, right, bottom], outline=(60, 70, 80))
    dt = cfg.grid.sample_interval_ms
    for k in range(6):
        f = fmax * k / 5
        x = left + (right - left) * f / fmax
        d.line([x, bottom, x, bottom + 4], fill=MUTED)
        d.text((x - 12, bottom + 6), f"{f:.1f}", fill=MUTED, font=font(10))
        d.text((x - 12, bottom + 18), f"{f * 1000 / dt:.0f}Hz", fill=MUTED, font=font(9))
    d.text((left, 6), "Amplitude spectra  (cycles/sample, Hz at "
           f"{dt:g} ms)", fill=TEXT, font=font(12, True))
    n = block.seis.shape[2]
    freqs = np.fft.rfftfreq(n)
    win = np.hanning(n)[:, None]
    for r in range(block.n_replicas):
        sec = block.seis[r, slice_index]
        p = np.abs(np.fft.rfft((sec - sec.mean(0)) * win, axis=0)).mean(1)
        p = np.convolve(p, np.ones(3) / 3, mode="same")
        p = p / (p.max() or 1)
        pts = [(left + (right - left) * f / fmax, bottom - (bottom - top) * v * 0.92) for f, v in zip(freqs, p)]
        col = REPLICA_COLORS[r % len(REPLICA_COLORS)]
        d.line(pts, fill=col, width=2)
        fc = block.meta["seismic"]["centers"][r]
        resp = amplitude_spectrum(freqs, fc, cfg.seismic.wavelet, cfg.seismic.bandwidth)
        resp = resp / (resp.max() or 1)
        for f, v in zip(freqs[::2], resp[::2]):
            x, y = left + (right - left) * f / fmax, bottom - (bottom - top) * v * 0.92
            d.point((x, y), fill=col)
        d.text((right - 150, top + 6 + 15 * r), f"R{r}  fc={fc:.3f} ({fc * 1000 / dt:.0f} Hz)", fill=col, font=font(11))
    return np.asarray(img)


def wiggle_chart(block: Block, size: tuple[int, int], slice_index: int, column: int) -> np.ndarray:
    """Classic wiggle traces (positive lobes filled) of one column for every replica."""
    W, H = size
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    R = block.n_replicas
    top, bottom = 44, H - 8
    d.text((8, 4), f"Trace x={column} (all replicas)", fill=(20, 20, 20), font=font(12, True))
    n = block.seis.shape[2]
    lane = (W - 16) / R
    for r in range(R):
        tr = block.seis[r, slice_index, :, column]
        x0 = 8 + lane * (r + 0.5)
        amp = lane * 0.42 / (np.abs(tr).max() or 1)
        ys = top + (bottom - top) * np.arange(n) / (n - 1)
        xs = x0 + tr * amp
        for i in range(n - 1):
            if tr[i] > 0 or tr[i + 1] > 0:
                d.polygon([(x0, ys[i]), (max(xs[i], x0), ys[i]), (max(xs[i + 1], x0), ys[i + 1]), (x0, ys[i + 1])],
                          fill=REPLICA_COLORS[r % len(REPLICA_COLORS)])
        d.line(list(zip(xs, ys)), fill=(20, 20, 20), width=1)
        d.line([(x0, top), (x0, bottom)], fill=(160, 160, 160))
        d.text((x0 - 8, 24), f"R{r}", fill=(20, 20, 20), font=font(11, True))
    return np.asarray(img)


def _finetuner_tools():
    try:
        from finetuning.quick_data import rgt_to_layers
        from seismic_app.preprocessing import normalize_to_uint8
        return normalize_to_uint8, rgt_to_layers
    except Exception:  # noqa: BLE001 - previews still work without the app packages

        def normalize_to_uint8(data):
            lo, hi = np.percentile(data, [2.0, 98.0])
            if hi <= lo:
                return np.zeros(data.shape, np.uint8)
            return ((np.clip(data, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)

        def rgt_to_layers(rgt, num_layers):
            lo, hi = float(rgt.min()), float(rgt.max())
            band = np.clip(np.floor((rgt - lo) / max(hi - lo, 1e-9) * num_layers).astype(int), 0, num_layers - 1)
            return (band % 2) == 1

        return normalize_to_uint8, rgt_to_layers


def csd_tiles(block: Block, s: int, S: int, replica: int, cmap: str) -> dict[str, Tile]:
    """Seismic + horizon tracks, GT dip, weight, prompt, target mask and distance for one slice and scale."""
    seis = block.seis[replica, s]
    c = block.csd[s]
    mask, dist = c[("csd_mask", None)], c[("csd_dist", None)]
    dip_s, wt = block.labels[s][("dip", S)], block.labels[s][("weight", S)]
    shown, prompt = c[("csd_prompt", S)], c[("csd_dip", S)]
    lo, hi = dip_range(dip_s)
    colored = colorize(dip_s, "csd", lo, hi)
    on_seis = seismic_rgb(seis, cmap)
    on_seis[shown] = colored[shown]
    prompt_rgb = np.full(seis.shape + (3,), 255, np.uint8)
    prompt_rgb[shown] = colorize(prompt, "csd", lo, hi)[shown]
    n_hor = block.meta.get("csd_horizons", [0] * block.n_slices)[s]
    return {
        "track": Tile(f"Seis+track S={S}", on_seis, seis, f"{int(shown.any(axis=0).sum())} cols", "Seis"),
        "dip": Tile(f"GT dip S={S}", colored, dip_s, f"{lo:.1f} .. {hi:.1f}", f"dip S={S}"),
        "weight": Tile(f"Weight S={S}", colorize(wt, "gray", 0, 1), wt, f"{wt.mean() * 100:.0f}% on", f"weight S={S}"),
        "prompt": Tile(f"Prompt S={S}", prompt_rgb, prompt, f"{int(shown.sum())} px", f"CsdDipXscale{S}"),
        "mask": Tile("GT mask", colorize(mask, "gray", 0, 1), mask, f"{n_hor} horizons", "CsdHorizon"),
        "dist": Tile("Distance", colorize(dist, "inferno", 0.0, float(dist.max()) or 1.0), dist,
                     f"max {dist.max():.3g}", "CsdDistance"),
    }


def view_grid(view: str, cfg: GeneratorConfig) -> tuple[int, int, float]:
    """(columns, rows, extra rows of chart height) used by ``render_view``."""
    if view == "arrays":
        n = max(1, len(products(cfg)))
        cols = 5 if n > 8 else min(4, n)
        return cols, (n + cols - 1) // cols, 0.0
    if view == "slices":
        return cfg.grid.n_slices, 2, 0.0
    if view == "replicas":
        return cfg.grid.n_replicas, 1, 0.9
    if view == "csd":
        return 6, max(1, len(cfg.grid.scales)), 0.0
    if view == "sam3":
        return 3, 3 if cfg.csd.enabled else 2, 0.0
    return 3, 2, 0.0


def fit_tile(view: str, cfg: GeneratorConfig, width: int, height: int, gap: int = 10) -> int:
    """Largest tile size so the whole view fits in ``width x height`` pixels."""
    cols, rows, extra = view_grid(view, cfg)
    tw = (width - (cols + 1) * gap) / cols
    th = (height - 34 - rows * (34 + gap) - gap - (12 if extra else 0)) / (rows + extra)
    return int(max(64, min(tw, th)))


def render_view(block: Block, cfg: GeneratorConfig, view: str, replica: int = 0, slice_index: int = 0,
                scale: int | None = None, tile: int = 320, cmap: str = "gray", show_faults: bool = True,
                facies_layers: int = 8) -> tuple[Image.Image, list[TileInfo]]:
    R, N = block.n_replicas, block.n_slices
    replica = int(np.clip(replica, 0, R - 1))
    s = int(np.clip(slice_index, 0, N - 1))
    scale = scale if scale in cfg.grid.scales else cfg.grid.scales[0]
    seis = block.seis[replica, s]
    fault = block.fault[s]
    tau = block.tau[s]
    meta = block.meta["seismic"]
    fc = meta["centers"][replica]
    rtxt = f"replica {replica}: fc {fc:.3f} c/s ({meta['centers_hz'][replica]:.0f} Hz), clip {meta['clips'][replica]:.2f}"
    head = f"Block {block.index}  |  slice {s + 1}/{N}  |  {rtxt}"

    def seis_tile(rr: int, ss: int, title: str, with_faults: bool) -> Tile:
        rgb = seismic_rgb(block.seis[rr, ss], cmap)
        if with_faults:
            rgb = overlay(rgb, block.fault[ss])
        return Tile(title, rgb, block.seis[rr, ss], f"slice {ss + 1}, replica {rr}", f"Seis s{ss + 1} r{rr}")

    if view == "overview":
        if cfg.csd.enabled and block.csd is not None:
            t = csd_tiles(block, s, scale, replica, cmap)
            tiles = [t["track"], t["dip"], t["weight"], t["prompt"], t["mask"], t["dist"]]
            return grid(tiles, 3, tile, head + f"  |  coarse-scale dips at S={scale}")
        mask_rgb = colorize(fault, "gray", 0, 1)
        tiles = [
            seis_tile(replica, s, "Seismic", False),
            seis_tile(replica, s, "Seismic + fault mask", True),
            Tile("Fault mask (as written)", mask_rgb, fault, f"{fault.mean() * 100:.2f}% foreground", "Fault"),
            Tile("Tau / RGT", tau_rgb(tau), tau, _fmt_range(tau), "TauAtT"),
            Tile(f"Dip S={scale}", array_rgb(Product("dip", scale), block.labels[s][("dip", scale)]),
                 block.labels[s][("dip", scale)], f"tau[y,x] - tau[y,x+{scale}]", f"dip S={scale}"),
            Tile(f"Label overlay S={scale}", _labels_composite(block, s, scale, cmap, replica),
                 block.labels[s][("weight", scale)], "cyan straddle, dark weight 0", f"weight S={scale}"),
        ]
        return grid(tiles, 3, tile, head)

    if view == "arrays":
        tiles = []
        for prod in products(cfg, cfg.output.arrays if cfg.output.arrays else None):
            a = block.array(prod, replica, s)
            tiles.append(Tile(prod.folder(cfg.output.suffix), array_rgb(prod, a, cmap), a, _fmt_range(a), prod.title))
        cols = 5 if len(tiles) > 8 else min(4, len(tiles))
        return grid(tiles, cols, tile, head + "  |  every array written for this id")

    if view == "slices":
        tiles = [seis_tile(replica, z, f"Slice {z + 1}", show_faults) for z in range(N)]
        tiles += [Tile(f"Fault {z + 1}", colorize(block.fault[z], "gray", 0, 1), block.fault[z],
                       f"{block.meta['faults_visible'][z]} faults, {block.fault[z].mean() * 100:.2f}%", f"Fault s{z + 1}")
                  for z in range(N)]
        return grid(tiles, N, tile, f"Block {block.index}  |  {N} slices of one model (replica {replica})")

    if view == "replicas":
        tiles = [seis_tile(r, s, f"Replica {r}", show_faults) for r in range(R)]
        img, infos = grid(tiles, R, tile, head)
        chart_h = int(tile * 0.9)
        spec = spectrum_chart(block, cfg, (int(img.width * 0.6), chart_h), s)
        wig = wiggle_chart(block, (img.width - spec.shape[1] - 30, chart_h), s, cfg.grid.width // 2)
        canvas = Image.new("RGB", (img.width, img.height + chart_h + 12), BG)
        canvas.paste(img, (0, 0))
        canvas.paste(Image.fromarray(spec), (10, img.height))
        canvas.paste(Image.fromarray(wig), (20 + spec.shape[1], img.height))
        return canvas, infos

    if view == "labels":
        tiles = [
            Tile(f"Dip S={scale}", array_rgb(Product("dip", scale), block.labels[s][("dip", scale)]),
                 block.labels[s][("dip", scale)], "shift to the same reflector S traces right", f"dip S={scale}"),
            Tile(f"Straddle S={scale}", colorize(block.labels[s][("straddle", scale)], "gray", 0, 1),
                 block.labels[s][("straddle", scale)], "x and x+S on opposite sides of a fault",
                 f"straddle S={scale}"),
            Tile(f"Scale weight S={scale}", colorize(block.labels[s][("weight", scale)], "gray", 0, 1),
                 block.labels[s][("weight", scale)], "0 = ignore in the loss", f"weight S={scale}"),
            Tile(f"Halo S={scale}", array_rgb(Product("halo", scale), block.labels[s][("halo", scale)]),
                 block.labels[s][("halo", scale)], "blur radius S, peaks at 1", f"halo S={scale}"),
            Tile("Overlay", _labels_composite(block, s, scale, cmap, replica), block.labels[s][("weight", scale)],
                 "red fault, cyan straddle, dark weight 0", f"weight S={scale}"),
            seis_tile(replica, s, "Seismic", show_faults),
        ]
        return grid(tiles, 3, tile, head + f"  |  labels at S={scale}")

    if view == "sam3":
        to_uint8, to_layers = _finetuner_tools()
        gray = to_uint8(seis)
        gray_rgb = np.repeat(gray[..., None], 3, axis=-1)
        prev_s, next_s = max(0, s - 1), min(N - 1, s + 1)
        rgb25 = np.stack([to_uint8(block.seis[replica, prev_s]), gray, to_uint8(block.seis[replica, next_s])], -1)
        target = np.zeros(seis.shape + (3,), np.uint8)
        target[:] = (24, 30, 38)
        target[fault > 0.5] = (255, 255, 255)
        layers = to_layers(tau, facies_layers)
        facies = overlay(gray_rgb, layers.astype(float), (52, 211, 153), 0.45)
        mosaic_ids = [max(0, s - 1), s, min(N - 1, s + 1), min(N - 1, s + 2)]
        mos = np.block([[to_uint8(block.seis[replica, mosaic_ids[0]]), to_uint8(block.seis[replica, mosaic_ids[1]])],
                        [to_uint8(block.seis[replica, mosaic_ids[2]]), to_uint8(block.seis[replica, mosaic_ids[3]])]])
        mmask = np.block([[block.fault[mosaic_ids[0]], block.fault[mosaic_ids[1]]],
                          [block.fault[mosaic_ids[2]], block.fault[mosaic_ids[3]]]])
        mosaic = overlay(np.repeat(mos[..., None], 3, -1), mmask, (255, 140, 0), 0.6)
        tiles = [
            Tile("Model input (gray, 2-98% clip)", gray_rgb, seis, "what the fine-tuner feeds SAM 3", "Seis"),
            Tile("Target: fault vs background", target, fault, "white = mask, dark = background", "Fault"),
            Tile("Input + target overlay", overlay(gray_rgb, fault, (255, 140, 0), 0.7), fault, "prompt: fault",
                 "Fault"),
            Tile("2.5D RGB (prev / this / next)", rgb25, seis, f"R=slice {prev_s + 1}, G={s + 1}, B={next_s + 1}",
                 "Seis"),
            Tile(f"Facies target ({facies_layers} layers)", facies, layers.astype(float),
                 "TauAtT -> rgt_layers, prompt: seismic facies", "facies"),
        ]
        if cfg.csd.enabled and block.csd is not None:
            horizon = block.csd[s][("csd_mask", None)]
            tiles.append(Tile(
                "CSD target: horizon tracks",
                overlay(gray_rgb, horizon, (0, 255, 255), 0.85),
                horizon,
                "CsdHorizon, prompt: horizon",
                "CsdHorizon",
            ))
        tiles.append(Tile("2x2 training mosaic", mosaic, None, "sections tiled like the fine-tuner", "mosaic"))
        return grid(tiles, 3, tile, head + "  |  what the Quick Fine-Tuner trains on")

    if view == "csd":
        if block.csd is None:
            raise ValueError("This block has no CSD arrays; generate it with csd=True.")
        tiles = []
        for S in cfg.grid.scales:
            t = csd_tiles(block, s, S, replica, cmap)
            tiles += [t["track"], t["dip"], t["weight"], t["prompt"], t["mask"], t["dist"]]
        note = "" if cfg.csd.enabled else " (preview only)"
        return grid(tiles, 6, tile, f"Block {block.index}  |  slice {s + 1}/{N}  |  replica {replica}  |  CSD{note}")

    raise ValueError(f"Unknown view {view!r}; choose from {list(VIEWS)}")
