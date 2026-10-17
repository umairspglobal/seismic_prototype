"""One block = one geological model rendered as slices x bandpass replicas.

``generate_block`` returns every array of every id in the block. Sample ids are
replica-major like the reference dataset::

    id = start_id + block * (R * N) + replica * N + slice      (R = 4, N = 6)

so ids 0-5 are the six slices of replica 0, ids 6-11 the same slices with the
second bandpass, and so on. Only ``Seis`` differs between replicas; every other
array is repeated across them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ARRAY_KINDS, GeneratorConfig
from .geology import BlockGeology, build_slice, crop, make_faults, make_stratigraphy
from .labels import section_labels
from .seismic import lowest_center, render_block

FOLDER_PREFIX = {
    "seis": "Seis",
    "fault": "Fault",
    "tau": "TauAtT",
    "dip": "dipXscale",
    "weight": "wtXscale",
    "straddle": "wtXStraddleScale",
    "halo": "wtXHaloScale",
}
KIND_TITLES = {
    "seis": "Seismic",
    "fault": "Fault mask",
    "tau": "Tau (RGT)",
    "dip": "Dip",
    "weight": "Scale weight",
    "straddle": "Straddle",
    "halo": "Halo",
}
SCALED_KINDS = ("dip", "weight", "straddle", "halo")


@dataclass(frozen=True)
class Product:
    kind: str
    scale: int | None = None

    def folder(self, suffix: str) -> str:
        prefix = FOLDER_PREFIX[self.kind]
        if self.scale is None:
            return f"{prefix}{suffix}"
        return f"{prefix}{self.scale}_{suffix}" if suffix else f"{prefix}{self.scale}"

    @property
    def title(self) -> str:
        base = KIND_TITLES[self.kind]
        return f"{base} S={self.scale}" if self.scale is not None else base


def products(cfg: GeneratorConfig, kinds: list[str] | tuple[str, ...] | None = None) -> list[Product]:
    wanted = set(cfg.output.arrays if kinds is None else kinds)
    out: list[Product] = []
    for kind in ARRAY_KINDS:
        if kind not in wanted:
            continue
        if kind in SCALED_KINDS:
            out.extend(Product(kind, s) for s in cfg.grid.scales)
        else:
            out.append(Product(kind))
    return out


@dataclass
class Block:
    index: int
    seis: np.ndarray
    tau: np.ndarray
    fault: np.ndarray
    labels: list[dict[tuple[str, int], np.ndarray]]
    meta: dict
    geology: BlockGeology | None = None

    @property
    def n_replicas(self) -> int:
        return self.seis.shape[0]

    @property
    def n_slices(self) -> int:
        return self.seis.shape[1]

    def array(self, product: Product, replica: int, slice_index: int) -> np.ndarray:
        if product.kind == "seis":
            return self.seis[replica, slice_index]
        if product.kind == "fault":
            return self.fault[slice_index]
        if product.kind == "tau":
            return self.tau[slice_index]
        return self.labels[slice_index][(product.kind, product.scale)]

    def sample_ids(self, start_id: int, block_offset: int) -> list[tuple[int, int, int]]:
        R, N = self.n_replicas, self.n_slices
        base = start_id + block_offset * R * N
        return [(base + r * N + s, r, s) for r in range(R) for s in range(N)]


def block_streams(seed: int, index: int) -> list[np.random.Generator]:
    """Independent streams so editing seismic settings keeps the same geology."""
    root = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(index)])
    return [np.random.default_rng(s) for s in root.spawn(3)]


def vertical_pad(cfg: GeneratorConfig) -> int:
    return int(np.clip(np.ceil(1.6 / max(lowest_center(cfg), 1e-3)) + 6, 16, 128))


def generate_block(cfg: GeneratorConfig, index: int, seed: int | None = None, keep_geology: bool = False) -> Block:
    seed = cfg.output.seed if seed is None else seed
    strat_rng, fault_rng, seis_rng = block_streams(seed, index)
    H, N = cfg.grid.height, cfg.grid.n_slices
    pad = vertical_pad(cfg)

    strat, geo_meta = make_stratigraphy(strat_rng, cfg, pad)
    faults = make_faults(fault_rng, cfg)
    padded = [build_slice(strat, faults, z, cfg, pad) for z in range(N)]
    sections = [crop(g, pad, H) for g in padded]
    geo_meta["faults"] = [f.meta for f in faults]

    seismic = render_block(seis_rng, cfg, [g.tau for g in padded], [s.mask for s in sections], pad)
    tau = np.stack([s.tau for s in sections])
    fault = np.stack([s.mask for s in sections]).astype(np.float64)
    labels = [section_labels(s.tau, s.mask, s.offsets, s.inside, cfg.grid.scales, cfg.labels, s.package)
              for s in sections]

    meta = {
        "block": int(index),
        "seed": int(seed),
        "geology": geo_meta,
        "seismic": seismic.meta,
        "fault_fraction": [round(float(m.mean()), 5) for m in fault],
        "faults_visible": [int(sum(st.any() for st in s.strokes)) for s in sections],
        "tau_range": [round(float(tau.min()), 3), round(float(tau.max()), 3)],
    }
    geology = BlockGeology(strat=strat, faults=faults, slices=sections, pad=pad, meta=geo_meta) if keep_geology else None
    return Block(index=int(index), seis=seismic.seis, tau=tau, fault=fault, labels=labels, meta=meta, geology=geology)
