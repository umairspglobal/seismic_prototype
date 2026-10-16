"""Data helpers for the quick SAM 3 fine-tuner. No GPU and no model weights."""

from __future__ import annotations

import numpy as np
import pytest

from finetuning.quick_data import (
    SectionSet,
    build_mosaic,
    list_pairs,
    load_sections,
    read_section,
    rgt_to_layers,
    split_stems,
)
from finetuning.quick_finetune import build_parser, lr_factor, summarize


def _write_dat(folder, stem: str, array: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.asarray(array, dtype="<f8").tofile(folder / f"{stem}.dat")


def test_read_and_match_square_sections(tmp_path):
    images, labels = tmp_path / "seis", tmp_path / "fault"
    seis = np.linspace(-2, 2, 16, dtype="<f8").reshape(4, 4)
    fault = np.zeros((4, 4), dtype="<f8")
    fault[:, 1] = 1
    _write_dat(images, "10", seis)
    _write_dat(labels, "10", fault)
    _write_dat(images, "2", seis)
    # "2" has no label and must be skipped; "10" matches.
    assert list_pairs(images, labels) == ["10"]

    loaded = read_section(images / "10.dat")
    assert loaded.shape == (4, 4)
    assert loaded.dtype == np.float32
    assert loaded[0, 0] == pytest.approx(seis[0, 0])


def test_rgt_layers_alternate_and_cover_half():
    rgt = np.linspace(0, 1, 8, dtype=np.float32).reshape(2, 4)
    mask = rgt_to_layers(rgt, num_layers=4)
    assert mask.dtype == bool
    # Equal bands, top band off, so about half the section is foreground.
    assert 0.3 < mask.mean() < 0.7
    assert rgt_to_layers(np.zeros((3, 3)), 8).sum() == 0


def test_load_sections_binary_and_layers(tmp_path):
    images, faults, rgt = tmp_path / "seis", tmp_path / "fault", tmp_path / "tau"
    seis = np.arange(16, dtype="<f8").reshape(4, 4)
    fault = np.zeros((4, 4), dtype="<f8")
    fault[0, :] = 1
    tau = np.tile(np.linspace(0, 100, 4), (4, 1))
    for stem in ("0", "1"):
        _write_dat(images, stem, seis)
        _write_dat(faults, stem, fault)
        _write_dat(rgt, stem, tau)

    binary = load_sections(images, faults, ["0", "1"], "binary")
    assert binary.images.shape == (2, 4, 4)
    assert binary.masks.dtype == bool
    assert binary.masks[:, 0, :].all()
    assert not binary.masks[:, 1, :].any()

    layers = load_sections(images, rgt, ["0"], "rgt_layers", num_layers=4)
    assert layers.masks.shape == (1, 4, 4)
    assert layers.positive_fraction > 0


def test_short_mosaic_repeats_instead_of_leaving_blanks():
    images = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    sections = SectionSet(["0", "1"], images, masks)
    image, mask = build_mosaic(sections, [0], grid=2, flips=[True, False, False, False])
    assert image.shape == (8, 8)
    # Every cell is the one available section (flipped only in cell 0).
    flipped = images[0][:, ::-1]
    assert image[:4, :4].tolist() == flipped.tolist()
    assert image[:4, 4:].tolist() == images[0].tolist()
    # The foreground pixel sits at column 0 and moves to column 3 when flipped.
    assert mask[0, 3] and not mask[0, 0]


def test_split_holds_out_validation_and_caps_it():
    stems = [str(i) for i in range(100)]
    train, val = split_stems(stems, max_samples=40, val_fraction=0.5, seed=1, max_val=6)
    assert len(val) == 6
    assert len(train) == 34
    assert set(train).isdisjoint(val)


def test_epoch_default_is_thirty():
    assert build_parser().get_default("epochs") == 30


def test_loss_schedule_and_scores():
    assert lr_factor(0, 100, 10) == pytest.approx(0.1)
    assert lr_factor(9, 100, 10) == pytest.approx(1.0)
    assert 0.05 <= lr_factor(99, 100, 10) <= 1.0
    stats = summarize(tp=50, fp=50, fn=0)
    assert stats["precision"] == pytest.approx(0.5)
    assert stats["recall"] == pytest.approx(1.0)
    assert stats["iou"] == pytest.approx(0.5)
