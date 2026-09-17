from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from server import main


def test_point_requests_reject_stale_family_during_switch(monkeypatch):
    monkeypatch.setattr(main, "_active_family", "sam31")
    monkeypatch.setattr(
        main, "_point_segmenter", SimpleNamespace(family="sam3")
    )
    monkeypatch.setitem(main._load_state, "error", None)
    monkeypatch.setitem(main._load_state, "stage", "Switching to SAM 3.1...")

    with pytest.raises(HTTPException, match="Switching to SAM 3.1"):
        main._require_point_segmenter()


def test_volume_requests_reject_stale_family_during_switch(monkeypatch):
    monkeypatch.setattr(main, "_active_family", "sam31")
    monkeypatch.setattr(
        main, "_propagator", SimpleNamespace(family="sam3")
    )
    monkeypatch.setitem(main._load_state, "error", None)
    monkeypatch.setitem(main._load_state, "stage", "Switching to SAM 3.1...")

    with pytest.raises(HTTPException, match="Switching to SAM 3.1"):
        main._require_propagator()


def test_slice_endpoint_does_not_require_a_loaded_tracker(monkeypatch):
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    section = np.zeros((4, 6), dtype=np.float32)
    geometry = SimpleNamespace(kind="2d")
    monkeypatch.setattr(
        main,
        "_get_file",
        lambda _name: (section, geometry, section.astype(np.uint8)),
    )
    monkeypatch.setattr(main, "_slice_rgb", lambda *_args, **_kwargs: rgb)
    monkeypatch.setattr(main, "_point_segmenter", None)
    monkeypatch.setattr(main, "_active_family", "sam31")
    monkeypatch.setitem(main._load_state, "stage", "Switching to SAM 3.1...")
    monkeypatch.setitem(main._load_state, "error", None)

    response = main.get_slice("1.sgy")

    assert response.media_type == "image/jpeg"
    assert response.body


def test_file_meta_does_not_require_a_loaded_tracker(monkeypatch):
    section = np.zeros((4, 6), dtype=np.float32)
    geometry = SimpleNamespace(kind="2d")
    monkeypatch.setattr(
        main,
        "_get_file",
        lambda _name: (section, geometry, section.astype(np.uint8)),
    )
    monkeypatch.setattr(main, "_point_segmenter", None)

    info = main.file_meta("line.sgy")

    assert info["name"] == "line.sgy"
    assert info["kind"] == "2d"
    assert info["shape"] == [4, 6]
