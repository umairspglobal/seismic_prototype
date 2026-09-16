from types import SimpleNamespace

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
