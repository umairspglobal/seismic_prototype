"""Uploads land in data/ and show up in the survey list."""

from __future__ import annotations

import io

import numpy as np
import pytest
import segyio
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import server.main as api


def _upload(payload: bytes, filename: str) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(payload))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    yield tmp_path


def test_upload_route_accepts_post():
    methods = {
        method
        for route in api.app.routes
        if getattr(route, "path", None) == "/api/files"
        for method in getattr(route, "methods", set())
    }
    assert "POST" in methods
    assert "GET" in methods


def test_upload_stores_npy_and_lists_it(data_dir):
    saved = api.upload_seismic_file(
        _upload(_npy_bytes(np.zeros((2, 3, 4), dtype=np.float32)), "volume.npy")
    )

    assert saved["name"] == "volume.npy"
    assert saved["format"] == "npy"
    assert saved["kind"] == "3d"
    assert saved["shape"] == [2, 3, 4]
    assert (data_dir / "volume.npy").is_file()
    assert api.list_files(format="npy") == [saved]
    assert "volume.npy" not in api._file_cache


def test_upload_stores_sgy(data_dir):
    path = data_dir / "built.sgy"
    section = np.arange(8, dtype=np.float32).reshape(2, 4)
    spec = segyio.spec()
    spec.format = 5
    spec.samples = list(range(4))
    spec.tracecount = 2
    with segyio.create(str(path), spec) as handle:
        handle.bin.update(
            {
                segyio.BinField.Samples: 4,
                segyio.BinField.Interval: 4000,
                segyio.BinField.Format: 5,
            }
        )
        for index in range(2):
            handle.header[index] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: index + 1,
                segyio.TraceField.CDP: 10 + index,
                segyio.TraceField.TRACE_SAMPLE_COUNT: 4,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: 4000,
            }
            handle.trace[index] = section[index]
    payload = path.read_bytes()
    path.unlink()

    saved = api.upload_seismic_file(_upload(payload, "line.sgy"))

    assert saved["name"] == "line.sgy"
    assert saved["format"] == "sgy"
    assert (data_dir / "line.sgy").is_file()
    assert saved["kind"] in {"2d", "3d"}


def test_upload_rejects_other_types_and_empty_files(data_dir):
    with pytest.raises(HTTPException) as rejected:
        api.upload_seismic_file(_upload(b"hello", "notes.txt"))
    assert rejected.value.status_code == 422
    assert list(data_dir.iterdir()) == []

    with pytest.raises(HTTPException) as empty:
        api.upload_seismic_file(_upload(b"", "empty.npy"))
    assert empty.value.status_code == 422
    assert list(data_dir.iterdir()) == []


def test_upload_drops_unreadable_files(data_dir):
    with pytest.raises(HTTPException) as bad:
        api.upload_seismic_file(_upload(b"this is not a numpy file", "broken.npy"))
    assert bad.value.status_code == 422
    assert list(data_dir.iterdir()) == []


def test_upload_strips_directories_and_avoids_clobbering(data_dir):
    first = api.upload_seismic_file(
        _upload(_npy_bytes(np.ones((2, 2), dtype=np.float32)), r"..\..\outside.npy")
    )
    second = api.upload_seismic_file(
        _upload(_npy_bytes(np.zeros((3, 2), dtype=np.float32)), "outside.npy")
    )

    assert first["name"] == "outside.npy"
    assert second["name"] == "outside_2.npy"
    assert (data_dir / "outside.npy").is_file()
    assert (data_dir / "outside_2.npy").is_file()
    assert not (data_dir.parent / "outside.npy").exists()
