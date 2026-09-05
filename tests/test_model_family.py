import pytest

from seismic_app import config
from seismic_app.inference import (
    _point_tracker_classes,
    _video_session_class,
    _video_tracker_classes,
)


def test_resolve_family_defaults_to_sam3():
    assert config.resolve_family(None) == "sam3"
    assert config.DEFAULT_MODEL_FAMILY == "sam3"
    assert config.checkpoint_for() == "facebook/sam3"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("sam3", "sam3"),
        ("SAM 3", "sam3"),
        ("sam2", "sam2"),
        ("SAM 2", "sam2"),
        ("sam2.1", "sam2"),
        ("SAM-2.1", "sam2"),
    ],
)
def test_resolve_family_aliases(raw, expected):
    assert config.resolve_family(raw) == expected


def test_resolve_family_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown model family"):
        config.resolve_family("sam1")


def test_sam2_checkpoint_is_the_improved_hiera_large():
    assert config.checkpoint_for("sam2") == "facebook/sam2.1-hiera-large"
    assert config.family_spec("sam2")["gated"] is False
    assert config.family_spec("sam3")["gated"] is True


@pytest.fixture(autouse=True)
def _reset_text_checkpoint_override():
    config._TEXT_CHECKPOINT_OVERRIDE = None
    yield
    config._TEXT_CHECKPOINT_OVERRIDE = None


def test_text_checkpoint_falls_back_to_stock_sam3(monkeypatch, tmp_path):
    monkeypatch.delenv("SAM3_TEXT_CHECKPOINT", raising=False)
    monkeypatch.setattr(config, "DEFAULT_TEXT_CHECKPOINT_DIR", tmp_path / "missing")
    assert config.text_checkpoint() == "facebook/sam3"


def test_text_checkpoint_env_override(monkeypatch):
    monkeypatch.setenv("SAM3_TEXT_CHECKPOINT", r"C:\weights\facies")
    assert config.text_checkpoint() == r"C:\weights\facies"


def test_text_checkpoint_local_converted_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("SAM3_TEXT_CHECKPOINT", raising=False)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(config, "DEFAULT_TEXT_CHECKPOINT_DIR", tmp_path)
    assert config.text_checkpoint() == str(tmp_path)


def test_list_text_checkpoints_includes_official_and_converted(monkeypatch, tmp_path):
    monkeypatch.delenv("SAM3_TEXT_CHECKPOINT", raising=False)
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "config.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("ignore")
    (tmp_path / "incomplete").mkdir()
    monkeypatch.setattr(config, "FINETUNED_CHECKPOINTS_DIR", tmp_path)
    listed = config.list_text_checkpoints()
    assert listed[0]["path"] == "facebook/sam3"
    assert listed[0]["source"] == "official"
    local = [item for item in listed if item["source"] == "local"]
    assert [item["id"] for item in local] == ["run_a"]
    assert local[0]["path"] == str(tmp_path / "run_a")


def test_set_text_checkpoint_accepts_folder_name(monkeypatch, tmp_path):
    (tmp_path / "seismic_facies_phase1").mkdir()
    (tmp_path / "seismic_facies_phase1" / "config.json").write_text("{}")
    monkeypatch.setattr(config, "FINETUNED_CHECKPOINTS_DIR", tmp_path)
    path = config.set_text_checkpoint("seismic_facies_phase1")
    assert path == str(tmp_path / "seismic_facies_phase1")
    assert config.text_checkpoint() == path


def test_resolve_text_checkpoint_rejects_raw_trainer_pt(tmp_path):
    (tmp_path / "checkpoint.pt").write_bytes(b"not-hf")
    with pytest.raises(ValueError, match="config.json"):
        config.resolve_text_checkpoint(str(tmp_path))


def test_tracker_classes_match_family():
    sam3_model, sam3_proc = _point_tracker_classes("sam3")
    sam2_model, sam2_proc = _point_tracker_classes("sam2")
    assert sam3_model.__name__ == "Sam3TrackerModel"
    assert sam3_proc.__name__ == "Sam3TrackerProcessor"
    assert sam2_model.__name__ == "Sam2Model"
    assert sam2_proc.__name__ == "Sam2Processor"

    sam3_vmodel, sam3_vproc = _video_tracker_classes("sam3")
    sam2_vmodel, sam2_vproc = _video_tracker_classes("sam2")
    assert sam3_vmodel.__name__ == "Sam3TrackerVideoModel"
    assert sam3_vproc.__name__ == "Sam3TrackerVideoProcessor"
    assert sam2_vmodel.__name__ == "Sam2VideoModel"
    assert sam2_vproc.__name__ == "Sam2VideoProcessor"

    assert _video_session_class("sam3").__name__ == "Sam3TrackerVideoInferenceSession"
    assert _video_session_class("sam2").__name__ == "Sam2VideoInferenceSession"
