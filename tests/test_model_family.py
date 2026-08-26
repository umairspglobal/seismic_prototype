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
