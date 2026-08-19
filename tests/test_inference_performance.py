from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from seismic_app.inference import Sam3PointSegmenter, Sam3VolumePropagator


class _Encoding(dict):
    def to(self, device):
        return _Encoding(
            {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in self.items()
            }
        )


class _PointProcessor:
    def __call__(self, images=None, original_sizes=None, **kwargs):
        if images is not None:
            return _Encoding(
                pixel_values=torch.zeros((1, 3, 8, 8)),
                original_sizes=torch.tensor([[8, 8]]),
            )
        return _Encoding(
            original_sizes=torch.as_tensor(original_sizes),
            input_points=torch.zeros((1, 1, 1, 2)),
            input_labels=torch.ones((1, 1, 1), dtype=torch.int64),
        )


class _PointModel:
    def __init__(self):
        self.encode_calls = 0

    def get_image_embeddings(self, pixel_values):
        self.encode_calls += 1
        return pixel_values.mean(dim=(-1, -2), keepdim=True)


def _fake_point_segmenter(cache_size=2):
    segmenter = Sam3PointSegmenter.__new__(Sam3PointSegmenter)
    segmenter.device = "cpu"
    segmenter.model = _PointModel()
    segmenter.processor = _PointProcessor()
    segmenter.embedding_cache_size = cache_size
    segmenter._prepared = OrderedDict()
    segmenter.last_timings = {}
    return segmenter


def test_point_embeddings_are_keyed_and_lru_bounded():
    segmenter = _fake_point_segmenter(cache_size=2)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    segmenter.prepare_image(image, ("file", "inline", 1))
    segmenter.prepare_image(image.copy(), ("file", "inline", 1))
    assert segmenter.model.encode_calls == 1

    segmenter.prepare_image(image, ("file", "inline", 2))
    segmenter.prepare_image(image, ("file", "inline", 3))
    assert segmenter.model.encode_calls == 3
    assert list(segmenter._prepared) == [
        ("file", "inline", 2),
        ("file", "inline", 3),
    ]


class _VideoProcessor:
    def init_video_session(self, video, **kwargs):
        return SimpleNamespace(video=video)

    def add_inputs_to_inference_session(self, session, **kwargs):
        return session

    def post_process_masks(self, masks, original_sizes, binarize):
        return [masks[0] > 0]


class _VideoModel:
    def __call__(self, inference_session, frame_idx):
        return SimpleNamespace(
            frame_idx=frame_idx,
            pred_masks=torch.ones((1, 1, 2, 2)),
        )

    def propagate_in_video_iterator(
        self, session, start_frame_idx, reverse=False
    ):
        indices = [start_frame_idx, 0] if reverse else [start_frame_idx, 2]
        for frame_idx in indices:
            yield SimpleNamespace(
                frame_idx=frame_idx,
                pred_masks=torch.ones((1, 1, 2, 2)),
            )


def test_propagation_reports_each_completed_frame():
    propagator = Sam3VolumePropagator.__new__(Sam3VolumePropagator)
    propagator.device = "cpu"
    propagator.model = _VideoModel()
    propagator.processor = _VideoProcessor()
    propagator.session_dtype = torch.float32
    propagator.last_timings = {}
    updates = []

    masks = propagator.propagate(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)],
        anchor_idx=1,
        points=[(0, 0)],
        labels=[1],
        progress=lambda done, total, frame_idx, mask: updates.append(
            (done, total, frame_idx, int(mask.sum()))
        ),
    )

    assert masks.shape == (3, 2, 2)
    assert [update[:3] for update in updates] == [
        (1, 3, 1),
        (2, 3, 2),
        (3, 3, 0),
    ]
    assert propagator.last_timings["fps"] > 0
