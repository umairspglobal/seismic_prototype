import numpy as np

from seismic_app.inference import Sam31PointSegmenter, Sam31VolumePropagator


class FakeMultiplexPredictor:
    def __init__(self):
        self.model = object()
        self.closed = []
        self.next_session = 1

    def handle_request(self, request):
        if request["type"] == "start_session":
            session_id = f"session-{self.next_session}"
            self.next_session += 1
            return {"session_id": session_id}
        if request["type"] == "close_session":
            self.closed.append(request["session_id"])
            return {"is_success": True}
        if request["type"] == "add_prompt":
            object_id = request["obj_id"]
            mask = np.full((4, 5), object_id, dtype=bool)
            return {
                "frame_index": request["frame_index"],
                "outputs": {
                    "out_obj_ids": np.array([object_id]),
                    "out_binary_masks": mask[None, ...],
                },
            }
        raise AssertionError(request)

    def handle_stream_request(self, request):
        for frame_index in range(3):
            yield {
                "frame_index": frame_index,
                "outputs": {
                    "out_obj_ids": np.array([1]),
                    "out_binary_masks": np.ones((1, 4, 5), dtype=bool),
                },
            }


def test_sam31_point_adapter_reuses_prepared_session():
    predictor = FakeMultiplexPredictor()
    segmenter = Sam31PointSegmenter(predictor, embedding_cache_size=1)
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    first = segmenter.segment(image, [(2, 1)], [1], image_key="slice-a")
    second = segmenter.segment(image, [(2, 1), (3, 1)], [1, 0], image_key="slice-a")

    assert first.all()
    assert second.all()
    assert predictor.next_session == 2


def test_sam31_volume_adapter_keeps_live_multiplex_session():
    predictor = FakeMultiplexPredictor()
    propagator = Sam31VolumePropagator(predictor)
    frames = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    updates = []

    masks = propagator.propagate(
        frames,
        anchor_idx=1,
        points_per_object=[[(2, 1)]],
        labels_per_object=[[1]],
        progress=lambda done, total, frame, _masks: updates.append(
            (done, total, frame)
        ),
        keep_live=True,
    )

    assert masks.shape == (1, 3, 4, 5)
    assert masks.all()
    assert updates == [(1, 3, 0), (2, 3, 1), (3, 3, 2)]
    assert propagator.live is not None
