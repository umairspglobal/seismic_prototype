import numpy as np
import torch

from seismic_app.inference import (
    Sam31PointSegmenter,
    Sam31VolumePropagator,
    _adapt_sam31_init_state,
    _crop_letterbox_mask,
    _install_sam31_point_mask_ranking,
    _letterbox_relative_points,
    _mask_for_native_id,
    _pad_to_square,
    _rerank_sam31_multimask_ious,
)


class FakeMultiplexPredictor:
    def __init__(self):
        self.model = object()
        self.closed = []
        self.next_session = 1
        self.requests = []
        self.session_shapes = {}

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            session_id = f"session-{self.next_session}"
            self.next_session += 1
            image = request["resource_path"][0]
            width, height = image.size
            self.session_shapes[session_id] = (height, width)
            return {"session_id": session_id}
        if request["type"] == "close_session":
            self.closed.append(request["session_id"])
            return {"is_success": True}
        if request["type"] == "remove_object":
            return {"is_success": True}
        if request["type"] == "add_prompt":
            object_id = request["obj_id"]
            height, width = self.session_shapes[request["session_id"]]
            mask = np.ones((height, width), dtype=bool)
            return {
                "frame_index": request["frame_index"],
                "outputs": {
                    "out_obj_ids": np.array([object_id]),
                    "out_binary_masks": mask[None, ...],
                },
            }
        raise AssertionError(request)

    def handle_stream_request(self, request):
        height, width = self.session_shapes[request["session_id"]]
        for frame_index in range(3):
            yield {
                "frame_index": frame_index,
                "outputs": {
                    "out_obj_ids": np.array([1]),
                    "out_binary_masks": np.ones((1, height, width), dtype=bool),
                },
            }


class FakeSam31Model:
    def init_state(self, resource_path, offload_video_to_cpu=False):
        return resource_path, offload_video_to_cpu


def test_sam31_init_adapter_discards_unsupported_state_offload_option():
    model = FakeSam31Model()
    _adapt_sam31_init_state(model)

    assert model.init_state(
        resource_path=["frame.png"],
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    ) == (["frame.png"], True)


def test_sam31_letterbox_keeps_wide_section_aspect():
    rgb = np.zeros((4, 16, 3), dtype=np.uint8)
    rgb[2, 8] = 255
    padded, box = _pad_to_square(rgb)

    assert padded.shape == (16, 16, 3)
    assert box == {"top": 6, "left": 0, "height": 4, "width": 16, "side": 16}
    assert np.array_equal(padded[6:10], rgb)
    assert _letterbox_relative_points([(8, 2)], box) == [[8 / 16, (2 + 6) / 16]]

    native = np.zeros((16, 16), dtype=bool)
    native[8, 8] = True
    cropped = _crop_letterbox_mask(native, box)
    assert cropped.shape == (4, 16)
    assert cropped[2, 8]


def test_sam31_point_adapter_reuses_prepared_session():
    predictor = FakeMultiplexPredictor()
    segmenter = Sam31PointSegmenter(predictor, embedding_cache_size=1)
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    first = segmenter.segment(image, [(2, 1)], [1], image_key="slice-a")
    second = segmenter.segment(image, [(2, 1), (3, 1)], [1, 0], image_key="slice-a")

    assert first.shape == (4, 5)
    assert first.all()
    assert second.all()
    assert predictor.next_session == 2
    session_image = [
        request for request in predictor.requests if request["type"] == "start_session"
    ][0]["resource_path"][0]
    assert session_image.size == (5, 5)
    prompt = [
        request for request in predictor.requests if request["type"] == "add_prompt"
    ][-1]
    assert prompt["points"] == [[0.4, 0.2], [0.6, 0.2]]
    assert prompt["rel_coordinates"] is True


def test_sam31_one_click_prefers_local_click_consistent_candidate():
    masks = -torch.ones((1, 3, 4, 4))
    masks[0, 0] = 1  # Incorrect full-frame candidate with the highest native IoU.
    masks[0, 1, 1:3, 1:3] = 1
    ious = torch.tensor([[0.99, 0.65, 0.98]])

    reranked = _rerank_sam31_multimask_ious(
        masks,
        ious,
        torch.tensor([[[2.0, 2.0]]]),
        torch.tensor([[1]], dtype=torch.int32),
        image_size=4,
    )

    assert int(reranked.argmax(dim=1).item()) == 1


def test_sam31_one_click_ranks_upsampled_thin_candidate():
    masks = -torch.ones((1, 3, 2, 2))
    masks[0, 0] = 1  # Full-frame candidate, highest native IoU.
    masks[0, 1, 1] = 1  # Bottom half at 2x2; a thin event after upsample.
    ious = torch.tensor([[0.99, 0.40, 0.10]])

    reranked = _rerank_sam31_multimask_ious(
        masks,
        ious,
        torch.tensor([[[2.0, 3.0]]]),
        torch.tensor([[1]], dtype=torch.int32),
        image_size=4,
    )

    assert int(reranked.argmax(dim=1).item()) == 1


def test_sam31_rerank_on_wide_aspect_logits():
    masks = -torch.ones((1, 3, 8, 8))
    masks[0, 0] = 1
    masks[0, 1, 3:5, 1:7] = 1
    ious = torch.tensor([[0.97, 0.41, 0.20]])

    reranked = _rerank_sam31_multimask_ious(
        masks,
        ious,
        torch.tensor([[[20.0, 16.0]]]),
        torch.tensor([[1]], dtype=torch.int32),
        image_size=32,
    )

    assert int(reranked.argmax(dim=1).item()) == 1


def test_sam31_uses_only_native_mask_when_object_id_differs():
    mask = np.ones((4, 5), dtype=bool)
    chosen = _mask_for_native_id(
        {
            "out_obj_ids": np.array([7]),
            "out_binary_masks": mask[None, ...],
        },
        native_id=1,
    )

    assert chosen is not None
    assert chosen.shape == (4, 5)
    assert chosen.all()


def test_sam31_decoder_wrap_prefers_local_mask():
    prompt = type("Enc", (), {})()
    prompt.forward = lambda points=None, boxes=None, masks=None: points

    def native_forward(*args, multimask_output=False, **kwargs):
        masks = -torch.ones((1, 3, 4, 4))
        masks[0, 0] = 1
        masks[0, 1, 1:3, 1:3] = 1
        ious = torch.tensor([[0.99, 0.65, 0.10]])
        return masks, ious, torch.zeros(1), torch.zeros(1)

    decoder = type("Dec", (), {})()
    decoder.forward = native_forward
    tracker_model = type("Tracker", (), {})()
    tracker_model.image_size = 4
    tracker_model.interactive_sam_prompt_encoder = prompt
    tracker_model.interactive_sam_mask_decoder = decoder
    wrapper = type("Wrapper", (), {})()
    wrapper.model = tracker_model
    demo = type("Demo", (), {})()
    demo.tracker = wrapper
    predictor = type("Predictor", (), {})()
    predictor.model = demo

    _install_sam31_point_mask_ranking(predictor)
    prompt.forward(
        points=(
            torch.tensor([[[2.0, 2.0]]]),
            torch.tensor([[1]], dtype=torch.int32),
        )
    )
    _masks, ious, *_ = decoder.forward(multimask_output=True)

    assert int(ious.argmax(dim=1).item()) == 1


def test_sam31_point_preview_replaces_native_object_state():
    predictor = FakeMultiplexPredictor()
    segmenter = Sam31PointSegmenter(predictor, embedding_cache_size=1)
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    segmenter.segment(
        image,
        [(2, 1)],
        [1],
        image_key="slice-a",
        object_id=3,
    )
    segmenter.segment(
        image,
        [(2, 1)],
        [1],
        image_key="slice-a",
        object_id=3,
    )
    segmenter.segment(
        image,
        [(1, 2)],
        [1],
        image_key="slice-a",
        object_id=3,
    )

    object_requests = [
        request
        for request in predictor.requests
        if request["type"] in ("add_prompt", "remove_object")
    ]
    assert [request["type"] for request in object_requests] == [
        "add_prompt",
        "remove_object",
        "add_prompt",
    ]
    assert all(request["obj_id"] == 4 for request in object_requests)


def test_sam31_point_adapter_closes_other_file_sessions():
    predictor = FakeMultiplexPredictor()
    segmenter = Sam31PointSegmenter(predictor, embedding_cache_size=4)
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    segmenter.prepare_image(image, image_key=("a.sgy", "inline", 0))
    segmenter.prepare_image(image, image_key=("b.sgy", "inline", 0))

    assert predictor.closed == ["session-1"]
    assert list(segmenter._prepared) == [("b.sgy", "inline", 0)]


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
    prompt = next(
        request for request in predictor.requests if request["type"] == "add_prompt"
    )
    assert prompt["points"] == [[0.4, 0.2]]
    assert prompt["rel_coordinates"] is True


def test_sam31_refinement_uses_relative_coordinates():
    predictor = FakeMultiplexPredictor()
    propagator = Sam31VolumePropagator(predictor)
    frames = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    propagator.propagate(
        frames,
        anchor_idx=1,
        points_per_object=[[(2, 1)]],
        labels_per_object=[[1]],
        keep_live=True,
    )

    propagator.refine_frame(propagator.live, 0, 2, [(4, 3)], [1])

    prompt = [
        request for request in predictor.requests if request["type"] == "add_prompt"
    ][-1]
    assert prompt["points"] == [[0.8, 0.6]]
    assert prompt["rel_coordinates"] is True
