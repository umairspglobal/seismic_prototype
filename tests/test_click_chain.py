from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from seismic_app.inference import (
    LiveTracker,
    MaskBase,
    Sam3PointSegmenter,
    Sam3VolumePropagator,
    _install_temporal_mask_selection,
    _keep_as_conditioning,
    _tidy_mask,
)

SIZE = 16


class _Encoding(dict):
    def to(self, device):
        return self


class _ChainProcessor:
    def __call__(self, images=None, original_sizes=None, input_points=None, input_labels=None, **kwargs):
        if images is not None:
            return _Encoding(pixel_values=torch.zeros((1, 3, SIZE, SIZE)), original_sizes=torch.tensor([[SIZE, SIZE]]))
        return _Encoding(
            original_sizes=torch.as_tensor(original_sizes),
            input_points=torch.tensor(input_points, dtype=torch.float32),
            input_labels=torch.tensor(input_labels),
        )

    def post_process_masks(self, masks, original_sizes, mask_threshold=0.0, binarize=True):
        return [masks[0] > mask_threshold]


def _boxes(points, labels, label_wanted, radius):
    mask = torch.zeros((SIZE, SIZE), dtype=torch.bool)
    for (x, y), label in zip(points.tolist(), labels.tolist()):
        if label == label_wanted:
            mask[max(0, int(y) - radius) : int(y) + radius + 1, max(0, int(x) - radius) : int(x) + radius + 1] = True
    return mask


class _ChainModel:
    """Masks are boxes around positive clicks minus the negative pixels.

    ``sloppy`` makes the single-mask output ignore negative clicks and
    ``flood`` makes it cover the whole image.
    """

    def __init__(self, sloppy=False, flood=False):
        self.sloppy = sloppy
        self.flood = flood
        self.calls = []

    def get_image_embeddings(self, pixel_values):
        return [torch.zeros((1, 4, 2, 2))]

    def __call__(self, input_points, input_labels, image_embeddings, multimask_output, input_masks=None, **kwargs):
        points, labels = input_points[0, 0], input_labels[0, 0]
        self.calls.append({"n": len(labels), "multimask": multimask_output, "prior": input_masks})
        positive = _boxes(points, labels, 1, radius=1)
        clean = positive & ~_boxes(points, labels, 0, radius=0)
        if multimask_output:
            candidates = [torch.ones_like(clean), clean, torch.zeros_like(clean)]
            iou = [0.99, 0.5, 0.1]
        else:
            candidates = [torch.ones_like(clean) if self.flood else positive if self.sloppy else clean]
            iou = [0.9]
        logits = torch.stack([torch.where(c, 5.0, -5.0) for c in candidates])
        return SimpleNamespace(pred_masks=logits[None, None], iou_scores=torch.tensor(iou)[None, None])


def _segmenter(model):
    segmenter = Sam3PointSegmenter.__new__(Sam3PointSegmenter)
    segmenter.device = "cpu"
    segmenter.model = model
    segmenter.processor = _ChainProcessor()
    segmenter.embedding_cache_size = 2
    segmenter._prepared = OrderedDict()
    segmenter.chain_cache_size = 8
    segmenter._chains = OrderedDict()
    segmenter.last_timings = {}
    return segmenter


IMAGE = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)


def test_clicks_are_decoded_one_at_a_time_on_top_of_the_previous_mask():
    model = _ChainModel()
    segmenter = _segmenter(model)

    segmenter.segment(IMAGE, [(4, 4)], [1], image_key="s", object_id=0)
    mask = segmenter.segment(IMAGE, [(4, 4), (10, 10)], [1, 1], image_key="s", object_id=0)

    assert [(c["n"], c["multimask"], c["prior"] is not None) for c in model.calls] == [
        (1, True, False),  # first click: ambiguous, ranked from three candidates
        (2, False, True),  # next click refines the previous mask logits
        (2, True, True),  # the mask grew, so the alternatives are checked too
    ]
    assert model.calls[1]["prior"].shape == (1, 1, SIZE, SIZE)
    assert mask[4, 4] and mask[10, 10] and not mask[0, 15]


def test_cached_chain_makes_repeats_and_undo_free():
    model = _ChainModel()
    segmenter = _segmenter(model)
    clicks = [(4, 4), (10, 10), (12, 4)]
    for n in range(1, 4):
        last = segmenter.segment(IMAGE, clicks[:n], [1] * n, image_key="s", object_id=0)
    decoded = len(model.calls)

    assert segmenter.segment(IMAGE, clicks, [1, 1, 1], image_key="s", object_id=0) is last
    undone = segmenter.segment(IMAGE, clicks[:2], [1, 1], image_key="s", object_id=0)
    assert len(model.calls) == decoded
    assert undone[10, 10] and not undone[4, 12]

    # Another object on the same slice keeps its own chain.
    segmenter.segment(IMAGE, [(4, 4)], [1], image_key="s", object_id=1)
    assert len(model.calls) == decoded + 1


def test_a_mask_that_breaks_a_click_falls_back_to_the_consistent_candidate():
    model = _ChainModel(sloppy=True)
    segmenter = _segmenter(model)
    segmenter.segment(IMAGE, [(4, 4)], [1], image_key="s")
    mask = segmenter.segment(IMAGE, [(4, 4), (5, 4)], [1, 0], image_key="s")

    assert [c["multimask"] for c in model.calls] == [True, False, True]
    assert mask[4, 3] and not mask[4, 5]


def test_a_click_obeying_flood_loses_to_a_candidate_that_keeps_the_mask():
    model = _ChainModel(flood=True)
    segmenter = _segmenter(model)
    segmenter.segment(IMAGE, [(4, 4)], [1], image_key="s")
    mask = segmenter.segment(IMAGE, [(4, 4), (6, 6)], [1, 1], image_key="s")

    assert mask[4, 4] and mask[6, 6]
    assert not mask[15, 15]


def test_refinement_starts_from_the_given_base_logits():
    model = _ChainModel()
    segmenter = _segmenter(model)
    base = MaskBase(key="tracked", logits=torch.full((SIZE, SIZE), 3.0))

    segmenter.segment(IMAGE, [(4, 4)], [0], image_key="s", base=base)

    assert model.calls[0]["multimask"] is False
    assert torch.equal(model.calls[0]["prior"][0, 0], base.logits)


def test_a_correction_click_only_changes_the_region_it_touches():
    model = _ChainModel()
    segmenter = _segmenter(model)
    tracked = np.zeros((SIZE, SIZE), dtype=bool)
    tracked[1:6, 1:6] = True
    tracked[10:15, 10:15] = True
    base = MaskBase(key="tracked", logits=torch.where(torch.from_numpy(tracked), 5.0, -5.0))

    # The fake decoder answers a lone negative click with an empty mask.
    mask = segmenter.segment(IMAGE, [(3, 3)], [0], image_key="s", base=base)

    assert not mask[1:6, 1:6].any()
    assert mask[10:15, 10:15].all()


def test_tidy_mask_keeps_the_clicked_object_and_drops_specks():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:25, 5:25] = True  # clicked object
    mask[10, 10] = False  # pinhole
    mask[14:18, 14:18] = False  # hole holding a negative click
    mask[30:32, 30:32] = True  # speck
    mask[28:40, 0:12] = True  # large second part, touches the border

    tidy = _tidy_mask(mask, [(6, 6), (15, 15)], [1, 0], max_hole_area=20)

    assert tidy[10, 10]
    assert not tidy[15, 15]
    assert not tidy[30:32, 30:32].any()
    assert tidy[35, 5]


def test_tidy_mask_leaves_masks_without_a_clicked_component_alone():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:4, 2:4] = True
    mask[10:12, 10:12] = True
    assert np.array_equal(_tidy_mask(mask, [(18, 18)], [1], max_hole_area=0), mask)


class _Store:
    """Minimal stand-in for the HF inference session's per-object outputs."""

    def __init__(self, n_objects=1):
        self.output_dict_per_obj = {
            i: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}} for i in range(n_objects)
        }
        self.obj_with_new_inputs = []

    def obj_id_to_idx(self, obj_id):
        return obj_id - 1


class _MaskVideoProcessor:
    def __init__(self):
        self.calls = []

    def video_processor(self, videos, **kwargs):
        return SimpleNamespace(pixel_values_videos=[torch.zeros((len(videos), 3, 4, 4))])

    def add_inputs_to_inference_session(self, session, frame_idx, obj_ids, **kwargs):
        self.calls.append((frame_idx, tuple(obj_ids), "input_masks" in kwargs))
        session.obj_with_new_inputs = list(obj_ids)

    def post_process_masks(self, masks, original_sizes, binarize):
        return [masks[0] > 0]


class _MaskVideoModel:
    """Stores conditioning outputs where HF would: tracked slices go to non-cond."""

    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.tracked = set()

    def _out(self, session, frame_idx):
        outputs = session.output_dict_per_obj[0]
        if frame_idx in outputs["cond_frame_outputs"] and not session.obj_with_new_inputs:
            entry = outputs["cond_frame_outputs"][frame_idx]
        else:
            entry = {"pred_masks": torch.full((1, 1, 2, 2), float(frame_idx + 1))}
            target = "non_cond_frame_outputs" if frame_idx in self.tracked else "cond_frame_outputs"
            outputs[target][frame_idx] = entry
        session.obj_with_new_inputs = []
        return SimpleNamespace(frame_idx=frame_idx, object_ids=[1], pred_masks=entry["pred_masks"])

    def __call__(self, inference_session, frame_idx):
        return self._out(inference_session, frame_idx)

    def propagate_in_video_iterator(self, session, start_frame_idx, reverse=False):
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, self.n_frames)
        for frame_idx in order:
            if frame_idx not in session.output_dict_per_obj[0]["cond_frame_outputs"]:
                self.tracked.add(frame_idx)
            yield self._out(session, frame_idx)


def _propagator(n_frames):
    propagator = Sam3VolumePropagator.__new__(Sam3VolumePropagator)
    propagator.device = "cpu"
    propagator.model = _MaskVideoModel(n_frames)
    propagator.processor = _MaskVideoProcessor()
    propagator.session_dtype = torch.float32
    propagator.last_timings = {}
    propagator.live = None
    propagator.family = "sam3"
    propagator._session_cls = lambda **kwargs: _Store()
    return propagator


def test_propagation_conditions_seeded_slices_on_the_preview_mask():
    propagator = _propagator(3)
    seed = np.ones((2, 2), dtype=bool)

    propagator.propagate(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)],
        anchor_idx=1,
        points_per_object=[[(0, 0)]],
        labels_per_object=[[1]],
        keep_live=True,
        seed_masks_per_object=[{1: seed}],
    )

    assert propagator.processor.calls == [(1, (1,), True)]
    assert (0, 1) in propagator.live.prompted
    assert propagator.refine_base(propagator.live, 0, 1) is None


def test_negative_only_slices_refine_the_tracked_mask_and_stay_pinned():
    propagator = _propagator(4)
    seen = []

    def refine_mask(position, frame, points, labels, base):
        seen.append((position, frame, labels, base))
        return np.zeros((2, 2), dtype=bool)

    masks = propagator.propagate(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)],
        anchor_idx=0,
        points_per_object=[[(0, 0), (1, 1)]],
        labels_per_object=[[1, 0]],
        frame_indices_per_object=[[0, 2]],
        keep_live=True,
        seed_masks_per_object=[{0: np.ones((2, 2), dtype=bool)}],
        refine_mask=refine_mask,
    )

    (position, frame, labels, base), = seen
    assert (position, frame, labels) == (0, 2, [0])
    assert torch.equal(base.logits, torch.full((2, 2), 3.0))  # the tracked mask on slice 2
    session = propagator.live.sessions[0]
    assert 2 in session.output_dict_per_obj[0]["cond_frame_outputs"]
    assert 2 not in session.output_dict_per_obj[0]["non_cond_frame_outputs"]
    assert propagator.live.bases[(0, 2)] is base
    assert masks.shape == (1, 4, 2, 2)


def test_pinned_corrections_become_conditioning_frames():
    propagator = _propagator(3)
    live = LiveTracker(1, 3, 2, 2)
    session = _Store()
    live.attach(session, {0: 1})
    propagator.model.tracked.add(2)
    outputs = session.output_dict_per_obj[0]
    for frame in (1, 2, 9, 10):
        outputs["non_cond_frame_outputs"][frame] = {"pred_masks": torch.full((1, 1, 2, 2), -1.0)}

    base = propagator.refine_base(live, 0, 2)
    frame_masks = propagator.pin_mask(live, 0, 2, np.ones((2, 2), dtype=bool))

    assert torch.equal(base.logits, torch.full((2, 2), -1.0))
    assert propagator.refine_base(live, 0, 2) is base  # captured once, before the first edit
    assert 2 in outputs["cond_frame_outputs"]
    # Stale tracked memories within num_maskmem slices of the correction are dropped.
    assert sorted(outputs["non_cond_frame_outputs"]) == [10]
    assert live.dirty_frames == {2}
    assert frame_masks.shape == (1, 2, 2) and frame_masks.all()

    live.masks[0, 1] = True
    shown = propagator.refine_base(live, 0, 1)
    assert torch.equal(shown.logits, torch.full((2, 2), 10.0))


class _Decoder:
    def forward(self, multimask_output):
        masks = torch.full((1, 1, 3, 4, 4), -1.0)
        masks[0, 0, 0] = 1.0  # most of the section
        masks[0, 0, 1, :2, :2] = 1.0  # the layer tracked so far
        return masks, torch.tensor([[[0.9, 0.5, 0.1]]]), None, torch.ones((1, 1))


class _Tracker:
    def __init__(self):
        self.mask_decoder = _Decoder()

    def _run_single_frame_inference(self, **kwargs):
        return self.mask_decoder.forward(multimask_output=True)


def test_tracked_slices_keep_the_candidate_matching_the_previous_slice():
    tracker = _Tracker()
    _install_temporal_mask_selection(tracker)
    session = _Store()
    previous = torch.full((1, 1, 4, 4), -1.0)
    previous[..., :2, :2] = 1.0
    session.output_dict_per_obj[0]["non_cond_frame_outputs"][2] = {"pred_masks": previous}
    common = dict(inference_session=session, obj_idx=0, frame_idx=3, reverse=False, mask_inputs=None)

    _, tracked_scores, _, _ = tracker._run_single_frame_inference(point_inputs=None, **common)
    _, prompted_scores, _, _ = tracker._run_single_frame_inference(point_inputs={"clicks": 1}, **common)

    assert int(tracked_scores.argmax()) == 1
    assert int(prompted_scores.argmax()) == 0


def test_keep_as_conditioning_moves_only_existing_entries():
    session = _Store()
    entry = {"pred_masks": torch.zeros(1)}
    session.output_dict_per_obj[0]["non_cond_frame_outputs"][4] = entry

    _keep_as_conditioning(session, [1], 4)
    _keep_as_conditioning(session, [1], 5)

    assert session.output_dict_per_obj[0]["cond_frame_outputs"] == {4: entry}
    assert session.output_dict_per_obj[0]["non_cond_frame_outputs"] == {}
