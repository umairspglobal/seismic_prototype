from types import SimpleNamespace

import numpy as np
import torch

from seismic_app.inference import (
    Sam31PointSegmenter,
    Sam31VolumePropagator,
    _install_sam31_feature_provider,
    _install_sam31_temporal_selection,
)


def _outputs():
    return {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}


class FakeDecoderTracker:
    """SAM 3.1 tracker heads for click decoding at a 4x4 mask resolution."""

    image_size = 8
    low_res_mask_size = 4

    def __init__(self):
        self.calls = []

    def _forward_sam_heads(self, **kwargs):
        self.calls.append(kwargs)
        full = torch.full((4, 4), 10.0)
        local = torch.full((4, 4), -10.0)
        local[0:2, 1:3] = 10.0
        empty = torch.full((4, 4), -10.0)
        if not kwargs["multimask_output"]:
            # Refines the prior: drops the column holding the newest negative click.
            refined = kwargs["mask_inputs"][0, 0].clone()
            x, _y = kwargs["point_inputs"]["point_coords"][0, -1].tolist()
            refined[:, int(x * 4 / 8)] = -10.0
            return {"low_res_multimasks": refined[None, None], "ious": torch.tensor([[0.9]])}
        return {
            "low_res_multimasks": torch.stack([full, local, empty])[None],
            "ious": torch.tensor([[0.99, 0.5, 0.1]]),
        }


def _segmenter(tracker):
    segmenter = Sam31PointSegmenter.__new__(Sam31PointSegmenter)
    segmenter.device = "cpu"
    segmenter.encoder = SimpleNamespace(tracker=tracker, size=tracker.image_size)
    segmenter._init_caches(4)
    return segmenter


def test_sam31_clicks_decode_one_at_a_time_on_stretched_coordinates():
    tracker = FakeDecoderTracker()
    segmenter = _segmenter(tracker)
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    segmenter._prepared["slice"] = {"pix": None, "high_res": None, "shape": (16, 32)}

    first = segmenter.segment(image, [(16, 4)], [1], image_key="slice")
    # The click-consistent local candidate wins over the frame-filling one.
    assert first[4, 16] and not first[12, 4]
    assert tracker.calls[0]["mask_inputs"] is None
    assert tracker.calls[0]["multimask_output"]
    assert tracker.calls[0]["point_inputs"]["point_coords"].tolist() == [[[4.0, 2.0]]]

    second = segmenter.segment(image, [(16, 4), (10, 2)], [1, 0], image_key="slice")
    refine = tracker.calls[1]
    assert refine["point_inputs"]["point_coords"].tolist() == [[[4.0, 2.0], [2.5, 1.0]]]
    prior = refine["mask_inputs"][0, 0]
    assert prior[0, 1] == 10.0 and prior[3, 0] == -10.0
    assert second[4, 16] and not second[2, 10]
    decoded = len(tracker.calls)
    again = segmenter.segment(image, [(16, 4), (10, 2)], [1, 0], image_key="slice")
    assert len(tracker.calls) == decoded
    assert (again == second).all()


class FakeVideoTracker:
    """SAM 3.1 tracker state API: each slice gets the nearest conditioning mask."""

    low_res_mask_size = 4

    def __init__(self):
        self.object_score_logit_threshold = 0.0
        self.states = []
        self.added = []
        self.sweeps = []

    def init_state(self, video_height, video_width, num_frames, cached_features, offload_state_to_cpu):
        state = {
            "num_frames": num_frames,
            "cached_features": cached_features,
            "multiplex_state": None,
            "output_dict": _outputs(),
            "output_dict_per_obj": {0: _outputs()},
            "masks": {},
        }
        self.states.append(state)
        return state

    def add_new_masks(self, state, frame_idx, obj_ids, masks):
        self.added.append((len(self.states) - 1, frame_idx, state["multiplex_state"]))
        state["masks"][frame_idx] = masks[0].numpy()
        state["multiplex_state"] = "multiplex"

    def propagate_in_video_preflight(self, state, run_mem_encoder):
        for frame_idx in state["masks"]:
            state["output_dict"]["cond_frame_outputs"][frame_idx] = {"pred_masks": torch.zeros(1, 1, 4, 4)}

    def propagate_in_video(self, state, start, max_frames, reverse, tqdm_disable):
        self.sweeps.append(([id(s) for s in self.states].index(id(state)), start, reverse))
        frames = range(start, -1, -1) if reverse else range(start, state["num_frames"])
        for frame_idx in frames:
            nearest = min(state["masks"], key=lambda f: abs(f - frame_idx))
            mask = torch.from_numpy(state["masks"][nearest])
            if frame_idx not in state["masks"]:
                state["output_dict"]["non_cond_frame_outputs"][frame_idx] = {
                    "pred_masks": torch.zeros(1, 1, 4, 4)
                }
            video_res = torch.where(mask, 1.0, -1.0)[None, None]
            yield frame_idx, [1], None, video_res, torch.tensor([[5.0]])


class FakeVolumeSegmenter:
    def __init__(self, tracker):
        self.encoder = SimpleNamespace(tracker=tracker, encode=lambda rgb: (None, {}))
        self.calls = []

    def segment(self, rgb, points, labels, image_key=None, object_id=0, base=None):
        self.calls.append((points, labels, image_key, object_id, base))
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[1, 1] = True
        return mask


def _propagator(tracker):
    segmenter = FakeVolumeSegmenter(tracker)
    propagator = Sam31VolumePropagator(SimpleNamespace(model=None), segmenter=segmenter)
    return propagator, segmenter


def _seed(shape, row):
    mask = np.zeros(shape, dtype=bool)
    mask[row] = True
    return mask


def test_sam31_tracks_from_preview_masks_without_presence_gate():
    tracker = FakeVideoTracker()
    propagator, segmenter = _propagator(tracker)
    frames = np.zeros((5, 4, 6, 3), dtype=np.uint8)
    seed = _seed((4, 6), 2)
    updates = []

    masks = propagator.propagate(
        frames,
        anchor_idx=2,
        points_per_object=[[(1, 2)], [(1, 2)]],
        labels_per_object=[[1], [1]],
        seed_masks_per_object=[{2: seed}, {2: seed}],
        progress=lambda done, total, frame, _masks: updates.append(frame),
        keep_live=True,
    )

    assert tracker.object_score_logit_threshold == Sam31VolumePropagator.presence_threshold
    assert segmenter.calls == []
    # One state per object, both seeded with the preview mask: objects may overlap.
    assert len(tracker.states) == 2
    assert [(f, mux) for _, f, mux in tracker.added] == [(2, None), (2, None)]
    assert masks.shape == (2, 5, 4, 6)
    assert all((masks[g, f] == seed).all() for g in range(2) for f in range(5))
    assert sorted(set(updates)) == [0, 1, 2, 3, 4]
    assert propagator.live.conditioning[0] == {2: seed}


def test_sam31_segments_a_clicked_slice_without_a_preview_mask():
    tracker = FakeVideoTracker()
    propagator, segmenter = _propagator(tracker)
    frames = np.zeros((3, 4, 6, 3), dtype=np.uint8)

    masks = propagator.propagate(frames, anchor_idx=1, points_per_object=[[(1, 1)]], labels_per_object=[[1]])

    assert len(segmenter.calls) == 1
    points, labels, _key, object_id, base = segmenter.calls[0]
    assert (points, labels, object_id, base) == ([(1, 1)], [1], 0, None)
    assert masks[0, :, 1, 1].all() and masks[0].sum() == 3


def test_sam31_resweep_rebuilds_edited_object_from_all_its_masks():
    tracker = FakeVideoTracker()
    propagator, _segmenter = _propagator(tracker)
    frames = np.zeros((6, 4, 6, 3), dtype=np.uint8)
    seed = _seed((4, 6), 0)
    propagator.propagate(
        frames,
        anchor_idx=1,
        points_per_object=[[(1, 0)], [(1, 0)]],
        labels_per_object=[[1], [1]],
        seed_masks_per_object=[{1: seed}, {1: seed}],
        keep_live=True,
    )
    live = propagator.live
    edit = _seed((4, 6), 3)

    shown = propagator.pin_mask(live, 1, 4, edit)
    assert (shown[1] == edit).all()
    tracker.added.clear()
    tracker.sweeps.clear()
    masks = propagator.resweep(live)

    # Only object 2 is rebuilt, from its preview mask and the pinned edit,
    # each added on the plain mask-prompt path.
    assert [(f, mux) for _, f, mux in tracker.added] == [(1, None), (4, None)]
    rebuilt = len(tracker.states) - 1
    assert tracker.sweeps == [(rebuilt, 4, False), (rebuilt, 4, True)]
    assert (masks[1, 4] == edit).all() and (masks[1, 5] == edit).all()
    assert (masks[1, 1] == seed).all()
    assert (masks[0, 5] == seed).all()
    assert not live.dirty_frames and not live.dirty_objects


def test_sam31_chunk_continues_from_masks_of_present_objects():
    tracker = FakeVideoTracker()
    propagator, _segmenter = _propagator(tracker)
    frames = np.zeros((4, 4, 6, 3), dtype=np.uint8)
    seeds = np.zeros((2, 4, 6), dtype=bool)
    seeds[0, 2] = True
    got = {}

    propagator.propagate_from_masks(frames, 0, seeds, False, lambda f, m: got.__setitem__(f, m.copy()))

    assert len(tracker.states) == 1
    assert sorted(got) == [0, 1, 2, 3]
    assert all((got[f][0] == seeds[0]).all() and not got[f][1].any() for f in got)


class IdentityMultiplex:
    def demux(self, x):
        return x

    def mux(self, x):
        return x


def test_sam31_tracking_prefers_candidate_matching_previous_slice():
    candidates = torch.full((1, 3, 4, 4), -10.0)
    candidates[0, 0] = 10.0  # the IoU head's favourite: the whole slice
    candidates[0, 1, 1, :] = 10.0  # the event tracked on the previous slice
    decoder = SimpleNamespace(
        forward=lambda **kw: {"masks": candidates, "iou_pred": torch.tensor([[0.9, 0.2, 0.1]])}
    )
    tracker = SimpleNamespace(sam_mask_decoder=decoder)
    tracker._run_single_frame_inference = lambda **kw: tracker.sam_mask_decoder.forward(multimask_output=True)
    _install_sam31_temporal_selection(tracker)
    previous = torch.full((1, 1, 4, 4), -10.0)
    previous[0, 0, 1, :] = 10.0
    output_dict = _outputs()
    output_dict["non_cond_frame_outputs"][4] = {"pred_masks": previous}

    out = tracker._run_single_frame_inference(
        frame_idx=5,
        reverse=False,
        output_dict=output_dict,
        inference_state={"multiplex_state": IdentityMultiplex()},
        point_inputs=None,
        mask_inputs=None,
    )
    assert int(out["iou_pred"].argmax()) == 1

    clicked = tracker._run_single_frame_inference(
        frame_idx=5,
        reverse=False,
        output_dict=output_dict,
        inference_state={"multiplex_state": IdentityMultiplex()},
        point_inputs={"point_coords": None},
        mask_inputs=None,
    )
    assert int(clicked["iou_pred"].argmax()) == 0


def test_sam31_states_fetch_missing_slice_features_from_provider():
    tracker = SimpleNamespace(
        _get_image_feature=lambda state, frame_idx, batch_size: state["cached_features"][frame_idx]
    )
    _install_sam31_feature_provider(tracker)
    cache = {}
    state = {"cached_features": cache, "feature_provider": lambda f: cache.__setitem__(f, f"slice-{f}")}

    assert tracker._get_image_feature(state, 3, 1) == "slice-3"
    assert tracker._get_image_feature(state, 3, 1) == "slice-3"
    assert list(cache) == [3]
