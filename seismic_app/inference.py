"""Step 5.1/5.2 of the pipeline: SAM 3 zero-shot inference engine.

Loads facebook/sam3 once, then applies the fixed SEISMIC_PROMPTS to every
tile in a single batched forward pass (each tile is repeated once per
prompt, since Sam3Processor batches one (image, text) pair per item).

No fine-tuning is performed here - SAM 3's open-vocabulary text encoder is
used directly ("zero-shot") against the five seismic noun phrases. This
will work best if the checkpoint has been fine-tuned on seismic data per
section 4 of the guide; out of the box it still runs and produces masks,
but quality against natural-image-trained weights will vary per concept.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from contextlib import nullcontext
import gc
import importlib.util
import math
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .logutil import get_logger

log = get_logger("inference")

# Transformers' lazy module loader is not thread-safe. FastAPI runs sync
# endpoints in a thread pool, so two concurrent imports of Sam3Tracker*
# can raise ImportError even when the classes exist. Serialize them.
_tf_import_lock = threading.Lock()
_tf_point_classes: tuple[type, type] | None = None
_tf_video_classes: tuple[type, type] | None = None
_tf_version: str | None = None


def transformers_version() -> str | None:
    """Import transformers under the shared lock and return its version."""
    global _tf_version
    with _tf_import_lock:
        if _tf_version is None:
            import transformers

            _tf_version = transformers.__version__
        return _tf_version


def _point_tracker_classes() -> tuple[type, type]:
    global _tf_point_classes
    with _tf_import_lock:
        if _tf_point_classes is None:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            _tf_point_classes = (Sam3TrackerModel, Sam3TrackerProcessor)
        return _tf_point_classes


def _video_tracker_classes() -> tuple[type, type]:
    global _tf_video_classes
    with _tf_import_lock:
        if _tf_video_classes is None:
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

            _tf_video_classes = (Sam3TrackerVideoModel, Sam3TrackerVideoProcessor)
        return _tf_video_classes


class Sam3SeismicSegmenter:
    """Wraps transformers' Sam3Model/Sam3Processor for the 5 fixed prompts."""

    def __init__(
        self,
        checkpoint: str = config.DEFAULT_CHECKPOINT,
        device: str | None = None,
        prompts: list[str] | None = None,
    ):
        from transformers import Sam3Model, Sam3Processor  # deferred: heavy import

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has no CUDA-enabled "
                "PyTorch build. Install a CUDA PyTorch wheel in the active environment "
                "or select CPU."
            )
        if (
            compile_model
            and hasattr(torch, "compile")
            and importlib.util.find_spec("triton") is None
        ):
            raise RuntimeError(
                "torch.compile requires Triton, which is unavailable in this "
                "environment; run without compile_model."
            )
        self.prompts = prompts or config.SEISMIC_PROMPTS

        log.info(
            "Loading Sam3Model from '%s' onto device=%s (first run may download "
            "weights from Hugging Face - this can take several minutes)...",
            checkpoint,
            self.device,
        )
        t0 = time.perf_counter()
        self.model = Sam3Model.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        log.info("Sam3Model weights loaded in %.1fs; loading processor...", time.perf_counter() - t0)
        self.processor = Sam3Processor.from_pretrained(checkpoint)
        log.info(
            "Text-prompt segmenter ready on %s (prompts=%s)",
            self.device,
            self.prompts,
        )

    @torch.no_grad()
    def segment_tile(self, tile_image: Image.Image) -> dict[str, np.ndarray]:
        """Run all fixed prompts against one tile in a single forward pass.

        Returns
        -------
        dict mapping noun_phrase -> float32 probability map of shape
        (tile_h, tile_w), resized back to the tile's own pixel size.
        """
        w, h = tile_image.size
        images = [tile_image] * len(self.prompts)

        inputs = self.processor(images=images, text=self.prompts, return_tensors="pt").to(
            self.device
        )

        outputs = self.model(**inputs)

        # semantic_seg: (num_prompts, 1, h_model, w_model) logits - one
        # channel per (image, text) pair in the batch, i.e. per prompt here.
        logits = outputs.semantic_seg
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)[:, 0].float().cpu().numpy()  # (num_prompts, h, w)

        return {prompt: probs[i] for i, prompt in enumerate(self.prompts)}


class Sam3PointSegmenter:
    """Interactive point-prompted segmentation (SAM3 Tracker / PVS head).

    This is the SAM2-style promptable-visual-segmentation interface of
    SAM 3: the user clicks positive/negative points on the section and
    the model segments the one object they indicated. The *whole*
    section image is passed in one go (seismic lines are small compared
    to SAM's 1024 input; the processor resizes internally), so click
    coordinates are plain full-resolution array indices - no tile
    bookkeeping required.
    """

    def __init__(
        self,
        checkpoint: str = config.DEFAULT_CHECKPOINT,
        device: str | None = None,
        embedding_cache_size: int = 2,
    ):
        Sam3TrackerModel, Sam3TrackerProcessor = _point_tracker_classes()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has no CUDA-enabled "
                "PyTorch build. Install a CUDA PyTorch wheel in the active environment "
                "or select CPU."
            )
        log.info(
            "Loading Sam3TrackerModel from '%s' onto device=%s (first run may "
            "download weights - can take several minutes on CPU)...",
            checkpoint,
            self.device,
        )
        t0 = time.perf_counter()
        self.model = Sam3TrackerModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        log.info(
            "Sam3TrackerModel weights loaded in %.1fs; loading processor...",
            time.perf_counter() - t0,
        )
        self.processor = Sam3TrackerProcessor.from_pretrained(checkpoint)
        log.info("Point-prompt tracker ready on %s", self.device)

        # Keep only a few sections resident: embeddings are large GPU tensors.
        self.embedding_cache_size = max(1, int(embedding_cache_size))
        self._prepared: OrderedDict[Hashable, dict] = OrderedDict()
        self.last_timings: dict[str, float] = {}

    def _sync_cuda(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _autocast(self):
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def prepare_image(
        self,
        rgb: np.ndarray,
        image_key: Hashable | None = None,
    ) -> Hashable:
        """Preprocess and encode a section before the first point is clicked."""
        key = image_key if image_key is not None else ("array", id(rgb), rgb.shape)
        if key in self._prepared:
            self._prepared.move_to_end(key)
            self.last_timings["prepare_image"] = 0.0
            return key

        log.info("Preparing image features for %r...", key)
        self._sync_cuda()
        started = time.perf_counter()
        inputs = self.processor(
            images=Image.fromarray(rgb),
            return_tensors="pt",
        ).to(self.device)
        with self._autocast():
            embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        self._sync_cuda()
        elapsed = time.perf_counter() - started
        self._prepared[key] = {
            "embeddings": embeddings,
            "original_sizes": inputs["original_sizes"].detach().cpu(),
            "shape": rgb.shape[:2],
        }
        self._prepared.move_to_end(key)
        while len(self._prepared) > self.embedding_cache_size:
            self._prepared.popitem(last=False)
        self.last_timings["prepare_image"] = elapsed
        log.info("Image features ready in %.3fs (%d cached)", elapsed, len(self._prepared))
        return key

    @torch.no_grad()
    def segment(
        self,
        rgb: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
        image_key: Hashable | None = None,
    ) -> np.ndarray:
        """Segment one object indicated by clicked points.

        Parameters
        ----------
        rgb : (H, W, 3) uint8 full-resolution section image, time down.
        points : (trace_idx, sample_idx) array-index pairs, i.e. (x, y)
            pixel coordinates on the section image.
        labels : 1 for positive (inside the object), 0 for negative.

        Returns
        -------
        (H, W) boolean mask at full section resolution.
        """
        if len(points) != len(labels) or not points:
            raise ValueError("points and labels must be equal-length and non-empty")

        h, w = rgb.shape[:2]
        n_pos = sum(1 for lab in labels if lab == 1)
        n_neg = len(labels) - n_pos
        log.info(
            "Point segmentation: image %dx%d, %d positive / %d negative points, device=%s",
            w,
            h,
            n_pos,
            n_neg,
            self.device,
        )
        t0 = time.perf_counter()

        key = self.prepare_image(rgb, image_key=image_key)
        prepared = self._prepared[key]
        # 4D: (image, object, point, xy) - one image, one object.
        input_points = [[[[float(x), float(y)] for x, y in points]]]
        input_labels = [[[int(l) for l in labels]]]

        self._sync_cuda()
        prompt_started = time.perf_counter()
        inputs = self.processor(
            input_points=input_points,
            input_labels=input_labels,
            original_sizes=prepared["original_sizes"],
            return_tensors="pt",
        ).to(self.device)

        model_inputs = dict(inputs)
        model_inputs["image_embeddings"] = prepared["embeddings"]

        log.info("Running prompt encoder + mask decoder...")
        with self._autocast():
            outputs = self.model(**model_inputs)
        self._sync_cuda()
        decoder_elapsed = time.perf_counter() - prompt_started

        # post_process_masks -> list per image of (n_objects, n_masks, H, W).
        post_started = time.perf_counter()
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        iou = outputs.iou_scores.float().cpu().numpy().reshape(-1)
        best = int(iou.argmax())
        mask = np.asarray(masks[0, best], dtype=bool)
        post_elapsed = time.perf_counter() - post_started
        self.last_timings.update(
            prompt_decode=decoder_elapsed,
            post_process=post_elapsed,
            total=time.perf_counter() - t0,
        )
        coverage = 100.0 * float(mask.mean())
        log.info(
            "Point segmentation done in %.1fs (best IoU=%.3f, mask coverage=%.2f%%)",
            time.perf_counter() - t0,
            float(iou[best]),
            coverage,
        )
        return mask


class Sam3VolumePropagator:
    """Propagate a point-picked object through a 3D volume, SAM2-video style.

    This is the seismic equivalent of SAM 2's video segmentation: the
    slices of a 3D volume along one axis (inlines, crosslines, or time
    slices) are treated as consecutive video frames. You click points on
    one slice; the memory-based tracker then follows the object through
    every other slice in both directions. No conversion of the .sgy to an
    actual video file is needed - the frames are fed in as arrays.

    Note this only makes sense for 3D volumes. A 2D line is a single
    frame: its image already contains the full time axis, so there is
    nothing to propagate through.
    """

    def __init__(
        self,
        checkpoint: str = config.DEFAULT_CHECKPOINT,
        device: str | None = None,
        use_bfloat16: bool = True,
        compile_model: bool = False,
    ):
        Sam3TrackerVideoModel, Sam3TrackerVideoProcessor = _video_tracker_classes()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has no CUDA-enabled "
                "PyTorch build. Install a CUDA PyTorch wheel in the active environment "
                "or select CPU."
            )
        log.info(
            "Loading Sam3TrackerVideoModel from '%s' onto device=%s...",
            checkpoint,
            self.device,
        )
        t0 = time.perf_counter()
        self.model = Sam3TrackerVideoModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        if compile_model and hasattr(torch, "compile"):
            log.info("Compiling SAM 3 video tracker (first propagation will warm up)...")
            self.model = torch.compile(self.model)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(checkpoint)
        self.session_dtype = (
            torch.bfloat16
            if use_bfloat16
            and self.device.startswith("cuda")
            and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.last_timings: dict[str, float] = {}
        log.info(
            "Volume propagator ready on %s (loaded in %.1fs)",
            self.device,
            time.perf_counter() - t0,
        )

    def _autocast(self):
        if self.session_dtype == torch.bfloat16 and self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _free_cuda(self) -> None:
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        text = str(exc).lower()
        return "out of memory" in text or "cuda oom" in text

    def _estimate_max_objects(self) -> int:
        """How many objects can share one tracker session on this GPU.

        Decoder work is per-object, but the memory encoder batches every
        object on the current slice. That batch is the VRAM ceiling.
        Video frames and mask-memory banks are stored on CPU so they do
        not count against this budget.
        """
        if not self.device.startswith("cuda"):
            return 10**9
        free, _total = torch.cuda.mem_get_info()
        # Keep ~20% free for the vision encoder + scratch; never claim more
        # than is actually available.
        headroom = max(int(0.20 * free), 256 * 1024 * 1024)
        usable = int(free) - headroom
        if usable <= 0:
            return 1
        per_object = 48 * 1024 * 1024  # batched high-res masks + mem-enc activations
        return max(1, usable // per_object)

    def _new_session(self, pixel_values: torch.Tensor, height: int, width: int):
        """Build a tracker session with CPU-side video + memory banks."""
        with _tf_import_lock:
            from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
                Sam3TrackerVideoInferenceSession,
            )

        state_device = "cpu" if self.device.startswith("cuda") else self.device
        return Sam3TrackerVideoInferenceSession(
            video=pixel_values,
            video_height=height,
            video_width=width,
            inference_device=self.device,
            inference_state_device=state_device,
            video_storage_device=state_device,
            dtype=self.session_dtype,
            # Features live on CPU; a handful of cached slices avoids
            # re-encoding the same frame on the reverse sweep.
            max_vision_features_cache_size=8,
        )

    def _propagate_wave(
        self,
        pixel_values: torch.Tensor,
        height: int,
        width: int,
        points_per_object: Sequence[Sequence[tuple[int, int]]],
        labels_per_object: Sequence[Sequence[int]],
        frame_indices_per_object: Sequence[Sequence[int]],
        object_indices: Sequence[int],
        masks: np.ndarray,
        on_frame: Callable[[int, int], None],
    ) -> None:
        """Track one GPU-sized batch of objects through the volume.

        Points may live on several slices, mirroring SAM2's video
        refinement. Slices where an object has at least one positive
        point anchor the object (conditioning frames). Slices with only
        negative points for an object cannot stand alone - segmenting
        from negatives alone is meaningless - so they are applied AFTER
        the first sweep, when the tracker already has memory of the
        object on that slice; the negative click then subtracts from the
        remembered mask instead of erasing the object. The corrections
        are swept outward afterwards.
        """
        # frame -> [(wave_position, points_on_frame, labels_on_frame)]
        anchor_groups: dict[int, list[tuple[int, list, list]]] = {}
        refine_groups: dict[int, list[tuple[int, list, list]]] = {}
        for wave_pos, global_idx in enumerate(object_indices):
            per_frame: dict[int, tuple[list, list]] = {}
            for (x, y), label, frame_idx in zip(
                points_per_object[global_idx],
                labels_per_object[global_idx],
                frame_indices_per_object[global_idx],
            ):
                bucket = per_frame.setdefault(int(frame_idx), ([], []))
                bucket[0].append((float(x), float(y)))
                bucket[1].append(int(label))
            for frame_idx, (pts, labs) in per_frame.items():
                target = anchor_groups if 1 in labs else refine_groups
                target.setdefault(frame_idx, []).append((wave_pos, pts, labs))

        n_frames = masks.shape[1]
        visited: set[int] = set()
        done = 0

        def record(out) -> None:
            nonlocal done
            rows = self._to_masks(out.pred_masks, height, width)
            # Output rows follow session registration order; out.object_ids
            # maps each row back to the wave position we assigned (pos + 1).
            for row, obj_id in zip(rows, out.object_ids):
                masks[object_indices[int(obj_id) - 1], out.frame_idx] = row
            if out.frame_idx not in visited:
                visited.add(out.frame_idx)
                done += 1
                if done == n_frames or done % max(1, n_frames // 10) == 0:
                    log.info("  slice %d/%d done", done, n_frames)
            # Re-emits refresh the browser overlay after refinements.
            on_frame(out.frame_idx, done)

        def prompt_frame(frame_idx: int, group: list[tuple[int, list, list]]) -> None:
            self.processor.add_inputs_to_inference_session(
                session,
                frame_idx=frame_idx,
                obj_ids=[wave_pos + 1 for wave_pos, _, _ in group],
                input_points=[[[[x, y] for x, y in pts] for _, pts, _ in group]],
                input_labels=[[labs for _, _, labs in group]],
                original_size=(height, width),
            )
            record(self.model(inference_session=session, frame_idx=frame_idx))

        session = self._new_session(pixel_values, height, width)
        try:
            with self._autocast():
                # Pass 1: condition every anchored slice, then sweep both ways.
                for frame_idx in sorted(anchor_groups):
                    prompt_frame(frame_idx, anchor_groups[frame_idx])
                log.info(
                    "Conditioned %d anchored slice(s); propagating...",
                    len(anchor_groups),
                )
                start_frame = min(anchor_groups)
                for reverse in (False, True):
                    for out in self.model.propagate_in_video_iterator(
                        session, start_frame_idx=start_frame, reverse=reverse
                    ):
                        record(out)
                    if not reverse:
                        log.info("Forward sweep complete; propagating backward...")

                # Pass 2: negative-only refinements. Every slice is tracked
                # now, so these run as memory-based refinements (subtractive),
                # not as fresh "the object is absent here" conditioning.
                if refine_groups:
                    log.info(
                        "Applying negative-only refinements on slice(s) %s and "
                        "re-sweeping...",
                        sorted(refine_groups),
                    )
                    for frame_idx in sorted(refine_groups):
                        prompt_frame(frame_idx, refine_groups[frame_idx])
                    for reverse, start in (
                        (False, min(refine_groups)),
                        (True, max(refine_groups)),
                    ):
                        for out in self.model.propagate_in_video_iterator(
                            session, start_frame_idx=start, reverse=reverse
                        ):
                            record(out)
        finally:
            del session
            self._free_cuda()

    @torch.no_grad()
    def propagate(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        anchor_idx: int,
        points_per_object: Sequence[Sequence[tuple[int, int]]],
        labels_per_object: Sequence[Sequence[int]],
        frame_indices_per_object: Sequence[Sequence[int]] | None = None,
        progress: Callable[[int, int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        """Track one or more objects through a stack of slices.

        Objects share a tracker session up to the GPU's VRAM budget.
        Video frames and per-object memory banks live on CPU; only the
        active decode runs on the GPU. Any objects that do not fit are
        queued and tracked in later waves (the vision encoder is reused
        per wave, not per object).

        Parameters
        ----------
        frames : list of (H, W, 3) uint8 slice images along the chosen
            volume axis, in order.
        anchor_idx : default slice for points without an explicit slice.
        points_per_object : one list of (col, row) pixel pairs per object.
        labels_per_object : matching lists of 1 = positive / 0 = negative.
        frame_indices_per_object : per-point slice index, aligned with
            points_per_object. Defaults to anchor_idx for every point.
            Points on several slices act as SAM2-style refinement clicks:
            each prompted slice becomes a conditioning frame.
        progress : called as (done, total, frame_idx, frame_masks) where
            frame_masks is (n_objects, H, W) bool for every object so far.

        Returns
        -------
        (n_objects, n_frames, H, W) boolean mask stack.
        """
        if not points_per_object or len(points_per_object) != len(labels_per_object):
            raise ValueError("need at least one object with matching points/labels")
        if frame_indices_per_object is None:
            frame_indices_per_object = [
                [anchor_idx] * len(pts) for pts in points_per_object
            ]
        if len(frame_indices_per_object) != len(points_per_object):
            raise ValueError("frame_indices_per_object must match points_per_object")
        n_frames = len(frames)
        for pts, labs, frs in zip(
            points_per_object, labels_per_object, frame_indices_per_object
        ):
            if len(pts) != len(labs) or len(pts) != len(frs) or not pts:
                raise ValueError(
                    "each object needs equal-length, non-empty points/labels/slices"
                )
            if any(not 0 <= int(f) < n_frames for f in frs):
                raise ValueError("point slice index out of range")
            if 1 not in labs:
                raise ValueError(
                    "each object needs at least one positive (+) point; "
                    "negative-only prompts cannot define an object"
                )

        n_objects = len(points_per_object)
        h, w = frames[0].shape[:2]
        batch_limit = min(n_objects, self._estimate_max_objects())
        prompted_slices = sorted(
            {int(f) for frs in frame_indices_per_object for f in frs}
        )
        log.info(
            "Volume propagation: %d slices of %dx%d, prompts on slice(s) %s, "
            "%d object(s) with %s point(s). GPU budget %d object(s)/wave "
            "(%d wave(s) queued).",
            n_frames,
            w,
            h,
            prompted_slices,
            n_objects,
            [len(p) for p in points_per_object],
            batch_limit,
            math.ceil(n_objects / batch_limit),
        )
        t0 = time.perf_counter()

        session_started = time.perf_counter()
        processed = self.processor.video_processor(
            videos=frames, device="cpu", return_tensors="pt"
        )
        pixel_values = processed.pixel_values_videos[0]
        self.last_timings["session_init"] = time.perf_counter() - session_started
        del frames  # the uint8 slices are no longer needed; free them early

        masks = np.zeros((n_objects, n_frames, h, w), dtype=bool)
        pending = list(range(n_objects))
        waves_done = 0
        wave_sizes: list[int] = []
        estimated_waves = max(1, math.ceil(n_objects / batch_limit))

        def on_frame(frame_idx: int, done_in_wave: int) -> None:
            if progress is None:
                return
            total = estimated_waves * n_frames
            done = waves_done * n_frames + done_in_wave
            progress(done, total, frame_idx, masks[:, frame_idx])

        while pending:
            wave = pending[:batch_limit]
            queued = pending[len(wave) :]
            log.info(
                "Tracker wave %d: %d object(s) on GPU, %d queued",
                waves_done + 1,
                len(wave),
                len(queued),
            )
            try:
                self._propagate_wave(
                    pixel_values,
                    h,
                    w,
                    points_per_object,
                    labels_per_object,
                    frame_indices_per_object,
                    wave,
                    masks,
                    on_frame,
                )
            except Exception as exc:
                if not self._is_cuda_oom(exc) or len(wave) == 1:
                    raise
                log.warning(
                    "GPU full with %d objects in one session (%s); "
                    "halving the wave and queueing the rest",
                    len(wave),
                    exc,
                )
                self._free_cuda()
                batch_limit = max(1, len(wave) // 2)
                estimated_waves = waves_done + math.ceil(len(pending) / batch_limit)
                continue
            pending = queued
            wave_sizes.append(len(wave))
            waves_done += 1
            estimated_waves = waves_done + (
                math.ceil(len(pending) / batch_limit) if pending else 0
            )

        total_elapsed = time.perf_counter() - t0
        self.last_timings.update(
            inference=total_elapsed - self.last_timings["session_init"],
            total=total_elapsed,
            fps=n_frames / total_elapsed if total_elapsed else float("inf"),
            object_waves=float(waves_done),
            objects_per_wave=float(max(wave_sizes) if wave_sizes else 0),
        )
        log.info(
            "Volume propagation finished in %.1fs (%d slices, %d object(s) "
            "in %d wave(s), total coverage %.2f%%)",
            total_elapsed,
            n_frames,
            n_objects,
            waves_done,
            100.0 * float(masks.any(axis=0).mean()),
        )
        return masks

    def _to_masks(self, pred_masks: torch.Tensor, h: int, w: int) -> np.ndarray:
        """Threshold + resize one frame's predicted logits to (n_rows, H, W).

        Row order matches the session's object registration order; callers
        map rows back to objects via the output's object_ids.
        """
        video_masks = self.processor.post_process_masks(
            [pred_masks], original_sizes=[(h, w)], binarize=True
        )[0]
        # (n_rows, 1, H, W) -> drop the per-object mask channel.
        return np.asarray(video_masks[:, 0].detach().cpu().numpy(), dtype=bool)
