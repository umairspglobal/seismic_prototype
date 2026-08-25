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
import importlib.util
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .logutil import get_logger

log = get_logger("inference")


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
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # deferred

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
        from transformers import (  # deferred: heavy import
            Sam3TrackerVideoModel,
            Sam3TrackerVideoProcessor,
        )

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

    @torch.no_grad()
    def propagate(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        anchor_idx: int,
        points_per_object: Sequence[Sequence[tuple[int, int]]],
        labels_per_object: Sequence[Sequence[int]],
        progress: Callable[[int, int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        """Track one or more objects through a stack of slices.

        All objects share a single video session, so the expensive
        per-slice vision encoding runs once regardless of object count -
        adding objects costs only the (cheap) per-object mask decoding.

        Parameters
        ----------
        frames : list of (H, W, 3) uint8 slice images along the chosen
            volume axis, in order.
        anchor_idx : index of the slice the points were picked on.
        points_per_object : one list of (col, row) pixel pairs per object,
            all on the anchor slice.
        labels_per_object : matching lists of 1 = positive / 0 = negative.
        progress : called as (done, total, frame_idx, frame_masks) where
            frame_masks is (n_objects, H, W) bool.

        Returns
        -------
        (n_objects, n_frames, H, W) boolean mask stack.
        """
        if not points_per_object or len(points_per_object) != len(labels_per_object):
            raise ValueError("need at least one object with matching points/labels")
        for pts, labs in zip(points_per_object, labels_per_object):
            if len(pts) != len(labs) or not pts:
                raise ValueError(
                    "each object needs equal-length, non-empty points and labels"
                )

        n_objects = len(points_per_object)
        n_frames = len(frames)
        h, w = frames[0].shape[:2]
        log.info(
            "Volume propagation: %d slices of %dx%d, anchor slice %d, "
            "%d object(s) with %s point(s). One shared tracker session - "
            "the per-slice cost is nearly independent of object count.",
            n_frames,
            w,
            h,
            anchor_idx,
            n_objects,
            [len(p) for p in points_per_object],
        )
        t0 = time.perf_counter()

        session_started = time.perf_counter()
        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            max_vision_features_cache_size=2,
            dtype=self.session_dtype,
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_timings["session_init"] = time.perf_counter() - session_started

        # Register each object separately (SAM2-demo style: objects may have
        # different point counts, so no padding is needed).
        for obj_i, (points, labels) in enumerate(
            zip(points_per_object, labels_per_object)
        ):
            self.processor.add_inputs_to_inference_session(
                session,
                frame_idx=anchor_idx,
                obj_ids=obj_i + 1,
                input_points=[[[[float(x), float(y)] for x, y in points]]],
                input_labels=[[[int(l) for l in labels]]],
                original_size=(h, w),
            )

        # Segment the anchor slice first, then sweep forward and backward.
        masks = np.zeros((n_objects, n_frames, h, w), dtype=bool)
        with self._autocast():
            anchor_out = self.model(inference_session=session, frame_idx=anchor_idx)
            masks[:, anchor_idx] = self._to_masks(anchor_out.pred_masks, n_objects, h, w)
            if progress is not None:
                progress(1, n_frames, anchor_idx, masks[:, anchor_idx])
            log.info("Anchor slice %d segmented; propagating forward...", anchor_idx)

            done = 1
            for reverse in (False, True):
                direction = "backward" if reverse else "forward"
                for out in self.model.propagate_in_video_iterator(
                    session, start_frame_idx=anchor_idx, reverse=reverse
                ):
                    if out.frame_idx == anchor_idx:
                        continue
                    masks[:, out.frame_idx] = self._to_masks(
                        out.pred_masks, n_objects, h, w
                    )
                    done += 1
                    if progress is not None:
                        progress(done, n_frames, out.frame_idx, masks[:, out.frame_idx])
                    if done == n_frames or done % max(1, n_frames // 10) == 0:
                        log.info("  slice %d/%d done (%s)", done, n_frames, direction)
                if not reverse:
                    log.info("Forward sweep complete; propagating backward...")

        total_elapsed = time.perf_counter() - t0
        self.last_timings.update(
            inference=total_elapsed - self.last_timings["session_init"],
            total=total_elapsed,
            fps=n_frames / total_elapsed if total_elapsed else float("inf"),
        )
        log.info(
            "Volume propagation finished in %.1fs (%d/%d slices, %d object(s), "
            "total coverage %.2f%%)",
            total_elapsed,
            done,
            n_frames,
            n_objects,
            100.0 * float(masks.any(axis=0).mean()),
        )
        return masks

    def _to_masks(
        self, pred_masks: torch.Tensor, n_objects: int, h: int, w: int
    ) -> np.ndarray:
        """Threshold + resize one frame's predicted logits to (n_objects, H, W)."""
        video_masks = self.processor.post_process_masks(
            [pred_masks], original_sizes=[(h, w)], binarize=True
        )[0]
        # (n_objects, 1, H, W) -> drop the per-object mask channel.
        out = np.asarray(video_masks[:, 0].detach().cpu().numpy(), dtype=bool)
        if out.shape[0] != n_objects:
            raise RuntimeError(
                f"tracker returned {out.shape[0]} object masks, expected {n_objects}"
            )
        return out
