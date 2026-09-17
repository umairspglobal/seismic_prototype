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
import inspect
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .logutil import get_logger

log = get_logger("inference")

# Transformers' lazy module loader is not thread-safe. FastAPI runs sync
# endpoints in a thread pool, so two concurrent imports of Sam* classes
# can raise ImportError even when the classes exist. Serialize them.
_tf_import_lock = threading.Lock()
_tf_point_classes: dict[str, tuple[type, type]] = {}
_tf_video_classes: dict[str, tuple[type, type]] = {}
_tf_session_classes: dict[str, type] = {}
_tf_version: str | None = None


def _adapt_sam31_init_state(model: Any) -> None:
    """Allow SAM 3.1 models without SAM 3's state-offload option."""
    try:
        parameters = inspect.signature(model.init_state).parameters
    except (TypeError, ValueError):
        log.warning("Could not inspect SAM 3.1 init_state signature")
        return
    if "offload_state_to_cpu" in parameters:
        return

    native_init_state = model.init_state

    def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("offload_state_to_cpu", None)
        return native_init_state(*args, **kwargs)

    model.init_state = compatible_init_state
    log.info("Adapted SAM 3.1 init_state for the shared predictor API")


def transformers_version() -> str | None:
    """Import transformers under the shared lock and return its version."""
    global _tf_version
    with _tf_import_lock:
        if _tf_version is None:
            import transformers

            _tf_version = transformers.__version__
        return _tf_version


def build_sam31_predictor(max_num_objects: int = 128) -> Any:
    """Build the repository-native SAM 3.1 Object Multiplex predictor."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SAM 3.1 Object Multiplex requires a CUDA GPU. Install the latest "
            "facebookresearch/sam3 package and a CUDA-enabled PyTorch build, or "
            "select SAM 3 / SAM 2."
        )
    # This project keeps a Windows-compatible SAM checkout beside the app.
    # Prefer it to site-packages: upstream imports Triton unconditionally,
    # while the local checkout contains the required no-Triton fallbacks.
    local_checkout = Path(__file__).resolve().parents[1] / "sam3"
    if (local_checkout / "sam3" / "model_builder.py").is_file():
        checkout_path = str(local_checkout)
        if checkout_path not in sys.path:
            sys.path.insert(0, checkout_path)
            log.info("Using repository-local SAM 3 code from %s", local_checkout)
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
    except ModuleNotFoundError as exc:
        if exc.name == "triton" and sys.platform == "win32":
            raise RuntimeError(
                "The installed SAM 3 package requires Triton, which is not bundled "
                "with Windows PyTorch. Keep the Windows-compatible sam3 checkout "
                "in this project, or install a compatible triton-windows build."
            ) from exc
        if exc.name and not exc.name.startswith("sam3"):
            raise RuntimeError(
                f"SAM 3.1 dependency {exc.name!r} is missing. "
                "Reinstall requirements.txt and restart the inference server."
            ) from exc
        raise RuntimeError(
            "SAM 3.1 needs the latest facebookresearch/sam3 model code. "
            "Reinstall requirements.txt, or run: "
            "pip install -U git+https://github.com/facebookresearch/sam3.git"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "The installed facebookresearch/sam3 package is incompatible with "
            f"SAM 3.1 ({exc}). Reinstall requirements.txt and restart the server."
        ) from exc

    log.info("Loading native SAM 3.1 Object Multiplex predictor...")
    predictor = build_sam3_multiplex_video_predictor(
        max_num_objects=max_num_objects,
        multiplex_count=16,
        # These optional kernels are not part of the base installation.
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    _adapt_sam31_init_state(predictor.model)
    _install_sam31_point_mask_ranking(predictor)
    return predictor


def _point_tracker_classes(family: str | None = None) -> tuple[type, type]:
    key = config.resolve_family(family)
    with _tf_import_lock:
        if key not in _tf_point_classes:
            if key == "sam2":
                from transformers import Sam2Model, Sam2Processor

                _tf_point_classes[key] = (Sam2Model, Sam2Processor)
            else:
                from transformers import Sam3TrackerModel, Sam3TrackerProcessor

                _tf_point_classes[key] = (Sam3TrackerModel, Sam3TrackerProcessor)
        return _tf_point_classes[key]


def _video_tracker_classes(family: str | None = None) -> tuple[type, type]:
    key = config.resolve_family(family)
    with _tf_import_lock:
        if key not in _tf_video_classes:
            if key == "sam2":
                from transformers import Sam2VideoModel, Sam2VideoProcessor

                _tf_video_classes[key] = (Sam2VideoModel, Sam2VideoProcessor)
            else:
                from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

                _tf_video_classes[key] = (Sam3TrackerVideoModel, Sam3TrackerVideoProcessor)
        return _tf_video_classes[key]


def _video_session_class(family: str | None = None) -> type:
    key = config.resolve_family(family)
    with _tf_import_lock:
        if key not in _tf_session_classes:
            if key == "sam2":
                from transformers.models.sam2_video.modeling_sam2_video import (
                    Sam2VideoInferenceSession,
                )

                _tf_session_classes[key] = Sam2VideoInferenceSession
            else:
                from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
                    Sam3TrackerVideoInferenceSession,
                )

                _tf_session_classes[key] = Sam3TrackerVideoInferenceSession
        return _tf_session_classes[key]


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
        self.prompts = prompts or config.SEISMIC_PROMPTS
        self.checkpoint = checkpoint

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
    """Interactive point-prompted segmentation (SAM 3 tracker or SAM 2).

    SAM 3 uses the tracker / PVS head; SAM 2 uses Sam2Model. Both share
    the same click interface: the user marks positive/negative points on
    the section and the model segments the one object they indicated.
    The *whole* section image is passed in one go (seismic lines are
    small compared to SAM's 1024 input; the processor resizes
    internally), so click coordinates are plain full-resolution array
    indices - no tile bookkeeping required.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        embedding_cache_size: int = 2,
        family: str | None = None,
    ):
        self.family = config.resolve_family(family)
        spec = config.family_spec(self.family)
        checkpoint = checkpoint or str(spec["checkpoint"])
        ModelCls, ProcessorCls = _point_tracker_classes(self.family)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has no CUDA-enabled "
                "PyTorch build. Install a CUDA PyTorch wheel in the active environment "
                "or select CPU."
            )
        log.info(
            "Loading %s from '%s' onto device=%s (first run may "
            "download weights - can take several minutes on CPU)...",
            spec["point_model"],
            checkpoint,
            self.device,
        )
        t0 = time.perf_counter()
        self.model = ModelCls.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        log.info(
            "%s weights loaded in %.1fs; loading processor...",
            spec["point_model"],
            time.perf_counter() - t0,
        )
        self.processor = ProcessorCls.from_pretrained(checkpoint)
        self.checkpoint = checkpoint
        log.info("Point-prompt tracker (%s) ready on %s", spec["label"], self.device)

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
        object_id: int = 0,
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
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            mask_threshold=0.0,
            binarize=True,
        )[0]
        candidates = np.asarray(masks[0].cpu().numpy(), dtype=bool)
        iou = outputs.iou_scores.float().cpu().numpy().reshape(-1)
        best = _select_prompt_mask(candidates, iou, points, labels)
        mask = candidates[best]
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


def _select_prompt_mask(
    masks: np.ndarray,
    iou_scores: np.ndarray,
    points: Sequence[tuple[int, int]],
    labels: Sequence[int],
) -> int:
    """Choose the most specific candidate that actually obeys the clicks.

    SAM emits several valid semantic levels for one point (part, object,
    surrounding region). Its natural-image IoU head can rank the whole seismic
    section above the local event, or rank an empty mask after negative clicks.
    Click consistency is therefore authoritative; predicted IoU only breaks
    ties between equally specific, click-consistent candidates.
    """
    candidates = np.asarray(masks, dtype=bool)
    scores = np.asarray(iou_scores, dtype=np.float32).reshape(-1)
    if candidates.ndim == 2:
        candidates = candidates[None, ...]
    if candidates.ndim != 3 or not len(candidates):
        raise ValueError("point decoder returned no mask candidates")
    if len(scores) != len(candidates):
        raise ValueError("mask candidates and IoU scores must have equal length")

    height, width = candidates.shape[-2:]
    samples = [
        (min(width - 1, max(0, int(x))), min(height - 1, max(0, int(y))), int(label))
        for (x, y), label in zip(points, labels)
    ]

    rankings: list[tuple[tuple[float, ...], int]] = []
    for index, (candidate, iou) in enumerate(zip(candidates, scores)):
        positive_hits = sum(candidate[y, x] for x, y, label in samples if label == 1)
        negative_hits = sum(candidate[y, x] for x, y, label in samples if label == 0)
        positive_count = sum(label == 1 for _, _, label in samples)
        negative_count = len(samples) - positive_count
        prompt_errors = (positive_count - positive_hits) + negative_hits
        coverage = float(candidate.mean())
        # Among masks with the same click agreement, prefer the most local
        # interpretation. IoU remains a final tie-breaker for equal areas.
        rank = (
            -float(prompt_errors),
            float(positive_hits),
            float(negative_count - negative_hits),
            -coverage,
            float(iou),
        )
        rankings.append((rank, index))

    return max(rankings)[1]


def _rerank_sam31_multimask_ious(
    mask_logits: torch.Tensor,
    iou_scores: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Make SAM 3.1 choose the smallest candidate that obeys the clicks.

    The native Object Multiplex decoder emits three masks for a one-point
    prompt, then immediately collapses them using its natural-image IoU head.
    On seismic sections that head can rank a nearly full-frame candidate first.
    Re-ranking the IoUs here lets the native decoder select the local,
    click-consistent candidate while keeping its matching object token.

    Ranking is done on logits upsampled to the tracker image size. The decoder
    itself is low-resolution (~1/3.5), which is too coarse to score a one-pixel
    seismic event, so sampling the raw decoder grid can make the full-frame
    candidate look like the only click-consistent mask.
    """
    if (
        mask_logits.ndim != 4
        or iou_scores.ndim != 2
        or point_coords.ndim != 3
        or point_labels.ndim != 2
        or mask_logits.shape[:2] != iou_scores.shape
        or mask_logits.shape[0] != point_coords.shape[0]
        or point_coords.shape[:2] != point_labels.shape
        or mask_logits.shape[1] <= 1
    ):
        return iou_scores

    size = max(int(image_size), 1)
    logits = mask_logits.float()
    if logits.shape[-2:] != (size, size):
        logits = F.interpolate(
            logits,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
    coords = point_coords.to(logits.device, dtype=torch.float32)
    labels = point_labels.to(logits.device)
    x = torch.clamp(torch.round(coords[..., 0]), 0, size - 1).long()
    y = torch.clamp(torch.round(coords[..., 1]), 0, size - 1).long()

    binary = logits > 0
    reranked = iou_scores.clone()
    floor = torch.finfo(reranked.dtype).min
    for batch_index in range(mask_logits.shape[0]):
        samples = binary[batch_index, :, y[batch_index], x[batch_index]]
        positive = labels[batch_index] == 1
        negative = labels[batch_index] == 0
        errors = (
            ((~samples) & positive.unsqueeze(0))
            | (samples & negative.unsqueeze(0))
        ).sum(dim=1)
        areas = binary[batch_index].flatten(1).sum(dim=1)

        eligible = errors == errors.min()
        if positive.any():
            nonempty = areas > 0
            if (eligible & nonempty).any():
                eligible &= nonempty
        eligible_areas = torch.where(
            eligible,
            areas,
            torch.full_like(areas, areas.max() + 1),
        )
        eligible &= areas == eligible_areas.min()

        native_scores = torch.nan_to_num(
            iou_scores[batch_index],
            nan=floor,
            neginf=floor,
        )
        best = torch.argmax(
            torch.where(
                eligible,
                native_scores,
                torch.full_like(native_scores, floor),
            )
        )
        reranked[batch_index].fill_(floor)
        reranked[batch_index, best] = native_scores[best]
    return reranked


def _sam31_interactive_modules(predictor: Any) -> tuple[Any, Any, int] | None:
    """Find SAM 3.1's interactive prompt encoder / mask decoder pair."""
    model = getattr(predictor, "model", None)
    if model is None:
        return None
    tracker = getattr(model, "tracker", model)
    inner = getattr(tracker, "model", tracker)
    seen: list[Any] = []
    for candidate in (inner, tracker, model):
        if candidate is None or any(candidate is item for item in seen):
            continue
        seen.append(candidate)
        encoder = getattr(candidate, "interactive_sam_prompt_encoder", None)
        decoder = getattr(candidate, "interactive_sam_mask_decoder", None)
        image_size = getattr(candidate, "image_size", None)
        if encoder is not None and decoder is not None and image_size is not None:
            return encoder, decoder, int(image_size)
    return None


def _install_sam31_point_mask_ranking(predictor: Any) -> None:
    """Intercept native one-click multimasks before SAM 3.1 collapses them."""
    located = _sam31_interactive_modules(predictor)
    if located is None:
        log.error("Could not locate SAM 3.1's interactive mask decoder")
        return
    prompt_encoder, mask_decoder, image_size = located
    if getattr(mask_decoder, "_seismic_point_rank_wrapped", False):
        return

    prompt_context: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {
        "points": None
    }
    original_encoder_forward = prompt_encoder.forward
    original_decoder_forward = mask_decoder.forward

    def _as_point_pair(points: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            isinstance(points, tuple)
            and len(points) == 2
            and all(isinstance(value, torch.Tensor) for value in points)
        ):
            return points
        return None

    def capture_and_encode(*args: Any, **kwargs: Any) -> Any:
        points = kwargs.get("points")
        if points is None and args:
            points = args[0]
        prompt_context["points"] = _as_point_pair(points)
        return original_encoder_forward(*args, **kwargs)

    def rank_and_decode(*args: Any, **kwargs: Any) -> Any:
        output = original_decoder_forward(*args, **kwargs)
        multimask_output = kwargs.get("multimask_output")
        if multimask_output is None and len(args) >= 5:
            multimask_output = args[4]
        points = prompt_context.pop("points", None)
        if not multimask_output or points is None or not isinstance(output, tuple):
            return output
        masks, ious, *rest = output
        coords, labels = points
        ranked = _rerank_sam31_multimask_ious(
            masks,
            ious,
            coords,
            labels,
            image_size,
        )
        return (masks, ranked, *rest)

    prompt_encoder.forward = capture_and_encode
    mask_decoder.forward = rank_and_decode
    mask_decoder._seismic_point_rank_wrapped = True
    log.info("Enabled seismic-specific SAM 3.1 point-mask candidate ranking")


class LiveTracker:
    """Tracker sessions kept alive after a propagation, for interactive edits.

    Sessions keep their video and per-object memory banks on CPU, so
    holding them resident costs host RAM rather than VRAM. Keeping them
    is what makes SAM2-style editing possible: a click on a slice that
    has already been tracked is applied *with* the tracker's memory of
    the object there, so it refines the propagated mask immediately
    instead of segmenting that slice from scratch.

    ``masks`` is the propagated volume, kept in sync with every edit, so
    it is also what gets exported to ParaView.
    """

    def __init__(self, n_objects: int, n_frames: int, height: int, width: int):
        self.masks = np.zeros((n_objects, n_frames, height, width), dtype=bool)
        self.height = height
        self.width = width
        # One session per GPU-sized wave of objects.
        self.sessions: list = []
        # global object index -> (session position, object id within that session)
        self.slots: dict[int, tuple[int, int]] = {}
        # Slices edited since the last sweep; the next sweep starts from these.
        self.dirty_frames: set[int] = set()

    @property
    def n_objects(self) -> int:
        return int(self.masks.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.masks.shape[1])

    def attach(self, session, slots: dict[int, int]) -> None:
        """Register a finished wave's session and its object id mapping."""
        session_idx = len(self.sessions)
        self.sessions.append(session)
        for global_idx, obj_id in slots.items():
            self.slots[global_idx] = (session_idx, obj_id)

    def objects_in(self, session_idx: int) -> dict[int, int]:
        """object id within the session -> global object index."""
        return {
            obj_id: global_idx
            for global_idx, (sess, obj_id) in self.slots.items()
            if sess == session_idx
        }

    def close(self) -> None:
        self.sessions.clear()
        self.slots.clear()


def _native_mask_rows(outputs: dict | None) -> dict[int, np.ndarray]:
    """Normalize repository-native predictor output to object-id mask rows."""
    if not outputs:
        return {}
    object_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return {
        int(object_id): np.asarray(mask, dtype=bool)
        for object_id, mask in zip(object_ids, masks)
    }


def _mask_for_native_id(outputs: dict | None, native_id: int) -> np.ndarray | None:
    """Return the mask for ``native_id``, or the only native mask if ids disagree."""
    rows = _native_mask_rows(outputs)
    mask = rows.get(int(native_id))
    if mask is None and len(rows) == 1:
        mask = next(iter(rows.values()))
    return mask


def _close_native_session(predictor: Any, session_id: str) -> None:
    try:
        predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": False,
            }
        )
    except Exception:
        log.debug("Could not close native SAM 3.1 session %s", session_id, exc_info=True)


def _relative_points(
    points: Sequence[tuple[int, int]],
    shape: tuple[int, int] | Sequence[int],
) -> list[list[float]]:
    """Map original-image pixels to the native predictor's [0, 1] coordinates."""
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    return [
        [
            min(1.0, max(0.0, float(x) / width)),
            min(1.0, max(0.0, float(y) / height)),
        ]
        for x, y in points
    ]


def _pad_to_square(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Letterbox a section so SAM 3.1's square resize cannot stretch it.

    The native Object Multiplex loader always does ``PIL.Image.resize((S, S))``.
    A 1750x231 seismic line stretched ~7x vertically turns thin horizons into
    full-frame texture, so one click segments the whole section. Padding to a
    square first keeps the original aspect ratio through that resize.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 1:
        raise ValueError("expected an HxWxC image array")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    letterbox = {
        "top": top,
        "left": left,
        "height": height,
        "width": width,
        "side": side,
    }
    if top == 0 and left == 0 and side == height and side == width:
        return rgb, letterbox
    fill = int(np.median(rgb.reshape(-1, rgb.shape[-1]).mean(axis=-1)))
    padded = np.full((side, side, rgb.shape[2]), fill, dtype=rgb.dtype)
    padded[top : top + height, left : left + width] = rgb
    return padded, letterbox


def _letterbox_relative_points(
    points: Sequence[tuple[int, int]],
    letterbox: dict[str, int],
) -> list[list[float]]:
    """Map original-image pixels into the letterboxed square's [0, 1] range."""
    shifted = [
        (int(x) + int(letterbox["left"]), int(y) + int(letterbox["top"]))
        for x, y in points
    ]
    return _relative_points(shifted, (int(letterbox["side"]), int(letterbox["side"])))


def _crop_letterbox_mask(mask: np.ndarray, letterbox: dict[str, int]) -> np.ndarray:
    """Crop a native square mask back to the original section shape."""
    height, width = int(letterbox["height"]), int(letterbox["width"])
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[-2:] == (height, width):
        return mask
    top, left, side = (
        int(letterbox["top"]),
        int(letterbox["left"]),
        int(letterbox["side"]),
    )
    if mask.shape[-2:] == (side, side):
        cropped = mask[..., top : top + height, left : left + width]
        if cropped.shape[-2:] == (height, width):
            return cropped
    log.warning(
        "SAM 3.1 mask shape %s did not match letterbox %s; resizing",
        mask.shape[-2:],
        (height, width),
    )
    plane = mask
    while plane.ndim > 2:
        plane = plane[0]
    resized = np.array(
        Image.fromarray(plane.astype(np.uint8) * 255).resize(
            (width, height), Image.NEAREST
        )
    )
    return resized > 0


class Sam31PointSegmenter:
    """Single-image point prompting through SAM 3.1's native video predictor."""

    family = "sam31"
    checkpoint = "facebook/sam3.1"
    device = "cuda"

    def __init__(self, predictor: Any, embedding_cache_size: int = 2):
        self.predictor = predictor
        self.model = predictor.model
        self.processor = None
        self.embedding_cache_size = max(1, int(embedding_cache_size))
        self._prepared: OrderedDict[Hashable, dict] = OrderedDict()
        self.last_timings: dict[str, float] = {}

    def prepare_image(
        self,
        rgb: np.ndarray,
        image_key: Hashable | None = None,
    ) -> Hashable:
        key = image_key if image_key is not None else ("array", id(rgb), rgb.shape)
        if key in self._prepared:
            self._prepared.move_to_end(key)
            self.last_timings["prepare_image"] = 0.0
            return key

        started = time.perf_counter()
        if isinstance(key, tuple) and key:
            current_file = key[0]
            stale_keys = [
                stale
                for stale in list(self._prepared)
                if isinstance(stale, tuple) and stale and stale[0] != current_file
            ]
            for stale in stale_keys:
                evicted = self._prepared.pop(stale)
                _close_native_session(self.predictor, evicted["session_id"])
        padded, letterbox = _pad_to_square(rgb)
        response = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": [Image.fromarray(padded).convert("RGB")],
                "offload_video_to_cpu": True,
            }
        )
        self._prepared[key] = {
            "session_id": response["session_id"],
            "shape": rgb.shape[:2],
            "letterbox": letterbox,
            "objects": {},
        }
        self._prepared.move_to_end(key)
        while len(self._prepared) > self.embedding_cache_size:
            _, evicted = self._prepared.popitem(last=False)
            _close_native_session(self.predictor, evicted["session_id"])
        self.last_timings["prepare_image"] = time.perf_counter() - started
        return key

    @torch.no_grad()
    def segment(
        self,
        rgb: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
        image_key: Hashable | None = None,
        object_id: int = 0,
    ) -> np.ndarray:
        if len(points) != len(labels) or not points:
            raise ValueError("points and labels must be equal-length and non-empty")

        started = time.perf_counter()
        key = self.prepare_image(rgb, image_key=image_key)
        prepared = self._prepared[key]
        session_id = prepared["session_id"]
        native_id = int(object_id) + 1
        signature = (
            tuple((int(x), int(y)) for x, y in points),
            tuple(int(label) for label in labels),
        )
        previous = prepared["objects"].get(native_id)
        if previous is not None and previous["signature"] == signature:
            self.last_timings["total"] = time.perf_counter() - started
            return previous["mask"].copy()
        if previous is not None:
            self.predictor.handle_request(
                {
                    "type": "remove_object",
                    "session_id": session_id,
                    "frame_index": 0,
                    "obj_id": native_id,
                }
            )
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "points": _letterbox_relative_points(points, prepared["letterbox"]),
                "point_labels": [int(label) for label in labels],
                "obj_id": native_id,
                "clear_old_points": True,
                "rel_coordinates": True,
            }
        )
        mask = _mask_for_native_id(response.get("outputs"), native_id)
        if mask is None:
            mask = np.zeros(rgb.shape[:2], dtype=bool)
        else:
            mask = _crop_letterbox_mask(mask, prepared["letterbox"])
        prepared["objects"][native_id] = {
            "signature": signature,
            "mask": mask.copy(),
        }
        self.last_timings["total"] = time.perf_counter() - started
        return mask

    def close(self) -> None:
        for prepared in self._prepared.values():
            _close_native_session(self.predictor, prepared["session_id"])
        self._prepared.clear()


class Sam31LiveTracker(LiveTracker):
    """Live SAM 3.1 session plus the app's persistent mask volume."""

    def __init__(
        self,
        predictor: Any,
        session_id: str,
        n_objects: int,
        n_frames: int,
        height: int,
        width: int,
        letterbox: dict[str, int],
    ):
        super().__init__(n_objects, n_frames, height, width)
        self.predictor = predictor
        self.session_id = session_id
        self.letterbox = letterbox

    def close(self) -> None:
        _close_native_session(self.predictor, self.session_id)
        super().close()


class Sam31VolumePropagator:
    """SAM 3.1 Object Multiplex adapter for seismic-slice video tracking."""

    family = "sam31"
    checkpoint = "facebook/sam3.1"
    device = "cuda"
    session_dtype = torch.bfloat16

    def __init__(self, predictor: Any):
        self.predictor = predictor
        self.model = predictor.model
        self.processor = None
        self.live: Sam31LiveTracker | None = None
        self.last_timings: dict[str, float] = {}

    @staticmethod
    def _apply_outputs(
        live: Sam31LiveTracker,
        frame_idx: int,
        outputs: dict | None,
    ) -> None:
        rows = _native_mask_rows(outputs)
        for native_id, mask in rows.items():
            object_index = native_id - 1
            if not (0 <= object_index < live.n_objects):
                if len(rows) != 1 or live.n_objects != 1:
                    continue
                object_index = 0
            live.masks[object_index, frame_idx] = _crop_letterbox_mask(
                mask, live.letterbox
            )

    @torch.no_grad()
    def propagate(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        anchor_idx: int,
        points_per_object: Sequence[Sequence[tuple[int, int]]],
        labels_per_object: Sequence[Sequence[int]],
        frame_indices_per_object: Sequence[Sequence[int]] | None = None,
        progress: Callable[[int, int, int, np.ndarray], None] | None = None,
        keep_live: bool = False,
    ) -> np.ndarray:
        if not points_per_object or len(points_per_object) != len(labels_per_object):
            raise ValueError("need at least one object with matching points/labels")
        if frame_indices_per_object is None:
            frame_indices_per_object = [
                [anchor_idx] * len(points) for points in points_per_object
            ]
        if len(frame_indices_per_object) != len(points_per_object):
            raise ValueError("frame_indices_per_object must match points_per_object")

        n_frames = len(frames)
        for points, labels, frame_indices in zip(
            points_per_object, labels_per_object, frame_indices_per_object
        ):
            if (
                len(points) != len(labels)
                or len(points) != len(frame_indices)
                or not points
            ):
                raise ValueError(
                    "each object needs equal-length, non-empty points/labels/slices"
                )
            if any(not 0 <= int(frame) < n_frames for frame in frame_indices):
                raise ValueError("point slice index out of range")
            if 1 not in labels:
                raise ValueError("each object needs at least one positive (+) point")

        if self.live is not None:
            self.live.close()
            self.live = None

        started = time.perf_counter()
        height, width = frames[0].shape[:2]
        padded_frames = []
        letterbox: dict[str, int] | None = None
        for frame in frames:
            padded, letterbox = _pad_to_square(frame)
            padded_frames.append(Image.fromarray(padded).convert("RGB"))
        assert letterbox is not None
        response = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": padded_frames,
                "offload_video_to_cpu": True,
            }
        )
        live = Sam31LiveTracker(
            self.predictor,
            response["session_id"],
            len(points_per_object),
            n_frames,
            height,
            width,
            letterbox,
        )
        self.last_timings["session_init"] = time.perf_counter() - started

        try:
            for object_index, (points, labels, frame_indices) in enumerate(
                zip(points_per_object, labels_per_object, frame_indices_per_object)
            ):
                grouped: dict[int, tuple[list[list[float]], list[int]]] = {}
                for (x, y), label, frame_idx in zip(points, labels, frame_indices):
                    frame_points, frame_labels = grouped.setdefault(
                        int(frame_idx), ([], [])
                    )
                    frame_points.extend(
                        _letterbox_relative_points([(x, y)], letterbox)
                    )
                    frame_labels.append(int(label))
                for frame_idx, (frame_points, frame_labels) in sorted(grouped.items()):
                    prompt_response = self.predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": live.session_id,
                            "frame_index": frame_idx,
                            "points": frame_points,
                            "point_labels": frame_labels,
                            "obj_id": object_index + 1,
                            "clear_old_points": True,
                            "rel_coordinates": True,
                        }
                    )
                    self._apply_outputs(
                        live, frame_idx, prompt_response.get("outputs")
                    )

            visited: set[int] = set()
            for event in self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": live.session_id,
                }
            ):
                frame_idx = int(event["frame_index"])
                self._apply_outputs(live, frame_idx, event.get("outputs"))
                visited.add(frame_idx)
                if progress is not None:
                    progress(
                        len(visited),
                        n_frames,
                        frame_idx,
                        live.masks[:, frame_idx],
                    )
        except Exception:
            live.close()
            raise

        elapsed = time.perf_counter() - started
        self.last_timings.update(
            inference=elapsed - self.last_timings["session_init"],
            total=elapsed,
            fps=n_frames / elapsed if elapsed else float("inf"),
            object_waves=1.0,
            objects_per_wave=float(len(points_per_object)),
        )
        if keep_live:
            self.live = live
        else:
            live.close()
        return live.masks

    @torch.no_grad()
    def refine_frame(
        self,
        live: Sam31LiveTracker,
        object_index: int,
        frame_idx: int,
        points: Sequence[tuple[int, int]],
        labels: Sequence[int],
    ) -> np.ndarray:
        if not points or len(points) != len(labels):
            raise ValueError("refinement needs equal-length, non-empty points/labels")
        started = time.perf_counter()
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": live.session_id,
                "frame_index": int(frame_idx),
                "points": _letterbox_relative_points(points, live.letterbox),
                "point_labels": [int(label) for label in labels],
                "obj_id": int(object_index) + 1,
                "clear_old_points": True,
                "rel_coordinates": True,
            }
        )
        self._apply_outputs(live, int(frame_idx), response.get("outputs"))
        live.dirty_frames.add(int(frame_idx))
        self.last_timings["refine_frame"] = time.perf_counter() - started
        return live.masks[:, int(frame_idx)]

    @torch.no_grad()
    def resweep(
        self,
        live: Sam31LiveTracker,
        progress: Callable[[int, int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        edited = sorted(live.dirty_frames)
        if not edited:
            raise ValueError("nothing was edited since the last propagation")

        started = time.perf_counter()
        visited: set[int] = set()
        for direction, start_frame in (
            ("forward", edited[0]),
            ("backward", edited[-1]),
        ):
            for event in self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": live.session_id,
                    "propagation_direction": direction,
                    "start_frame_index": start_frame,
                }
            ):
                frame_idx = int(event["frame_index"])
                self._apply_outputs(live, frame_idx, event.get("outputs"))
                visited.add(frame_idx)
                if progress is not None:
                    progress(
                        len(visited),
                        live.n_frames,
                        frame_idx,
                        live.masks[:, frame_idx],
                    )
        live.dirty_frames.clear()
        elapsed = time.perf_counter() - started
        self.last_timings.update(
            total=elapsed,
            fps=live.n_frames / elapsed if elapsed else float("inf"),
        )
        return live.masks

    def close(self) -> None:
        if self.live is not None:
            self.live.close()
            self.live = None


class Sam3VolumePropagator:
    """Propagate a point-picked object through a 3D volume, video-tracker style.

    This is the seismic equivalent of SAM 2's video segmentation: the
    slices of a 3D volume along one axis (inlines, crosslines, or time
    slices) are treated as consecutive video frames. You click points on
    one slice; the memory-based tracker then follows the object through
    every other slice in both directions. No conversion of the .sgy to an
    actual video file is needed - the frames are fed in as arrays.

    ``family`` selects SAM 3's tracker video model (default) or SAM 2.1.

    Note this only makes sense for 3D volumes. A 2D line is a single
    frame: its image already contains the full time axis, so there is
    nothing to propagate through.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        use_bfloat16: bool = True,
        compile_model: bool = False,
        family: str | None = None,
    ):
        self.family = config.resolve_family(family)
        spec = config.family_spec(self.family)
        checkpoint = checkpoint or str(spec["checkpoint"])
        ModelCls, ProcessorCls = _video_tracker_classes(self.family)
        self._session_cls = _video_session_class(self.family)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has no CUDA-enabled "
                "PyTorch build. Install a CUDA PyTorch wheel in the active environment "
                "or select CPU."
            )
        log.info(
            "Loading %s from '%s' onto device=%s...",
            spec["video_model"],
            checkpoint,
            self.device,
        )
        t0 = time.perf_counter()
        self.model = ModelCls.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        if compile_model and hasattr(torch, "compile"):
            log.info(
                "Compiling %s video tracker (first propagation will warm up)...",
                spec["label"],
            )
            self.model = torch.compile(self.model)
        self.processor = ProcessorCls.from_pretrained(checkpoint)
        self.checkpoint = checkpoint
        self.session_dtype = (
            torch.bfloat16
            if use_bfloat16
            and self.device.startswith("cuda")
            and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.last_timings: dict[str, float] = {}
        # Sessions from the last propagation, kept for interactive edits.
        self.live: LiveTracker | None = None
        log.info(
            "Volume propagator (%s) ready on %s (loaded in %.1fs)",
            spec["label"],
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
        state_device = "cpu" if self.device.startswith("cuda") else self.device
        return self._session_cls(
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
        live: LiveTracker | None = None,
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
        keep_session = False
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
            keep_session = live is not None
        finally:
            if keep_session:
                # Hand the session (CPU-resident) to the caller so later
                # clicks refine with memory instead of rebuilding.
                live.attach(
                    session,
                    {g: pos + 1 for pos, g in enumerate(object_indices)},
                )
            else:
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
        keep_live: bool = False,
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
        keep_live : keep the tracker sessions resident afterwards (on
            ``self.live``) so ``refine_frame``/``resweep`` can edit the
            result interactively. Costs host RAM until replaced.

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

        # Drop any previous live sessions before allocating new ones so the
        # old video + memory banks are freed rather than doubled up.
        self.live = None
        live = LiveTracker(n_objects, n_frames, h, w) if keep_live else None
        masks = live.masks if live is not None else np.zeros(
            (n_objects, n_frames, h, w), dtype=bool
        )
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
                    live,
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
        self.live = live
        return masks

    @torch.no_grad()
    def refine_frame(
        self,
        live: LiveTracker,
        object_index: int,
        frame_idx: int,
        points: Sequence[tuple[int, int]],
        labels: Sequence[int],
    ) -> np.ndarray:
        """Re-decode one slice for one object from its clicks on that slice.

        The slice has already been tracked, so the tracker conditions on
        its memory of the object *and* the new clicks - this is SAM2's
        refinement path. A negative click therefore carves away part of
        the propagated mask rather than redefining the object, and a
        positive click extends it, both visible immediately.

        Clicks replace whatever was previously prompted on this slice for
        this object, so callers should send the object's full point list
        for the slice. Returns the (n_objects, H, W) stack for the slice.
        """
        if not points or len(points) != len(labels):
            raise ValueError("refinement needs equal-length, non-empty points/labels")
        slot = live.slots.get(object_index)
        if slot is None:
            raise ValueError(
                f"object {object_index} is not part of the live tracker session; "
                "re-propagate to add it"
            )
        session_idx, obj_id = slot
        session = live.sessions[session_idx]
        started = time.perf_counter()

        with self._autocast():
            self.processor.add_inputs_to_inference_session(
                session,
                frame_idx=int(frame_idx),
                obj_ids=[obj_id],
                input_points=[[[[float(x), float(y)] for x, y in points]]],
                input_labels=[[[int(l) for l in labels]]],
                original_size=(live.height, live.width),
            )
            out = self.model(inference_session=session, frame_idx=int(frame_idx))

        rows = self._to_masks(out.pred_masks, live.height, live.width)
        by_session_id = live.objects_in(session_idx)
        for row, out_id in zip(rows, out.object_ids):
            global_idx = by_session_id.get(int(out_id))
            if global_idx is not None:
                live.masks[global_idx, frame_idx] = row
        live.dirty_frames.add(int(frame_idx))
        self.last_timings["refine_frame"] = time.perf_counter() - started
        log.info(
            "Refined object %d on slice %d with %d point(s) in %.2fs "
            "(coverage %.2f%%)",
            object_index + 1,
            frame_idx,
            len(points),
            self.last_timings["refine_frame"],
            100.0 * float(live.masks[object_index, frame_idx].mean()),
        )
        return live.masks[:, int(frame_idx)]

    @torch.no_grad()
    def resweep(
        self,
        live: LiveTracker,
        progress: Callable[[int, int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        """Re-track the volume outward from the slices edited since the last sweep.

        Reuses the live sessions, so the refinement clicks already applied
        stay in place and only the tracking is redone. This is what SAM 2's
        demo does when you refine a frame and hit 'Track objects' again.
        """
        if not live.sessions:
            raise ValueError("no live tracker sessions; run a full propagation first")
        edited = sorted(live.dirty_frames)
        if not edited:
            raise ValueError("nothing was edited since the last propagation")

        n_frames = live.n_frames
        total = len(live.sessions) * n_frames
        t0 = time.perf_counter()
        log.info(
            "Re-sweeping %d slice(s) from edit(s) on slice(s) %s across "
            "%d session(s)...",
            n_frames,
            edited,
            len(live.sessions),
        )
        done = 0
        for session_idx, session in enumerate(live.sessions):
            by_session_id = live.objects_in(session_idx)
            visited: set[int] = set()
            with self._autocast():
                # Forward from the first edit, backward from the last: together
                # they cover every slice while starting where the change is.
                for reverse, start in ((False, edited[0]), (True, edited[-1])):
                    for out in self.model.propagate_in_video_iterator(
                        session, start_frame_idx=start, reverse=reverse
                    ):
                        rows = self._to_masks(out.pred_masks, live.height, live.width)
                        for row, out_id in zip(rows, out.object_ids):
                            global_idx = by_session_id.get(int(out_id))
                            if global_idx is not None:
                                live.masks[global_idx, out.frame_idx] = row
                        if out.frame_idx not in visited:
                            visited.add(out.frame_idx)
                            done += 1
                        if progress is not None:
                            progress(
                                min(done, total),
                                total,
                                int(out.frame_idx),
                                live.masks[:, out.frame_idx],
                            )
            self._free_cuda()

        live.dirty_frames.clear()
        elapsed = time.perf_counter() - t0
        self.last_timings.update(
            total=elapsed, fps=n_frames / elapsed if elapsed else float("inf")
        )
        log.info(
            "Re-sweep finished in %.1fs (total coverage %.2f%%)",
            elapsed,
            100.0 * float(live.masks.any(axis=0).mean()),
        )
        return live.masks

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
