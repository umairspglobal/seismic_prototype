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
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import importlib.util
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any
import uuid

import numpy as np
from scipy import ndimage
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
    _enable_sam31_efficient_attention()
    tracker = predictor.model.tracker.model
    _install_sam31_feature_provider(tracker)
    _install_sam31_temporal_selection(tracker)
    return predictor


def _enable_sam31_efficient_attention() -> None:
    """Run SAM 3.1's memory attention on the memory-efficient SDPA kernel.

    The Windows checkout pins this attention to the MATH kernel because
    PyTorch's Windows wheels have no flash kernel. MATH materialises the
    full attention matrix: ~1.8 GB per remembered slice here, which spills
    past VRAM within a few slices and slows tracking ~30x. The
    memory-efficient kernel is available on Windows and gives the same
    result.
    """
    if sys.platform != "win32":
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from sam3.model import decoder

    if getattr(decoder, "_seismic_efficient_sdpa", False):
        return

    def kernel(backend: Any) -> Any:
        if backend == SDPBackend.MATH:
            return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        return sdpa_kernel(backend)

    decoder.sdpa_kernel = kernel
    decoder._seismic_efficient_sdpa = True


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


@dataclass(frozen=True)
class MaskBase:
    """Low-res mask logits that a click chain refines instead of starting empty.

    ``key`` names where the logits came from (one tracked slice of one
    propagation), so a cached chain is only extended from the same start.
    """

    key: Hashable
    logits: torch.Tensor  # (h, w) float32 on CPU, at the decoder's mask resolution


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
        self._init_caches(embedding_cache_size)

    def _init_caches(self, embedding_cache_size: int) -> None:
        # Keep only a few sections resident: embeddings are large GPU tensors.
        self.embedding_cache_size = max(1, int(embedding_cache_size))
        self._prepared: OrderedDict[Hashable, dict] = OrderedDict()
        # Decoded click chains per (slice, object, base). Small CPU tensors, so
        # they outlive the embeddings: propagation re-reads the exact preview
        # masks from here even after the user has scrubbed to other slices.
        self.chain_cache_size = 64
        self._chains: OrderedDict[Hashable, list[dict]] = OrderedDict()
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
        base: MaskBase | None = None,
    ) -> np.ndarray:
        """Segment one object indicated by clicked points.

        Clicks are decoded one at a time, in the order given, each together
        with the mask logits of the click before it. That is how SAM is
        trained and how Meta's SAM 2 / SAM 3 demos refine a mask; decoding
        a long click list in one pass is far outside the training range and
        returns speckled or frame-filling masks on seismic sections. Every
        step is cached, so a new click costs one decode and undo none, and
        the same clicks always give the same mask.

        Parameters
        ----------
        rgb : (H, W, 3) uint8 full-resolution section image, time down.
        points : (trace_idx, sample_idx) array-index pairs, i.e. (x, y)
            pixel coordinates on the section image, in click order.
        labels : 1 for positive (inside the object), 0 for negative.
        object_id : clicks of different objects form separate chains.
        base : mask logits the first click refines (e.g. the tracked mask
            on this slice) instead of starting from an empty mask; each
            click then only changes the region it touches.

        Returns
        -------
        (H, W) boolean mask at full section resolution.
        """
        if len(points) != len(labels) or not points:
            raise ValueError("points and labels must be equal-length and non-empty")

        h, w = rgb.shape[:2]
        n_pos = sum(1 for lab in labels if lab == 1)
        t0 = time.perf_counter()
        key = image_key if image_key is not None else ("array", id(rgb), rgb.shape)
        clicks = [(int(x), int(y), int(l)) for (x, y), l in zip(points, labels)]
        chain_key = (key, int(object_id), None if base is None else base.key)
        chain = self._chains.pop(chain_key, [])
        self._chains[chain_key] = chain
        while len(self._chains) > self.chain_cache_size:
            self._chains.popitem(last=False)
        keep = 0
        while keep < min(len(chain), len(clicks)) and chain[keep]["click"] == clicks[keep]:
            keep += 1
        del chain[keep:]

        decoded = len(clicks) - keep
        if decoded or chain[-1]["mask"] is None:
            prepared = self._prepared[self.prepare_image(rgb, image_key=key)]
            self._sync_cuda()
            if not decoded:
                chain[-1]["mask"] = self._mask_from_logits(prepared, chain[-1]["logits"])
            previous = chain[-1]["logits"] if chain else (None if base is None else base.logits)
            for k in range(keep, len(clicks)):
                logits, mask = self._click_step(prepared, points[: k + 1], labels[: k + 1], previous)
                if base is not None:
                    logits, mask = self._local_edit(prepared, previous, logits, mask, clicks[k])
                if chain:
                    chain[-1]["mask"] = None  # only the newest step keeps its full-size mask
                chain.append({"click": clicks[k], "logits": logits, "mask": mask})
                previous = logits
            self._sync_cuda()
            low_h, low_w = chain[-1]["logits"].shape[-2:]
            # 4x4 decoder cells: anything smaller is upsampling noise.
            chain[-1]["mask"] = _tidy_mask(
                chain[-1]["mask"], points, labels, max_hole_area=16 * (h / low_h) * (w / low_w)
            )
        mask = chain[-1]["mask"]
        self.last_timings.update(total=time.perf_counter() - t0, clicks_decoded=float(decoded))
        log.info(
            "Point segmentation %dx%d: %d positive / %d negative clicks, %d decoded%s "
            "in %.2fs (mask coverage %.2f%%)",
            w,
            h,
            n_pos,
            len(clicks) - n_pos,
            decoded,
            " from a tracked mask" if base is not None else "",
            time.perf_counter() - t0,
            100.0 * float(mask.mean()),
        )
        return mask

    def _decode(
        self,
        prepared: dict,
        points: Sequence[tuple[int, int]],
        labels: Sequence[int],
        prior: torch.Tensor | None,
        multimask: bool,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """One prompt-encoder + mask-decoder pass.

        Returns (n, h, w) float32 CPU low-res logits, (n,) predicted IoUs
        and (n, H, W) full-resolution boolean masks.
        """
        inputs = self.processor(
            # 4D: (image, object, point, xy) - one image, one object.
            input_points=[[[[float(x), float(y)] for x, y in points]]],
            input_labels=[[[int(l) for l in labels]]],
            original_sizes=prepared["original_sizes"],
            return_tensors="pt",
        ).to(self.device)
        model_inputs = dict(inputs)
        embeddings = prepared["embeddings"]
        model_inputs["image_embeddings"] = embeddings
        if prior is not None:
            reference = embeddings[-1] if isinstance(embeddings, (list, tuple)) else embeddings
            # Clamped as in SAM's own click refinement, against rare large logits.
            model_inputs["input_masks"] = torch.clamp(prior.float(), -32.0, 32.0)[None, None].to(
                self.device, reference.dtype
            )
        with self._autocast():
            outputs = self.model(**model_inputs, multimask_output=multimask)
        logits = outputs.pred_masks.float().cpu()
        masks = self.processor.post_process_masks(
            logits, inputs["original_sizes"], mask_threshold=0.0, binarize=True
        )[0]
        iou = outputs.iou_scores.float().cpu().numpy().reshape(-1)
        return logits[0, 0], iou, np.asarray(masks[0].numpy(), dtype=bool)

    def _mask_from_logits(self, prepared: dict, logits: torch.Tensor) -> np.ndarray:
        masks = self.processor.post_process_masks(
            logits[None, None, None], prepared["original_sizes"], mask_threshold=0.0, binarize=True
        )[0]
        return np.asarray(masks[0, 0].numpy(), dtype=bool)

    def _local_edit(
        self,
        prepared: dict,
        previous: torch.Tensor,
        logits: torch.Tensor,
        mask: np.ndarray,
        click: tuple[int, int, int],
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Apply a correction click only to the region it touches.

        On a tracked slice the decoder sometimes answers one click by
        redrawing the whole mask, e.g. a negative click on a layer's tail
        also drops half the layer. Only the removed (negative click) or
        added (positive click) region connected to the click is taken from
        the new decode; the rest of the tracked mask stays as it was.
        """
        x, y, label = click
        before = self._mask_from_logits(prepared, previous)
        changed = before & ~mask if label == 0 else mask & ~before
        parts, n = ndimage.label(changed, structure=np.ones((3, 3), dtype=bool))
        height, width = changed.shape
        x, y = min(width - 1, max(0, x)), min(height - 1, max(0, y))
        touched = np.zeros(n + 1, dtype=bool)
        touched[parts[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3].ravel()] = True
        touched[0] = False
        region = torch.from_numpy(touched[parts])[None, None].float()
        region = F.interpolate(region, size=tuple(previous.shape[-2:]), mode="area")[0, 0] > 0
        edited = torch.where(region, logits, previous)
        return edited, self._mask_from_logits(prepared, edited)

    def _click_step(
        self,
        prepared: dict,
        points: Sequence[tuple[int, int]],
        labels: Sequence[int],
        previous: torch.Tensor | None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Decode the newest click on top of ``previous`` mask logits.

        The first click of a chain is ambiguous (part, object, region), so it
        uses SAM's three-mask output ranked by click agreement. Later clicks
        refine the previous mask, preferably through the single-mask output
        as SAM was trained. That output can also redraw the mask as most of
        the section, which obeys every positive click (SAM 2.1 does so on
        the second click of a thin horizon). So among the candidates with
        the fewest violated clicks, the one that keeps most of the previous
        mask wins; the single-mask output wins near-ties.
        """
        if previous is None:
            logits, iou, masks = self._decode(prepared, points, labels, None, multimask=True)
            best = _select_prompt_mask(masks, iou, points, labels)
            return logits[best], masks[best]
        margin = 0.05
        before = self._mask_from_logits(prepared, previous)
        logits, _iou, masks = self._decode(prepared, points, labels, previous, multimask=False)
        scores = [(_click_errors(masks[0], points, labels), _mask_iou(masks[0], before))]
        if scores[0][0] == 0 and scores[0][1] >= 1.0 - margin:
            return logits[0], masks[0]
        multi_logits, _multi_iou, multi_masks = self._decode(
            prepared, points, labels, previous, multimask=True
        )
        candidates = [(logits[0], masks[0])] + list(zip(multi_logits, multi_masks))
        scores += [(_click_errors(m, points, labels), _mask_iou(m, before)) for m in multi_masks]
        best = min(range(len(candidates)), key=lambda i: (scores[i][0], -scores[i][1]))
        if scores[0][0] == scores[best][0] and scores[0][1] >= scores[best][1] - margin:
            best = 0
        return candidates[best]


def _tidy_mask(
    mask: np.ndarray,
    points: Sequence[tuple[int, int]],
    labels: Sequence[int],
    max_hole_area: float,
) -> np.ndarray:
    """Drop stray fragments and fill pinholes, as SAM 2 cleans its masks.

    Upsampled low-res logits leave hundreds of specks around the clicked
    object on seismic. Besides looking noisy, a speckled mask used as a
    tracking prompt gives the memory bank no clear object, and the tracker
    floods the section within a few slices. Components holding a positive
    click are the object; other components stay only when at least a
    quarter of its size (a second part SAM grouped with it). Enclosed holes
    up to ``max_hole_area`` pixels without a negative click are filled.
    """
    height, width = mask.shape

    def at(lab: np.ndarray, x: int, y: int) -> int:
        return int(lab[min(height - 1, max(0, int(y))), min(width - 1, max(0, int(x)))])

    parts, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    clicked = sorted({at(parts, x, y) for (x, y), l in zip(points, labels) if l == 1} - {0})
    if n > 1 and clicked:
        sizes = np.bincount(parts.ravel())
        keep = sizes >= 0.25 * sizes[clicked].sum()
        keep[clicked] = True
        keep[0] = False
        mask = keep[parts]
    holes, n = ndimage.label(~mask)
    if n:
        fill = np.bincount(holes.ravel()) <= max_hole_area
        fill[0] = False
        edges = np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]])
        fill[edges] = False
        for (x, y), l in zip(points, labels):
            if l == 0:
                fill[at(holes, x, y)] = False
        mask = mask | fill[holes]
    return mask


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def _click_errors(
    mask: np.ndarray,
    points: Sequence[tuple[int, int]],
    labels: Sequence[int],
) -> int:
    """Count clicks the mask disobeys: positives outside plus negatives inside."""
    height, width = mask.shape[-2:]
    errors = 0
    for (x, y), label in zip(points, labels):
        inside = bool(mask[min(height - 1, max(0, int(y))), min(width - 1, max(0, int(x)))])
        errors += inside != (int(label) == 1)
    return errors


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
        # (object, slice) pairs whose mask came from the object's own positive
        # clicks; edits there redo the click chain instead of refining tracking.
        self.prompted: set[tuple[int, int]] = set()
        # (object, slice) -> tracked mask logits that slice's edits refine,
        # captured before its first edit so undo can return to it.
        self.bases: dict[tuple[int, int], MaskBase | None] = {}
        self.uid = uuid.uuid4().hex

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


def _install_temporal_mask_selection(model: Any) -> None:
    """Pick each tracked slice's mask by agreement with the slice before it.

    On tracked slices SAM's decoder proposes three masks and the tracker
    keeps the one its IoU head rates highest. That head was trained on
    natural video; on seismic it rates a layer, part of it and most of the
    section nearly alike, so the mask flips between them from slice to slice
    or floods the section. Neighbouring slices of a volume are nearly
    identical, so the candidate overlapping the previous slice's mask most
    is kept instead. The object score still decides whether the object is
    present at all.
    """
    run_frame = model._run_single_frame_inference
    decode = model.mask_decoder.forward
    previous: dict[str, torch.Tensor] = {}

    def run_single_frame_inference(*args: Any, **kwargs: Any) -> Any:
        previous.clear()
        if kwargs.get("point_inputs") is None and kwargs.get("mask_inputs") is None:
            outputs = kwargs["inference_session"].output_dict_per_obj[kwargs["obj_idx"]]
            neighbour = int(kwargs["frame_idx"]) + (1 if kwargs.get("reverse") else -1)
            entry = outputs["non_cond_frame_outputs"].get(neighbour)
            if entry is None:
                entry = outputs["cond_frame_outputs"].get(neighbour)
            if entry is not None:
                logits = entry["pred_masks"]
                previous["mask"] = logits.reshape(-1, *logits.shape[-2:])[0] > 0
        return run_frame(*args, **kwargs)

    def mask_decoder_forward(*args: Any, **kwargs: Any) -> Any:
        masks, iou_scores, tokens, object_scores = decode(*args, **kwargs)
        prior = previous.pop("mask", None)
        if (
            kwargs.get("multimask_output")
            and prior is not None
            and prior.any()
            and masks.shape[:2] == (1, 1)
            and masks.shape[2] > 1
            and tuple(prior.shape) == tuple(masks.shape[-2:])
        ):
            candidates = masks[0, 0] > 0
            prior = prior.to(candidates.device)
            overlap = (candidates & prior).flatten(1).sum(1) / (candidates | prior).flatten(1).sum(
                1
            ).clamp(min=1)
            iou_scores = overlap.to(iou_scores.dtype).reshape(iou_scores.shape)
        return masks, iou_scores, tokens, object_scores

    model._run_single_frame_inference = run_single_frame_inference
    model.mask_decoder.forward = mask_decoder_forward


def _check_object_prompts(
    n_frames: int,
    anchor_idx: int,
    points_per_object: Sequence[Sequence[tuple[int, int]]],
    labels_per_object: Sequence[Sequence[int]],
    frame_indices_per_object: Sequence[Sequence[int]] | None,
    seed_masks_per_object: Sequence[Mapping[int, np.ndarray]] | None,
) -> Sequence[Sequence[int]]:
    """Validate per-object propagation prompts; returns each point's slice."""
    if not points_per_object or len(points_per_object) != len(labels_per_object):
        raise ValueError("need at least one object with matching points/labels")
    if frame_indices_per_object is None:
        frame_indices_per_object = [[anchor_idx] * len(pts) for pts in points_per_object]
    if len(frame_indices_per_object) != len(points_per_object):
        raise ValueError("frame_indices_per_object must match points_per_object")
    if seed_masks_per_object is not None and len(seed_masks_per_object) != len(points_per_object):
        raise ValueError("seed_masks_per_object must match points_per_object")
    for pts, labs, frs in zip(points_per_object, labels_per_object, frame_indices_per_object):
        if len(pts) != len(labs) or len(pts) != len(frs) or not pts:
            raise ValueError("each object needs equal-length, non-empty points/labels/slices")
        if any(not 0 <= int(f) < n_frames for f in frs):
            raise ValueError("point slice index out of range")
        if 1 not in labs:
            raise ValueError(
                "each object needs at least one positive (+) point; "
                "negative-only prompts cannot define an object"
            )
    return frame_indices_per_object


def _keep_as_conditioning(session: Any, obj_ids: Sequence[int], frame_idx: int) -> None:
    """Hold a corrected slice's latest output as a conditioning frame.

    The HF session files inputs on an already-tracked slice as an ordinary
    tracked output: the next sweep re-predicts it and drops the correction,
    and it leaves the memory bank after a few slices. Meta's SAM 2 / SAM 3
    predictors keep every corrected frame as conditioning
    (``add_all_frames_to_correct_as_cond``); this does the same.
    """
    for obj_id in obj_ids:
        outputs = session.output_dict_per_obj[session.obj_id_to_idx(obj_id)]
        entry = outputs["non_cond_frame_outputs"].pop(frame_idx, None)
        if entry is not None:
            outputs["cond_frame_outputs"][frame_idx] = entry


def _install_sam31_feature_provider(tracker: Any) -> None:
    """Let SAM 3.1 tracker states fetch slice features from the caller.

    On a cache miss the native tracker reads ``inference_state["images"]``,
    which only its own video loader fills. States built here carry a
    ``feature_provider`` that encodes the slice into the shared cache first.
    """
    native = tracker._get_image_feature

    def get_image_feature(inference_state: dict, frame_idx: int, batch_size: int) -> Any:
        provider = inference_state.get("feature_provider")
        if provider is not None and frame_idx not in inference_state["cached_features"]:
            provider(frame_idx)
        return native(inference_state, frame_idx, batch_size)

    tracker._get_image_feature = get_image_feature


def _install_sam31_temporal_selection(tracker: Any) -> None:
    """SAM 3.1 version of ``_install_temporal_mask_selection``.

    SAM 3.1's tracking decoder works in multiplexed space, so its three
    candidates are demultiplexed to score them against the neighbouring
    slice's mask, and the scores are multiplexed back. The tracker then
    takes the mask and its object pointer from the same winning token.
    """
    run_frame = tracker._run_single_frame_inference
    decode = tracker.sam_mask_decoder.forward
    previous: dict[str, Any] = {}

    def run_single_frame_inference(*args: Any, **kwargs: Any) -> Any:
        previous.clear()
        tracked = (
            kwargs.get("point_inputs") is None
            and kwargs.get("mask_inputs") is None
            and not kwargs.get("add_to_existing_state")
            and not kwargs.get("reconditioning")
        )
        if tracked and "output_dict" in kwargs and "inference_state" in kwargs:
            outputs = kwargs["output_dict"]
            neighbour = int(kwargs["frame_idx"]) + (1 if kwargs.get("reverse") else -1)
            entry = outputs["non_cond_frame_outputs"].get(neighbour)
            if entry is None:
                entry = outputs["cond_frame_outputs"].get(neighbour)
            multiplex = kwargs["inference_state"].get("multiplex_state")
            if entry is not None and multiplex is not None:
                previous["mask"] = entry["pred_masks"]
                previous["multiplex"] = multiplex
        return run_frame(*args, **kwargs)

    def mask_decoder_forward(*args: Any, **kwargs: Any) -> Any:
        out = decode(*args, **kwargs)
        prior = previous.pop("mask", None)
        multiplex = previous.pop("multiplex", None)
        masks = out["masks"]
        if prior is None or not kwargs.get("multimask_output") or masks.shape[2] < 2:
            return out
        candidates = multiplex.demux(masks.float()) > 0
        prior = prior.to(candidates.device).reshape(candidates.shape[0], -1, *prior.shape[-2:])
        prior = prior[:, 0] > 0
        if prior.shape[-2:] != candidates.shape[-2:]:
            return out
        overlap = (candidates & prior[:, None]).flatten(2).sum(-1) / (
            candidates | prior[:, None]
        ).flatten(2).sum(-1).clamp(min=1)
        native = multiplex.demux(out["iou_pred"].float())
        # Objects absent from the neighbour keep the IoU head's ranking.
        present = prior.flatten(1).any(1)[:, None]
        scores = torch.where(present, overlap.to(native.dtype), native)
        out["iou_pred"] = multiplex.mux(scores).to(out["iou_pred"].dtype)
        return out

    tracker._run_single_frame_inference = run_single_frame_inference
    tracker.sam_mask_decoder.forward = mask_decoder_forward


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
        _install_temporal_mask_selection(self.model)
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
        seed_masks_per_object: Sequence[Mapping[int, np.ndarray]] | None = None,
        refine_mask: Callable[..., np.ndarray] | None = None,
    ) -> None:
        """Track one GPU-sized batch of objects through the volume.

        Points may live on several slices, mirroring SAM2's video
        refinement. Slices where an object has at least one positive
        point anchor the object (conditioning frames): with the preview
        mask from ``seed_masks_per_object`` when given, so the tracker
        starts from exactly what the user saw, else from the clicks.
        Slices with only negative points for an object cannot stand alone -
        segmenting from negatives alone is meaningless - so they are applied
        AFTER the first sweep, when the tracker already has memory of the
        object on that slice; the negative click then subtracts from the
        remembered mask instead of erasing the object. Each correction is
        kept as a conditioning frame and swept outward afterwards.
        """
        # frame -> [(wave_position, points_on_frame, labels_on_frame)]
        anchor_groups: dict[int, list[tuple[int, list, list]]] = {}
        refine_groups: dict[int, list[tuple[int, list, list]]] = {}
        # frame -> [(wave_position, preview mask)]
        seed_groups: dict[int, list[tuple[int, np.ndarray]]] = {}
        for wave_pos, global_idx in enumerate(object_indices):
            seeds = seed_masks_per_object[global_idx] if seed_masks_per_object else {}
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
                seed = seeds.get(frame_idx)
                if seed is not None and seed.any():
                    seed_groups.setdefault(frame_idx, []).append((wave_pos, seed))
                else:
                    target = anchor_groups if 1 in labs else refine_groups
                    target.setdefault(frame_idx, []).append((wave_pos, pts, labs))
                if live is not None and 1 in labs:
                    live.prompted.add((global_idx, frame_idx))

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

        def add_points(frame_idx: int, group: list[tuple[int, list, list]]) -> None:
            self.processor.add_inputs_to_inference_session(
                session,
                frame_idx=frame_idx,
                obj_ids=[wave_pos + 1 for wave_pos, _, _ in group],
                input_points=[[[[x, y] for x, y in pts] for _, pts, _ in group]],
                input_labels=[[labs for _, _, labs in group]],
                original_size=(height, width),
            )

        def condition_frame(frame_idx: int) -> None:
            seeded = seed_groups.get(frame_idx, [])
            clicked = anchor_groups.get(frame_idx, [])
            if seeded:
                self.processor.add_inputs_to_inference_session(
                    session,
                    frame_idx=frame_idx,
                    obj_ids=[wave_pos + 1 for wave_pos, _ in seeded],
                    input_masks=[mask for _, mask in seeded],
                )
            if clicked:
                add_points(frame_idx, clicked)
            if seeded and clicked:
                # Each add replaces the session's to-decode list; decode both.
                session.obj_with_new_inputs = [pos + 1 for pos, _ in seeded] + [
                    pos + 1 for pos, _, _ in clicked
                ]
            record(self.model(inference_session=session, frame_idx=frame_idx))

        def apply_refinement(frame_idx: int, group: list[tuple[int, list, list]]) -> None:
            if refine_mask is None:
                add_points(frame_idx, group)
                record(self.model(inference_session=session, frame_idx=frame_idx))
                _keep_as_conditioning(session, [pos + 1 for pos, _, _ in group], frame_idx)
                return
            for wave_pos, pts, labs in group:
                global_idx = object_indices[wave_pos]
                key = (global_idx, frame_idx)
                base = self._mask_base(
                    session, wave_pos + 1, frame_idx, (live.uid if live else uuid.uuid4().hex, *key)
                )
                if live is not None:
                    live.bases[key] = base
                mask = refine_mask(global_idx, frame_idx, pts, labs, base)
                record(self._condition_on_mask(session, wave_pos + 1, frame_idx, mask))

        session = self._new_session(pixel_values, height, width)
        keep_session = False
        try:
            with self._autocast():
                # Pass 1: condition every anchored slice, then sweep both ways.
                conditioned = sorted(set(seed_groups) | set(anchor_groups))
                for frame_idx in conditioned:
                    condition_frame(frame_idx)
                log.info(
                    "Conditioned %d anchored slice(s) (%d from preview masks); propagating...",
                    len(conditioned),
                    len(seed_groups),
                )
                start_frame = conditioned[0]
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
                        apply_refinement(frame_idx, refine_groups[frame_idx])
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
        seed_masks_per_object: Sequence[Mapping[int, np.ndarray]] | None = None,
        refine_mask: Callable[..., np.ndarray] | None = None,
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
            ``self.live``) so ``pin_mask``/``resweep`` can edit the result
            interactively. Costs host RAM until replaced.
        seed_masks_per_object : per object, {slice: (H, W) bool mask} for
            prompted slices. A seeded slice is conditioned on that mask (the
            preview the user saw) instead of re-decoding its clicks.
        refine_mask : ``(object, slice, points, labels, base) -> mask`` used
            for slices with only negative clicks: it should refine the
            tracked mask logits ``base`` with the clicks, e.g. via
            ``Sam3PointSegmenter.segment(..., base=base)``. Without it those
            clicks go to the tracker directly.

        Returns
        -------
        (n_objects, n_frames, H, W) boolean mask stack.
        """
        n_frames = len(frames)
        frame_indices_per_object = _check_object_prompts(
            n_frames,
            anchor_idx,
            points_per_object,
            labels_per_object,
            frame_indices_per_object,
            seed_masks_per_object,
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
                    seed_masks_per_object,
                    refine_mask,
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

    def _live_slot(self, live: LiveTracker, object_index: int) -> tuple[int, int]:
        slot = live.slots.get(object_index)
        if slot is None:
            raise ValueError(
                f"object {object_index} is not part of the live tracker session; "
                "re-propagate to add it"
            )
        return slot

    @staticmethod
    def _mask_base(
        session: Any,
        obj_id: int,
        frame_idx: int,
        key: Hashable,
        shown: np.ndarray | None = None,
    ) -> MaskBase | None:
        """The session's mask logits for one object on one slice.

        Slices whose memory was cleared by a nearby correction have no
        output until the next sweep; ``shown`` (the mask on screen) then
        stands in, as SAM-scale +/-10 logits.
        """
        outputs = session.output_dict_per_obj[session.obj_id_to_idx(obj_id)]
        entry = outputs["cond_frame_outputs"].get(frame_idx)
        if entry is None:
            entry = outputs["non_cond_frame_outputs"].get(frame_idx)
        if entry is not None:
            logits = entry["pred_masks"]
            return MaskBase(key=key, logits=logits.reshape(-1, *logits.shape[-2:])[0].float().cpu())
        stored = next(iter(outputs["cond_frame_outputs"].values()), None)
        if shown is None or stored is None:
            return None
        cover = F.interpolate(
            torch.from_numpy(np.asarray(shown, dtype=np.float32))[None, None],
            size=tuple(stored["pred_masks"].shape[-2:]),
            mode="area",
        )[0, 0]
        return MaskBase(key=key, logits=torch.where(cover >= 0.5, 10.0, -10.0))

    def _condition_on_mask(self, session: Any, obj_id: int, frame_idx: int, mask: np.ndarray):
        """Make ``mask`` the object's conditioning mask on one slice."""
        outputs = session.output_dict_per_obj[session.obj_id_to_idx(obj_id)]
        # The memory encoder writes into an existing conditioning entry for
        # this slice rather than the new output; drop the old one first.
        outputs["cond_frame_outputs"].pop(frame_idx, None)
        self.processor.add_inputs_to_inference_session(
            session, frame_idx=frame_idx, obj_ids=[obj_id], input_masks=[mask]
        )
        out = self.model(inference_session=session, frame_idx=frame_idx)
        _keep_as_conditioning(session, [obj_id], frame_idx)
        # As Meta's SAM 3 tracker (clear_non_cond_mem_around_input): tracked
        # memories next to a correction still show the old object and would
        # outvote it on the next sweep.
        reach = int(getattr(getattr(self.model, "config", None), "num_maskmem", 7))
        for t in range(frame_idx - reach, frame_idx + reach + 1):
            outputs["non_cond_frame_outputs"].pop(t, None)
        return out

    def refine_base(self, live: LiveTracker, object_index: int, frame_idx: int) -> MaskBase | None:
        """Mask logits that clicks on this slice should refine.

        ``None`` on slices the object was prompted on with positive clicks:
        their mask is the click chain itself. Elsewhere, the tracked mask as
        it was before the slice's first edit, so the clicks refine what the
        user saw and removing them all returns to it.
        """
        key = (int(object_index), int(frame_idx))
        if key not in live.bases:
            if key in live.prompted:
                live.bases[key] = None
            else:
                session_idx, obj_id = self._live_slot(live, object_index)
                live.bases[key] = self._mask_base(
                    live.sessions[session_idx],
                    obj_id,
                    int(frame_idx),
                    (live.uid, *key),
                    shown=live.masks[object_index, int(frame_idx)],
                )
        return live.bases[key]

    @torch.no_grad()
    def pin_mask(
        self,
        live: LiveTracker,
        object_index: int,
        frame_idx: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Set one object's mask on a slice and keep it as a conditioning frame.

        This is SAM 2's frame correction: the slice then anchors tracking like
        the original prompts, so later sweeps propagate it and never overwrite
        it. Returns the (n_objects, H, W) stack for the slice.
        """
        session_idx, obj_id = self._live_slot(live, object_index)
        session = live.sessions[session_idx]
        frame_idx = int(frame_idx)
        started = time.perf_counter()
        with self._autocast():
            out = self._condition_on_mask(session, obj_id, frame_idx, np.asarray(mask, dtype=bool))
        rows = self._to_masks(out.pred_masks, live.height, live.width)
        by_session_id = live.objects_in(session_idx)
        for row, out_id in zip(rows, out.object_ids):
            global_idx = by_session_id.get(int(out_id))
            if global_idx is not None:
                live.masks[global_idx, frame_idx] = row
        live.dirty_frames.add(frame_idx)
        self.last_timings["refine_frame"] = time.perf_counter() - started
        log.info(
            "Pinned object %d on slice %d in %.2fs (coverage %.2f%%)",
            object_index + 1,
            frame_idx,
            self.last_timings["refine_frame"],
            100.0 * float(live.masks[object_index, frame_idx].mean()),
        )
        return live.masks[:, frame_idx]

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

    @torch.no_grad()
    def propagate_from_masks(
        self,
        frames: Sequence[np.ndarray],
        seed_idx: int,
        seed_masks: np.ndarray,
        reverse: bool,
        progress: Callable[[int, np.ndarray], None] | None = None,
    ) -> None:
        """Continue a propagation through a further chunk of slices.

        Volumes too long to track in one session are covered chunk by
        chunk: the mask the previous chunk produced on the shared boundary
        slice ``seed_idx`` conditions this chunk, which is then swept in
        one direction only. ``progress(frame_idx, masks)`` receives every
        slice's (n_objects, H, W) masks as it is tracked; nothing is kept
        live. Objects whose seed mask is empty stay empty.
        """
        n_objects = int(seed_masks.shape[0])
        n_frames = len(frames)
        h, w = frames[0].shape[:2]
        t0 = time.perf_counter()
        processed = self.processor.video_processor(
            videos=frames, device="cpu", return_tensors="pt"
        )
        pixel_values = processed.pixel_values_videos[0]
        del frames, processed
        order = range(seed_idx, -1, -1) if reverse else range(seed_idx, n_frames)
        present = [i for i in range(n_objects) if seed_masks[i].any()]
        masks = np.zeros((n_objects, n_frames, h, w), dtype=bool)
        batch_limit = max(1, min(len(present) or 1, self._estimate_max_objects()))
        waves = [present[i : i + batch_limit] for i in range(0, len(present), batch_limit)]
        log.info(
            "Continuing propagation %s through %d slices from a mask seed "
            "(%d of %d object(s) still present, %d wave(s))",
            "backward" if reverse else "forward",
            n_frames,
            len(present),
            n_objects,
            len(waves),
        )
        if not waves:
            if progress is not None:
                for frame_idx in order:
                    progress(frame_idx, masks[:, frame_idx])
            return
        for wave_no, wave in enumerate(waves):
            last_wave = wave_no == len(waves) - 1
            session = self._new_session(pixel_values, h, w)
            try:
                with self._autocast():
                    self.processor.add_inputs_to_inference_session(
                        session,
                        frame_idx=int(seed_idx),
                        obj_ids=[pos + 1 for pos in range(len(wave))],
                        input_masks=[seed_masks[g] for g in wave],
                    )
                    self.model(inference_session=session, frame_idx=int(seed_idx))
                    for out in self.model.propagate_in_video_iterator(
                        session, start_frame_idx=int(seed_idx), reverse=reverse
                    ):
                        rows = self._to_masks(out.pred_masks, h, w)
                        for row, obj_id in zip(rows, out.object_ids):
                            masks[wave[int(obj_id) - 1], out.frame_idx] = row
                        if last_wave and progress is not None:
                            progress(int(out.frame_idx), masks[:, out.frame_idx])
            finally:
                del session
                self._free_cuda()
        elapsed = time.perf_counter() - t0
        log.info(
            "Chunk tracked in %.1fs (%.2f slices/s, coverage %.2f%%)",
            elapsed,
            n_frames / elapsed if elapsed else float("inf"),
            100.0 * float(masks.any(axis=0).mean()),
        )

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


class _Sam31Encoder:
    """SAM 3.1's vision backbone, producing the features its tracker reads.

    Sections are stretched to the square model input exactly as SAM 3.1's
    own video loader (and SAM 3's processor) do, so clicks and tracking
    see the same pixels.
    """

    def __init__(self, predictor: Any):
        from sam3.model.data_misc import NestedTensor

        self.tracker = predictor.model.tracker.model
        self.backbone = predictor.model.detector.backbone
        self.size = int(self.tracker.image_size)
        self._nested = NestedTensor

    def encode(self, rgb: np.ndarray) -> tuple[torch.Tensor, dict]:
        """``(image, backbone_out)`` as SAM 3.1 tracker states cache them.

        Run under bf16 autocast, as SAM 3.1's predictor does.
        """
        size = self.size
        pixels = np.asarray(Image.fromarray(rgb).convert("RGB").resize((size, size)), dtype=np.float32)
        image = torch.from_numpy(pixels).to("cuda").permute(2, 0, 1)[None]
        image = (image / 255.0 - 0.5) / 0.5
        out = self.backbone.forward_image(
            image, need_sam3_out=False, need_interactive_out=True, need_propagation_out=True
        )
        features = {}
        for key, decoder in (
            ("interactive", self.tracker.interactive_sam_mask_decoder),
            ("sam2_backbone_out", self.tracker.sam_mask_decoder),
        ):
            fpn = [level.tensors.to(torch.bfloat16) for level in out[key]["backbone_fpn"]]
            # The tracker expects the high-res levels already projected by each decoder.
            fpn = [decoder.conv_s0(fpn[0]), decoder.conv_s1(fpn[1]), fpn[2]]
            features[key] = {
                "vision_features": fpn[-1],
                "vision_mask": None,
                "vision_pos_enc": out[key]["vision_pos_enc"],
                "backbone_fpn": [self._nested(level, None) for level in fpn],
            }
        return image, features


class Sam31PointSegmenter(Sam3PointSegmenter):
    """Click segmentation with SAM 3.1's interactive decoder.

    Click handling is SAM 3's: one click per decode on top of the previous
    mask's logits, click-consistent candidate choice, local edits of
    tracked masks and mask tidying. Only the encoder and decoder are
    SAM 3.1's.
    """

    family = "sam31"
    checkpoint = "facebook/sam3.1"

    def __init__(self, predictor: Any, embedding_cache_size: int = 4):
        self.predictor = predictor
        self.model = predictor.model
        self.processor = None
        self.device = "cuda"
        self.encoder = _Sam31Encoder(predictor)
        self._init_caches(embedding_cache_size)

    @torch.no_grad()
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

        self._sync_cuda()
        started = time.perf_counter()
        tracker = self.encoder.tracker
        with self._autocast():
            _image, backbone_out = self.encoder.encode(rgb)
            feats = tracker._prepare_backbone_features(backbone_out)["interactive"]
            pix = tracker._get_interactive_pix_mem(feats["vision_feats"], feats["feat_sizes"])
            high_res = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(feats["vision_feats"][:-1], feats["feat_sizes"][:-1])
            ]
        self._sync_cuda()
        elapsed = time.perf_counter() - started
        self._prepared[key] = {"pix": pix, "high_res": high_res, "shape": rgb.shape[:2]}
        self._prepared.move_to_end(key)
        while len(self._prepared) > self.embedding_cache_size:
            self._prepared.popitem(last=False)
        self.last_timings["prepare_image"] = elapsed
        log.info("SAM 3.1 image features ready in %.3fs (%d cached)", elapsed, len(self._prepared))
        return key

    def _decode(
        self,
        prepared: dict,
        points: Sequence[tuple[int, int]],
        labels: Sequence[int],
        prior: torch.Tensor | None,
        multimask: bool,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        height, width = prepared["shape"]
        size = self.encoder.size
        coords = torch.tensor(
            [[[float(x) * size / width, float(y) * size / height] for x, y in points]],
            device=self.device,
        )
        point_labels = torch.tensor([[int(l) for l in labels]], dtype=torch.int32, device=self.device)
        mask_inputs = None
        if prior is not None:
            mask_inputs = torch.clamp(prior.float(), -32.0, 32.0)[None, None].to(self.device)
        with self._autocast():
            out = self.encoder.tracker._forward_sam_heads(
                backbone_features=prepared["pix"],
                point_inputs={"point_coords": coords, "point_labels": point_labels},
                mask_inputs=mask_inputs,
                interactive_high_res_features=prepared["high_res"],
                multimask_output=multimask,
                objects_to_interact=[0],
                multiplex_state=None,
            )
        logits = out["low_res_multimasks"][0].float()
        masks = self._upsample(logits, (height, width))
        return logits.cpu(), out["ious"][0].float().cpu().numpy(), masks

    @staticmethod
    def _upsample(logits: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
        """(n, h, w) low-res logits -> (n, H, W) boolean masks."""
        up = F.interpolate(logits[:, None].float(), size=tuple(shape), mode="bilinear", align_corners=False)
        return np.asarray((up[:, 0] > 0).cpu().numpy(), dtype=bool)

    def _mask_from_logits(self, prepared: dict, logits: torch.Tensor) -> np.ndarray:
        return self._upsample(logits[None].to(self.device), prepared["shape"])[0]

    def close(self) -> None:
        self._prepared.clear()
        self._chains.clear()


class Sam31LiveTracker(LiveTracker):
    """Live SAM 3.1 tracking: one tracker state per object plus its inputs.

    ``conditioning[obj]`` holds every mask the object is conditioned on
    (preview masks and pinned corrections) so a re-sweep can rebuild the
    object's state from them.
    """

    def __init__(self, frames: Sequence[np.ndarray], n_objects: int):
        height, width = frames[0].shape[:2]
        super().__init__(n_objects, len(frames), height, width)
        self.frames = list(frames)
        self.sessions = [None] * n_objects
        self.conditioning: list[dict[int, np.ndarray]] = [{} for _ in range(n_objects)]
        self.dirty_objects: set[int] = set()

    def close(self) -> None:
        super().close()
        self.frames = []


class Sam31VolumePropagator:
    """Volume tracking with SAM 3.1's tracker, driven like ``Sam3VolumePropagator``.

    Each object gets its own tracker state, as in SAM 3.1's interactive
    demo, so objects never compete for pixels (one multiplexed state makes
    them non-overlapping). States are seeded with the preview masks and
    tracked with temporal mask selection. Corrections become conditioning
    masks; a re-sweep rebuilds the edited objects' states from all of them.
    """

    family = "sam31"
    checkpoint = "facebook/sam3.1"
    device = "cuda"
    session_dtype = torch.bfloat16
    # Host RAM per object and tracked slice: mask memory, image features
    # and their position encoding, all kept for the memory attention.
    state_bytes_per_object_frame = 8 * 1024**2
    presence_threshold = -1000.0

    def __init__(self, predictor: Any, segmenter: Sam31PointSegmenter | None = None):
        self.predictor = predictor
        self.model = predictor.model
        self.processor = None
        self.segmenter = segmenter if segmenter is not None else Sam31PointSegmenter(predictor, 1)
        self.encoder = self.segmenter.encoder
        self.tracker = self.encoder.tracker
        # SAM 3.1's presence head drifts negative as a reflector changes
        # across slices; once gated, "absent" enters memory and the object
        # never recovers. The mask logits already fall to empty where an
        # event fades, so presence is left to them (Meta's pipeline instead
        # re-anchors objects on detector masks, which clicks don't have).
        self.tracker.object_score_logit_threshold = self.presence_threshold
        self.live: Sam31LiveTracker | None = None
        self.last_timings: dict[str, float] = {}

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _features(self, frames: Sequence[np.ndarray]) -> tuple[dict, Callable[[int], None]]:
        """A feature cache shared by every object's state, and its provider."""
        cache: dict = {}

        def provide(frame_idx: int) -> None:
            # States track the same slice in lockstep, so a few slices suffice.
            cache[frame_idx] = self.encoder.encode(frames[frame_idx])
            while len(cache) > 3:
                cache.pop(next(iter(cache)))

        return cache, provide

    def _build_state(
        self,
        frames: Sequence[np.ndarray],
        cache: dict,
        provide: Callable[[int], None],
        conditioning: Mapping[int, np.ndarray],
    ) -> dict:
        height, width = frames[0].shape[:2]
        state = self.tracker.init_state(
            video_height=height,
            video_width=width,
            num_frames=len(frames),
            cached_features=cache,
            offload_state_to_cpu=True,
        )
        state["feature_provider"] = provide
        for frame_idx, mask in sorted(conditioning.items()):
            # Given a multiplex state, SAM 3.1 adds the mask as a new object
            # joining a bucket. A fresh one (it is deterministic) keeps every
            # mask of the object on the plain mask-prompt path.
            multiplex = state["multiplex_state"]
            state["multiplex_state"] = None
            self.tracker.add_new_masks(
                state, int(frame_idx), [1], torch.from_numpy(np.asarray(mask, dtype=bool))[None]
            )
            if multiplex is not None:
                state["multiplex_state"] = multiplex
        self.tracker.propagate_in_video_preflight(state, run_mem_encoder=True)
        return state

    def _sweep(
        self,
        states: Mapping[int, dict],
        conditioning: Sequence[Mapping[int, np.ndarray]],
        sweeps: Sequence[tuple[bool, int]],
        masks: np.ndarray,
        on_frame: Callable[[int], None],
    ) -> None:
        """Track every state through ``sweeps`` of (reverse, start) in lockstep."""
        objects = list(states)
        for reverse, start in sweeps:
            runs = [
                self.tracker.propagate_in_video(
                    states[g], int(start), None, reverse, tqdm_disable=True
                )
                for g in objects
            ]
            for outputs in zip(*runs):
                frame_idx = int(outputs[0][0])
                for g, (_frame, _ids, _low_res, video_res, _scores) in zip(objects, outputs):
                    # Conditioning slices keep their mask as given; SAM 3.1
                    # stores it only at the decoder's 288x288 resolution.
                    given = conditioning[g].get(frame_idx)
                    masks[g, frame_idx] = (
                        given if given is not None else (video_res[0, 0] > 0).cpu().numpy()
                    )
                on_frame(frame_idx)

    def _retrack(
        self,
        live: Sam31LiveTracker,
        objects: Sequence[int],
        sweeps: Sequence[tuple[bool, int]],
        on_frame: Callable[[int], None],
    ) -> None:
        """Rebuild ``objects``' states from their conditioning masks and track them."""
        objects = [g for g in objects if live.conditioning[g]]
        for g in objects:
            live.sessions[g] = None
        self._free_cuda()
        cache, provide = self._features(live.frames)
        try:
            with self._autocast():
                for g in objects:
                    live.sessions[g] = self._build_state(live.frames, cache, provide, live.conditioning[g])
                self._sweep(
                    {g: live.sessions[g] for g in objects}, live.conditioning, sweeps, live.masks, on_frame
                )
        finally:
            cache.clear()
            self._free_cuda()

    @staticmethod
    def _free_cuda() -> None:
        gc.collect()
        torch.cuda.empty_cache()

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
        seed_masks_per_object: Sequence[Mapping[int, np.ndarray]] | None = None,
        refine_mask: Callable[..., np.ndarray] | None = None,
    ) -> np.ndarray:
        """Same contract as ``Sam3VolumePropagator.propagate``.

        Positively clicked slices without a preview mask are segmented from
        their clicks with the point segmenter first.
        """
        frames = list(frames)
        n_frames = len(frames)
        frame_indices_per_object = _check_object_prompts(
            n_frames,
            anchor_idx,
            points_per_object,
            labels_per_object,
            frame_indices_per_object,
            seed_masks_per_object,
        )
        if self.live is not None:
            self.live.close()
            self.live = None

        t0 = time.perf_counter()
        n_objects = len(points_per_object)
        live = Sam31LiveTracker(frames, n_objects)
        del frames
        refinements: list[tuple[int, int, list, list]] = []
        for g in range(n_objects):
            seeds = seed_masks_per_object[g] if seed_masks_per_object else {}
            per_frame: dict[int, tuple[list, list]] = {}
            for (x, y), label, frame_idx in zip(
                points_per_object[g], labels_per_object[g], frame_indices_per_object[g]
            ):
                bucket = per_frame.setdefault(int(frame_idx), ([], []))
                bucket[0].append((int(x), int(y)))
                bucket[1].append(int(label))
            for frame_idx, (pts, labs) in sorted(per_frame.items()):
                if 1 not in labs:
                    refinements.append((g, frame_idx, pts, labs))
                    continue
                live.prompted.add((g, frame_idx))
                seed = seeds.get(frame_idx)
                if seed is None or not seed.any():
                    seed = self.segmenter.segment(
                        live.frames[frame_idx], pts, labs, image_key=(live.uid, frame_idx), object_id=g
                    )
                live.conditioning[g][frame_idx] = np.asarray(seed, dtype=bool)
        self.last_timings["session_init"] = time.perf_counter() - t0

        conditioned = sorted({f for cond in live.conditioning for f in cond})
        log.info(
            "SAM 3.1 volume propagation: %d slices of %dx%d, %d object(s) conditioned "
            "on slice(s) %s",
            n_frames,
            live.width,
            live.height,
            n_objects,
            conditioned,
        )
        visited: set[int] = set()

        def on_frame(frame_idx: int) -> None:
            visited.add(frame_idx)
            if len(visited) == n_frames or len(visited) % max(1, n_frames // 10) == 0:
                log.info("  slice %d/%d done", len(visited), n_frames)
            if progress is not None:
                progress(len(visited), n_frames, frame_idx, live.masks[:, frame_idx])

        try:
            if conditioned:
                start = conditioned[0]
                self._retrack(live, range(n_objects), ((False, start), (True, start)), on_frame)
            if refinements:
                # Negative-only slices refine the tracked mask there, then
                # anchor the objects like a correction would.
                log.info(
                    "Applying negative-only refinements on slice(s) %s and re-sweeping...",
                    sorted({f for _, f, _, _ in refinements}),
                )
                for g, frame_idx, pts, labs in refinements:
                    base = self._mask_base(live, g, frame_idx)
                    live.bases[(g, frame_idx)] = base
                    if refine_mask is not None:
                        mask = refine_mask(g, frame_idx, pts, labs, base)
                    else:
                        mask = self.segmenter.segment(
                            live.frames[frame_idx],
                            pts,
                            labs,
                            image_key=(live.uid, frame_idx),
                            object_id=g,
                            base=base,
                        )
                    live.conditioning[g][frame_idx] = np.asarray(mask, dtype=bool)
                    live.masks[g, frame_idx] = live.conditioning[g][frame_idx]
                refined = sorted({f for _, f, _, _ in refinements})
                self._retrack(
                    live,
                    sorted({g for g, _, _, _ in refinements}),
                    ((False, refined[0]), (True, refined[-1])),
                    on_frame,
                )
        except Exception:
            live.close()
            raise

        elapsed = time.perf_counter() - t0
        self.last_timings.update(
            inference=elapsed - self.last_timings["session_init"],
            total=elapsed,
            fps=n_frames / elapsed if elapsed else float("inf"),
            object_waves=1.0,
            objects_per_wave=float(n_objects),
        )
        log.info(
            "SAM 3.1 volume propagation finished in %.1fs (total coverage %.2f%%)",
            elapsed,
            100.0 * float(live.masks.any(axis=0).mean()),
        )
        if keep_live:
            self.live = live
            return live.masks
        masks = live.masks
        live.close()
        return masks

    def _mask_base(
        self,
        live: Sam31LiveTracker,
        object_index: int,
        frame_idx: int,
        shown: np.ndarray | None = None,
    ) -> MaskBase | None:
        """The tracker's mask logits for one object on one slice (see SAM 3's)."""
        key = (live.uid, int(object_index), int(frame_idx))
        state = live.sessions[object_index]
        if state is not None:
            outputs = state["output_dict"]
            entry = outputs["cond_frame_outputs"].get(frame_idx)
            if entry is None:
                entry = outputs["non_cond_frame_outputs"].get(frame_idx)
            if entry is not None:
                logits = entry["pred_masks"]
                return MaskBase(key=key, logits=logits.reshape(-1, *logits.shape[-2:])[0].float().cpu())
        if shown is None:
            return None
        size = int(self.tracker.low_res_mask_size)
        cover = F.interpolate(
            torch.from_numpy(np.asarray(shown, dtype=np.float32))[None, None],
            size=(size, size),
            mode="area",
        )[0, 0]
        return MaskBase(key=key, logits=torch.where(cover >= 0.5, 10.0, -10.0))

    def refine_base(self, live: Sam31LiveTracker, object_index: int, frame_idx: int) -> MaskBase | None:
        """Mask logits that clicks on this slice refine (see ``Sam3VolumePropagator``)."""
        key = (int(object_index), int(frame_idx))
        if key not in live.bases:
            live.bases[key] = (
                None
                if key in live.prompted
                else self._mask_base(live, *key, shown=live.masks[key[0], key[1]])
            )
        return live.bases[key]

    def pin_mask(
        self,
        live: Sam31LiveTracker,
        object_index: int,
        frame_idx: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Set one object's mask on a slice and keep it as a conditioning mask.

        The next re-sweep tracks from it. Returns the (n_objects, H, W)
        stack for the slice.
        """
        object_index, frame_idx = int(object_index), int(frame_idx)
        mask = np.asarray(mask, dtype=bool)
        live.conditioning[object_index][frame_idx] = mask
        live.masks[object_index, frame_idx] = mask
        live.dirty_frames.add(frame_idx)
        live.dirty_objects.add(object_index)
        self.last_timings["refine_frame"] = 0.0
        log.info(
            "Pinned object %d on slice %d (coverage %.2f%%)",
            object_index + 1,
            frame_idx,
            100.0 * float(mask.mean()),
        )
        return live.masks[:, frame_idx]

    @torch.no_grad()
    def resweep(
        self,
        live: Sam31LiveTracker,
        progress: Callable[[int, int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        """Re-track the edited objects outward from the slices edited since the last sweep."""
        edited = sorted(live.dirty_frames)
        if not edited:
            raise ValueError("nothing was edited since the last propagation")
        objects = sorted(live.dirty_objects) or list(range(live.n_objects))
        t0 = time.perf_counter()
        log.info(
            "Re-sweeping object(s) %s from edit(s) on slice(s) %s...",
            [g + 1 for g in objects],
            edited,
        )
        visited: set[int] = set()

        def on_frame(frame_idx: int) -> None:
            visited.add(frame_idx)
            if progress is not None:
                progress(len(visited), live.n_frames, frame_idx, live.masks[:, frame_idx])

        self._retrack(live, objects, ((False, edited[0]), (True, edited[-1])), on_frame)
        live.dirty_frames.clear()
        live.dirty_objects.clear()
        elapsed = time.perf_counter() - t0
        self.last_timings.update(
            total=elapsed, fps=live.n_frames / elapsed if elapsed else float("inf")
        )
        log.info(
            "Re-sweep finished in %.1fs (total coverage %.2f%%)",
            elapsed,
            100.0 * float(live.masks.any(axis=0).mean()),
        )
        return live.masks

    @torch.no_grad()
    def propagate_from_masks(
        self,
        frames: Sequence[np.ndarray],
        seed_idx: int,
        seed_masks: np.ndarray,
        reverse: bool,
        progress: Callable[[int, np.ndarray], None] | None = None,
    ) -> None:
        """Continue a propagation through a further chunk of slices (see SAM 3's)."""
        frames = list(frames)
        n_objects, n_frames = int(seed_masks.shape[0]), len(frames)
        height, width = frames[0].shape[:2]
        t0 = time.perf_counter()
        masks = np.zeros((n_objects, n_frames, height, width), dtype=bool)
        present = [g for g in range(n_objects) if seed_masks[g].any()]
        conditioning = [
            {int(seed_idx): np.asarray(seed_masks[g], dtype=bool)} if g in present else {}
            for g in range(n_objects)
        ]
        log.info(
            "Continuing SAM 3.1 propagation %s through %d slices from a mask seed "
            "(%d of %d object(s) still present)",
            "backward" if reverse else "forward",
            n_frames,
            len(present),
            n_objects,
        )

        def on_frame(frame_idx: int) -> None:
            if progress is not None:
                progress(frame_idx, masks[:, frame_idx])

        if not present:
            for frame_idx in (range(seed_idx, -1, -1) if reverse else range(seed_idx, n_frames)):
                on_frame(frame_idx)
            return
        cache, provide = self._features(frames)
        try:
            with self._autocast():
                states = {g: self._build_state(frames, cache, provide, conditioning[g]) for g in present}
                self._sweep(states, conditioning, ((reverse, int(seed_idx)),), masks, on_frame)
        finally:
            states = None
            cache.clear()
            self._free_cuda()
        elapsed = time.perf_counter() - t0
        log.info(
            "Chunk tracked in %.1fs (%.2f slices/s, coverage %.2f%%)",
            elapsed,
            n_frames / elapsed if elapsed else float("inf"),
            100.0 * float(masks.any(axis=0).mean()),
        )

    def close(self) -> None:
        if self.live is not None:
            self.live.close()
            self.live = None
