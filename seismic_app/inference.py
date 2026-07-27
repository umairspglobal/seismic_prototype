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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config


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

        self.model = Sam3Model.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(checkpoint)

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
    ):
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # deferred

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has no CUDA-enabled "
                "PyTorch build. Install a CUDA PyTorch wheel in the active environment "
                "or select CPU."
            )
        self.model = Sam3TrackerModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.processor = Sam3TrackerProcessor.from_pretrained(checkpoint)

    @torch.no_grad()
    def segment(
        self,
        rgb: np.ndarray,
        points: list[tuple[int, int]],
        labels: list[int],
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

        image = Image.fromarray(rgb)
        # 4D: (image, object, point, xy) - one image, one object.
        input_points = [[[[float(x), float(y)] for x, y in points]]]
        input_labels = [[[int(l) for l in labels]]]

        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        # post_process_masks -> list per image of (n_objects, n_masks, H, W).
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        iou = outputs.iou_scores.cpu().numpy().reshape(-1)
        best = int(iou.argmax())
        return np.asarray(masks[0, best], dtype=bool)
