"""Quick SAM 3 fine-tuning for one text prompt ("fault", "seismic facies", ...).

Trains the part of ``transformers.Sam3Model`` that produces ``semantic_seg``
- the per-prompt probability map the app's auto-segment path thresholds -
against binary masks. The 1008-px ViT backbone and the text encoder stay
frozen (optionally the last few ViT blocks are unfrozen), the prompt is
encoded once, and only the DETR encoder / mask decoder (+ optional FPN neck)
receive gradients. That keeps a run to minutes instead of the official
trainer's hours per epoch, and the result is written with
``save_pretrained`` so it drops straight into ``finetuned_checkpoints/``
where the app lists it.

Run directly or through ``finetuning/finetune_gui.py``. Lines starting with
``@@`` are JSON progress events for the GUI; everything else is a log line.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from finetuning.quick_data import (  # noqa: E402
    LABEL_MODES,
    SectionSet,
    build_mosaic,
    fixed_mosaics,
    list_pairs,
    load_sections,
    split_stems,
)

EVENT_PREFIX = "@@"
MODEL_SIDE = 1008
TRAIN_SCOPES = {
    "heads": "mask decoder only",
    "heads_encoder": "mask decoder + DETR encoder",
    "heads_encoder_neck": "mask decoder + DETR encoder + FPN neck",
}
VIT_LR_SCALE = 0.1
MAX_BAD_STEPS = 10

log = logging.getLogger("quick_finetune")


def emit(kind: str, **data) -> None:
    print(EVENT_PREFIX + json.dumps({"type": kind, **data}), flush=True)


def setup_logging(log_file: Path | None) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    log.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.propagate = False


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# --- Model pieces -----------------------------------------------------------


class Sam3SemanticTrainer:
    """Wraps a Sam3Model so only the semantic-seg path is trained."""

    def __init__(self, args: argparse.Namespace, device: str):
        import torch
        from transformers import Sam3Model, Sam3Processor
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        self.torch = torch
        self.device = device
        self.args = args

        log.info("Loading base model '%s' (about 3.4 GB; first load can take a minute)...", args.base_model)
        t0 = time.perf_counter()
        self.processor = Sam3Processor.from_pretrained(args.base_model)
        self.model = Sam3Model.from_pretrained(args.base_model, dtype=torch.float32)
        log.info("Base model loaded in %.1fs", time.perf_counter() - t0)

        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)

        self._encode_prompt(args.prompt)
        # The prompt is cached, so the 350M-parameter text encoder can leave the GPU.
        self.model.text_encoder.to("cpu")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        self.vit = self.model.vision_encoder.backbone
        n_blocks = len(self.vit.layers)
        self.n_unfrozen = max(0, min(int(args.unfreeze_vit_blocks), n_blocks))
        self.split = n_blocks - self.n_unfrozen
        self.train_neck = args.train_scope == "heads_encoder_neck"

        self.trainable_modules = [self.model.mask_decoder]
        if args.train_scope in ("heads_encoder", "heads_encoder_neck"):
            self.trainable_modules.append(self.model.detr_encoder)
        if self.train_neck:
            self.trainable_modules.append(self.model.vision_encoder.neck)
        self.vit_trainable = list(self.vit.layers[self.split :])

        head_params, vit_params = [], []
        for module in self.trainable_modules:
            for p in module.parameters():
                p.requires_grad_(True)
                head_params.append(p)
        for module in self.vit_trainable:
            for p in module.parameters():
                p.requires_grad_(True)
                vit_params.append(p)

        groups = [{"params": head_params, "lr": args.lr, "base_lr": args.lr}]
        if vit_params:
            groups.append(
                {"params": vit_params, "lr": args.lr * VIT_LR_SCALE, "base_lr": args.lr * VIT_LR_SCALE}
            )
        self.optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

        n_train = sum(p.numel() for p in head_params) + sum(p.numel() for p in vit_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        log.info(
            "Training %s%s: %.1fM of %.0fM parameters (%.1f%%)",
            TRAIN_SCOPES[args.train_scope],
            f" + last {self.n_unfrozen} of {n_blocks} ViT blocks" if self.n_unfrozen else "",
            n_train / 1e6,
            n_total / 1e6,
            100.0 * n_train / n_total,
        )

    def _encode_prompt(self, prompt: str) -> None:
        torch = self.torch
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        self._pooled_cls = BaseModelOutputWithPooling
        text = self.processor(text=prompt, return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        self.text_pooled = feats.pooler_output.detach()
        self.text_mask = attention_mask
        log.info("Prompt '%s' encoded once (%d tokens) and cached", prompt, int(attention_mask.sum()))

    def set_train_mode(self, training: bool) -> None:
        self.model.eval()
        if training:
            for module in self.trainable_modules + self.vit_trainable:
                module.train()

    def trainable_state(self) -> dict:
        return {
            name: p.detach().to("cpu", copy=True)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

    def load_trainable_state(self, state: dict) -> None:
        params = dict(self.model.named_parameters())
        with self.torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value.to(params[name].device))

    def pixel_values(self, images: list[np.ndarray]):
        """uint8 (H, W) sections -> normalized (B, 3, 1008, 1008), as Sam3Processor does."""
        torch = self.torch
        x = torch.from_numpy(np.stack(images)).to(self.device, non_blocking=True)
        x = x.float().div_(255.0).unsqueeze(1)
        x = torch.nn.functional.interpolate(
            x, size=(MODEL_SIDE, MODEL_SIDE), mode="bilinear", align_corners=False, antialias=True
        )
        x = (x - 0.5) / 0.5
        return x.expand(-1, 3, -1, -1).contiguous()

    def _vision(self, pixel_values, training: bool):
        torch = self.torch
        from torch.utils.checkpoint import checkpoint
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        vit = self.vit
        b = pixel_values.shape[0]
        side = MODEL_SIDE // vit.config.patch_size
        with torch.no_grad():
            h = vit.embeddings(pixel_values).view(b, side, side, -1)
            h = vit.layer_norm(h)
            for layer in vit.layers[: self.split]:
                h = layer(h)
        for layer in vit.layers[self.split :]:
            h = checkpoint(layer, h, use_reentrant=False) if training else layer(h)
        seq = h.reshape(b, side * side, -1)
        spatial = h.permute(0, 3, 1, 2)
        with nullcontext() if (self.train_neck and training) else torch.no_grad():
            fpn, pos = self.model.vision_encoder.neck(spatial)
        return Sam3VisionEncoderOutput(
            last_hidden_state=seq, fpn_hidden_states=fpn, fpn_position_encoding=pos
        )

    def semantic_logits(self, pixel_values, training: bool):
        b = pixel_values.shape[0]
        vision = self._vision(pixel_values, training)
        text = self._pooled_cls(pooler_output=self.text_pooled.expand(b, -1, -1))
        out = self.model(
            vision_embeds=vision,
            text_embeds=text,
            attention_mask=self.text_mask.expand(b, -1),
        )
        return out.semantic_seg

    def save(self, out_dir: Path) -> None:
        torch = self.torch
        self.model.to("cpu")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.save_pretrained(out_dir)
        self.processor.save_pretrained(out_dir)


# --- Loss and metrics -------------------------------------------------------


def seg_loss(torch, logits, target, pos_weight: float):
    F = torch.nn.functional
    logits = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device)
    )
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return bce + dice.mean(), logits


def summarize(tp: float, fp: float, fn: float) -> dict:
    iou = tp / max(tp + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def overlay(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    tint = np.array(color, dtype=np.float32)
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * tint
    return rgb.astype(np.uint8)


def save_preview(path: Path, samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], title: str) -> None:
    from PIL import Image, ImageDraw

    rows = []
    for gray, truth, pred in samples:
        rows.append(
            np.concatenate(
                [
                    np.repeat(gray[..., None], 3, axis=-1),
                    overlay(gray, truth, (0, 200, 255)),
                    overlay(gray, pred, (255, 60, 0)),
                ],
                axis=1,
            )
        )
    grid = np.concatenate(rows, axis=0)
    scale = max(1, 900 // max(1, 3 * samples[0][0].shape[1]))
    img = Image.fromarray(grid).resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    banner = Image.new("RGB", (img.width, 22), (30, 30, 30))
    ImageDraw.Draw(banner).text((6, 5), f"{title}    seismic | label (blue) | prediction (orange)", fill=(230, 230, 230))
    canvas = Image.new("RGB", (img.width, img.height + 22))
    canvas.paste(banner, (0, 0))
    canvas.paste(img, (0, 22))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# --- Training ---------------------------------------------------------------


def evaluate(trainer: Sam3SemanticTrainer, val_mosaics, args, pos_weight: float, tile_hw: tuple[int, int]):
    """Global-pixel IoU/F1 at the app's 0.5 threshold, plus a few single-section previews."""
    torch = trainer.torch
    trainer.set_train_mode(False)
    th, tw = tile_hw
    grid = val_mosaics[0][0].shape[0] // th if val_mosaics else 1
    tp = fp = fn = 0.0
    losses = []
    previews = []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if trainer.device.startswith("cuda") else nullcontext()
    with torch.no_grad():
        for start in range(0, len(val_mosaics), args.batch_size):
            chunk = val_mosaics[start : start + args.batch_size]
            images = [img for img, _ in chunk]
            target = torch.from_numpy(np.stack([m for _, m in chunk])).to(trainer.device).float().unsqueeze(1)
            with amp:
                logits = trainer.semantic_logits(trainer.pixel_values(images), training=False)
            loss, up = seg_loss(torch, logits, target, pos_weight)
            losses.append(float(loss))
            pred = up.sigmoid() > 0.5
            truth = target > 0.5
            tp += float((pred & truth).sum())
            fp += float((pred & ~truth).sum())
            fn += float((~pred & truth).sum())
            for k in range(len(chunk)):
                pred_np = None
                for cell in range(grid * grid):
                    if len(previews) >= args.preview_count:
                        break
                    if pred_np is None:
                        pred_np = pred[k, 0].cpu().numpy()
                    r, c = divmod(cell, grid)
                    win = (slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw))
                    previews.append((images[k][win], chunk[k][1][win], pred_np[win]))
    metrics = summarize(tp, fp, fn)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, previews


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def stop_requested(args) -> bool:
    return bool(args.stop_file) and Path(args.stop_file).exists()


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    image_dir, label_dir = Path(args.image_dir), Path(args.label_dir)
    out_dir = Path(args.output_dir) / args.model_name

    if (out_dir / "config.json").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already holds a model. Pick another name or allow overwriting.")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "train.log")
    emit("stage", stage="setup", message="Preparing")

    log.info("=" * 64)
    log.info("SAM 3 quick fine-tune: '%s'", args.model_name)
    log.info("  prompt        : %s", args.prompt)
    log.info("  images        : %s", image_dir)
    log.info("  labels        : %s  (%s%s)", label_dir, args.label_mode,
             f", {args.num_layers} layers" if args.label_mode == "rgt_layers" else "")
    log.info("  output        : %s", out_dir)
    log.info("=" * 64)

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment; choose CPU or install CUDA PyTorch.")
    device = args.device
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s, %.1f GB VRAM, torch %s", props.name, props.total_memory / 1024**3, torch.__version__)
    else:
        log.warning("Training on CPU - expect this to be very slow.")

    # Data
    emit("stage", stage="data", message="Loading data")
    stems = list_pairs(image_dir, label_dir)
    log.info("Found %d matched image/label files", len(stems))
    train_stems, val_stems = split_stems(stems, args.max_samples, args.val_fraction, args.seed, args.max_val)
    log.info("Using %d for training and %d for validation", len(train_stems), len(val_stems))

    def data_progress(done: int, total: int) -> None:
        emit("data", done=done, total=total)

    t0 = time.perf_counter()
    train_set = load_sections(image_dir, label_dir, train_stems, args.label_mode, args.num_layers, data_progress)
    val_set = load_sections(image_dir, label_dir, val_stems, args.label_mode, args.num_layers)
    h, w = train_set.images.shape[1:]
    log.info(
        "Loaded %d sections of %dx%d in %.1fs; foreground covers %.1f%% of pixels",
        len(train_set) + len(val_set), h, w, time.perf_counter() - t0, 100 * train_set.positive_fraction,
    )
    (out_dir / "val_samples.txt").write_text("\n".join(val_stems), encoding="utf-8")

    pos_weight = args.pos_weight
    if pos_weight <= 0:
        p = min(max(train_set.positive_fraction, 1e-4), 0.5)
        pos_weight = float(min(10.0, math.sqrt((1 - p) / p)))
    log.info("Positive-pixel loss weight: %.2f", pos_weight)

    grid = args.mosaic
    per_input = grid * grid
    log.info(
        "Each model input is a %dx%d mosaic of %d sections (%d px -> %d px)",
        grid, grid, per_input, grid * h, MODEL_SIDE,
    )
    val_mosaics = fixed_mosaics(val_set, grid)
    steps_per_epoch = max(1, len(train_set) // (per_input * args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(5, int(0.05 * total_steps))
    log.info(
        "Plan: %d epochs x %d steps (batch %d) = %d steps; %d validation inputs",
        args.epochs, steps_per_epoch, args.batch_size, total_steps, len(val_mosaics),
    )

    # Model
    emit("stage", stage="model", message="Loading SAM 3")
    trainer = Sam3SemanticTrainer(args, device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()

    history: list[dict] = []
    baseline = None
    if args.eval_base:
        emit("stage", stage="validate", message="Scoring base model")
        log.info("Scoring the base model on the validation set before training...")
        baseline, previews = evaluate(trainer, val_mosaics, args, pos_weight, (h, w))
        log.info(
            "Base model    : IoU %.3f | F1 %.3f | precision %.3f | recall %.3f | loss %.4f",
            baseline["iou"], baseline["f1"], baseline["precision"], baseline["recall"], baseline["loss"],
        )
        preview_path = out_dir / "previews" / "epoch_00_base.png"
        save_preview(preview_path, previews, "Base model (before fine-tuning)")
        emit("val", epoch=0, **baseline)
        emit("preview", path=str(preview_path), title="Base model")

    best_state, best_iou, best_epoch = None, -1.0, 0
    rng = np.random.default_rng(args.seed)
    step = 0
    bad_steps = 0
    stopped = False
    loss_window: list[float] = []
    train_start = time.perf_counter()
    last_emit = 0.0
    log_every = max(1, steps_per_epoch // 10)

    emit("stage", stage="train", message="Training")
    log.info("Training started")
    for epoch in range(1, args.epochs + 1):
        trainer.set_train_mode(True)
        order = rng.permutation(len(train_set))
        epoch_losses = []
        for i in range(steps_per_epoch):
            if stop_requested(args):
                stopped = True
                break
            images, masks = [], []
            for b in range(args.batch_size):
                start = (i * args.batch_size + b) * per_input
                idx = order[start : start + per_input].tolist()
                flips = (rng.random(per_input) < 0.5).tolist() if args.hflip else None
                img, msk = build_mosaic(train_set, idx, grid, flips)
                images.append(img)
                masks.append(msk)
            target = torch.from_numpy(np.stack(masks)).to(device).float().unsqueeze(1)

            factor = lr_factor(step, total_steps, warmup)
            for group in trainer.optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor

            with amp:
                logits = trainer.semantic_logits(trainer.pixel_values(images), training=True)
            loss, _ = seg_loss(torch, logits, target, pos_weight)

            if not torch.isfinite(loss):
                bad_steps += 1
                trainer.optimizer.zero_grad(set_to_none=True)
                log.warning("Non-finite loss at step %d (skipped, %d in a row)", step + 1, bad_steps)
                if bad_steps >= MAX_BAD_STEPS:
                    raise RuntimeError(
                        f"{MAX_BAD_STEPS} non-finite losses in a row - lower the learning rate and retry."
                    )
                step += 1
                continue
            bad_steps = 0

            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for g in trainer.optimizer.param_groups for p in g["params"]], args.grad_clip
            )
            trainer.optimizer.step()
            step += 1

            value = float(loss.detach())
            epoch_losses.append(value)
            loss_window = (loss_window + [value])[-50:]
            elapsed = time.perf_counter() - train_start
            sec_per_step = elapsed / step
            eta = sec_per_step * (total_steps - step)
            now = time.perf_counter()
            if now - last_emit > 0.5 or step == total_steps:
                last_emit = now
                emit(
                    "progress",
                    epoch=epoch, epochs=args.epochs, step=step, total=total_steps,
                    loss=value, avg_loss=float(np.mean(loss_window)),
                    lr=trainer.optimizer.param_groups[0]["lr"],
                    sec_per_step=sec_per_step, eta=eta,
                    gpu_gb=(torch.cuda.max_memory_allocated() / 1024**3) if device.startswith("cuda") else 0.0,
                )
            if (i + 1) % log_every == 0 or i == 0:
                log.info(
                    "Epoch %d/%d  step %4d/%d  loss %.4f (avg %.4f)  grad %.2f  lr %.2e  %.2fs/step  ETA %s%s",
                    epoch, args.epochs, step, total_steps, value, np.mean(loss_window), float(grad_norm),
                    trainer.optimizer.param_groups[0]["lr"], sec_per_step, fmt_duration(eta),
                    f"  GPU {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB" if device.startswith("cuda") else "",
                )

        if stopped:
            log.warning("Stop requested - validating the current weights and saving the best model so far")
        if epoch_losses or not stopped:
            emit("stage", stage="validate", message=f"Validating epoch {epoch}")
            metrics, previews = evaluate(trainer, val_mosaics, args, pos_weight, (h, w))
            metrics["train_loss"] = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            metrics["epoch"] = epoch
            history.append(metrics)
            improved = metrics["iou"] > best_iou
            log.info(
                "Epoch %d done : IoU %.3f | F1 %.3f | precision %.3f | recall %.3f | val loss %.4f | train loss %.4f%s",
                epoch, metrics["iou"], metrics["f1"], metrics["precision"], metrics["recall"],
                metrics["loss"], metrics["train_loss"], "  <- best so far" if improved else "",
            )
            preview_path = out_dir / "previews" / f"epoch_{epoch:02d}.png"
            save_preview(preview_path, previews, f"Epoch {epoch}  IoU {metrics['iou']:.3f}")
            emit("val", **metrics)
            emit("preview", path=str(preview_path), title=f"Epoch {epoch}")
            if improved:
                best_iou, best_epoch = metrics["iou"], epoch
                best_state = trainer.trainable_state() if args.keep_best else None
            if not stopped:
                trainer.set_train_mode(True)
        if stopped:
            break

    if not history:
        log.warning("Stopped before any training step finished - nothing to save.")
        emit("done", saved=False, output=str(out_dir))
        return 0

    if args.keep_best and best_state is not None and best_epoch != history[-1]["epoch"]:
        log.info("Restoring the best epoch (%d, IoU %.3f) before saving", best_epoch, best_iou)
        trainer.load_trainable_state(best_state)
    saved_epoch = best_epoch if args.keep_best else history[-1]["epoch"]

    emit("stage", stage="save", message="Saving model")
    log.info("Saving model to %s ...", out_dir)
    t0 = time.perf_counter()
    trainer.save(out_dir)
    final = next(m for m in history if m["epoch"] == saved_epoch)
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "prompt": args.prompt,
        "base_model": args.base_model,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "label_mode": args.label_mode,
        "num_layers": args.num_layers if args.label_mode == "rgt_layers" else None,
        "trained": TRAIN_SCOPES[args.train_scope],
        "unfrozen_vit_blocks": trainer.n_unfrozen,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "saved_epoch": saved_epoch,
        "stopped_early": stopped,
        "metrics": final,
        "base_metrics": baseline,
        "history": history,
        "settings": vars(args),
        "minutes": (time.perf_counter() - started) / 60,
    }
    (out_dir / "finetune_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("Saved in %.1fs", time.perf_counter() - t0)
    log.info("=" * 64)
    if baseline:
        log.info("IoU %.3f -> %.3f (base -> fine-tuned, epoch %d)", baseline["iou"], final["iou"], saved_epoch)
    else:
        log.info("Fine-tuned IoU %.3f (epoch %d)", final["iou"], saved_epoch)
    log.info("Finished in %s. Model: %s", fmt_duration(time.perf_counter() - started), out_dir)
    log.info("=" * 64)
    emit("done", saved=True, output=str(out_dir), iou=final["iou"], base_iou=baseline["iou"] if baseline else None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", required=True, help="Folder of seismic .dat sections (e.g. Seis75).")
    p.add_argument("--label-dir", required=True, help="Folder of matching label .dat files.")
    p.add_argument("--label-mode", choices=LABEL_MODES, default="binary")
    p.add_argument("--num-layers", type=int, default=8, help="Bands per section for rgt_layers.")
    p.add_argument("--prompt", required=True, help='Text prompt to train, e.g. "fault".')
    p.add_argument("--base-model", default="facebook/sam3", help="Hub id or local Sam3Model folder.")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "finetuned_checkpoints"))
    p.add_argument("--model-name", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-samples", type=int, default=2000, help="0 = use every matched file.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--max-val", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--mosaic", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=0.0, help="0 = automatic from label balance.")
    p.add_argument("--train-scope", choices=tuple(TRAIN_SCOPES), default="heads_encoder")
    p.add_argument("--unfreeze-vit-blocks", type=int, default=0)
    p.add_argument("--no-hflip", dest="hflip", action="store_false")
    p.add_argument("--no-eval-base", dest="eval_base", action="store_false")
    p.add_argument("--keep-last", dest="keep_best", action="store_false",
                   help="Save the last epoch instead of the best validation epoch.")
    p.add_argument("--preview-count", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stop-file", default="", help="Training stops cleanly once this file exists.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        return run(args)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            if not log.handlers:
                setup_logging(None)
            log.error(exc.code)
            emit("error", message=exc.code)
            return 2
        raise
    except Exception as exc:  # noqa: BLE001 - report everything to the GUI
        if not log.handlers:
            setup_logging(None)
        log.error("Training failed: %s", exc)
        for line in traceback.format_exc().rstrip().splitlines():
            log.error("  %s", line)
        if "out of memory" in str(exc).lower():
            log.error("GPU ran out of memory: lower the batch size or unfreeze fewer ViT blocks.")
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
