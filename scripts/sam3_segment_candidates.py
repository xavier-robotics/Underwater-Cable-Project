#!/usr/bin/env python3
"""SAM3 candidate helpers and compatibility CLI for damage-only YOLO annotation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sam3 import model_builder
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


@dataclass
class Candidate:
    prompt: str
    score: float
    box_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    area_ratio: float


def resolve_bpe_path(path: Path | None) -> Path:
    # The project's outer sam3/ directory can be a namespace package with
    # __file__ = None. Resolve assets beside the actual loaded model module.
    if path is None:
        path = Path(model_builder.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not path.is_file():
        raise SystemExit(f"SAM3 tokenizer vocabulary does not exist: {path}. Specify --bpe-path.")
    return path.resolve()


def register_dtype_alignment_hooks(model: torch.nn.Module) -> None:
    def align_input(module: torch.nn.Linear, inputs: tuple) -> tuple:
        if inputs and hasattr(inputs[0], "dtype") and inputs[0].dtype != module.weight.dtype:
            return (inputs[0].to(dtype=module.weight.dtype), *inputs[1:])
        return inputs

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_pre_hook(align_input)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def clean_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    mask = (mask > 0.5).astype(np.uint8)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union else 0.0


def collect_candidates(
    output: dict,
    prompt: str,
    height: int,
    width: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[Candidate]:
    masks = to_numpy(output.get("masks", []))
    boxes = to_numpy(output.get("boxes", []))
    scores = to_numpy(output.get("scores", [])).reshape(-1)
    frame_area = max(1, height * width)
    candidates: list[Candidate] = []

    for index, score in enumerate(scores):
        if index >= len(masks):
            continue
        mask = clean_mask(masks[index], height, width)
        area_ratio = float(np.count_nonzero(mask) / frame_area)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        bbox = bbox_from_mask(mask)
        if bbox is None and index < len(boxes):
            x1, y1, x2, y2 = [int(round(float(v))) for v in boxes[index].reshape(-1)[:4]]
            bbox = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if bbox is None:
            continue
        candidates.append(Candidate(prompt, float(score), bbox, mask, area_ratio))
    return candidates


def dedupe_candidates(candidates: list[Candidate], iou_threshold: float) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(mask_iou(candidate.mask, old.mask) < iou_threshold for old in kept):
            kept.append(candidate)
    return kept


def main() -> None:
    # Keep this existing command usable, but export damage-only YOLO labels.
    from scripts.build_sam3_damage_dataset import main as damage_main
    damage_main()


if __name__ == "__main__":
    main()
