#!/usr/bin/env python3
"""Generate SAM3 mask candidates by sweeping text prompts on one image/frame."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


DEFAULT_PROMPTS = [
    "pipe",
    "cable",
    "underwater cable",
    "submarine cable",
    "black pipe",
    "rubber hose",
    "cylindrical object",
    "object",
    "thing",
    "marker",
    "tag",
    "floor",
    "wall",
    "grid",
    "panel",
]

COLORS = [
    (40, 180, 255),
    (80, 220, 80),
    (255, 130, 60),
    (220, 80, 220),
    (60, 220, 220),
    (255, 220, 80),
    (180, 120, 255),
]


@dataclass
class Candidate:
    prompt: str
    score: float
    box_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input image or video.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sam3_candidates"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to sam3.pt.")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to sweep. Repeat this option; defaults to a built-in broad list.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Text file with one prompt per line. Empty lines and # comments are ignored.",
    )
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou-threshold", type=float, default=0.75, help="Deduplicate masks above this IoU.")
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.85)
    parser.add_argument("--time-sec", type=float, default=0.0, help="Video timestamp to read.")
    parser.add_argument("--frame-index", type=int, help="Video frame index; overrides --time-sec.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompts or DEFAULT_PROMPTS)
    if args.prompt_file:
        lines = args.prompt_file.read_text(encoding="utf-8").splitlines()
        prompts.extend(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))

    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        key = prompt.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(prompt.strip())
    return deduped


def register_dtype_alignment_hooks(model: torch.nn.Module) -> None:
    def align_input(module: torch.nn.Linear, inputs: tuple) -> tuple:
        if inputs and hasattr(inputs[0], "dtype") and inputs[0].dtype != module.weight.dtype:
            return (inputs[0].to(dtype=module.weight.dtype), *inputs[1:])
        return inputs

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_pre_hook(align_input)


def read_input(path: Path, time_sec: float, frame_index: int | None) -> tuple[np.ndarray, str]:
    image = cv2.imread(str(path))
    if image is not None:
        return image, path.stem

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open image or video: {path}")
    if frame_index is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        label = f"frame_{frame_index:06d}"
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        label = f"time_{time_sec:.2f}s"
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Cannot read {label} from video: {path}")
    return frame, f"{path.stem}_{label}"


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


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower() or "mask"


def save_outputs(frame: np.ndarray, candidates: list[Candidate], out_dir: Path, source_name: str) -> None:
    masks_dir = out_dir / "masks"
    crops_dir = out_dir / "crops"
    masks_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    overlay = frame.copy()
    results = []
    for index, candidate in enumerate(candidates, start=1):
        color = np.asarray(COLORS[(index - 1) % len(COLORS)], dtype=np.uint8)
        overlay[candidate.mask > 0] = (0.55 * overlay[candidate.mask > 0] + 0.45 * color).astype(np.uint8)

        x1, y1, x2, y2 = candidate.box_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(v) for v in color), 2)
        cv2.putText(
            overlay,
            f"{index}:{candidate.prompt} {candidate.score:.2f}",
            (x1, max(24, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            tuple(int(v) for v in color),
            2,
            cv2.LINE_AA,
        )

        stem = f"{index:03d}_{safe_name(candidate.prompt)}_{candidate.score:.3f}"
        cv2.imwrite(str(masks_dir / f"{stem}.png"), candidate.mask * 255)
        cv2.imwrite(str(crops_dir / f"{stem}.jpg"), frame[y1:y2, x1:x2])
        results.append(
            {
                "id": index,
                "prompt": candidate.prompt,
                "score": round(candidate.score, 5),
                "box_xyxy": [x1, y1, x2, y2],
                "area_ratio": round(candidate.area_ratio, 5),
                "mask_path": str((masks_dir / f"{stem}.png").relative_to(out_dir)),
                "crop_path": str((crops_dir / f"{stem}.jpg").relative_to(out_dir)),
            }
        )

    cv2.imwrite(str(out_dir / f"{source_name}_source.jpg"), frame)
    cv2.imwrite(str(out_dir / f"{source_name}_all_candidates.jpg"), overlay)
    (out_dir / "candidates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible. Run this in your normal shell with conda activate sam3.")
    if not args.checkpoint.exists():
        raise SystemExit(f"SAM3 checkpoint does not exist: {args.checkpoint}")

    prompts = read_prompts(args)
    frame, source_name = read_input(args.input, args.time_sec, args.frame_index)
    height, width = frame.shape[:2]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=args.device,
    )
    register_dtype_alignment_hooks(model)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.confidence)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    state = processor.set_image(image)

    all_candidates: list[Candidate] = []
    for prompt in prompts:
        output = processor.set_text_prompt(prompt, state)
        candidates = collect_candidates(
            output,
            prompt,
            height,
            width,
            args.min_area_ratio,
            args.max_area_ratio,
        )
        all_candidates.extend(candidates)
        print(f"{prompt!r}: {len(candidates)} candidates")

    kept = dedupe_candidates(all_candidates, args.iou_threshold)
    save_outputs(frame, kept, args.out_dir, source_name)
    print(f"Kept {len(kept)} / {len(all_candidates)} candidates after IoU dedupe.")
    print(f"Results written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
