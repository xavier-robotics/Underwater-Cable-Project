#!/usr/bin/env python3
"""Compare SAM3 text prompts on one image or one video frame."""

from __future__ import annotations

import argparse
import json
import re
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
]

COLORS = [
    (40, 180, 255),
    (80, 220, 80),
    (255, 130, 60),
    (220, 80, 220),
    (60, 220, 220),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/home/nvidia/uw_detection/data/frames_1080p/mmexport1779157141457/mmexport1779157141457_f000450_t00015.00.jpg"), help="Input image or video.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sam3_prompt_test"))
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to test. Repeat this option for multiple prompts.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpts/sam3/sam3.pt"), help="Path to sam3.pt.")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--time-sec", type=float, default=0.0, help="Video timestamp to test.")
    parser.add_argument("--frame-index", type=int, help="Video frame index; overrides --time-sec.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def register_dtype_alignment_hooks(model: torch.nn.Module) -> None:
    """Align Linear inputs with their weights for mixed-dtype SAM3 modules."""

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
        selected = f"frame_{frame_index:06d}"
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        selected = f"time_{time_sec:.2f}s"
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Cannot read {selected} from video: {path}")
    return frame, f"{path.stem}_{selected}"


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


def safe_name(text: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return name or "prompt"


def visualize(
    frame: np.ndarray, output: dict, prompt: str, confidence: float
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    height, width = frame.shape[:2]
    masks = to_numpy(output.get("masks", []))
    boxes = to_numpy(output.get("boxes", []))
    scores = to_numpy(output.get("scores", [])).reshape(-1)
    overlay = frame.copy()
    union_mask = np.zeros((height, width), dtype=np.uint8)
    detections: list[dict] = []

    for index, score in enumerate(scores):
        if float(score) < confidence:
            continue
        mask = clean_mask(masks[index], height, width) if index < len(masks) else None
        if mask is not None:
            color = np.asarray(COLORS[index % len(COLORS)], dtype=np.uint8)
            overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * color).astype(np.uint8)
            union_mask[mask > 0] = 255

        box = boxes[index].reshape(-1)[:4] if index < len(boxes) else None
        if box is not None:
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS[index % len(COLORS)], 3)
            cv2.putText(
                overlay,
                f"{float(score):.3f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLORS[index % len(COLORS)],
                2,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "score": round(float(score), 5),
                    "box_xyxy": [x1, y1, x2, y2],
                    "mask_area_ratio": round(float(np.count_nonzero(mask) / mask.size), 5)
                    if mask is not None
                    else None,
                }
            )

    cv2.rectangle(overlay, (0, 0), (width, 48), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"{prompt} | detections: {len(detections)}",
        (14, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay, union_mask, detections


def make_contact_sheet(images: list[np.ndarray], columns: int = 2) -> np.ndarray:
    if not images:
        raise ValueError("No images for contact sheet")
    target_width = 800
    resized = [
        cv2.resize(image, (target_width, round(image.shape[0] * target_width / image.shape[1])))
        for image in images
    ]
    tile_height = max(image.shape[0] for image in resized)
    rows = (len(resized) + columns - 1) // columns
    sheet = np.zeros((rows * tile_height, columns * target_width, 3), dtype=np.uint8)
    for index, image in enumerate(resized):
        row, column = divmod(index, columns)
        sheet[row * tile_height : row * tile_height + image.shape[0], column * target_width : (column + 1) * target_width] = image
    return sheet


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible. Run this script in the normal shell with the sam3 environment.")
    if not args.checkpoint.exists():
        raise SystemExit(f"SAM3 checkpoint does not exist: {args.checkpoint}")

    prompts = args.prompts or DEFAULT_PROMPTS
    frame, source_name = read_input(args.input, args.time_sec, args.frame_index)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_dir / f"{source_name}_source.jpg"), frame)

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=args.device,
    )
    register_dtype_alignment_hooks(model)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.confidence)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    state = processor.set_image(Image.fromarray(rgb))
    previews: list[np.ndarray] = []
    report = {"input": str(args.input), "source": source_name, "results": []}

    for index, prompt in enumerate(prompts, start=1):
        output = processor.set_text_prompt(prompt, state)
        preview, mask, detections = visualize(frame, output, prompt, args.confidence)
        stem = f"{index:02d}_{safe_name(prompt)}"
        cv2.imwrite(str(args.out_dir / f"{stem}_overlay.jpg"), preview)
        cv2.imwrite(str(args.out_dir / f"{stem}_mask.png"), mask)
        previews.append(preview)
        report["results"].append({"prompt": prompt, "detections": detections})
        print(f"{prompt!r}: {len(detections)} detections")

    cv2.imwrite(str(args.out_dir / "prompt_comparison.jpg"), make_contact_sheet(previews))
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Results written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
