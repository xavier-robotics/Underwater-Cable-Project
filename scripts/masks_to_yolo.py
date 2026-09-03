#!/usr/bin/env python
"""Convert SAM-style binary masks to YOLO box labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def mask_to_boxes(mask_path: Path, min_area: int) -> list[tuple[int, int, int, int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if int(area) < min_area:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def write_yolo_label(label_path: Path, boxes: list[tuple[int, int, int, int]], width: int, height: int, class_id: int) -> None:
    lines = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        nw = w / width
        nh = h / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--min-area", type=int, default=500)
    args = parser.parse_args()

    mask_paths = sorted(p for p in args.masks_dir.rglob("*") if p.suffix.lower() in MASK_EXTS)
    if not mask_paths:
        raise SystemExit(f"No masks found under {args.masks_dir}")

    converted = 0
    skipped = 0
    for mask_path in mask_paths:
        image = find_image(args.images_dir, mask_path.stem)
        if image is None:
            skipped += 1
            continue
        img = cv2.imread(str(image))
        if img is None:
            skipped += 1
            continue
        height, width = img.shape[:2]
        boxes = mask_to_boxes(mask_path, args.min_area)
        write_yolo_label(args.labels_dir / f"{image.stem}.txt", boxes, width, height, args.class_id)
        converted += 1

    print(f"converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
