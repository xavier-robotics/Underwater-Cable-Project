#!/usr/bin/env python
"""Split annotated images and YOLO labels into train/val/test folders."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def linked_label(image: Path, labels_dir: Path) -> Path:
    rel = image.relative_to(image.parents[0]) if image.parent == image.parents[0] else image.name
    return labels_dir / Path(rel).with_suffix(".txt").name


def copy_pair(image: Path, label: Path, out_root: Path, split: str) -> None:
    image_dst = out_root / "images" / split / image.name
    label_dst = out_root / "labels" / split / f"{image.stem}.txt"
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dst)
    if label.exists():
        shutil.copy2(label, label_dst)
    else:
        label_dst.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("data/annotated/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/annotated/labels"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = collect_images(args.images_dir)
    if not images:
        raise SystemExit(f"No images found under {args.images_dir}")

    random.Random(args.seed).shuffle(images)
    n_total = len(images)
    n_test = int(round(n_total * args.test_ratio))
    n_val = int(round(n_total * args.val_ratio))
    split_map = {
        "test": images[:n_test],
        "val": images[n_test : n_test + n_val],
        "train": images[n_test + n_val :],
    }

    for split, split_images in split_map.items():
        for image in split_images:
            label = args.labels_dir / f"{image.stem}.txt"
            copy_pair(image, label, args.out_dir, split)
        print(f"{split}: {len(split_images)} images")


if __name__ == "__main__":
    main()
