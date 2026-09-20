#!/usr/bin/env python3
"""Apply conservative underwater white balance and local contrast enhancement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("data/extracted_original_image"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed_images"))
    parser.add_argument("--gain_min", type=float, default=0.75)
    parser.add_argument("--gain_max", type=float, default=1.35)
    parser.add_argument("--clahe_clip", type=float, default=1.5)
    parser.add_argument("--clahe_grid", type=int, default=8)
    parser.add_argument("--jpeg_quality", type=int, default=96)
    return parser.parse_args()


def channel_means(image: np.ndarray) -> np.ndarray:
    """Calculate robust BGR means while ignoring extreme dark/bright pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, [5.0, 95.0])
    selection = (gray >= low) & (gray <= high)
    pixels = image[selection]
    if len(pixels) < 64:
        pixels = image.reshape(-1, 3)
    return pixels.astype(np.float32).mean(axis=0)


def enhance_underwater_image(
    image: np.ndarray,
    *,
    gain_min: float,
    gain_max: float,
    clahe_clip: float,
    clahe_grid: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply bounded gray-world white balance followed by LAB CLAHE."""
    means_before = channel_means(image)
    target = float(means_before.mean())
    gains = np.clip(
        target / np.maximum(means_before, 1.0),
        gain_min,
        gain_max,
    )
    balanced = np.clip(
        image.astype(np.float32) * gains.reshape(1, 1, 3),
        0,
        255,
    ).astype(np.uint8)

    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, clahe_clip),
        tileGridSize=(max(1, clahe_grid), max(1, clahe_grid)),
    )
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    means_after = channel_means(enhanced)
    diagnostics = {
        "mean_bgr_before": [round(float(value), 3) for value in means_before],
        "gain_bgr": [round(float(value), 4) for value in gains],
        "mean_bgr_after": [round(float(value), 3) for value in means_after],
        "green_red_ratio_before": round(
            float(means_before[1] / max(means_before[2], 1.0)),
            4,
        ),
        "green_red_ratio_after": round(
            float(means_after[1] / max(means_after[2], 1.0)),
            4,
        ),
    }
    return enhanced, diagnostics


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / max(1, image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def comparison_image(
    original: np.ndarray,
    enhanced: np.ndarray,
    name: str,
    target_height: int = 420,
) -> np.ndarray:
    """Create a labeled side-by-side preview with equal panel dimensions."""
    original_view = resize_to_height(original, target_height)
    enhanced_view = resize_to_height(enhanced, target_height)
    panel_width = max(original_view.shape[1], enhanced_view.shape[1])
    header_height = 42
    canvas = np.full(
        (target_height + header_height, panel_width * 2, 3),
        24,
        dtype=np.uint8,
    )
    for index, view in enumerate((original_view, enhanced_view)):
        x0 = index * panel_width + (panel_width - view.shape[1]) // 2
        canvas[header_height:, x0 : x0 + view.shape[1]] = view
    cv2.putText(
        canvas,
        f"ORIGINAL | {name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "ENHANCED: WB + CLAHE",
        (panel_width + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def contact_sheet(images: list[np.ndarray], columns: int = 3) -> np.ndarray:
    """Lay comparison previews out in a compact overview grid."""
    if not images:
        raise ValueError("No comparison images were provided")
    tile_width = 600
    resized = [
        cv2.resize(
            image,
            (tile_width, max(1, round(image.shape[0] * tile_width / image.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
        for image in images
    ]
    tile_height = max(image.shape[0] for image in resized)
    rows = (len(resized) + columns - 1) // columns
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        18,
        dtype=np.uint8,
    )
    for index, image in enumerate(resized):
        row, column = divmod(index, columns)
        y0 = row * tile_height
        x0 = column * tile_width
        sheet[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
    return sheet


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = args.out_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    previews: list[np.ndarray] = []
    for index, input_path in enumerate(image_paths, start=1):
        original = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if original is None:
            rows.append({"input": str(input_path), "error": "unreadable_image"})
            continue
        enhanced, diagnostics = enhance_underwater_image(
            original,
            gain_min=args.gain_min,
            gain_max=args.gain_max,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
        )
        relative_path = input_path.relative_to(args.input_dir)
        output_path = args.out_dir / relative_path
        comparison_path = comparisons_dir / relative_path.parent / f"{input_path.stem}_comparison.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        preview = comparison_image(original, enhanced, input_path.name)
        output_params = (
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            if output_path.suffix.lower() in {".jpg", ".jpeg"}
            else []
        )
        if not cv2.imwrite(str(output_path), enhanced, output_params):
            raise RuntimeError(f"Failed to write enhanced image: {output_path}")
        cv2.imwrite(
            str(comparison_path),
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), 94],
        )
        previews.append(preview)
        rows.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "comparison": str(comparison_path),
                **diagnostics,
            }
        )
        print(f"[{index:02d}/{len(image_paths):02d}] {input_path.name}")

    overview_path = args.out_dir / "comparison_overview.jpg"
    cv2.imwrite(
        str(overview_path),
        contact_sheet(previews),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )
    valid_rows = [row for row in rows if "error" not in row]
    report = {
        "method": "bounded_gray_world_plus_lab_clahe",
        "input": str(args.input_dir.resolve()),
        "output": str(args.out_dir.resolve()),
        "images": len(valid_rows),
        "parameters": {
            "gain_min": args.gain_min,
            "gain_max": args.gain_max,
            "clahe_clip": args.clahe_clip,
            "clahe_grid": args.clahe_grid,
        },
        "mean_green_red_ratio_before": round(
            float(np.mean([row["green_red_ratio_before"] for row in valid_rows])),
            4,
        ),
        "mean_green_red_ratio_after": round(
            float(np.mean([row["green_red_ratio_after"] for row in valid_rows])),
            4,
        ),
        "files": rows,
    }
    (args.out_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Enhanced {len(valid_rows)} images -> {args.out_dir.resolve()}")
    print(f"Overview -> {overview_path.resolve()}")


if __name__ == "__main__":
    main()
