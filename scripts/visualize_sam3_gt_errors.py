#!/usr/bin/env python3
"""Render compact three-panel visualizations for SAM3 GT mistakes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SAMPLE_PATTERN = re.compile(r"sample0*(\d+)(?:_[012])?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sample_number(row: dict[str, Any]) -> int:
    match = SAMPLE_PATTERN.fullmatch(str(row.get("sample_id", "")))
    if match is None:
        raise ValueError(f"Cannot parse sample ID: {row.get('sample_id')!r}")
    return int(match.group(1))


def read_image(path: Path | None, fallback_shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path is not None else None
    if image is not None:
        return image
    height, width = fallback_shape
    missing = np.full((height, width, 3), 35, dtype=np.uint8)
    cv2.putText(
        missing,
        "missing visualization",
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return missing


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def panel_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (15, 15, 15), -1)
    cv2.putText(
        canvas,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def evidence_values(result: dict[str, Any]) -> tuple[float, float]:
    frames = result.get("diagnostics", {}).get("frames", [])
    if not frames:
        return 0.0, 0.0
    frame = frames[0]
    direct = frame.get("damage", {}).get("direct_prompt_evidence", [])
    damage_score = max((float(item.get("score", 0.0)) for item in direct), default=0.0)
    geometry_score = float(
        frame.get("position", {}).get("support_geometry_score", 0.0)
    )
    return damage_score, geometry_score


def diagnostic_paths(result: dict[str, Any]) -> tuple[Path | None, Path | None]:
    frames = result.get("diagnostics", {}).get("frames", [])
    if not frames:
        return None, None
    crop_path = Path(str(frames[0].get("crop_path", "")))
    if not crop_path.is_absolute():
        crop_path = Path.cwd() / crop_path
    cable_overlay = crop_path.parent / "cable_mask.jpg"
    diagnostics_dir = (
        crop_path.parents[2] / "diagnostics"
        if len(crop_path.parents) >= 3
        else None
    )
    damage_visual = None
    if diagnostics_dir is not None and diagnostics_dir.is_dir():
        matches = sorted(
            diagnostics_dir.glob(
                f"*_{result['sample_id']}_f*_damage_full.jpg"
            )
        )
        damage_visual = matches[0] if matches else None
    return cable_overlay, damage_visual


def main() -> None:
    args = parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    indexed = {sample_number(row): row for row in results}
    errors = list(evaluation.get("errors", []))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    for error in errors:
        number = int(error["sample"])
        result = indexed[number]
        source = cv2.imread(str(result["video"]), cv2.IMREAD_COLOR)
        if source is None:
            raise SystemExit(f"Cannot read source image: {result['video']}")
        cable_path, damage_path = diagnostic_paths(result)
        cable = read_image(cable_path, source.shape[:2])
        damage = read_image(damage_path, source.shape[:2])
        panels = [
            panel_title(letterbox(source, 600, 338), "Enhanced input"),
            panel_title(letterbox(cable, 600, 338), "SAM3 pipe mask"),
            panel_title(letterbox(damage, 600, 338), "Damage evidence"),
        ]
        body = np.hstack(panels)
        header = np.full((76, body.shape[1], 3), 22, dtype=np.uint8)
        damage_score, geometry_score = evidence_values(result)
        title = (
            f"{result['sample_id']} | GT={error['truth']} "
            f"PRED={error['prediction']} | position={result['position']} "
            f"damage={result['damage']}"
        )
        detail = (
            f"metal-patch score={damage_score:.4f} | "
            f"support geometry={geometry_score:.4f}"
        )
        cv2.putText(
            header,
            title,
            (16, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            detail,
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (205, 225, 245),
            2,
            cv2.LINE_AA,
        )
        visualization = np.vstack([header, body])
        output_path = args.out_dir / f"{result['sample_id']}_error.jpg"
        cv2.imwrite(str(output_path), visualization, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(cv2.resize(visualization, (1440, 331), interpolation=cv2.INTER_AREA))
        manifest.append(
            {
                "sample_id": result["sample_id"],
                "gt": int(error["truth"]),
                "prediction": int(error["prediction"]),
                "damage_score": damage_score,
                "support_geometry_score": geometry_score,
                "output": str(output_path.resolve()),
            }
        )

    if rows:
        contact_sheet = np.vstack(rows)
        contact_path = args.out_dir / "error_contact_sheet.jpg"
        cv2.imwrite(str(contact_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        contact_path = None
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "error_count": len(manifest),
                "contact_sheet": str(contact_path.resolve()) if contact_path else None,
                "errors": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(manifest)} errors -> {args.out_dir}")


if __name__ == "__main__":
    main()
