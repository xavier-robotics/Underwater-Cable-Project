#!/usr/bin/env python
"""Print cable-relative bright-line support features from saved GT crops."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2

try:
    from scripts.classify_samples_sam3_only import (
        load_config,
        support_geometry_features,
    )
except ModuleNotFoundError:
    from classify_samples_sam3_only import load_config, support_geometry_features


LABEL_PATTERN = re.compile(r"_([012])$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sam3_only_classifier.yaml"),
    )
    parser.add_argument(
        "--visual-dir",
        type=Path,
        help="Optionally save cable masks and accepted/rejected support lines.",
    )
    args = parser.parse_args()
    position_config = load_config(args.config)["position"]
    if args.visual_dir is not None:
        args.visual_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_dir in sorted(args.samples_dir.iterdir()):
        match = LABEL_PATTERN.search(sample_dir.name)
        if not sample_dir.is_dir() or match is None:
            continue
        crop = cv2.imread(str(sample_dir / "crop.jpg"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if crop is None or mask is None:
            continue
        features = support_geometry_features(crop, mask, position_config)
        rows.append(
            {
                "sample_id": sample_dir.name,
                "gt": int(match.group(1)),
                **features,
            }
        )
        if args.visual_dir is not None:
            overlay = crop.copy()
            tinted = crop.copy()
            tinted[mask > 0] = (0, 220, 255)
            overlay = cv2.addWeighted(tinted, 0.30, overlay, 0.70, 0.0)
            line = features.get("line_xyxy")
            positive = float(features["score"]) >= float(
                position_config.get("support_geometry_score_threshold", 0.11)
            )
            if isinstance(line, list) and len(line) == 4:
                x1, y1, x2, y2 = map(int, line)
                color = (0, 255, 0) if positive else (0, 165, 255)
                cv2.line(overlay, (x1, y1), (x2, y2), color, 5, cv2.LINE_AA)
            text = (
                f"support={features['score']:.3f} "
                f"{'suspended' if positive else 'exposed'}"
            )
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 36), (20, 20, 20), -1)
            cv2.putText(
                overlay,
                text,
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(args.visual_dir / f"{sample_dir.name}.jpg"), overlay)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
