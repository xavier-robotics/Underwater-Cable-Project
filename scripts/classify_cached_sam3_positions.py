#!/usr/bin/env python
"""Add SAM3 support-based positions to cached damage classifications."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scripts.classify_samples_gpt import load_json
    from scripts.classify_samples_sam3_only import (
        SAM3PromptDetector,
        aggregate_support_position,
        load_config,
        resolve_sam3_class,
        support_geometry_features,
        support_position_frame_decision,
        top_detection_rows,
    )
    from scripts.classify_samples_zero_shot import (
        resolve_config_path,
        write_csv_results,
        write_json,
        write_review,
    )
except ModuleNotFoundError:
    from classify_samples_gpt import load_json
    from classify_samples_sam3_only import (
        SAM3PromptDetector,
        aggregate_support_position,
        load_config,
        resolve_sam3_class,
        support_geometry_features,
        support_position_frame_decision,
        top_detection_rows,
    )
    from classify_samples_zero_shot import (
        resolve_config_path,
        write_csv_results,
        write_json,
        write_review,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


def classify_positions(
    results: list[dict[str, Any]],
    detector: SAM3PromptDetector | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run only support prompts for results not already marked damaged."""
    output = copy.deepcopy(results)
    position_config = config["position"]
    prompts = list(map(str, position_config.get("support_prompts", [])))
    for result in output:
        decisions: list[dict[str, Any]] = []
        frame_diagnostics: list[dict[str, Any]] = []
        cached_frames = result.get("diagnostics", {}).get("frames", [])
        for frame in cached_frames if isinstance(cached_frames, list) else []:
            if not isinstance(frame, dict):
                continue
            raw_crop_path = frame.get("crop_path")
            raw_mask_path = frame.get("mask_path")
            if not raw_crop_path or not raw_mask_path:
                continue
            crop_path = resolve_config_path(str(raw_crop_path))
            mask_path = resolve_config_path(str(raw_mask_path))
            crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if crop is None or mask is None:
                continue
            geometry = support_geometry_features(crop, mask, position_config)
            decision = support_position_frame_decision(
                [],
                prompts,
                position_config,
                geometry=geometry,
            )
            decisions.append(decision)
            frame_diagnostics.append(
                {
                    "crop_path": str(crop_path),
                    "mask_path": str(mask_path),
                    "decision": decision,
                    "detections": [],
                }
            )

        # Older cached results may not contain crop/mask diagnostics. Rebuild
        # one pipe mask only in that compatibility path.
        if decisions:
            raw_paths: list[str] = []
        else:
            raw_paths = list(map(str, result.get("image_paths", [])))
        for raw_path in raw_paths:
            if detector is None:
                detector = SAM3PromptDetector(config["model"])
            image_path = resolve_config_path(str(raw_path))
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            cable = detector.adapter.extract_crop(image)
            if cable is None or cable.mask is None:
                continue
            reference_mask = cable.mask
            geometry = support_geometry_features(
                image,
                reference_mask,
                position_config,
            )
            if prompts:
                search_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
                detections = detector.predict(image, prompts, search_mask)
            else:
                detections = []
            decision = support_position_frame_decision(
                detections,
                prompts,
                position_config,
                reference_mask=reference_mask,
                geometry=geometry,
            )
            decisions.append(decision)
            frame_diagnostics.append(
                {
                    "image_path": str(image_path),
                    "decision": decision,
                    "detections": top_detection_rows(detections),
                }
            )

        position, confidence, agreement = aggregate_support_position(decisions)
        damage = str(result.get("damage", "intact"))
        class_id, class_name = resolve_sam3_class(
            position,
            damage,
            config,
        )
        result.update(
            {
                "position": position,
                "position_confidence": confidence,
                "position_evaluated": True,
                "class_id": class_id,
                "class_name": class_name,
                "method": "sam3_direct_marker_with_support_position",
            }
        )
        diagnostics = result.setdefault("diagnostics", {})
        diagnostics["position_agreement"] = agreement
        diagnostics["support_position_frames"] = frame_diagnostics
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sam3_only_classifier.yaml",
    )
    args = parser.parse_args()

    source = load_json(args.results)
    if not isinstance(source, list):
        raise SystemExit(f"Expected a JSON list: {args.results}")
    config = load_config(args.config)
    results = classify_positions(
        source,
        None,
        config,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "results.json", results)
    write_csv_results(args.out_dir / "results.csv", results)
    write_review(args.out_dir / "review.md", results)
    write_json(
        args.out_dir / "summary.json",
        {
            "method": "sam3_direct_marker_with_support_position",
            "total": len(results),
            "classes": {
                str(class_id): sum(
                    result["class_id"] == class_id for result in results
                )
                for class_id in (0, 1, 2)
            },
        },
    )
    print(
        f"Position classification complete: {len(results)} samples -> "
        f"{args.out_dir / 'results.json'}"
    )


if __name__ == "__main__":
    main()
