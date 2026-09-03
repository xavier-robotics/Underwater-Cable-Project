#!/usr/bin/env python3
"""Evaluate the SAM3-only three-class method on filename-labeled images."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from scripts.classify_samples_sam3_only import (
        SAM3PromptDetector,
        classify,
        load_config,
    )
    from scripts.classify_samples_zero_shot import (
        FrameInput,
        SampleInput,
        write_csv_results,
        write_json,
        write_review,
    )
    from scripts.select_sample_frames import expand_bbox
except ModuleNotFoundError:
    from classify_samples_sam3_only import (
        SAM3PromptDetector,
        classify,
        load_config,
    )
    from classify_samples_zero_shot import (
        FrameInput,
        SampleInput,
        write_csv_results,
        write_json,
        write_review,
    )
    from select_sample_frames import expand_bbox


REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = {
    0: "damaged",
    1: "exposed_intact",
    2: "suspended_intact",
}
LABEL_PATTERN = re.compile(r"_([012])$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sam3_only_classifier.yaml",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--label",
        type=int,
        choices=sorted(CLASS_NAMES),
        action="append",
        help="Evaluate only selected filename labels; repeat for multiple labels.",
    )
    parser.add_argument("--crop-pad-ratio", type=float, default=0.18)
    parser.add_argument("--crop-min-pad", type=int, default=24)
    parser.add_argument(
        "--keep-position-disabled",
        action="store_true",
        help="Keep the config's position.enabled value instead of evaluating all three classes.",
    )
    return parser.parse_args()


def discover_gt_images(input_dir: Path, limit: int | None) -> list[tuple[Path, int]]:
    """Return images whose stem ends in _0, _1, or _2."""
    rows: list[tuple[Path, int]] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = LABEL_PATTERN.search(path.stem)
        if match is None:
            continue
        rows.append((path.resolve(), int(match.group(1))))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def save_cable_sample(
    image_path: Path,
    frame_idx: int,
    detector: SAM3PromptDetector,
    samples_dir: Path,
    crop_pad_ratio: float,
    crop_min_pad: int,
) -> SampleInput | None:
    """Detect the cable once and create the inputs required by the classifier."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    cable = detector.adapter.extract_crop(image)
    if cable is None:
        return None

    height, width = image.shape[:2]
    x1, y1, x2, y2 = expand_bbox(
        cable.bbox_xyxy,
        width,
        height,
        crop_pad_ratio,
        crop_min_pad,
    )
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    full_mask = cable.mask
    if full_mask is None:
        full_mask = np.zeros((height, width), dtype=np.uint8)
        raw_x1, raw_y1, raw_x2, raw_y2 = cable.bbox_xyxy
        full_mask[raw_y1:raw_y2, raw_x1:raw_x2] = 255
    mask_crop = full_mask[y1:y2, x1:x2]

    sample_dir = samples_dir / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    crop_path = sample_dir / "crop.jpg"
    mask_path = sample_dir / "mask.png"
    overlay_path = sample_dir / "cable_mask.jpg"
    cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(mask_path), mask_crop)
    overlay = image.copy()
    tinted = image.copy()
    tinted[full_mask > 0] = (0, 220, 255)
    overlay = cv2.addWeighted(tinted, 0.35, overlay, 0.65, 0.0)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    frame = FrameInput(
        frame_idx=frame_idx,
        full_path=image_path,
        crop_path=crop_path,
        mask_path=mask_path,
        crop_bbox_xyxy=(x1, y1, x2, y2),
        sam3_score=float(cable.confidence),
        sam3_area_ratio=float(cable.area_ratio),
    )
    return SampleInput(
        video=str(image_path),
        sample_id=image_path.stem,
        start_sec=None,
        end_sec=None,
        frames=[frame],
    )


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def calculate_metrics(rows: list[dict[str, Any]], elapsed_sec: float) -> dict[str, Any]:
    labels = sorted(CLASS_NAMES)
    confusion = {
        str(gt): {str(pred): 0 for pred in labels + [-1]}
        for gt in labels
    }
    for row in rows:
        confusion[str(row["gt_label"])][str(row["pred_label"])] += 1

    per_class = []
    for label in labels:
        true_positive = confusion[str(label)][str(label)]
        support = sum(confusion[str(label)].values())
        false_positive = sum(
            confusion[str(other)][str(label)]
            for other in labels
            if other != label
        )
        precision = safe_ratio(true_positive, true_positive + false_positive)
        recall = safe_ratio(true_positive, support)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "class_id": label,
                "class_name": CLASS_NAMES[label],
                "support": support,
                "correct": true_positive,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )

    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    classified = sum(int(row["pred_label"]) >= 0 for row in rows)
    return {
        "total": total,
        "classified": classified,
        "unclassified": total - classified,
        "correct": correct,
        "accuracy": round(safe_ratio(correct, total), 6),
        "balanced_accuracy": round(
            float(np.mean([item["recall"] for item in per_class])),
            6,
        ),
        "macro_f1": round(
            float(np.mean([item["f1"] for item in per_class])),
            6,
        ),
        "elapsed_sec": round(elapsed_sec, 3),
        "images_per_sec": round(safe_ratio(total, elapsed_sec), 6),
        "seconds_per_image": round(safe_ratio(elapsed_sec, total), 6),
        "class_order": [CLASS_NAMES[label] for label in labels],
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image",
        "gt_label",
        "gt_class",
        "pred_label",
        "pred_class",
        "correct",
        "damage",
        "damage_confidence",
        "position",
        "position_confidence",
        "reason_codes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fields}
            output["reason_codes"] = json.dumps(
                output["reason_codes"], ensure_ascii=False
            )
            writer.writerow(output)


def write_report(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    order = list(CLASS_NAMES)
    lines = [
        "# SAM3 GT evaluation",
        "",
        f"- Total: {metrics['total']}",
        f"- Classified: {metrics['classified']}",
        f"- Correct: {metrics['correct']}",
        f"- Accuracy: {metrics['accuracy']:.2%}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.2%}",
        f"- Macro F1: {metrics['macro_f1']:.2%}",
        f"- Elapsed: {metrics['elapsed_sec']:.3f} s",
        f"- Throughput: {metrics['images_per_sec']:.4f} image/s",
        "",
        "## Per-class metrics",
        "",
        "| id | class | support | correct | precision | recall | F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["per_class"]:
        lines.append(
            f"| {item['class_id']} | {item['class_name']} | "
            f"{item['support']} | {item['correct']} | "
            f"{item['precision']:.2%} | {item['recall']:.2%} | "
            f"{item['f1']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows are ground truth; columns are predictions.",
            "",
            "| GT / Pred | damaged | exposed_intact | suspended_intact | unclassified |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    confusion = metrics["confusion_matrix"]
    for label in order:
        row = confusion[str(label)]
        lines.append(
            f"| {CLASS_NAMES[label]} | {row['0']} | {row['1']} | "
            f"{row['2']} | {row['-1']} |"
        )

    lines.extend(["", "## Errors", "", "| image | GT | prediction |", "|---|---|---|"])
    errors = [row for row in rows if not row["correct"]]
    if errors:
        for row in errors:
            lines.append(
                f"| {Path(row['image']).name} | {row['gt_class']} | "
                f"{row['pred_class']} |"
            )
    else:
        lines.append("| - | - | - |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_rows = discover_gt_images(args.input, args.limit)
    if args.label:
        selected_labels = set(args.label)
        gt_rows = [row for row in gt_rows if row[1] in selected_labels]
    if not gt_rows:
        raise SystemExit(f"No filename-labeled images found in {args.input}")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if not args.keep_position_disabled:
        config["position"]["enabled"] = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    started = time.perf_counter()
    detector = SAM3PromptDetector(config["model"])
    samples: list[SampleInput] = []
    failures: dict[str, str] = {}
    for index, (image_path, _) in enumerate(gt_rows):
        sample = save_cable_sample(
            image_path,
            index,
            detector,
            samples_dir,
            args.crop_pad_ratio,
            args.crop_min_pad,
        )
        if sample is None:
            failures[str(image_path)] = "cable_not_detected"
        else:
            samples.append(sample)
        print(
            f"[{index + 1:02d}/{len(gt_rows):02d}] cable "
            f"{'ok' if sample is not None else 'failed'}: {image_path.name}",
            flush=True,
        )

    results = classify(samples, detector, config, args.out_dir) if samples else []
    elapsed_sec = time.perf_counter() - started
    result_by_sample = {str(item["sample_id"]): item for item in results}

    predictions: list[dict[str, Any]] = []
    for image_path, gt_label in gt_rows:
        result = result_by_sample.get(image_path.stem)
        pred_label = int(result["class_id"]) if result is not None else -1
        pred_class = (
            CLASS_NAMES[pred_label] if pred_label in CLASS_NAMES else "unclassified"
        )
        predictions.append(
            {
                "image": str(image_path),
                "gt_label": gt_label,
                "gt_class": CLASS_NAMES[gt_label],
                "pred_label": pred_label,
                "pred_class": pred_class,
                "correct": pred_label == gt_label,
                "damage": result.get("damage") if result else None,
                "damage_confidence": (
                    result.get("damage_confidence") if result else None
                ),
                "position": result.get("position") if result else None,
                "position_confidence": (
                    result.get("position_confidence") if result else None
                ),
                "reason_codes": (
                    result.get("reason_codes", [])
                    if result
                    else [failures.get(str(image_path), "unclassified")]
                ),
            }
        )

    metrics = calculate_metrics(predictions, elapsed_sec)
    write_json(args.out_dir / "predictions.json", predictions)
    write_predictions_csv(args.out_dir / "predictions.csv", predictions)
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(args.out_dir / "classification_results.json", results)
    write_csv_results(args.out_dir / "classification_results.csv", results)
    write_review(args.out_dir / "classification_review.md", results)
    write_report(args.out_dir / "report.md", metrics, predictions)
    print(
        f"Evaluation complete: {metrics['correct']}/{metrics['total']} "
        f"({metrics['accuracy']:.2%}) -> {args.out_dir / 'report.md'}"
    )


if __name__ == "__main__":
    main()
