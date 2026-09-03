#!/usr/bin/env python
"""Compare midterm classifications with class suffixes in the GT directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


GT_PATTERN = re.compile(r"^sample(?P<sample>\d+)_(?P<class_id>[012])$")
CLASS_NAMES = {0: "damaged", 1: "exposed_intact", 2: "suspended_intact"}
RESULT_CLASS_NAME_TO_GT_ID = {
    "damaged": 0,
    "exposed_intact": 1,
    "suspended_intact": 2,
}


def load_ground_truth(gt_dir: Path) -> dict[int, int]:
    """Read sample number and class ID from sampleNN_CLASS image names."""
    labels: dict[int, int] = {}
    for path in sorted(gt_dir.iterdir()):
        if not path.is_file():
            continue
        match = GT_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        sample = int(match.group("sample"))
        class_id = int(match.group("class_id"))
        if sample in labels and labels[sample] != class_id:
            raise ValueError(f"Conflicting labels for sample {sample}: {gt_dir}")
        labels[sample] = class_id
    if not labels:
        raise ValueError(f"No sampleNN_CLASS images found in {gt_dir}")
    return labels


def sample_number(result: dict[str, Any]) -> int:
    video = Path(str(result["video"]))
    if video.stem.isdigit():
        return int(video.stem)
    sample_id = str(result.get("sample_id", ""))
    match = re.fullmatch(
        r"(?:S|sample)?0*(\d+)(?:_[012])?",
        sample_id,
        re.IGNORECASE,
    )
    if match is not None:
        return int(match.group(1))
    raise ValueError(
        f"Neither video stem nor sample_id identifies a sample: {video}, "
        f"sample_id={sample_id!r}"
    )


def prediction_gt_class_id(result: dict[str, Any]) -> int:
    """Translate project result classes to the GT filename suffix scheme."""
    class_name = str(result.get("class_name", ""))
    if class_name in RESULT_CLASS_NAME_TO_GT_ID:
        return RESULT_CLASS_NAME_TO_GT_ID[class_name]
    return int(result["class_id"])


def evaluate(
    results: list[dict[str, Any]],
    labels: dict[int, int],
) -> dict[str, Any]:
    predictions = {
        sample_number(item): prediction_gt_class_id(item) for item in results
    }
    missing = sorted(set(labels) - set(predictions))
    extra = sorted(set(predictions) - set(labels))
    confusion = [[0 for _ in range(3)] for _ in range(3)]
    errors: list[dict[str, int]] = []
    per_sample: list[dict[str, Any]] = []
    for sample in sorted(set(labels) & set(predictions)):
        truth = labels[sample]
        prediction = predictions[sample]
        confusion[truth][prediction] += 1
        per_sample.append(
            {
                "sample": sample,
                "truth": truth,
                "truth_name": CLASS_NAMES[truth],
                "prediction": prediction,
                "prediction_name": CLASS_NAMES[prediction],
                "correct": truth == prediction,
            }
        )
        if truth != prediction:
            errors.append(
                {"sample": sample, "truth": truth, "prediction": prediction}
            )
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(3))
    true_positive = confusion[0][0]
    false_negative = sum(confusion[0][1:])
    false_positive = sum(confusion[row][0] for row in (1, 2))
    true_negative = total - true_positive - false_negative - false_positive
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    damage_precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    damage_recall = (
        true_positive / recall_denominator if recall_denominator else 0.0
    )
    f1_denominator = damage_precision + damage_recall
    return {
        "class_names": CLASS_NAMES,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "confusion_matrix_rows_truth_columns_prediction": confusion,
        "damage_binary": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "accuracy": round(
                (true_positive + true_negative) / total,
                4,
            ) if total else None,
            "precision": round(damage_precision, 4),
            "recall": round(damage_recall, 4),
            "f1": round(
                2 * damage_precision * damage_recall / f1_denominator,
                4,
            ) if f1_denominator else 0.0,
        },
        "per_sample": per_sample,
        "errors": errors,
        "missing_predictions": missing,
        "extra_predictions": extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("/home/nvidia/DATA/UW/mid/gt"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise SystemExit(f"Expected a JSON list: {args.results}")
    report = evaluate(results, load_ground_truth(args.gt_dir))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
