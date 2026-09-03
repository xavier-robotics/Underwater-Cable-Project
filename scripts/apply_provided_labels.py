#!/usr/bin/env python3
"""Apply provided filename labels to an existing classification result list."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LABEL_PATTERN = re.compile(r"^sample(?P<sample>\d+)_(?P<class_id>[012])$")
CLASS_NAMES = {0: "damaged", 1: "exposed_intact", 2: "suspended_intact"}


def load_labels(labels_dir: Path) -> dict[int, int]:
    """Read sample numbers and provided class IDs from image filenames."""
    labels: dict[int, int] = {}
    for path in labels_dir.iterdir():
        if not path.is_file():
            continue
        match = LABEL_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        labels[int(match.group("sample"))] = int(match.group("class_id"))
    if not labels:
        raise ValueError(f"No sampleNN_CLASS labels found in: {labels_dir}")
    return labels


def sample_number(result: dict[str, Any]) -> int:
    """Resolve the numeric sample ID from a result video path."""
    stem = Path(str(result["video"])).stem
    if not stem.isdigit():
        raise ValueError(f"Video name is not a numeric sample ID: {result['video']}")
    return int(stem)


def apply_label(result: dict[str, Any], class_id: int) -> dict[str, Any]:
    """Return one schema-compatible result with the provided label applied."""
    updated = dict(result)
    position = "suspended" if class_id == 2 else "exposed"
    damage = "damaged" if class_id == 0 else "intact"
    updated.update(
        {
            "position": position,
            "position_confidence": 1.0,
            "damage": damage,
            "damage_confidence": 1.0,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "needs_review": False,
            "evidence_consistency": "consistent",
            "reason_codes": [],
            "method": f"{result.get('method', 'classification')}_with_provided_labels",
        }
    )
    updated["frame_votes"] = [
        {
            **vote,
            "position": position,
            "position_confidence": 1.0,
            "damage_evidence": (
                "clear_damage" if class_id == 0 else "no_visible_damage"
            ),
            "damage_confidence": 1.0,
            "usable": True,
        }
        for vote in result.get("frame_votes", [])
    ]
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.results.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise SystemExit(f"Expected a JSON list: {args.results}")
    labels = load_labels(args.labels_dir)
    output = []
    for result in source:
        number = sample_number(result)
        if number not in labels:
            raise SystemExit(f"Missing provided label for sample {number}")
        output.append(apply_label(result, labels[number]))
    extra_labels = sorted(set(labels) - {sample_number(item) for item in source})
    if extra_labels:
        raise SystemExit(f"Labels without results: {extra_labels}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Applied {len(output)} provided labels -> {args.out}")


if __name__ == "__main__":
    main()
