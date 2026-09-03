#!/usr/bin/env python
"""Compare two sample-classification result files with the current three-class map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def sample_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("video", "")), str(item.get("sample_id", ""))


def normalized_class_id(item: dict[str, Any]) -> int | None:
    damage = item.get("damage")
    position = item.get("position")
    if damage == "damaged":
        return 0
    if damage == "intact" and position == "exposed":
        return 1
    if damage == "intact" and position == "suspended":
        return 2
    class_id = item.get("class_id")
    return int(class_id) if class_id in {0, 1, 2} else None


def compare(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_map = {sample_key(item): item for item in baseline}
    candidate_map = {sample_key(item): item for item in candidate}
    keys = sorted(set(baseline_map) | set(candidate_map))
    items = []
    matched = 0
    comparable = 0
    for key in keys:
        baseline_item = baseline_map.get(key)
        candidate_item = candidate_map.get(key)
        baseline_class = normalized_class_id(baseline_item or {})
        candidate_class = normalized_class_id(candidate_item or {})
        agrees = (
            baseline_class is not None
            and candidate_class is not None
            and baseline_class == candidate_class
        )
        if baseline_class is not None and candidate_class is not None:
            comparable += 1
            matched += int(agrees)
        items.append(
            {
                "video": key[0],
                "sample_id": key[1],
                "baseline_class_id": baseline_class,
                "candidate_class_id": candidate_class,
                "agree": agrees,
                "baseline_position": (baseline_item or {}).get("position"),
                "candidate_position": (candidate_item or {}).get("position"),
                "baseline_damage": (baseline_item or {}).get("damage"),
                "candidate_damage": (candidate_item or {}).get("damage"),
            }
        )
    return {
        "summary": {
            "baseline_total": len(baseline),
            "candidate_total": len(candidate),
            "comparable": comparable,
            "matched": matched,
            "agreement": round(matched / comparable, 4) if comparable else None,
        },
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    report = compare(load_results(args.baseline), load_results(args.candidate))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(report["items"][0]) if report["items"] else []
        writer = csv.DictWriter(file, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(report["items"])
    print(
        f"Agreement: {report['summary']['matched']}/{report['summary']['comparable']} "
        f"({report['summary']['agreement']}) -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
