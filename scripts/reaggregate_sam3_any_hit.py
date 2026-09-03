#!/usr/bin/env python
"""Reaggregate cached SAM3 frame evidence with one-hit damage voting."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

try:
    from scripts.classify_samples_gpt import load_json
    from scripts.classify_samples_sam3_only import load_config
    from scripts.classify_samples_zero_shot import (
        CLASS_MAP,
        aggregate_damage,
        write_csv_results,
        write_json,
        write_review,
    )
except ModuleNotFoundError:
    from classify_samples_gpt import load_json
    from classify_samples_sam3_only import load_config
    from classify_samples_zero_shot import (
        CLASS_MAP,
        aggregate_damage,
        write_csv_results,
        write_json,
        write_review,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


def apply_cached_decision_mode(
    decision: dict[str, Any],
    damage_config: dict[str, Any],
) -> None:
    """Reapply config-only decisions to cached SAM3 prompt scores."""
    mode = str(damage_config.get("decision_mode", "fused"))
    if mode == "fused":
        return
    if mode != "direct_prompt_threshold":
        raise ValueError(f"Unsupported damage decision_mode: {mode}")
    raw_thresholds = damage_config.get(
        "direct_positive_prompt_thresholds",
        {},
    )
    thresholds = (
        {
            str(prompt): float(threshold)
            for prompt, threshold in raw_thresholds.items()
        }
        if isinstance(raw_thresholds, dict)
        else {}
    )
    scores = decision.get("local_prompt_scores", {})
    hits = [
        prompt
        for prompt, threshold in thresholds.items()
        if float(scores.get(prompt, 0.0)) >= threshold
    ]
    decision["decision_mode"] = mode
    decision["direct_positive_prompts"] = hits
    decision["strong"] = bool(hits)
    decision["positive"] = bool(hits)


def reaggregate(
    source_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply one-hit aggregation without rerunning SAM3 image inference."""
    damage_config = {**config["damage"], "min_positive_frames": 1}
    output_method = (
        "sam3_direct_marker_any_hit"
        if damage_config.get("decision_mode") == "direct_prompt_threshold"
        else "sam3_damage_any_hit"
    )
    results = copy.deepcopy(source_results)
    for result in results:
        frame_rows = result.get("diagnostics", {}).get("frames", [])
        frame_decisions = [row["damage"] for row in frame_rows]
        for decision in frame_decisions:
            apply_cached_decision_mode(decision, damage_config)
        damage, confidence = aggregate_damage(frame_decisions, damage_config)
        class_id, class_name = CLASS_MAP[("exposed", damage)]
        result.update(
            {
                "position": "exposed",
                "position_confidence": 1.0,
                "position_evaluated": False,
                "damage": damage,
                "damage_confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "method": output_method,
                "evidence_consistency": "consistent",
            }
        )
        reason_codes: list[str] = []
        if damage == "intact" and any(
            float(item["positive_score"])
            >= float(damage_config.get("weak_prompt_score", 0.0))
            for item in frame_decisions
        ):
            reason_codes.append("weak_damage_evidence")
        result["reason_codes"] = reason_codes
        result["needs_review"] = bool(reason_codes)
        for vote, decision in zip(
            result.get("frame_votes", []),
            frame_decisions,
        ):
            vote["position"] = "exposed"
            vote["position_confidence"] = 1.0
            vote["damage_evidence"] = (
                "clear_damage"
                if bool(decision["positive"])
                else "no_visible_damage"
            )
        diagnostics = result.setdefault("diagnostics", {})
        diagnostics["position_agreement"] = 1.0
        diagnostics["positive_damage_frames"] = sum(
            bool(item["positive"]) for item in frame_decisions
        )
        diagnostics["strong_damage_frames"] = sum(
            bool(item["strong"]) for item in frame_decisions
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sam3_only_classifier.yaml",
    )
    args = parser.parse_args()

    source = load_json(args.input_results)
    if not isinstance(source, list):
        raise SystemExit(f"Expected a JSON list: {args.input_results}")
    results = reaggregate(source, load_config(args.config))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "results.json", results)
    write_csv_results(args.out_dir / "results.csv", results)
    write_review(args.out_dir / "review.md", results)
    write_json(
        args.out_dir / "summary.json",
        {
            "method": (
                results[0]["method"]
                if results
                else "sam3_damage_any_hit"
            ),
            "total": len(results),
            "damaged": sum(item["damage"] == "damaged" for item in results),
            "non_damaged": sum(item["damage"] != "damaged" for item in results),
        },
    )
    print(
        f"One-hit reaggregation complete: {len(results)} samples -> "
        f"{args.out_dir / 'results.json'}"
    )


if __name__ == "__main__":
    main()
