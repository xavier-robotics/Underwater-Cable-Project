#!/usr/bin/env python
"""Validate Agent classifications and decide which samples need another attempt."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CLASS_MAP = {
    ("exposed", "intact"): (1, "exposed_intact"),
    ("exposed", "damaged"): (0, "damaged"),
    ("suspended", "intact"): (2, "suspended_intact"),
    ("suspended", "damaged"): (0, "damaged"),
}
EVIDENCE_CONSISTENCY_VALUES = {"consistent", "mixed", "insufficient"}
DAMAGE_EVIDENCE_VALUES = {"clear_damage", "no_visible_damage", "uncertain"}


def load_json_list(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if missing_ok and not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def sample_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("video", "")), str(item.get("sample_id", ""))


def valid_confidence(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= number <= 1.0


def majority_value(values: list[str], threshold: float) -> tuple[str | None, float]:
    if not values:
        return None, 0.0
    counts = Counter(values)
    winner, winner_count = counts.most_common(1)[0]
    ratio = winner_count / len(values)
    tied = sum(count == winner_count for count in counts.values()) > 1
    if tied or ratio < threshold:
        return None, ratio
    return winner, ratio


def request_frame_count(request: dict[str, Any]) -> int:
    value = request.get("selected_frame_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value

    images = request.get("images", [])
    if isinstance(images, list):
        frame_ids = {
            image.get("frame_idx")
            for image in images
            if isinstance(image, dict) and image.get("frame_idx") is not None
        }
        if frame_ids:
            return len(frame_ids)
    image_paths = request.get("image_paths", [])
    return len(image_paths) if isinstance(image_paths, list) else 0


def request_frame_ids(request: dict[str, Any]) -> set[Any]:
    declared_frame_ids = request.get("frame_ids")
    if isinstance(declared_frame_ids, list):
        return set(declared_frame_ids)
    images = request.get("images", [])
    if not isinstance(images, list):
        return set()
    return {
        image.get("frame_idx")
        for image in images
        if isinstance(image, dict) and image.get("frame_idx") is not None
    }


def validate_results(
    requests: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    confidence_threshold: float = 0.65,
    min_frames: int = 3,
    position_majority_threshold: float = 0.6,
    clear_damage_threshold: float = 0.75,
) -> dict[str, Any]:
    """Return per-sample pass/retry decisions without trusting Agent review flags."""
    result_counts = Counter(sample_key(result) for result in results)
    results_by_key = {sample_key(result): result for result in results}
    request_keys = {sample_key(request) for request in requests}
    items = []

    for request in requests:
        key = sample_key(request)
        result = results_by_key.get(key)
        reasons: list[str] = []
        diagnostics = {
            "usable_frames": 0,
            "position_agreement": 0.0,
            "clear_damage_frames": 0,
        }

        if not key[0] or not key[1]:
            reasons.append("invalid_request_identity")
        if result_counts[key] == 0:
            reasons.append("missing_result")
        elif result_counts[key] > 1:
            reasons.append("duplicate_result")

        frame_count = request_frame_count(request)
        if frame_count < min_frames:
            reasons.append("insufficient_frames")

        request_paths = request.get("image_paths", [])
        if not isinstance(request_paths, list) or not request_paths:
            reasons.append("missing_image_paths")
            request_paths = []
        elif any(
            not Path(str(path)).is_file() or Path(str(path)).stat().st_size <= 0
            for path in request_paths
        ):
            reasons.append("unreadable_image")

        frame_ids = request_frame_ids(request)
        request_images = request.get("images", [])
        if request.get("evidence_mode") == "full-and-crop":
            roles_by_frame: dict[Any, set[str]] = {frame_id: set() for frame_id in frame_ids}
            image_manifest_paths = []
            for image in request_images if isinstance(request_images, list) else []:
                if not isinstance(image, dict):
                    continue
                roles_by_frame.setdefault(image.get("frame_idx"), set()).add(str(image.get("kind", "")))
                image_manifest_paths.append(image.get("path"))
            if any("context" not in roles for roles in roles_by_frame.values()):
                reasons.append("missing_context_evidence")
            if any("detail" not in roles for roles in roles_by_frame.values()):
                reasons.append("missing_detail_evidence")
            if image_manifest_paths != request_paths:
                reasons.append("image_manifest_mismatch")
        elif request.get("evidence_mode") == "contact-sheet":
            if (
                not isinstance(request_images, list)
                or len(request_images) != 1
                or not isinstance(request_images[0], dict)
                or request_images[0].get("kind") != "contact_sheet"
            ):
                reasons.append("invalid_contact_sheet")
            source_frames = request.get("source_frames")
            if (
                not isinstance(source_frames, list)
                or len(source_frames) != frame_count
                or any(not isinstance(frame, dict) for frame in source_frames)
            ):
                reasons.append("invalid_source_frames")
            else:
                if any(not frame.get("has_context") for frame in source_frames):
                    reasons.append("missing_context_evidence")
                if any(not frame.get("has_detail") for frame in source_frames):
                    reasons.append("missing_detail_evidence")

        if result is not None:
            position = result.get("position")
            damage = result.get("damage")
            if position not in {"exposed", "suspended"}:
                reasons.append("invalid_position")
            if damage not in {"intact", "damaged"}:
                reasons.append("invalid_damage")

            if position in {"exposed", "suspended"} and damage in {"intact", "damaged"}:
                expected_id, expected_name = CLASS_MAP[(position, damage)]
                if result.get("class_id") != expected_id or result.get("class_name") != expected_name:
                    reasons.append("class_mapping_mismatch")

            for field in ("position_confidence", "damage_confidence"):
                if not valid_confidence(result.get(field)):
                    reasons.append(f"invalid_{field}")
                elif float(result[field]) < confidence_threshold:
                    reasons.append(f"low_{field}")

            needs_review = result.get("needs_review")
            if not isinstance(needs_review, bool):
                reasons.append("invalid_needs_review")
            elif needs_review:
                reasons.append("agent_requested_review")

            consistency = result.get("evidence_consistency")
            if consistency not in EVIDENCE_CONSISTENCY_VALUES:
                reasons.append("invalid_evidence_consistency")
            elif consistency == "insufficient":
                reasons.append("insufficient_evidence")

            if not isinstance(result.get("reason_codes"), list):
                reasons.append("invalid_reason_codes")

            if frame_ids:
                frame_votes = result.get("frame_votes")
                if not isinstance(frame_votes, list):
                    reasons.append("invalid_frame_votes")
                    frame_votes = []
                vote_ids = [
                    vote.get("frame_idx")
                    for vote in frame_votes
                    if isinstance(vote, dict)
                ]
                if len(vote_ids) != len(set(vote_ids)):
                    reasons.append("duplicate_frame_vote")
                if set(vote_ids) != frame_ids:
                    reasons.append("frame_vote_coverage_mismatch")

                usable_votes = []
                valid_position_votes = []
                clear_damage_votes = []
                for vote in frame_votes:
                    if not isinstance(vote, dict):
                        reasons.append("invalid_frame_vote")
                        continue
                    if not isinstance(vote.get("usable"), bool):
                        reasons.append("invalid_frame_vote_usable")
                        continue
                    if vote["usable"]:
                        if vote.get("position") not in {"exposed", "suspended"}:
                            reasons.append("invalid_frame_vote_position")
                        elif not valid_confidence(vote.get("position_confidence")):
                            reasons.append("invalid_frame_vote_position_confidence")
                        else:
                            valid_position_votes.append(str(vote["position"]))

                        damage_evidence = vote.get("damage_evidence")
                        if damage_evidence not in DAMAGE_EVIDENCE_VALUES:
                            reasons.append("invalid_frame_vote_damage_evidence")
                        elif not valid_confidence(vote.get("damage_confidence")):
                            reasons.append("invalid_frame_vote_damage_confidence")
                        elif (
                            damage_evidence == "clear_damage"
                            and float(vote["damage_confidence"]) >= clear_damage_threshold
                        ):
                            clear_damage_votes.append(vote)
                        usable_votes.append(vote)

                diagnostics["usable_frames"] = len(usable_votes)
                diagnostics["clear_damage_frames"] = len(clear_damage_votes)
                if len(usable_votes) < min_frames:
                    reasons.append("insufficient_usable_frames")

                majority_position, agreement = majority_value(
                    valid_position_votes,
                    position_majority_threshold,
                )
                diagnostics["position_agreement"] = round(agreement, 4)
                if valid_position_votes and majority_position is None:
                    reasons.append("uncertain_position_votes")
                elif majority_position is not None and position != majority_position:
                    reasons.append("aggregate_position_mismatch")

                if damage == "damaged" and not clear_damage_votes:
                    reasons.append("unsupported_damage_evidence")
                elif damage == "intact" and clear_damage_votes:
                    reasons.append("aggregate_damage_mismatch")

            if result.get("image_paths") != request_paths:
                reasons.append("image_paths_mismatch")
            if result.get("attempt_id") != request.get("attempt_id"):
                reasons.append("attempt_id_mismatch")

        unique_reasons = list(dict.fromkeys(reasons))
        status = "pass" if not unique_reasons else "retry"
        items.append(
            {
                "video": key[0],
                "sample_id": key[1],
                "attempt_id": request.get("attempt_id"),
                "status": status,
                "reason_codes": unique_reasons,
                "retry_action": "none" if status == "pass" else "more_evidence",
                "diagnostics": diagnostics,
            }
        )

    unexpected_results = [
        {"video": key[0], "sample_id": key[1]}
        for key in sorted(set(results_by_key) - request_keys)
    ]
    pass_count = sum(item["status"] == "pass" for item in items)
    return {
        "summary": {
            "total": len(items),
            "pass": pass_count,
            "retry": len(items) - pass_count,
            "confidence_threshold": confidence_threshold,
            "min_frames": min_frames,
            "position_majority_threshold": position_majority_threshold,
            "clear_damage_threshold": clear_damage_threshold,
            "unexpected_results": len(unexpected_results),
        },
        "items": items,
        "unexpected_results": unexpected_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--position-majority-threshold", type=float, default=0.6)
    parser.add_argument("--clear-damage-threshold", type=float, default=0.75)
    parser.add_argument(
        "--missing-results-ok",
        action="store_true",
        help="Treat a missing results file as an empty result list.",
    )
    args = parser.parse_args()

    requests = load_json_list(args.requests)
    results = load_json_list(args.results, missing_ok=args.missing_results_ok)
    report = validate_results(
        requests,
        results,
        confidence_threshold=args.confidence_threshold,
        min_frames=args.min_frames,
        position_majority_threshold=args.position_majority_threshold,
        clear_damage_threshold=args.clear_damage_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        f"Validated {summary['total']} samples: "
        f"{summary['pass']} pass, {summary['retry']} retry -> {args.out}"
    )


if __name__ == "__main__":
    main()
