#!/usr/bin/env python
"""Run batched Codex classification with independent result validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.validate_agent_results import CLASS_MAP, load_json_list, sample_key, validate_results
except ModuleNotFoundError:
    from validate_agent_results import CLASS_MAP, load_json_list, sample_key, validate_results


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_SCRIPT = REPO_ROOT / "scripts" / "classify_samples_codex_agent.sh"
TECHNICAL_RETRY_REASONS = {
    "missing_result",
    "duplicate_result",
    "invalid_position",
    "invalid_damage",
    "class_mapping_mismatch",
    "invalid_position_confidence",
    "invalid_damage_confidence",
    "invalid_needs_review",
    "invalid_evidence_consistency",
    "invalid_reason_codes",
    "invalid_frame_votes",
    "duplicate_frame_vote",
    "frame_vote_coverage_mismatch",
    "invalid_frame_vote",
    "invalid_frame_vote_usable",
    "invalid_frame_vote_position",
    "invalid_frame_vote_position_confidence",
    "invalid_frame_vote_damage_evidence",
    "invalid_frame_vote_damage_confidence",
    "image_paths_mismatch",
    "attempt_id_mismatch",
}
UNCERTAINTY_RETRY_REASONS = {
    "low_position_confidence",
    "low_damage_confidence",
    "agent_requested_review",
    "insufficient_evidence",
    "insufficient_usable_frames",
    "uncertain_position_votes",
    "aggregate_position_mismatch",
    "unsupported_damage_evidence",
    "aggregate_damage_mismatch",
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_signature(
    requests: list[dict[str, Any]],
    *,
    model: str,
    task_file: Path,
) -> str:
    image_hashes = []
    for request in requests:
        for raw_path in request.get("image_paths", []):
            path = Path(str(raw_path))
            image_hashes.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path) if path.is_file() else None,
                }
            )
    payload = {
        "model": model,
        "task_sha256": file_sha256(task_file),
        "requests": requests,
        "images": image_hashes,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_matches(cache_path: Path, results_path: Path, signature: str) -> bool:
    if not cache_path.is_file() or not results_path.is_file():
        return False
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        cache.get("signature") == signature
        and cache.get("results_sha256") == file_sha256(results_path)
    )


def is_technical_retry(item: dict[str, Any]) -> bool:
    return any(reason in TECHNICAL_RETRY_REASONS for reason in item.get("reason_codes", []))


def should_retry_item(item: dict[str, Any], policy: str) -> bool:
    reasons = set(item.get("reason_codes", []))
    if policy == "none":
        return False
    if policy == "all_failures":
        return True
    if policy == "uncertain":
        return bool(reasons & (TECHNICAL_RETRY_REASONS | UNCERTAINTY_RETRY_REASONS))
    return bool(reasons & TECHNICAL_RETRY_REASONS)


def normalize_class_mapping(result: dict[str, Any]) -> bool:
    """Repair the deterministic class fields when aggregate labels are valid."""
    position = result.get("position")
    damage = result.get("damage")
    if not isinstance(position, str) or not isinstance(damage, str):
        return False
    if (position, damage) not in CLASS_MAP:
        return False
    class_id, class_name = CLASS_MAP[(position, damage)]
    result["class_id"] = class_id
    result["class_name"] = class_name
    return True


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Closed-loop config must be a mapping: {path}")
    rounds = data.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Closed-loop config must define at least one round")
    max_attempts = int(data.get("max_attempts", len(rounds)))
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    data["rounds"] = rounds[:max_attempts]
    data["max_attempts"] = min(max_attempts, len(rounds))
    return data


def run_agent_attempt(
    *,
    input_path: Path,
    attempt_dir: Path,
    model: str,
    image_field: str,
    attempt_id: int,
    round_config: dict[str, Any],
    include_file: Path | None,
    requests_file: Path | None,
    limit: int | None,
    prepare_only: bool,
    execute_only: bool,
) -> None:
    command = [
        str(AGENT_SCRIPT),
        "--input",
        str(input_path),
        "--out-dir",
        str(attempt_dir),
        "--model",
        model,
        "--image-field",
        image_field,
        "--attempt-id",
        str(attempt_id),
        "--evidence-mode",
        str(round_config.get("evidence_mode", "contact-sheet")),
        "--max-frames-per-sample",
        str(int(round_config.get("max_frames_per_sample", 3))),
    ]
    if include_file is not None:
        command.extend(["--include-file", str(include_file)])
    if requests_file is not None:
        command.extend(["--requests-file", str(requests_file)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if prepare_only:
        command.append("--prepare-only")
    if execute_only:
        command.append("--execute-only")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def result_map(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {sample_key(result): result for result in results}


def execute_agent_batch(
    *,
    input_path: Path,
    batch_dir: Path,
    batch_manifest: Path,
    batch_requests: list[dict[str, Any]],
    model: str,
    image_field: str,
    attempt_id: int,
    round_config: dict[str, Any],
    confidence_threshold: float,
    position_majority_threshold: float,
    clear_damage_threshold: float,
    cache_enabled: bool,
    finalization_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    write_json(batch_manifest, batch_requests)
    run_agent_attempt(
        input_path=input_path,
        attempt_dir=batch_dir,
        model=model,
        image_field=image_field,
        attempt_id=attempt_id,
        round_config=round_config,
        include_file=None,
        requests_file=batch_manifest,
        limit=None,
        prepare_only=True,
        execute_only=False,
    )

    task_file = batch_dir / "codex_classification_task.md"
    results_path = batch_dir / "results.json"
    cache_path = batch_dir / "result_cache.json"
    signature = build_cache_signature(batch_requests, model=model, task_file=task_file)
    cache_hit = cache_enabled and cache_matches(cache_path, results_path, signature)
    agent_returncode = 0
    if not cache_hit:
        results_path.unlink(missing_ok=True)
        try:
            run_agent_attempt(
                input_path=input_path,
                attempt_dir=batch_dir,
                model=model,
                image_field=image_field,
                attempt_id=attempt_id,
                round_config=round_config,
                include_file=None,
                requests_file=batch_manifest,
                limit=None,
                prepare_only=False,
                execute_only=True,
            )
        except subprocess.CalledProcessError as error:
            agent_returncode = error.returncode

    results = load_json_list(results_path, missing_ok=True)
    if results:
        for result in results:
            normalize_class_mapping(result)
        write_json(results_path, results)
    min_frames = int(
        round_config.get(
            "min_frames",
            round_config.get("max_frames_per_sample", 3),
        )
    )
    report = validate_results(
        batch_requests,
        results,
        confidence_threshold=confidence_threshold,
        min_frames=min_frames,
        position_majority_threshold=position_majority_threshold,
        clear_damage_threshold=clear_damage_threshold,
    )
    validation_path = batch_dir / "validation.json"
    write_json(validation_path, report)

    if finalization_policy == "best_effort":
        result_counts = Counter(sample_key(result) for result in results)
        results_by_key = result_map(results)
        has_technical_errors = (
            report["summary"]["unexpected_results"] > 0
            or any(
                result_counts[sample_key(request)] != 1
                or not normalize_class_mapping(
                    results_by_key.get(sample_key(request), {})
                )
                for request in batch_requests
            )
        )
    else:
        has_technical_errors = (
            report["summary"]["unexpected_results"] > 0
            or any(
                item["status"] == "retry" and is_technical_retry(item)
                for item in report["items"]
            )
        )
    if (
        cache_enabled
        and not cache_hit
        and agent_returncode == 0
        and results_path.is_file()
        and not has_technical_errors
    ):
        write_json(
            cache_path,
            {
                "signature": signature,
                "results_sha256": file_sha256(results_path),
                "sample_count": len(batch_requests),
            },
        )

    metadata = {
        "directory": str(batch_dir),
        "sample_count": len(batch_requests),
        "cache_hit": cache_hit,
        "agent_returncode": agent_returncode,
        "validation": str(validation_path),
        "summary": report["summary"],
    }
    return results, metadata


def write_final_outputs(
    work_dir: Path,
    original_requests: list[dict[str, Any]],
    latest_results: dict[tuple[str, str], dict[str, Any]],
    final_decisions: dict[tuple[str, str], dict[str, Any]],
    *,
    finalization_policy: str,
) -> None:
    ordered_results = [
        latest_results[sample_key(request)]
        for request in original_requests
        if sample_key(request) in latest_results
    ]
    write_json(work_dir / "results.json", ordered_results)

    fields = [
        "video",
        "sample_id",
        "attempt_id",
        "start_sec",
        "end_sec",
        "position",
        "position_confidence",
        "damage",
        "damage_confidence",
        "class_id",
        "class_name",
        "needs_review",
        "evidence_consistency",
        "reason_codes",
        "frame_votes",
        "image_paths",
    ]
    with (work_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in ordered_results:
            row = {field: result.get(field) for field in fields}
            for field in ("reason_codes", "frame_votes", "image_paths"):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)

    review_lines = [
        "| video | sample_id | attempt | status | position | damage | class_id | reason_codes |",
        "|---|---|---:|---|---|---|---:|---|",
    ]
    human_review = []
    warnings = []
    unclassified = []
    for request in original_requests:
        key = sample_key(request)
        decision = final_decisions[key]
        result = latest_results.get(key, {})
        reasons = decision.get("reason_codes", [])
        status = decision["status"]
        review_lines.append(
            "| {video} | {sample_id} | {attempt} | {status} | {position} | {damage} | "
            "{class_id} | {reasons} |".format(
                video=Path(key[0]).name,
                sample_id=key[1],
                attempt=decision.get("attempt_id", ""),
                status=status,
                position=result.get("position", ""),
                damage=result.get("damage", ""),
                class_id=result.get("class_id", ""),
                reasons=", ".join(reasons),
            )
        )
        if status == "human_review":
            human_review.append({**decision, "result": result or None})
        elif status == "classified_with_warnings":
            warnings.append({**decision, "result": result or None})
        elif status == "unclassified":
            unclassified.append({**decision, "result": result or None})

    (work_dir / "review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    if finalization_policy == "best_effort":
        (work_dir / "human_review.json").unlink(missing_ok=True)
        write_json(work_dir / "warnings.json", warnings)
        write_json(work_dir / "unclassified.json", unclassified)
    else:
        write_json(work_dir / "human_review.json", human_review)
        (work_dir / "warnings.json").unlink(missing_ok=True)
        (work_dir / "unclassified.json").unlink(missing_ok=True)

    clean = sum(item["status"] == "pass" for item in final_decisions.values())
    classified = clean + len(warnings)
    quality_lines = [
        "# Classification quality check",
        "",
        f"- total_samples: {len(original_requests)}",
        f"- classified: {classified}",
        f"- clean: {clean}",
        f"- classified_with_warnings: {len(warnings)}",
        f"- unclassified: {len(unclassified)}",
        f"- human_review: {len(human_review)}",
        f"- results_json: {(work_dir / 'results.json').is_file()}",
        f"- results_csv: {(work_dir / 'results.csv').is_file()}",
        f"- review_md: {(work_dir / 'review.md').is_file()}",
    ]
    if finalization_policy == "best_effort":
        quality_lines.extend(
            [
                f"- warnings_json: {(work_dir / 'warnings.json').is_file()}",
                f"- unclassified_json: {(work_dir / 'unclassified.json').is_file()}",
            ]
        )
    else:
        quality_lines.append(
            f"- human_review_json: {(work_dir / 'human_review.json').is_file()}"
        )
    (work_dir / "quality_check.md").write_text("\n".join(quality_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "closed_loop.yaml",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--image-field",
        default="full_frame_path",
        choices=["masked_crop_path", "crop_path", "full_frame_path", "path"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the first Agent request without invoking Codex.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    confidence_threshold = float(config.get("confidence_threshold", 0.65))
    position_majority_threshold = float(config.get("position_majority_threshold", 0.6))
    clear_damage_threshold = float(config.get("clear_damage_threshold", 0.75))
    retry_policy = str(config.get("retry_policy", "all_failures"))
    finalization_policy = str(config.get("finalization_policy", "strict"))
    if finalization_policy not in {"strict", "best_effort"}:
        raise ValueError("finalization_policy must be strict or best_effort")
    cache_enabled = bool(config.get("cache_results", True))
    agent_batch_size = max(1, int(config.get("agent_batch_size", 6)))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.work_dir / "closed_loop_state.json"
    state: dict[str, Any] = {
        "status": "running",
        "input": str(args.input),
        "config": str(args.config),
        "model": args.model,
        "max_attempts": config["max_attempts"],
        "retry_policy": retry_policy,
        "finalization_policy": finalization_policy,
        "cache_results": cache_enabled,
        "agent_batch_size": agent_batch_size,
        "attempts": [],
    }
    write_json(state_path, state)

    include_file: Path | None = None
    original_requests: list[dict[str, Any]] = []
    latest_results: dict[tuple[str, str], dict[str, Any]] = {}
    final_decisions: dict[tuple[str, str], dict[str, Any]] = {}

    for attempt_index, round_config in enumerate(config["rounds"], start=1):
        attempt_dir = args.work_dir / f"attempt_{attempt_index}"
        try:
            run_agent_attempt(
                input_path=args.input,
                attempt_dir=attempt_dir,
                model=args.model,
                image_field=args.image_field,
                attempt_id=attempt_index,
                round_config=round_config,
                include_file=include_file,
                requests_file=None,
                limit=args.limit,
                prepare_only=True,
                execute_only=False,
            )
        except subprocess.CalledProcessError:
            state["status"] = "failed"
            state["error"] = f"Agent attempt {attempt_index} failed before requests were prepared"
            write_json(state_path, state)
            raise

        requests = load_json_list(attempt_dir / "requests.json")
        if attempt_index == 1:
            original_requests = requests

        attempt_state = {
            "attempt_id": attempt_index,
            "directory": str(attempt_dir),
            "sample_count": len(requests),
            "max_frames_per_sample": int(round_config.get("max_frames_per_sample", 3)),
            "evidence_mode": str(round_config.get("evidence_mode", "contact-sheet")),
            "batch_size": agent_batch_size,
            "batches": [],
        }
        state["attempts"].append(attempt_state)

        request_batches = chunked(requests, agent_batch_size)
        if args.prepare_only:
            for batch_index, batch_requests in enumerate(request_batches, start=1):
                batch_dir = attempt_dir / "batches" / f"batch_{batch_index:03d}"
                batch_manifest = (
                    attempt_dir / "batch_manifests" / f"batch_{batch_index:03d}.json"
                )
                write_json(batch_manifest, batch_requests)
                run_agent_attempt(
                    input_path=args.input,
                    attempt_dir=batch_dir,
                    model=args.model,
                    image_field=args.image_field,
                    attempt_id=attempt_index,
                    round_config=round_config,
                    include_file=None,
                    requests_file=batch_manifest,
                    limit=None,
                    prepare_only=True,
                    execute_only=False,
                )
                attempt_state["batches"].append(
                    {
                        "batch_id": batch_index,
                        "directory": str(batch_dir),
                        "sample_count": len(batch_requests),
                        "status": "prepared",
                    }
                )
            state["status"] = "prepared"
            write_json(state_path, state)
            print(
                f"Prepared closed-loop attempt 1: {len(requests)} samples in "
                f"{len(request_batches)} batches -> {attempt_dir}"
            )
            return

        results_path = attempt_dir / "results.json"
        results = []
        for batch_index, batch_requests in enumerate(request_batches, start=1):
            batch_dir = attempt_dir / "batches" / f"batch_{batch_index:03d}"
            batch_manifest = attempt_dir / "batch_manifests" / f"batch_{batch_index:03d}.json"
            batch_results, batch_metadata = execute_agent_batch(
                input_path=args.input,
                batch_dir=batch_dir,
                batch_manifest=batch_manifest,
                batch_requests=batch_requests,
                model=args.model,
                image_field=args.image_field,
                attempt_id=attempt_index,
                round_config=round_config,
                confidence_threshold=confidence_threshold,
                position_majority_threshold=position_majority_threshold,
                clear_damage_threshold=clear_damage_threshold,
                cache_enabled=cache_enabled,
                finalization_policy=finalization_policy,
            )
            batch_metadata["batch_id"] = batch_index
            attempt_state["batches"].append(batch_metadata)
            results.extend(batch_results)
            write_json(state_path, state)
            if batch_metadata["agent_returncode"] != 0:
                attempt_state["status"] = "interrupted"
                state["status"] = "agent_interrupted"
                state["failed_batch"] = {
                    "attempt_id": attempt_index,
                    "batch_id": batch_index,
                    "directory": str(batch_dir),
                    "returncode": batch_metadata["agent_returncode"],
                }
                write_json(state_path, state)
                raise SystemExit(
                    "Codex batch stopped. Completed batch caches were preserved; "
                    "rerun the same command to resume."
                )

        for result in results:
            normalize_class_mapping(result)
        write_json(results_path, results)
        attempt_state["cache_hits"] = sum(
            bool(batch["cache_hit"]) for batch in attempt_state["batches"]
        )
        latest_results.update(result_map(results))
        report = validate_results(
            requests,
            results,
            confidence_threshold=confidence_threshold,
            min_frames=int(round_config.get("min_frames", round_config.get("max_frames_per_sample", 3))),
            position_majority_threshold=position_majority_threshold,
            clear_damage_threshold=clear_damage_threshold,
        )
        validation_path = attempt_dir / "validation.json"
        write_json(validation_path, report)
        attempt_state["validation"] = str(validation_path)
        attempt_state["summary"] = report["summary"]

        has_next_attempt = attempt_index < config["max_attempts"]
        retry_items = []
        for item in report["items"]:
            key = sample_key(item)
            if item["status"] == "pass":
                final_decisions[key] = item
                continue

            should_retry = should_retry_item(item, retry_policy)
            if has_next_attempt and should_retry:
                retry_items.append(item)
            elif (
                finalization_policy == "best_effort"
                and normalize_class_mapping(latest_results.get(key, {}))
            ):
                item["status"] = "classified_with_warnings"
                item["retry_action"] = "none"
                final_decisions[key] = item
            else:
                item["status"] = (
                    "unclassified" if finalization_policy == "best_effort" else "human_review"
                )
                item["retry_action"] = (
                    "none" if finalization_policy == "best_effort" else "human_review"
                )
                final_decisions[key] = item

        if not retry_items:
            break

        include_file = args.work_dir / f"retry_attempt_{attempt_index + 1}.json"
        write_json(include_file, retry_items)
        attempt_state["retry_file"] = str(include_file)
        write_json(state_path, state)

    for request in original_requests:
        key = sample_key(request)
        if key not in final_decisions:
            final_decisions[key] = {
                "video": key[0],
                "sample_id": key[1],
                "attempt_id": None,
                "status": (
                    "unclassified" if finalization_policy == "best_effort" else "human_review"
                ),
                "reason_codes": ["missing_final_decision"],
                "retry_action": (
                    "none" if finalization_policy == "best_effort" else "human_review"
                ),
            }

    write_final_outputs(
        args.work_dir,
        original_requests,
        latest_results,
        final_decisions,
        finalization_policy=finalization_policy,
    )
    clean_count = sum(item["status"] == "pass" for item in final_decisions.values())
    warning_count = sum(
        item["status"] == "classified_with_warnings"
        for item in final_decisions.values()
    )
    unclassified_count = sum(
        item["status"] == "unclassified" for item in final_decisions.values()
    )
    human_review_count = sum(item["status"] == "human_review" for item in final_decisions.values())
    if finalization_policy == "best_effort":
        state["status"] = "complete" if unclassified_count == 0 else "incomplete"
        state["summary"] = {
            "total": len(original_requests),
            "classified": clean_count + warning_count,
            "clean": clean_count,
            "with_warnings": warning_count,
            "unclassified": unclassified_count,
        }
    else:
        state["status"] = "complete" if human_review_count == 0 else "human_review_required"
        state["summary"] = {
            "total": len(original_requests),
            "pass": len(original_requests) - human_review_count,
            "human_review": human_review_count,
        }
    write_json(state_path, state)
    if finalization_policy == "best_effort":
        print(
            f"Classification complete: {state['summary']['classified']} classified, "
            f"{unclassified_count} unclassified -> {args.work_dir}"
        )
    else:
        print(
            f"Closed loop complete: {state['summary']['pass']} pass, "
            f"{human_review_count} human review -> {args.work_dir}"
        )


if __name__ == "__main__":
    main()
