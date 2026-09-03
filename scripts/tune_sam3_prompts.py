#!/usr/bin/env python
"""Sweep and rank SAM3 text-prompt combinations on cached cable samples."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from scripts.classify_contact import classify_box
    from scripts.classify_samples_gpt import load_json
    from scripts.classify_samples_sam3_only import (
        PromptDetection,
        SAM3PromptDetector,
        damage_frame_decision,
        flatten_prompt_groups,
        load_config as load_model_config,
    )
    from scripts.classify_samples_zero_shot import (
        aggregate_damage,
        aggregate_position,
        cable_bbox,
        collect_samples,
        crop_mask_for_frame,
        file_signature,
        position_frame_decision,
        read_frame_images,
        resolve_config_path,
        visual_damage_features,
        write_json,
    )
except ModuleNotFoundError:
    from classify_contact import classify_box
    from classify_samples_gpt import load_json
    from classify_samples_sam3_only import (
        PromptDetection,
        SAM3PromptDetector,
        damage_frame_decision,
        flatten_prompt_groups,
        load_config as load_model_config,
    )
    from classify_samples_zero_shot import (
        aggregate_damage,
        aggregate_position,
        cable_bbox,
        collect_samples,
        crop_mask_for_frame,
        file_signature,
        position_frame_decision,
        read_frame_images,
        resolve_config_path,
        visual_damage_features,
        write_json,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
DUMMY_MASK = np.ones((1, 1), dtype=np.uint8)


def load_search_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt-search config must be a mapping: {path}")
    required = (
        "nouns",
        "localization_prompts",
        "damage_templates",
        "intact_templates",
        "exposed_templates",
        "suspended_templates",
        "search",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in {path}")
    return data


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))


def expand_template_prompts(
    nouns: list[str],
    templates: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    families: dict[str, list[str]] = {}
    all_prompts: list[str] = []
    for noun in nouns:
        prompts = unique(
            [str(template).format(noun=noun) for template in templates]
        )
        families[noun] = prompts
        all_prompts.extend(prompts)
    return unique(all_prompts), families


def prompt_catalog(config: dict[str, Any]) -> dict[str, Any]:
    nouns = list(map(str, config["nouns"]))
    damage, damage_families = expand_template_prompts(
        nouns,
        list(map(str, config["damage_templates"])),
    )
    damage = unique(
        damage
        + list(map(str, config.get("damage_additional_prompts", [])))
    )
    for prompt in config.get("damage_additional_prompts", []):
        prompt = str(prompt)
        matched = False
        for noun in nouns:
            if noun in prompt.split():
                damage_families[noun] = unique(
                    damage_families[noun] + [prompt]
                )
                matched = True
        if not matched:
            damage_families.setdefault("semantic", []).append(prompt)
    damage_families = {
        family: unique(prompts)
        for family, prompts in damage_families.items()
    }
    candidate_groups = {
        str(name): unique(list(map(str, prompts)))
        for name, prompts in config.get(
            "damage_candidate_groups",
            {},
        ).items()
        if isinstance(prompts, list)
    }
    for name, prompts in candidate_groups.items():
        damage = unique(damage + prompts)
        damage_families[f"evidence_{name}"] = prompts

    intact, intact_families = expand_template_prompts(
        nouns,
        list(map(str, config["intact_templates"])),
    )
    exposed, exposed_families = expand_template_prompts(
        nouns,
        list(map(str, config["exposed_templates"])),
    )
    suspended, suspended_families = expand_template_prompts(
        nouns,
        list(map(str, config["suspended_templates"])),
    )
    return {
        "nouns": nouns,
        "localization": unique(
            list(map(str, config["localization_prompts"]))
        ),
        "damage": damage,
        "damage_families": damage_families,
        "damage_candidate_groups": candidate_groups,
        "intact": intact,
        "intact_families": intact_families,
        "exposed": exposed,
        "exposed_families": exposed_families,
        "suspended": suspended,
        "suspended_families": suspended_families,
    }


def reference_results(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    references: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected result list: {path}")
        for item in data:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("video", "")), str(item.get("sample_id", "")))
            references[key] = {
                "video": key[0],
                "sample_id": key[1],
                "damage": item.get("damage"),
                "position": item.get("position"),
                "class_id": item.get("class_id"),
                "source": str(path),
            }
    return references


def summarize_prompt(
    detections: list[PromptDetection],
    reference_area_ratio: float,
    local_max_area_ratio: float,
) -> dict[str, Any]:
    if not detections:
        return {
            "detections": 0,
            "local_score": 0.0,
            "global_score": 0.0,
            "best_iou": 0.0,
            "best_score": 0.0,
            "best_area_ratio": 0.0,
            "best_mask_overlap": 0.0,
            "best_reference_coverage": 0.0,
        }
    local_score = max(
        (
            detection.score
            for detection in detections
            if detection.area_ratio <= local_max_area_ratio
        ),
        default=0.0,
    )
    best_detection = detections[0]
    best_iou = -1.0
    for detection in detections:
        intersection_ratio = (
            detection.reference_coverage * reference_area_ratio
        )
        union_ratio = (
            detection.area_ratio
            + reference_area_ratio
            - intersection_ratio
        )
        iou = (
            intersection_ratio / union_ratio if union_ratio > 0 else 0.0
        )
        if iou > best_iou:
            best_iou = iou
            best_detection = detection
    return {
        "detections": len(detections),
        "local_score": round(float(local_score), 7),
        "global_score": round(
            max(detection.score for detection in detections),
            7,
        ),
        "best_iou": round(max(0.0, best_iou), 7),
        "best_score": round(best_detection.score, 7),
        "best_area_ratio": round(best_detection.area_ratio, 7),
        "best_mask_overlap": round(best_detection.mask_overlap, 7),
        "best_reference_coverage": round(
            best_detection.reference_coverage,
            7,
        ),
    }


def sweep_signature(
    samples,
    search_config_path: Path,
    model_config_path: Path,
    checkpoint_path: Path,
) -> str:
    inputs: list[dict[str, Any]] = []
    for sample in samples:
        for frame in sample.frames:
            for path in (frame.full_path, frame.crop_path, frame.mask_path):
                if path is not None and path.is_file():
                    inputs.append(file_signature(path))
    payload = {
        "tuner": file_signature(Path(__file__)),
        "detector": file_signature(
            REPO_ROOT / "scripts" / "classify_samples_sam3_only.py"
        ),
        "search_config": file_signature(search_config_path),
        "model_config": file_signature(model_config_path),
        "checkpoint": file_signature(checkpoint_path),
        "inputs": inputs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def sweep_features(
    samples,
    references: dict[tuple[str, str], dict[str, Any]],
    detector: SAM3PromptDetector,
    catalog: dict[str, Any],
    model_config: dict[str, Any],
) -> list[dict[str, Any]]:
    damage_config = model_config["damage"]
    position_config = model_config["position"]
    crop_prompts = unique(catalog["damage"] + catalog["intact"])
    full_prompts = unique(
        catalog["localization"]
        + catalog["exposed"]
        + catalog["suspended"]
    )
    local_max_area = float(
        damage_config.get("local_max_area_ratio", 0.25)
    )
    total_frames = sum(len(sample.frames) for sample in samples)
    processed = 0
    features: list[dict[str, Any]] = []
    for sample in samples:
        key = (sample.video, sample.sample_id)
        sample_row = {
            "video": sample.video,
            "sample_id": sample.sample_id,
            "reference": references.get(key),
            "frames": [],
        }
        for frame in sample.frames:
            full, crop, full_mask = read_frame_images(frame)
            crop_mask = crop_mask_for_frame(frame, full_mask, crop.shape)
            crop_reference_area = float(
                np.count_nonzero(crop_mask) / max(1, crop_mask.size)
            )
            full_reference_area = float(
                np.count_nonzero(full_mask) / max(1, full_mask.size)
            )

            crop_features: dict[str, Any] = {}
            for prompt, detections in detector.predict_each(
                crop,
                crop_prompts,
                crop_mask,
            ):
                crop_features[prompt] = summarize_prompt(
                    detections,
                    crop_reference_area,
                    local_max_area,
                )

            full_features: dict[str, Any] = {}
            for prompt, detections in detector.predict_each(
                full,
                full_prompts,
                full_mask,
            ):
                full_features[prompt] = summarize_prompt(
                    detections,
                    full_reference_area,
                    1.0,
                )

            bbox = cable_bbox(full_mask)
            contact_result = (
                classify_box(full, bbox, position_config.get("contact", {}))
                if bbox is not None
                else {
                    "state": "unknown",
                    "distance_px": None,
                    "contact_ratio": 0.0,
                }
            )
            visual = visual_damage_features(
                crop,
                crop_mask,
                damage_config,
            )
            sample_row["frames"].append(
                {
                    "frame_idx": frame.frame_idx,
                    "sam3_area_ratio": frame.sam3_area_ratio,
                    "crop_reference_area_ratio": round(
                        crop_reference_area,
                        7,
                    ),
                    "full_reference_area_ratio": round(
                        full_reference_area,
                        7,
                    ),
                    "visual_damage": visual,
                    "contact_result": contact_result,
                    "crop_prompts": crop_features,
                    "full_prompts": full_features,
                }
            )
            processed += 1
            print(
                f"[sweep] {processed}/{total_frames} "
                f"{Path(sample.video).name}:{sample.sample_id}:"
                f"f{frame.frame_idx}",
                flush=True,
            )
        features.append(sample_row)
    return features


def fake_detection(
    prompt: str,
    score: float,
    area_ratio: float,
) -> PromptDetection:
    return PromptDetection(
        prompt=prompt,
        score=score,
        box_xyxy=(0, 0, 1, 1),
        mask=DUMMY_MASK,
        area_ratio=area_ratio,
        mask_overlap=1.0,
        reference_coverage=0.1,
    )


def damage_detections_from_features(
    frame: dict[str, Any],
    prompts: list[str],
    local_max_area: float,
) -> list[PromptDetection]:
    detections: list[PromptDetection] = []
    for prompt in prompts:
        row = frame["crop_prompts"].get(prompt, {})
        local_score = float(row.get("local_score", 0.0))
        global_score = float(row.get("global_score", 0.0))
        if local_score > 0:
            detections.append(
                fake_detection(
                    prompt,
                    local_score,
                    min(0.10, local_max_area),
                )
            )
        if global_score > local_score + 1e-9:
            detections.append(
                fake_detection(
                    prompt,
                    global_score,
                    min(0.80, local_max_area + 0.20),
                )
            )
    return detections


def evaluate_damage(
    features: list[dict[str, Any]],
    positive_prompts: list[str],
    intact_prompts: list[str],
    base_config: dict[str, Any],
    positive_prompt_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["positive_prompts"] = positive_prompts
    config["positive_prompt_groups"] = (
        positive_prompt_groups
        if positive_prompt_groups is not None
        else {"candidate": positive_prompts}
    )
    config["intact_prompts"] = intact_prompts
    # Prompt-combination sweeps evaluate the fused candidate prompts supplied
    # above, independently of any deployment-only direct-trigger rule.
    config["decision_mode"] = "fused"
    local_max_area = float(config.get("local_max_area_ratio", 0.25))
    correct = false_positive = false_negative = total = 0
    false_positive_frames = 0
    damaged_positive_frame_counts: list[int] = []
    evidence_values: list[float] = []
    items = []
    for sample in features:
        reference = sample.get("reference")
        if not isinstance(reference, dict) or reference.get("damage") not in {
            "damaged",
            "intact",
        }:
            continue
        frame_decisions = []
        for frame in sample["frames"]:
            detections = damage_detections_from_features(
                frame,
                positive_prompts + intact_prompts,
                local_max_area,
            )
            frame_decisions.append(
                damage_frame_decision(
                    detections,
                    float(frame["visual_damage"]["score"]),
                    config,
                )
            )
        predicted, confidence = aggregate_damage(frame_decisions, config)
        expected = str(reference["damage"])
        is_correct = predicted == expected
        correct += int(is_correct)
        total += 1
        false_positive += int(predicted == "damaged" and expected == "intact")
        false_negative += int(predicted == "intact" and expected == "damaged")
        positive_frames = sum(
            bool(decision["positive"]) for decision in frame_decisions
        )
        if expected == "intact":
            false_positive_frames += positive_frames
        else:
            damaged_positive_frame_counts.append(positive_frames)
        strongest = max(
            (
                float(decision["combined_score"])
                for decision in frame_decisions
            ),
            default=0.0,
        )
        evidence_values.append(
            strongest if expected == "damaged" else 1.0 - strongest
        )
        items.append(
            {
                "video": sample["video"],
                "sample_id": sample["sample_id"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "confidence": confidence,
                "positive_frames": positive_frames,
                "strongest_combined_score": round(strongest, 6),
            }
        )
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "false_positive_frames": false_positive_frames,
        "min_damage_positive_frames": min(
            damaged_positive_frame_counts,
            default=0,
        ),
        "mean_damage_positive_frames": round(
            sum(damaged_positive_frame_counts)
            / max(1, len(damaged_positive_frame_counts)),
            4,
        ),
        "evidence_quality": round(
            sum(evidence_values) / len(evidence_values),
            6,
        )
        if evidence_values
        else 0.0,
        "positive_prompts": positive_prompts,
        "positive_prompt_groups": config["positive_prompt_groups"],
        "intact_prompts": intact_prompts,
        "items": items,
    }


def max_feature_score(
    frame: dict[str, Any],
    prompts: list[str],
) -> float:
    return max(
        (
            float(
                frame["full_prompts"].get(prompt, {}).get(
                    "global_score",
                    0.0,
                )
            )
            for prompt in prompts
        ),
        default=0.0,
    )


def evaluate_position(
    features: list[dict[str, Any]],
    exposed_prompts: list[str],
    suspended_prompts: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    correct = total = 0
    items = []
    for sample in features:
        reference = sample.get("reference")
        if not isinstance(reference, dict) or reference.get("position") not in {
            "exposed",
            "suspended",
        }:
            continue
        frame_decisions = [
            position_frame_decision(
                max_feature_score(frame, exposed_prompts),
                max_feature_score(frame, suspended_prompts),
                float(frame["sam3_area_ratio"]),
                frame["contact_result"],
                config,
            )
            for frame in sample["frames"]
        ]
        predicted, confidence, agreement = aggregate_position(frame_decisions)
        expected = str(reference["position"])
        is_correct = predicted == expected
        correct += int(is_correct)
        total += 1
        items.append(
            {
                "video": sample["video"],
                "sample_id": sample["sample_id"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "confidence": confidence,
                "agreement": agreement,
            }
        )
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "exposed_prompts": exposed_prompts,
        "suspended_prompts": suspended_prompts,
        "items": items,
    }


def localization_ranking(
    features: list[dict[str, Any]],
    prompts: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        values = [
            frame["full_prompts"].get(prompt, {})
            for sample in features
            for frame in sample["frames"]
        ]
        count = len(values)
        detected = sum(int(value.get("detections", 0) > 0) for value in values)
        rows.append(
            {
                "prompt": prompt,
                "frames": count,
                "detected_frames": detected,
                "detection_rate": round(detected / count, 4)
                if count
                else 0.0,
                "mean_iou": round(
                    sum(float(value.get("best_iou", 0.0)) for value in values)
                    / max(1, count),
                    6,
                ),
                "mean_score": round(
                    sum(
                        float(value.get("global_score", 0.0))
                        for value in values
                    )
                    / max(1, count),
                    6,
                ),
                "mean_mask_overlap": round(
                    sum(
                        float(value.get("best_mask_overlap", 0.0))
                        for value in values
                    )
                    / max(1, count),
                    6,
                ),
                "mean_reference_coverage": round(
                    sum(
                        float(
                            value.get("best_reference_coverage", 0.0)
                        )
                        for value in values
                    )
                    / max(1, count),
                    6,
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["mean_iou"],
            row["detection_rate"],
            row["mean_reference_coverage"],
        ),
        reverse=True,
    )


def damage_rank_key(row: dict[str, Any]) -> tuple:
    return (
        float(row["accuracy"] or 0.0),
        -int(row["false_positive"]),
        -int(row["false_negative"]),
        -int(row["false_positive_frames"]),
        int(row["min_damage_positive_frames"]),
        float(row["evidence_quality"]),
        -len(row["positive_prompts"]),
        -len(row["intact_prompts"]),
    )


def compact_damage_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "items"}


def tune_damage(
    features: list[dict[str, Any]],
    catalog: dict[str, Any],
    config: dict[str, Any],
    search_config: dict[str, Any],
) -> dict[str, Any]:
    intact_options = {
        f"{noun}_family": prompts
        for noun, prompts in catalog["intact_families"].items()
    }
    intact_options["all_intact"] = catalog["intact"]
    current_intact = [
        prompt
        for prompt in config["intact_prompts"]
        if prompt in catalog["intact"]
    ]
    if current_intact:
        intact_options["current"] = current_intact

    family_rows = []
    for family, positive_prompts in catalog["damage_families"].items():
        matching_intact = intact_options.get(
            f"{family}_family",
            intact_options["all_intact"],
        )
        row = evaluate_damage(
            features,
            positive_prompts,
            matching_intact,
            config,
        )
        row["name"] = f"{family}_family"
        family_rows.append(row)
    family_rows.sort(key=damage_rank_key, reverse=True)

    individual_rows = []
    for prompt in catalog["damage"]:
        best = None
        best_intact_name = ""
        for intact_name, intact_prompts in intact_options.items():
            row = evaluate_damage(
                features,
                [prompt],
                intact_prompts,
                config,
            )
            if best is None or damage_rank_key(row) > damage_rank_key(best):
                best = row
                best_intact_name = intact_name
        assert best is not None
        best["intact_option"] = best_intact_name
        individual_rows.append(best)
    individual_rows.sort(key=damage_rank_key, reverse=True)

    top_count = int(
        search_config["search"].get(
            "top_individual_damage_prompts",
            18,
        )
    )
    pool = [
        row["positive_prompts"][0]
        for row in individual_rows[:top_count]
    ]
    for prompt in config["positive_prompts"]:
        if prompt in catalog["damage"] and prompt not in pool:
            pool.append(prompt)

    max_prompt_count = int(
        search_config["search"].get("max_damage_prompt_count", 4)
    )
    combination_rows = []
    for size in range(1, max_prompt_count + 1):
        for combination in itertools.combinations(pool, size):
            best = None
            best_intact_name = ""
            for intact_name, intact_prompts in intact_options.items():
                row = evaluate_damage(
                    features,
                    list(combination),
                    intact_prompts,
                    config,
                )
                if (
                    best is None
                    or damage_rank_key(row) > damage_rank_key(best)
                ):
                    best = row
                    best_intact_name = intact_name
            assert best is not None
            best["intact_option"] = best_intact_name
            combination_rows.append(best)
    combination_rows.sort(key=damage_rank_key, reverse=True)
    max_rows = int(
        search_config["search"].get(
            "max_reported_combinations",
            25,
        )
    )
    deployment_rows = []
    for name, groups in search_config.get(
        "deployment_candidates",
        {},
    ).items():
        if not isinstance(groups, dict):
            continue
        normalized_groups = {
            str(group_name): unique(list(map(str, prompts)))
            for group_name, prompts in groups.items()
            if isinstance(prompts, list)
        }
        positive_prompts = flatten_prompt_groups(normalized_groups)
        if not positive_prompts:
            continue
        row = evaluate_damage(
            features,
            positive_prompts,
            current_intact or intact_options["all_intact"],
            config,
            positive_prompt_groups=normalized_groups,
        )
        row["name"] = str(name)
        deployment_rows.append(row)
    deployment_rows.sort(key=damage_rank_key, reverse=True)

    return {
        "family_ranking": [
            compact_damage_row(row) for row in family_rows
        ],
        "individual_ranking": [
            compact_damage_row(row) for row in individual_rows
        ],
        "combination_pool": pool,
        "top_combinations": [
            compact_damage_row(row)
            for row in combination_rows[:max_rows]
        ],
        "recommended": compact_damage_row(combination_rows[0])
        if combination_rows
        else None,
        "deployment_candidates": [
            compact_damage_row(row) for row in deployment_rows
        ],
    }


def tune_position(
    features: list[dict[str, Any]],
    catalog: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for noun in catalog["nouns"]:
        row = evaluate_position(
            features,
            catalog["exposed_families"][noun],
            catalog["suspended_families"][noun],
            config,
        )
        row["name"] = f"{noun}_family"
        rows.append(row)
    rows.append(
        {
            **evaluate_position(
                features,
                catalog["exposed"],
                catalog["suspended"],
                config,
            ),
            "name": "all_families",
        }
    )
    rows.sort(
        key=lambda row: (
            float(row["accuracy"] or 0.0),
            -len(row["exposed_prompts"]) - len(row["suspended_prompts"]),
        ),
        reverse=True,
    )
    return {
        "family_ranking": rows,
        "recommended": {
            key: value
            for key, value in rows[0].items()
            if key != "items"
        }
        if rows
        else None,
    }


def make_report(
    features: list[dict[str, Any]],
    catalog: dict[str, Any],
    model_config: dict[str, Any],
    search_config: dict[str, Any],
) -> dict[str, Any]:
    localization = localization_ranking(
        features,
        catalog["localization"],
    )
    damage = tune_damage(
        features,
        catalog,
        model_config["damage"],
        search_config,
    )
    position = tune_position(
        features,
        catalog,
        model_config["position"],
    )
    selected_config = search_config.get("selected", {})
    selected = None
    if isinstance(selected_config, dict) and selected_config:
        localization_prompt = str(
            selected_config.get("localization_prompt", "")
        )
        localization_row = next(
            (
                row
                for row in localization
                if row["prompt"] == localization_prompt
            ),
            None,
        )
        selected_groups = selected_config.get("damage_prompt_groups")
        if isinstance(selected_groups, dict) and selected_groups:
            selected_groups = {
                str(name): list(map(str, prompts))
                for name, prompts in selected_groups.items()
                if isinstance(prompts, list)
            }
            selected_positive_prompts = flatten_prompt_groups(selected_groups)
        else:
            selected_groups = None
            selected_positive_prompts = list(
                map(str, selected_config["damage_positive_prompts"])
            )
        selected_damage = evaluate_damage(
            features,
            selected_positive_prompts,
            list(map(str, selected_config["intact_prompts"])),
            model_config["damage"],
            positive_prompt_groups=selected_groups,
        )
        selected_position = evaluate_position(
            features,
            list(map(str, selected_config["exposed_prompts"])),
            list(map(str, selected_config["suspended_prompts"])),
            model_config["position"],
        )
        selected = {
            "localization": localization_row,
            "damage": compact_damage_row(selected_damage),
            "position": {
                key: value
                for key, value in selected_position.items()
                if key != "items"
            },
            "rationale": list(map(str, selected_config.get("rationale", []))),
        }
    return {
        "summary": {
            "samples": len(features),
            "labeled_samples": sum(
                isinstance(sample.get("reference"), dict)
                for sample in features
            ),
            "frames": sum(len(sample["frames"]) for sample in features),
            "candidate_counts": {
                "localization": len(catalog["localization"]),
                "damage": len(catalog["damage"]),
                "intact": len(catalog["intact"]),
                "position_exposed": len(catalog["exposed"]),
                "position_suspended": len(catalog["suspended"]),
            },
        },
        "selected": selected,
        "localization_ranking": localization,
        "damage": damage,
        "position": position,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# SAM3 prompt search report",
        "",
        (
            f"Samples: {summary['samples']}; labeled: "
            f"{summary['labeled_samples']}; frames: {summary['frames']}."
        ),
        "",
        "## Selected robust configuration",
        "",
        "```json",
        json.dumps(
            report["selected"],
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Localization prompts",
        "",
        "| rank | prompt | detect | mean IoU | mean score | coverage |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(
        report["localization_ranking"][:14],
        start=1,
    ):
        lines.append(
            f"| {index} | {row['prompt']} | "
            f"{row['detection_rate']:.3f} | {row['mean_iou']:.3f} | "
            f"{row['mean_score']:.3f} | "
            f"{row['mean_reference_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Damage noun families",
            "",
        "| rank | family | accuracy | FP | FN | FP frames | min positive frames | prompts |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(
        report["damage"]["family_ranking"],
        start=1,
    ):
        lines.append(
            f"| {index} | {row['name']} | {row['accuracy']:.3f} | "
            f"{row['false_positive']} | {row['false_negative']} | "
            f"{row['false_positive_frames']} | "
            f"{row['min_damage_positive_frames']} | "
            f"{len(row['positive_prompts'])} |"
        )
    lines.extend(
        [
            "",
            "## Search-score leader (diagnostic only)",
            "",
            (
                "This row is the raw in-sample score leader. It is not "
                "automatically deployed because tiny datasets can reward "
                "semantically implausible prompt combinations."
            ),
            "",
            "```json",
            json.dumps(
                report["damage"]["recommended"],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## Grouped deployment candidates",
            "",
            "| rank | profile | accuracy | FP | FN | FP frames | min positive frames | prompts |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(
        report["damage"].get("deployment_candidates", []),
        start=1,
    ):
        lines.append(
            f"| {index} | {row['name']} | {row['accuracy']:.3f} | "
            f"{row['false_positive']} | {row['false_negative']} | "
            f"{row['false_positive_frames']} | "
            f"{row['min_damage_positive_frames']} | "
            f"{len(row['positive_prompts'])} |"
        )
    lines.extend(
        [
            "",
            "## Position noun families",
            "",
            "| rank | family | accuracy |",
            "|---:|---|---:|",
        ]
    )
    for index, row in enumerate(
        report["position"]["family_ranking"],
        start=1,
    ):
        lines.append(
            f"| {index} | {row['name']} | {row['accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recommended position combination",
            "",
            "```json",
            json.dumps(
                report["position"]["recommended"],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            (
                "The reported accuracy is agreement with the supplied "
                "reference results, not ground-truth accuracy."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="SAM3 sample manifest directory; repeat for more sets.",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        action="append",
        required=True,
        help="Reference results JSON; repeat to merge sets.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--search-config",
        type=Path,
        default=REPO_ROOT / "configs" / "sam3_prompt_search.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "sam3_only_classifier.yaml",
    )
    parser.add_argument("--force-sweep", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    search_config_path = args.search_config.expanduser().resolve()
    model_config_path = args.model_config.expanduser().resolve()
    search_config = load_search_config(search_config_path)
    model_config = load_model_config(model_config_path)
    catalog = prompt_catalog(search_config)
    references = reference_results(
        [path.expanduser().resolve() for path in args.reference_results]
    )

    sample_map = {}
    for input_path in args.input:
        for sample in collect_samples(
            input_path,
            max_frames_per_sample=int(
                model_config["sampling"].get("max_frames_per_sample", 3)
            ),
        ):
            sample_map[(sample.video, sample.sample_id)] = sample
    samples = list(sample_map.values())
    if not samples:
        raise SystemExit("No SAM3 samples with readable frames were found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.out_dir / "prompt_features.json"
    checkpoint_path = resolve_config_path(
        str(model_config["model"]["checkpoint"])
    )
    signature = sweep_signature(
        samples,
        search_config_path,
        model_config_path,
        checkpoint_path,
    )
    feature_payload = None
    if features_path.is_file() and args.evaluate_only:
        loaded = load_json(features_path)
        if isinstance(loaded, dict) and isinstance(
            loaded.get("features"),
            list,
        ):
            feature_payload = loaded
            print(
                f"[cache] Evaluate existing prompt sweep -> {features_path}"
            )
    elif features_path.is_file() and not args.force_sweep:
        loaded = load_json(features_path)
        if isinstance(loaded, dict) and loaded.get("signature") == signature:
            feature_payload = loaded
            print(f"[cache] Reuse prompt sweep -> {features_path}")
    if feature_payload is None:
        if args.evaluate_only:
            raise SystemExit(
                "No matching prompt_features.json; remove --evaluate-only."
            )
        detector = SAM3PromptDetector(model_config["model"])
        features = sweep_features(
            samples,
            references,
            detector,
            catalog,
            model_config,
        )
        feature_payload = {
            "signature": signature,
            "catalog": catalog,
            "features": features,
        }
        write_json(features_path, feature_payload)
    else:
        features = feature_payload["features"]
        for sample in features:
            key = (sample["video"], sample["sample_id"])
            sample["reference"] = references.get(key)

    report = make_report(
        features,
        catalog,
        model_config,
        search_config,
    )
    write_json(args.out_dir / "prompt_report.json", report)
    write_markdown(args.out_dir / "prompt_report.md", report)
    print(
        f"Prompt tuning complete: {len(features)} samples -> "
        f"{args.out_dir / 'prompt_report.json'}"
    )


if __name__ == "__main__":
    main()
