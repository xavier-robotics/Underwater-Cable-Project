#!/usr/bin/env python
"""Classify SAM3-selected cable samples using only SAM3 prompts and fixed rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from scripts.classify_samples_gpt import load_json
    from scripts.classify_samples_zero_shot import (
        CLASS_MAP,
        SampleInput,
        aggregate_damage,
        cable_bbox,
        collect_samples,
        crop_mask_for_frame,
        file_signature,
        read_frame_images,
        resolve_config_path,
        visual_damage_features,
        write_csv_results,
        write_json,
        write_review,
    )
    from scripts.select_sample_frames import SAM3Scorer
except ModuleNotFoundError:
    from classify_samples_gpt import load_json
    from classify_samples_zero_shot import (
        CLASS_MAP,
        SampleInput,
        aggregate_damage,
        cable_bbox,
        collect_samples,
        crop_mask_for_frame,
        file_signature,
        read_frame_images,
        resolve_config_path,
        visual_damage_features,
        write_csv_results,
        write_json,
        write_review,
    )
    from select_sample_frames import SAM3Scorer


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PromptDetection:
    prompt: str
    score: float
    box_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    area_ratio: float
    mask_overlap: float
    reference_coverage: float
    axial_position: float | None = None
    center_inside_reference: bool = True


def mask_center_inside_reference(
    candidate_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> bool:
    """Return whether the candidate's median mask point lies on the cable."""
    points_yx = np.column_stack(np.where(candidate_mask > 0))
    if len(points_yx) == 0:
        return False
    center_y, center_x = np.rint(np.median(points_yx, axis=0)).astype(int)
    height, width = reference_mask.shape[:2]
    if not (0 <= center_y < height and 0 <= center_x < width):
        return False
    return bool(reference_mask[center_y, center_x] > 0)


def mask_axial_position(
    candidate_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> float | None:
    """Locate a candidate from 0 to 1 along the cable's principal axis."""
    reference_points_yx = np.column_stack(np.where(reference_mask > 0))
    if len(reference_points_yx) < 2:
        return None
    reference_points = reference_points_yx[:, ::-1].astype(np.float64)
    center = reference_points.mean(axis=0)
    centered = reference_points - center
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    reference_projection = centered @ axis
    low, high = np.percentile(reference_projection, [1.0, 99.0])
    if high - low < 1e-6:
        return None

    candidate_points_yx = np.column_stack(
        np.where((candidate_mask > 0) & (reference_mask > 0))
    )
    if len(candidate_points_yx) == 0:
        return None
    candidate_points = candidate_points_yx[:, ::-1].astype(np.float64)
    candidate_center = np.median((candidate_points - center) @ axis)
    position = float((candidate_center - low) / (high - low))
    return float(np.clip(position, 0.0, 1.0))


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"SAM3-only config must be a mapping: {path}")
    for section in ("model", "sampling", "damage", "position", "quality"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"Missing mapping '{section}' in {path}")
    return data


def resolve_sam3_class(
    position: str,
    damage: str,
    config: dict[str, Any],
) -> tuple[int, str]:
    """Map position/damage states using the configured final-class priority."""
    classification = config.get("classification", {})
    priority = str(classification.get("priority", "damage"))
    if priority == "suspended" and position == "suspended":
        return 2, "suspended_intact"
    if priority not in {"damage", "suspended"}:
        raise ValueError(f"Unsupported classification priority: {priority}")
    return CLASS_MAP[(position, damage)]


class SAM3PromptDetector:
    """Run several text prompts against one cached SAM3 image embedding."""

    def __init__(self, config: dict[str, Any]) -> None:
        checkpoint = resolve_config_path(str(config["checkpoint"]))
        if not checkpoint.is_file():
            raise SystemExit(f"SAM3 checkpoint does not exist: {checkpoint}")
        self.min_area_ratio = float(config.get("min_candidate_area_ratio", 0.0015))
        self.max_area_ratio = float(config.get("max_candidate_area_ratio", 0.85))
        self.min_mask_overlap = float(config.get("min_mask_overlap", 0.20))
        self.min_reference_coverage = float(
            config.get("min_reference_coverage", 0.0)
        )
        self.adapter = SAM3Scorer(
            prompt="pipe",
            device=str(config.get("device", "auto")),
            confidence_threshold=float(config.get("confidence", 0.01)),
            min_area_ratio=0.0,
            min_span_ratio=0.0,
            checkpoint_path=checkpoint,
            dtype=str(config.get("dtype", "float32")),
        )

    @property
    def device(self) -> str:
        return self.adapter.device

    @property
    def dtype(self) -> str:
        return self.adapter.dtype

    def predict(
        self,
        image: np.ndarray,
        prompts: list[str],
        reference_mask: np.ndarray,
    ) -> list[PromptDetection]:
        return [
            detection
            for _, prompt_detections in self.predict_each(
                image,
                prompts,
                reference_mask,
            )
            for detection in prompt_detections
        ]

    def predict_each(
        self,
        image: np.ndarray,
        prompts: list[str],
        reference_mask: np.ndarray,
    ) -> Iterator[tuple[str, list[PromptDetection]]]:
        """Yield one prompt's detections while reusing one image embedding."""
        height, width = image.shape[:2]
        reference = (reference_mask > 0).astype(np.uint8)
        if reference.shape != (height, width):
            reference = cv2.resize(
                reference,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        reference_pixels = max(1, int(np.count_nonzero(reference)))
        frame_area = max(1, height * width)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = self.adapter._image_cls.fromarray(rgb)
        use_autocast = self.device == "cuda" and self.dtype == "bfloat16"
        with self.adapter._torch.autocast(
            device_type=self.device,
            dtype=self.adapter._autocast_dtype,
            enabled=use_autocast,
        ):
            state = self.adapter.processor.set_image(pil_image)
            for prompt in prompts:
                detections: list[PromptDetection] = []
                output = self.adapter.processor.set_text_prompt(prompt, state)
                masks = self.adapter._to_numpy(output.get("masks", []))
                boxes = self.adapter._to_numpy(output.get("boxes", []))
                scores = self.adapter._to_numpy(output.get("scores", [])).reshape(-1)
                for index, raw_score in enumerate(scores):
                    if index >= len(masks):
                        continue
                    mask = self.adapter._clean_mask(masks[index], (height, width))
                    candidate_pixels = int(np.count_nonzero(mask))
                    if candidate_pixels == 0:
                        continue
                    area_ratio = candidate_pixels / frame_area
                    if not self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                        continue
                    intersection = int(np.count_nonzero((mask > 0) & (reference > 0)))
                    overlap = intersection / candidate_pixels
                    if overlap < self.min_mask_overlap:
                        continue
                    reference_coverage = intersection / reference_pixels
                    if reference_coverage < self.min_reference_coverage:
                        continue
                    bbox = self.adapter._bbox_from_mask(mask)
                    if bbox is None and index < len(boxes):
                        bbox = self.adapter._bbox_from_box(boxes[index], width, height)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    detections.append(
                        PromptDetection(
                            prompt=prompt,
                            score=float(raw_score),
                            box_xyxy=(
                                max(0, x1),
                                max(0, y1),
                                min(width, x2),
                                min(height, y2),
                            ),
                            mask=mask,
                            area_ratio=float(area_ratio),
                            mask_overlap=float(overlap),
                            reference_coverage=float(reference_coverage),
                            axial_position=mask_axial_position(mask, reference),
                            center_inside_reference=(
                                mask_center_inside_reference(mask, reference)
                            ),
                        )
                    )
                yield prompt, detections


def prompt_best_scores(
    detections: list[PromptDetection],
    prompts: list[str],
    *,
    max_area_ratio: float | None = None,
    max_reference_coverage: float | None = None,
) -> dict[str, float]:
    scores = {prompt: 0.0 for prompt in prompts}
    for detection in detections:
        if detection.prompt not in scores:
            continue
        if max_area_ratio is not None and detection.area_ratio > max_area_ratio:
            continue
        if (
            max_reference_coverage is not None
            and detection.reference_coverage > max_reference_coverage
        ):
            continue
        scores[detection.prompt] = max(scores[detection.prompt], detection.score)
    return scores


def damage_prompt_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    """Return positive prompts grouped by independent evidence concept."""
    configured = config.get("positive_prompt_groups")
    if isinstance(configured, dict) and configured:
        groups = {
            str(name): list(
                dict.fromkeys(
                    str(prompt).strip()
                    for prompt in prompts
                    if str(prompt).strip()
                )
            )
            for name, prompts in configured.items()
            if isinstance(prompts, list)
        }
        groups = {name: prompts for name, prompts in groups.items() if prompts}
        if groups:
            return groups
    # Backward compatibility: legacy flat prompts remain independent evidence.
    return {
        str(prompt): [str(prompt)]
        for prompt in config.get("positive_prompts", [])
        if str(prompt).strip()
    }


def flatten_prompt_groups(groups: dict[str, list[str]]) -> list[str]:
    """Flatten prompt groups without evaluating duplicate phrases twice."""
    return list(
        dict.fromkeys(
            prompt
            for prompts in groups.values()
            for prompt in prompts
        )
    )


def group_best_scores(
    prompt_scores: dict[str, float],
    groups: dict[str, list[str]],
) -> dict[str, float]:
    """Take one score per concept so synonyms cannot inflate consensus."""
    return {
        name: max(
            (prompt_scores.get(prompt, 0.0) for prompt in prompts),
            default=0.0,
        )
        for name, prompts in groups.items()
    }


def strongest_detections_by_prompt(
    detections: list[PromptDetection],
    prompts: list[str],
    *,
    max_area_ratio: float,
    max_reference_coverage: float = 1.0,
    min_score: float,
) -> list[PromptDetection]:
    """Keep one report-quality local detection for each active prompt."""
    prompt_set = set(prompts)
    strongest: dict[str, PromptDetection] = {}
    for detection in detections:
        if (
            detection.prompt not in prompt_set
            or detection.area_ratio > max_area_ratio
            or detection.reference_coverage > max_reference_coverage
            or detection.score < min_score
        ):
            continue
        previous = strongest.get(detection.prompt)
        if previous is None or detection.score > previous.score:
            strongest[detection.prompt] = detection
    return sorted(
        strongest.values(),
        key=lambda detection: detection.score,
        reverse=True,
    )


def project_crop_detections(
    detections: list[PromptDetection],
    crop_shape: tuple[int, ...],
    full_shape: tuple[int, ...],
    crop_bbox_xyxy: tuple[int, int, int, int] | None,
) -> list[PromptDetection]:
    """Project crop-coordinate masks and boxes back onto the original frame."""
    crop_height, crop_width = crop_shape[:2]
    full_height, full_width = full_shape[:2]
    if crop_bbox_xyxy is None:
        x_offset = y_offset = 0
        target_width, target_height = full_width, full_height
    else:
        x1, y1, x2, y2 = crop_bbox_xyxy
        x_offset = max(0, min(full_width, x1))
        y_offset = max(0, min(full_height, y1))
        x2 = max(0, min(full_width, x2))
        y2 = max(0, min(full_height, y2))
        target_width = max(0, x2 - x_offset)
        target_height = max(0, y2 - y_offset)
    if target_width == 0 or target_height == 0:
        return []

    scale_x = target_width / max(1, crop_width)
    scale_y = target_height / max(1, crop_height)
    projected: list[PromptDetection] = []
    for detection in detections:
        resized_mask = cv2.resize(
            detection.mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        full_mask = np.zeros((full_height, full_width), dtype=np.uint8)
        full_mask[
            y_offset : y_offset + target_height,
            x_offset : x_offset + target_width,
        ] = resized_mask
        x1, y1, x2, y2 = detection.box_xyxy
        projected_box = (
            max(0, min(full_width, x_offset + round(x1 * scale_x))),
            max(0, min(full_height, y_offset + round(y1 * scale_y))),
            max(0, min(full_width, x_offset + round(x2 * scale_x))),
            max(0, min(full_height, y_offset + round(y2 * scale_y))),
        )
        projected.append(
            PromptDetection(
                prompt=detection.prompt,
                score=detection.score,
                box_xyxy=projected_box,
                mask=full_mask,
                area_ratio=float(
                    np.count_nonzero(full_mask) / max(1, full_height * full_width)
                ),
                mask_overlap=detection.mask_overlap,
                reference_coverage=detection.reference_coverage,
                axial_position=detection.axial_position,
                center_inside_reference=detection.center_inside_reference,
            )
        )
    return projected


def damage_support_mask(reference_mask: np.ndarray) -> np.ndarray:
    """Fill gaps inside the cable silhouette so exposed material stays valid."""
    binary = (reference_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return binary * 255
    largest_area = max(cv2.contourArea(contour) for contour in contours)
    minimum_area = max(64.0, largest_area * 0.05)
    significant = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= minimum_area
    ]
    if not significant:
        significant = [max(contours, key=cv2.contourArea)]
    points = np.vstack(significant)
    hull = cv2.convexHull(points)
    support = np.zeros_like(binary)
    cv2.drawContours(support, [hull], -1, 255, thickness=-1)
    return support


def mask_overlap_coefficient(first: np.ndarray, second: np.ndarray) -> float:
    """Return intersection over the smaller mask, robust to nested masks."""
    first_binary = first > 0
    second_binary = second > 0
    denominator = min(
        int(np.count_nonzero(first_binary)),
        int(np.count_nonzero(second_binary)),
    )
    if denominator == 0:
        return 0.0
    intersection = int(np.count_nonzero(first_binary & second_binary))
    return intersection / denominator


def filter_excluded_damage_detections(
    detections: list[PromptDetection],
    positive_prompts: list[str],
    exclusion_prompts: list[str],
    config: dict[str, Any],
) -> tuple[list[PromptDetection], list[dict[str, Any]]]:
    """Suppress positive regions that spatially match a known non-damage object."""
    positive_set = set(positive_prompts)
    exclusion_set = set(exclusion_prompts)
    local_max_area = float(config.get("local_max_area_ratio", 0.25))
    minimum_score = float(config.get("exclusion_min_score", 0.12))
    score_ratio = float(config.get("exclusion_score_ratio", 0.72))
    configured_bypass = config.get("positive_prompt_exclusion_bypass_scores")
    bypass_scores = (
        {
            str(prompt): float(score)
            for prompt, score in configured_bypass.items()
        }
        if isinstance(configured_bypass, dict)
        else {}
    )
    configured_bypass_margins = config.get(
        "positive_prompt_exclusion_bypass_margins"
    )
    bypass_margins = (
        {
            str(prompt): float(margin)
            for prompt, margin in configured_bypass_margins.items()
        }
        if isinstance(configured_bypass_margins, dict)
        else {}
    )
    overlap_threshold = float(
        config.get("exclusion_overlap_coefficient", 0.30)
    )
    exclusions = [
        detection
        for detection in detections
        if detection.prompt in exclusion_set
        and detection.area_ratio <= local_max_area
        and detection.score >= minimum_score
    ]
    filtered: list[PromptDetection] = []
    suppressed: list[dict[str, Any]] = []
    for detection in detections:
        if (
            detection.prompt not in positive_set
            or detection.area_ratio > local_max_area
        ):
            filtered.append(detection)
            continue
        matches: list[tuple[PromptDetection, float]] = []
        for exclusion in exclusions:
            overlap = mask_overlap_coefficient(detection.mask, exclusion.mask)
            if (
                overlap >= overlap_threshold
                and exclusion.score >= detection.score * score_ratio
            ):
                matches.append((exclusion, overlap))
        if not matches:
            filtered.append(detection)
            continue
        exclusion, overlap = max(
            matches,
            key=lambda item: (item[0].score, item[1]),
        )
        bypass_score = bypass_scores.get(detection.prompt)
        bypass_margin = bypass_margins.get(detection.prompt, 0.0)
        if (
            bypass_score is not None
            and detection.score >= bypass_score
            and detection.score - exclusion.score >= bypass_margin
        ):
            filtered.append(detection)
            continue
        suppressed.append(
            {
                "positive_prompt": detection.prompt,
                "positive_score": round(detection.score, 6),
                "exclusion_prompt": exclusion.prompt,
                "exclusion_score": round(exclusion.score, 6),
                "overlap_coefficient": round(overlap, 6),
            }
        )
    return filtered, suppressed


def filter_endpoint_damage_detections(
    detections: list[PromptDetection],
    positive_prompts: list[str],
    config: dict[str, Any],
) -> tuple[list[PromptDetection], list[dict[str, Any]]]:
    """Ignore positive evidence located in either labeled cable end."""
    positive_set = set(positive_prompts)
    middle_min = float(config.get("middle_axis_min_fraction", 0.0))
    middle_max = float(config.get("middle_axis_max_fraction", 1.0))
    if not 0.0 <= middle_min < middle_max <= 1.0:
        raise ValueError(
            "damage middle-axis fractions must satisfy "
            "0 <= min < max <= 1"
        )

    filtered: list[PromptDetection] = []
    ignored: list[dict[str, Any]] = []
    for detection in detections:
        position = detection.axial_position
        if (
            detection.prompt not in positive_set
            or position is None
            or middle_min <= position <= middle_max
        ):
            filtered.append(detection)
            continue
        ignored.append(
            {
                "positive_prompt": detection.prompt,
                "positive_score": round(detection.score, 6),
                "axial_position": round(position, 6),
                "terminal_side": "end_a" if position < middle_min else "end_b",
            }
        )
    return filtered, ignored


def position_support_mask(
    reference_mask: np.ndarray,
    margin_px: int,
) -> np.ndarray:
    """Build a cable-neighbourhood mask in which a support may be found."""
    height, width = reference_mask.shape[:2]
    bbox = cable_bbox(reference_mask)
    if bbox is None:
        return np.full((height, width), 255, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    margin = max(0, int(margin_px))
    search = np.zeros((height, width), dtype=np.uint8)
    search[
        max(0, y1 - margin) : min(height, y2 + margin),
        max(0, x1 - margin) : min(width, x2 + margin),
    ] = 255
    return search


def support_geometry_features(
    image: np.ndarray,
    reference_mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Find a bright support rod near a cable end at a steep angle."""
    empty = {
        "score": 0.0,
        "line_length_ratio": 0.0,
        "line_contrast": 0.0,
        "perpendicularity": 0.0,
        "endpoint_distance_ratio": None,
        "outside_fraction": 0.0,
        "cable_thickness_ratio": None,
        "line_xyxy": None,
    }
    binary = (reference_mask > 0).astype(np.uint8)
    points_yx = np.column_stack(np.where(binary > 0))
    if len(points_yx) < 16:
        return empty
    points = points_yx[:, ::-1].astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    cable_axis = eigenvectors[:, -1]
    projections = centered @ cable_axis
    cable_length = float(
        np.percentile(projections, 99.0) - np.percentile(projections, 1.0)
    )
    if cable_length < 8.0:
        return empty

    cable_area = float(np.count_nonzero(binary))
    cable_thickness = cable_area / max(cable_length, 1.0)
    cable_thickness_ratio = cable_thickness / max(cable_length, 1.0)
    empty["cable_thickness_ratio"] = round(cable_thickness_ratio, 6)
    if cable_thickness_ratio > float(
        config.get("geometry_max_cable_thickness_ratio", 0.30)
    ):
        return empty
    low_projection, high_projection = np.percentile(projections, [1.0, 99.0])
    endpoints = np.stack(
        [
            center + cable_axis * low_projection,
            center + cable_axis * high_projection,
        ]
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    edges = cv2.Canny(
        blurred,
        int(config.get("geometry_canny_low", 20)),
        int(config.get("geometry_canny_high", 70)),
    )
    minimum_length = max(
        18,
        int(
            round(
                cable_length
                * float(config.get("geometry_min_line_length_ratio", 0.10))
            )
        ),
    )
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=int(config.get("geometry_hough_threshold", 12)),
        minLineLength=minimum_length,
        maxLineGap=int(config.get("geometry_hough_max_line_gap", 30)),
    )
    if lines is None:
        return empty

    endpoint_margin = max(
        16.0,
        cable_thickness
        * float(config.get("geometry_endpoint_thickness_ratio", 0.80)),
    )
    maximum_axis_dot = float(config.get("geometry_max_axis_dot", 0.45))
    minimum_outside_fraction = float(
        config.get("geometry_min_outside_fraction", 0.55)
    )
    best: dict[str, Any] | None = None
    height, width = gray.shape[:2]
    for raw_line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, raw_line)
        delta = np.asarray([x2 - x1, y2 - y1], dtype=np.float32)
        length = float(np.linalg.norm(delta))
        if length < minimum_length:
            continue
        unit = delta / max(length, 1e-6)
        axis_dot = abs(float(np.dot(unit, cable_axis)))
        if axis_dot > maximum_axis_dot:
            continue
        sample_count = max(8, int(round(length / 4.0)))
        xs = np.clip(
            np.rint(np.linspace(x1, x2, sample_count)).astype(int),
            0,
            width - 1,
        )
        ys = np.clip(
            np.rint(np.linspace(y1, y2, sample_count)).astype(int),
            0,
            height - 1,
        )
        line_points = np.column_stack([xs, ys]).astype(np.float32)
        endpoint_distance = float(
            np.min(np.linalg.norm(line_points[:, None, :] - endpoints, axis=2))
        )
        if endpoint_distance > endpoint_margin:
            continue
        outside_fraction = float(np.mean(binary[ys, xs] == 0))
        if outside_fraction < minimum_outside_fraction:
            continue
        line_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.line(
            line_mask,
            (x1, y1),
            (x2, y2),
            255,
            thickness=max(3, int(round(cable_thickness * 0.06))),
        )
        line_pixels = blurred[line_mask > 0]
        if line_pixels.size == 0:
            continue
        contrast = float(np.percentile(line_pixels, 70.0) - median)
        if contrast < float(config.get("geometry_min_line_contrast", 8.0)):
            continue
        length_ratio = length / max(cable_length, 1.0)
        perpendicularity = 1.0 - axis_dot
        score = float(
            np.clip(
                length_ratio
                * perpendicularity
                * min(1.0, contrast / 40.0),
                0.0,
                1.0,
            )
        )
        row = {
            "score": round(score, 6),
            "line_length_ratio": round(length_ratio, 6),
            "line_contrast": round(contrast, 4),
            "perpendicularity": round(perpendicularity, 6),
            "endpoint_distance_ratio": round(
                endpoint_distance / max(cable_length, 1.0),
                6,
            ),
            "outside_fraction": round(outside_fraction, 6),
            "cable_thickness_ratio": round(cable_thickness_ratio, 6),
            "line_xyxy": [x1, y1, x2, y2],
        }
        if best is None or score > float(best["score"]):
            best = row
    return best or empty


def support_position_frame_decision(
    detections: list[PromptDetection],
    support_prompts: list[str],
    config: dict[str, Any],
    reference_mask: np.ndarray | None = None,
    geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse semantic support evidence with cable-relative line geometry."""
    minimum_overlap = float(
        config.get("minimum_support_reference_overlap", 0.0)
    )
    reference_overlaps: dict[int, float] = {}
    eligible: list[PromptDetection] = []
    for detection in detections:
        overlap = (
            mask_overlap_coefficient(detection.mask, reference_mask)
            if reference_mask is not None
            else 1.0
        )
        reference_overlaps[id(detection)] = overlap
        if overlap >= minimum_overlap:
            eligible.append(detection)

    prompt_scores = prompt_best_scores(eligible, support_prompts)
    support_score = max(prompt_scores.values(), default=0.0)
    configured_thresholds = config.get("support_prompt_thresholds")
    thresholds = (
        {
            str(prompt): float(threshold)
            for prompt, threshold in configured_thresholds.items()
            if str(prompt) in prompt_scores
        }
        if isinstance(configured_thresholds, dict)
        else {}
    )
    if thresholds:
        normalized_scores = [
            prompt_scores[prompt] / max(threshold, 1e-6)
            for prompt, threshold in thresholds.items()
        ]
        consensus_strength = min(normalized_scores, default=0.0)
        prompt_positive = consensus_strength >= 1.0
    else:
        threshold = float(config.get("support_score_threshold", 0.50))
        consensus_strength = support_score / max(threshold, 1e-6)
        prompt_positive = consensus_strength >= 1.0

    geometry = geometry or {
        "score": 0.0,
        "line_xyxy": None,
    }
    geometry_score = float(geometry.get("score", 0.0))
    geometry_threshold = float(
        config.get("support_geometry_score_threshold", 0.11)
    )
    geometry_strength = geometry_score / max(geometry_threshold, 1e-6)
    geometry_positive = geometry_strength >= 1.0
    decision_mode = str(config.get("support_decision_mode", "prompt_only"))
    if decision_mode == "prompt_only":
        support_positive = prompt_positive
        decision_strength = consensus_strength
    elif decision_mode == "geometry_only":
        support_positive = geometry_positive
        decision_strength = geometry_strength
    elif decision_mode == "geometry_and_prompt":
        support_positive = geometry_positive and prompt_positive
        decision_strength = min(geometry_strength, consensus_strength)
    elif decision_mode == "geometry_or_prompt":
        support_positive = geometry_positive or prompt_positive
        decision_strength = max(geometry_strength, consensus_strength)
    else:
        raise ValueError(
            f"Unsupported position support_decision_mode: {decision_mode}"
        )
    if support_positive:
        strength = float(
            np.clip(
                decision_strength - 1.0,
                0.0,
                1.0,
            )
        )
        confidence = 0.80 + 0.19 * strength
        position = "suspended"
    else:
        distance = float(np.clip(1.0 - decision_strength, 0.0, 1.0))
        confidence = 0.65 + 0.25 * distance
        position = "exposed"
    return {
        "position": position,
        "confidence": round(confidence, 4),
        "suspended_probability": round(
            confidence if support_positive else 1.0 - confidence,
            6,
        ),
        "support_score": round(support_score, 6),
        "support_consensus_strength": round(consensus_strength, 6),
        "support_decision_mode": decision_mode,
        "support_decision_strength": round(decision_strength, 6),
        "support_positive": support_positive,
        "support_prompt_positive": prompt_positive,
        "support_geometry_positive": geometry_positive,
        "support_geometry_score": round(geometry_score, 6),
        "support_geometry_threshold": round(geometry_threshold, 6),
        "support_geometry": geometry,
        "support_prompt_scores": {
            prompt: round(score, 6)
            for prompt, score in prompt_scores.items()
        },
        "eligible_support_detections": len(eligible),
        "maximum_support_reference_overlap": round(
            max(reference_overlaps.values(), default=0.0),
            6,
        ),
        "active_cues": (
            [
                cue
                for cue, active in (
                    ("support_prompt", prompt_positive),
                    ("perpendicular_endpoint_support", geometry_positive),
                )
                if active
            ]
            if support_positive
            else []
        ),
    }


def aggregate_support_position(
    frame_decisions: list[dict[str, Any]],
) -> tuple[str, float, float]:
    """Aggregate direct support evidence; absence in another frame is neutral."""
    if not frame_decisions:
        return "exposed", 0.5, 0.0
    positives = [
        decision
        for decision in frame_decisions
        if bool(decision.get("support_positive"))
    ]
    if positives:
        confidence = max(float(item["confidence"]) for item in positives)
        evidence_ratio = len(positives) / len(frame_decisions)
        return "suspended", round(confidence, 4), round(evidence_ratio, 4)
    confidence = max(float(item["confidence"]) for item in frame_decisions)
    return "exposed", round(confidence, 4), 1.0


def damage_frame_decision(
    detections: list[PromptDetection],
    visual_score: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    positive_groups = damage_prompt_groups(config)
    positive_prompts = flatten_prompt_groups(positive_groups)
    exclusion_prompts = list(map(str, config.get("exclusion_prompts", [])))
    intact_prompts = list(map(str, config["intact_prompts"]))
    local_max_area = float(config.get("local_max_area_ratio", 0.25))
    positive_max_coverage = float(
        config.get("positive_max_reference_coverage", 1.0)
    )
    filtered_detections, suppressed = filter_excluded_damage_detections(
        detections,
        positive_prompts,
        exclusion_prompts,
        config,
    )
    filtered_detections, ignored_endpoint_regions = (
        filter_endpoint_damage_detections(
            filtered_detections,
            positive_prompts,
            config,
        )
    )
    configured_minimums = config.get("positive_prompt_min_scores")
    prompt_minimums = (
        {
            str(prompt): float(score)
            for prompt, score in configured_minimums.items()
        }
        if isinstance(configured_minimums, dict)
        else {}
    )
    configured_coverage = config.get("positive_prompt_max_reference_coverage")
    coverage_limits = (
        {
            str(prompt): float(coverage)
            for prompt, coverage in configured_coverage.items()
        }
        if isinstance(configured_coverage, dict)
        else {}
    )
    positive_set = set(positive_prompts)
    filtered_detections = [
        detection
        for detection in filtered_detections
        if detection.prompt not in positive_set
        or (
            detection.score >= prompt_minimums.get(detection.prompt, 0.0)
            and detection.reference_coverage
            <= coverage_limits.get(
                detection.prompt,
                positive_max_coverage,
            )
        )
    ]
    local_prompt_scores = prompt_best_scores(
        filtered_detections,
        positive_prompts,
        max_area_ratio=local_max_area,
    )
    local_intact = prompt_best_scores(
        detections,
        intact_prompts,
        max_area_ratio=local_max_area,
    )
    global_prompt_scores = prompt_best_scores(
        filtered_detections,
        positive_prompts,
    )
    global_intact = prompt_best_scores(detections, intact_prompts)
    exclusion_scores = prompt_best_scores(
        detections,
        exclusion_prompts,
        max_area_ratio=local_max_area,
    )
    local_positive = group_best_scores(local_prompt_scores, positive_groups)
    global_positive = group_best_scores(global_prompt_scores, positive_groups)

    consensus_floor = float(config.get("consensus_min_score", 0.020))
    supporting_groups = sum(
        score >= consensus_floor for score in local_positive.values()
    )
    raw_positive_score = max(local_positive.values(), default=0.0)
    consensus_bonus = float(config.get("consensus_bonus", 0.004))
    positive_score = min(
        1.0,
        raw_positive_score
        + consensus_bonus * max(0, supporting_groups - 1),
    )
    intact_score = max(local_intact.values(), default=0.0)
    margin = positive_score - intact_score

    weak = float(config.get("weak_prompt_score", 0.020))
    strong = float(config.get("strong_prompt_score", 0.045))
    margin_threshold = float(config.get("prompt_margin", 0.003))
    relative = positive_score / max(positive_score + intact_score, 1e-6)
    absolute = float(
        np.clip(
            (positive_score - weak) / max(strong - weak, 1e-6),
            0.0,
            1.0,
        )
    )
    prompt_component = 0.55 * absolute + 0.45 * relative
    prompt_weight = float(config.get("prompt_weight", 0.92))
    visual_weight = float(config.get("visual_weight", 0.08))
    combined = float(
        np.clip(
            prompt_weight * prompt_component + visual_weight * visual_score,
            0.0,
            1.0,
        )
    )

    global_positive_score = max(global_positive.values(), default=0.0)
    global_intact_score = max(global_intact.values(), default=0.0)
    global_margin = global_positive_score - global_intact_score
    global_supporting_groups = sum(
        score >= float(config.get("global_consensus_min_score", 0.020))
        for score in global_positive.values()
    )
    global_strong = (
        bool(config.get("enable_global_strong", False))
        and global_positive_score
        >= float(config.get("global_strong_score", 0.060))
        and global_margin >= float(config.get("global_margin", 0.010))
        and global_supporting_groups
        >= int(config.get("min_global_supporting_prompts", 2))
    )
    is_strong = global_strong or (
        positive_score >= strong and margin >= margin_threshold
    )
    is_positive = is_strong or (
        positive_score >= weak
        and margin >= margin_threshold
        and combined >= float(config.get("frame_score_threshold", 0.40))
    )
    direct_thresholds_raw = config.get("direct_positive_prompt_thresholds", {})
    direct_thresholds = (
        {
            str(prompt): float(threshold)
            for prompt, threshold in direct_thresholds_raw.items()
        }
        if isinstance(direct_thresholds_raw, dict)
        else {}
    )
    direct_prompt_evidence: list[dict[str, Any]] = []
    for prompt, threshold in direct_thresholds.items():
        candidates = [
            detection
            for detection in filtered_detections
            if detection.prompt == prompt
            and detection.area_ratio <= local_max_area
        ]
        if not candidates:
            continue
        candidate = max(candidates, key=lambda item: item.score)
        if candidate.score < threshold:
            continue
        direct_prompt_evidence.append(
            {
                "prompt": prompt,
                "score": round(candidate.score, 6),
                "threshold": round(threshold, 6),
                "axial_position": (
                    round(candidate.axial_position, 6)
                    if candidate.axial_position is not None
                    else None
                ),
                "center_inside_reference": (
                    candidate.center_inside_reference
                ),
            }
        )
    require_center = bool(
        config.get("direct_require_center_inside_reference", False)
    )
    direct_positive_prompts = [
        str(item["prompt"])
        for item in direct_prompt_evidence
        if not require_center or bool(item["center_inside_reference"])
    ]
    decision_mode = str(config.get("decision_mode", "fused"))
    if decision_mode == "direct_prompt_threshold":
        is_strong = bool(direct_positive_prompts)
        is_positive = is_strong
    elif decision_mode != "fused":
        raise ValueError(f"Unsupported damage decision_mode: {decision_mode}")
    suppressed_count = len(suppressed)
    suppressed = sorted(
        suppressed,
        key=lambda item: (
            float(item["positive_score"]),
            float(item["exclusion_score"]),
        ),
        reverse=True,
    )[: int(config.get("max_suppressed_regions_report", 16))]
    return {
        "positive_score": round(positive_score, 6),
        "raw_positive_score": round(raw_positive_score, 6),
        "intact_score": round(intact_score, 6),
        "margin": round(margin, 6),
        "supporting_prompts": supporting_groups,
        "supporting_groups": supporting_groups,
        "prompt_component": round(prompt_component, 6),
        "combined_score": round(combined, 6),
        "global_positive_score": round(global_positive_score, 6),
        "global_intact_score": round(global_intact_score, 6),
        "global_margin": round(global_margin, 6),
        "global_supporting_prompts": global_supporting_groups,
        "global_supporting_groups": global_supporting_groups,
        "global_strong": global_strong,
        "strong": is_strong,
        "positive": is_positive,
        "decision_mode": decision_mode,
        "direct_positive_prompts": direct_positive_prompts,
        "direct_prompt_evidence": direct_prompt_evidence,
        "exclusion_score": round(max(exclusion_scores.values(), default=0.0), 6),
        "exclusion_prompt_scores": {
            prompt: round(score, 6)
            for prompt, score in exclusion_scores.items()
        },
        "suppressed_positive_regions": suppressed,
        "suppressed_positive_region_count": suppressed_count,
        "ignored_endpoint_regions": sorted(
            ignored_endpoint_regions,
            key=lambda item: float(item["positive_score"]),
            reverse=True,
        )[: int(config.get("max_suppressed_regions_report", 16))],
        "ignored_endpoint_region_count": len(ignored_endpoint_regions),
        "local_prompt_scores": {
            prompt: round(score, 6)
            for prompt, score in {
                **local_prompt_scores,
                **local_intact,
                **exclusion_scores,
            }.items()
        },
        "local_group_scores": {
            name: round(score, 6)
            for name, score in local_positive.items()
        },
    }


def axial_distance(first: float, second: float) -> float:
    """Compare cable-axis positions while tolerating PCA direction flips."""
    return min(abs(first - second), abs((1.0 - first) - second))


def aggregate_sam3_damage(
    frame_decisions: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[str, float]:
    """Confirm weak direct-prompt hits over time without blocking strong hits."""
    if str(config.get("decision_mode", "fused")) != "direct_prompt_threshold":
        return aggregate_damage(frame_decisions, config)
    if not frame_decisions:
        return "intact", 0.5

    base_thresholds = {
        str(prompt): float(threshold)
        for prompt, threshold in config.get(
            "direct_positive_prompt_thresholds", {}
        ).items()
    }
    single_thresholds = {
        str(prompt): float(threshold)
        for prompt, threshold in config.get(
            "single_frame_direct_prompt_thresholds", {}
        ).items()
    }
    strong_thresholds = {
        str(prompt): float(threshold)
        for prompt, threshold in config.get(
            "direct_strong_prompt_thresholds", {}
        ).items()
    }
    require_center = bool(
        config.get("direct_require_center_inside_reference", False)
    )
    minimum_frames = max(
        2,
        int(config.get("direct_borderline_min_matching_frames", 2)),
    )
    axial_tolerance = float(
        config.get("direct_borderline_axial_tolerance", 0.10)
    )

    evidence_by_frame: list[list[dict[str, Any]]] = []
    for decision in frame_decisions:
        eligible = [
            item
            for item in decision.get("direct_prompt_evidence", [])
            if (
                not require_center
                or bool(item.get("center_inside_reference", False))
            )
        ]
        evidence_by_frame.append(eligible)

    all_evidence = [
        item for frame_evidence in evidence_by_frame for item in frame_evidence
    ]
    maximum_score = max(
        (float(item["score"]) for item in all_evidence),
        default=0.0,
    )

    if len(frame_decisions) == 1:
        confirmed = any(
            float(item["score"])
            >= single_thresholds.get(
                str(item["prompt"]),
                base_thresholds.get(str(item["prompt"]), 1.0),
            )
            for item in all_evidence
        )
        confirmation_ratio = 1.0 if confirmed else 0.0
    else:
        strong = any(
            float(item["score"])
            >= strong_thresholds.get(
                str(item["prompt"]),
                max(
                    0.05,
                    base_thresholds.get(str(item["prompt"]), 1.0),
                ),
            )
            for item in all_evidence
        )
        matching_frames = 0
        for frame_index, frame_evidence in enumerate(evidence_by_frame):
            frame_matches = False
            for item in frame_evidence:
                position = item.get("axial_position")
                if position is None:
                    continue
                for other_frame in evidence_by_frame[frame_index + 1 :]:
                    for other in other_frame:
                        other_position = other.get("axial_position")
                        if (
                            other_position is not None
                            and str(other["prompt"]) == str(item["prompt"])
                            and axial_distance(
                                float(position),
                                float(other_position),
                            )
                            <= axial_tolerance
                        ):
                            frame_matches = True
                            break
                    if frame_matches:
                        break
                if frame_matches:
                    break
            if frame_matches:
                matching_frames += 1
        # Each pair is counted from its earlier frame, hence N-1 matches prove
        # that N frames contain a stable cable-axis response.
        temporal = matching_frames >= minimum_frames - 1
        confirmed = strong or temporal
        confirmation_ratio = (
            1.0
            if strong
            else min(1.0, (matching_frames + 1) / minimum_frames)
            if temporal
            else 0.0
        )

    if confirmed:
        confidence = 0.55 + 0.44 * max(maximum_score, confirmation_ratio)
        return "damaged", round(min(confidence, 0.99), 4)
    intact_strength = max(1.0 - maximum_score, 0.5)
    confidence = 0.55 + 0.40 * intact_strength
    if maximum_score > 0.0:
        confidence = min(confidence, 0.72)
    return "intact", round(min(confidence, 0.95), 4)


def top_detection_rows(
    detections: list[PromptDetection],
    limit: int = 16,
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": detection.prompt,
            "score": round(detection.score, 6),
            "box_xyxy": list(detection.box_xyxy),
            "area_ratio": round(detection.area_ratio, 6),
            "mask_overlap": round(detection.mask_overlap, 4),
            "reference_coverage": round(detection.reference_coverage, 4),
            "axial_position": (
                round(detection.axial_position, 4)
                if detection.axial_position is not None
                else None
            ),
            "center_inside_reference": detection.center_inside_reference,
        }
        for detection in sorted(
            detections,
            key=lambda item: item.score,
            reverse=True,
        )[:limit]
    ]


def annotate_frame(
    image: np.ndarray,
    reference_mask: np.ndarray,
    detections: list[PromptDetection],
    label: str,
) -> np.ndarray:
    canvas = image.copy()
    contours, _ = cv2.findContours(
        (reference_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, (0, 220, 255), 2)
    colors = [
        (80, 220, 80),
        (255, 150, 60),
        (220, 90, 220),
        (80, 210, 230),
        (230, 210, 80),
        (170, 120, 255),
    ]
    for index, detection in enumerate(
        sorted(detections, key=lambda item: item.score, reverse=True)[:6]
    ):
        color = colors[index % len(colors)]
        overlay = canvas.copy()
        overlay[detection.mask > 0] = color
        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0)
        x1, y1, x2, y2 = detection.box_xyxy
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        legend = f"{detection.prompt}: {detection.score:.3f}"
        legend_y = 60 + index * 24
        (text_width, text_height), baseline = cv2.getTextSize(
            legend,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            1,
        )
        cv2.rectangle(
            canvas,
            (6, legend_y - text_height - 5),
            (12 + text_width, legend_y + baseline + 3),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            legend,
            (9, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def classify(
    samples: list[SampleInput],
    detector: SAM3PromptDetector,
    config: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    damage_config = config["damage"]
    position_config = config["position"]
    position_enabled = bool(position_config.get("enabled", True))
    damage_groups = damage_prompt_groups(damage_config)
    damage_prompts = list(
        dict.fromkeys(
            flatten_prompt_groups(damage_groups)
            + list(map(str, damage_config.get("exclusion_prompts", [])))
            + list(map(str, damage_config["intact_prompts"]))
        )
    )
    support_prompts = list(map(str, position_config.get("support_prompts", [])))

    frame_data: dict[tuple[str, str, int], dict[str, Any]] = {}
    for sample in samples:
        for frame in sample.frames:
            full, crop, full_mask = read_frame_images(frame)
            crop_mask = crop_mask_for_frame(frame, full_mask, crop.shape)
            damage_mask = damage_support_mask(crop_mask)
            damage_detections = detector.predict(
                crop,
                damage_prompts,
                damage_mask,
            )
            visual = visual_damage_features(crop, damage_mask, damage_config)
            damage_decision = damage_frame_decision(
                damage_detections,
                float(visual["score"]),
                damage_config,
            )

            support_geometry = support_geometry_features(
                crop,
                crop_mask,
                position_config,
            )

            support_mask = position_support_mask(
                full_mask,
                int(position_config.get("support_search_margin_px", 220)),
            )
            if position_enabled and support_prompts:
                position_detections = detector.predict(
                    full,
                    support_prompts,
                    support_mask,
                )
            else:
                position_detections = []
            if position_enabled:
                position_decision = support_position_frame_decision(
                    position_detections,
                    support_prompts,
                    position_config,
                    full_mask,
                    support_geometry,
                )
            else:
                position_decision = {
                    "position": "exposed",
                    "confidence": 1.0,
                    "suspended_probability": 0.0,
                    "support_score": 0.0,
                    "support_consensus_strength": 0.0,
                    "support_positive": False,
                    "support_prompt_scores": {},
                    "eligible_support_detections": 0,
                    "maximum_support_reference_overlap": 0.0,
                    "active_cues": [],
                    "evaluated": False,
                }
            frame_data[(sample.video, sample.sample_id, frame.frame_idx)] = {
                "full": full,
                "crop": crop,
                "full_mask": full_mask,
                "crop_mask": crop_mask,
                "damage_mask": damage_mask,
                "damage_detections": damage_detections,
                "damage_visual": visual,
                "damage_decision": damage_decision,
                "support_mask": support_mask,
                "support_geometry": support_geometry,
                "position_detections": position_detections,
                "position_decision": position_decision,
            }

    results: list[dict[str, Any]] = []
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    low_confidence = float(config["quality"].get("low_confidence", 0.60))
    for sample in samples:
        damage_frames = [
            frame_data[
                (sample.video, sample.sample_id, frame.frame_idx)
            ]["damage_decision"]
            for frame in sample.frames
        ]
        position_frames = [
            frame_data[
                (sample.video, sample.sample_id, frame.frame_idx)
            ]["position_decision"]
            for frame in sample.frames
        ]
        damage, damage_confidence = aggregate_sam3_damage(
            damage_frames,
            damage_config,
        )
        if position_enabled:
            position, position_confidence, position_agreement = (
                aggregate_support_position(position_frames)
            )
        else:
            position, position_confidence, position_agreement = (
                "exposed",
                1.0,
                1.0,
            )
        class_id, class_name = resolve_sam3_class(
            position,
            damage,
            config,
        )
        reason_codes: list[str] = []
        if damage_confidence < low_confidence:
            reason_codes.append("low_damage_confidence")
        if position_confidence < low_confidence:
            reason_codes.append("low_position_confidence")
        if position == "suspended" and position_agreement < 2 / 3:
            reason_codes.append("support_visible_in_subset_of_frames")
        if damage == "intact" and any(
            float(item["positive_score"])
            >= float(damage_config.get("weak_prompt_score", 0.020))
            for item in damage_frames
        ):
            reason_codes.append("weak_damage_evidence")

        frame_votes: list[dict[str, Any]] = []
        frame_diagnostics: list[dict[str, Any]] = []
        safe_video = hashlib.sha1(
            sample.video.encode("utf-8")
        ).hexdigest()[:10]
        for frame in sample.frames:
            key = (sample.video, sample.sample_id, frame.frame_idx)
            item = frame_data[key]
            damage_decision = item["damage_decision"]
            position_decision = item["position_decision"]
            frame_votes.append(
                {
                    "frame_idx": frame.frame_idx,
                    "position": position_decision["position"],
                    "position_confidence": position_decision["confidence"],
                    "damage_evidence": (
                        "clear_damage"
                        if damage_decision["positive"]
                        else "no_visible_damage"
                    ),
                    "damage_confidence": round(
                        0.5
                        + abs(
                            float(damage_decision["combined_score"]) - 0.5
                        ),
                        4,
                    ),
                    "usable": True,
                }
            )
            frame_diagnostics.append(
                {
                    "frame_idx": frame.frame_idx,
                    "full_path": str(frame.full_path),
                    "crop_path": str(frame.crop_path),
                    "mask_path": (
                        str(frame.mask_path) if frame.mask_path else None
                    ),
                    "sam3_score": frame.sam3_score,
                    "sam3_area_ratio": frame.sam3_area_ratio,
                    "damage": damage_decision,
                    "visual_damage": item["damage_visual"],
                    "damage_detections": top_detection_rows(
                        item["damage_detections"]
                    ),
                    "position": position_decision,
                    "position_detections": top_detection_rows(
                        item["position_detections"]
                    ),
                }
            )

            accepted_damage_detections, _ = filter_excluded_damage_detections(
                item["damage_detections"],
                flatten_prompt_groups(damage_groups),
                list(map(str, damage_config.get("exclusion_prompts", []))),
                damage_config,
            )
            accepted_damage_detections, _ = filter_endpoint_damage_detections(
                accepted_damage_detections,
                flatten_prompt_groups(damage_groups),
                damage_config,
            )
            local_damage_detections = strongest_detections_by_prompt(
                accepted_damage_detections,
                flatten_prompt_groups(damage_groups),
                max_area_ratio=float(
                    damage_config.get("local_max_area_ratio", 0.25)
                ),
                max_reference_coverage=float(
                    damage_config.get(
                        "gallery_max_reference_coverage",
                        0.35,
                    )
                ),
                min_score=float(
                    damage_config.get(
                        "gallery_min_score",
                        damage_config.get("consensus_min_score", 0.020),
                    )
                ),
            )
            damage_canvas = annotate_frame(
                item["crop"],
                item["damage_mask"],
                local_damage_detections,
                (
                    f"{sample.sample_id} f={frame.frame_idx} "
                    f"damage={damage_decision['combined_score']:.3f}"
                ),
            )
            full_damage_canvas = annotate_frame(
                item["full"],
                damage_support_mask(item["full_mask"]),
                project_crop_detections(
                    local_damage_detections,
                    item["crop"].shape,
                    item["full"].shape,
                    frame.crop_bbox_xyxy,
                ),
                (
                    f"{sample.sample_id} f={frame.frame_idx} "
                    f"damage={damage_decision['combined_score']:.3f}"
                ),
            )
            position_canvas = annotate_frame(
                item["full"],
                item["support_mask"],
                item["position_detections"],
                (
                    f"{sample.sample_id} f={frame.frame_idx} "
                    f"position={position_decision['position']}"
                ),
            )
            support_geometry_canvas = annotate_frame(
                item["crop"],
                item["crop_mask"],
                [],
                (
                    f"{sample.sample_id} f={frame.frame_idx} "
                    f"support_geometry="
                    f"{position_decision.get('support_geometry_score', 0.0):.3f}"
                ),
            )
            geometry_line = item["support_geometry"].get("line_xyxy")
            if isinstance(geometry_line, list) and len(geometry_line) == 4:
                x1, y1, x2, y2 = map(int, geometry_line)
                line_color = (
                    (0, 255, 0)
                    if position_decision.get("support_geometry_positive")
                    else (0, 165, 255)
                )
                cv2.line(
                    support_geometry_canvas,
                    (x1, y1),
                    (x2, y2),
                    line_color,
                    5,
                    cv2.LINE_AA,
                )
            cv2.imwrite(
                str(
                    diagnostics_dir
                    / (
                        f"{safe_video}_{sample.sample_id}_"
                        f"f{frame.frame_idx:06d}_damage.jpg"
                    )
                ),
                damage_canvas,
            )
            cv2.imwrite(
                str(
                    diagnostics_dir
                    / (
                        f"{safe_video}_{sample.sample_id}_"
                        f"f{frame.frame_idx:06d}_damage_full.jpg"
                    )
                ),
                full_damage_canvas,
            )
            cv2.imwrite(
                str(
                    diagnostics_dir
                    / (
                        f"{safe_video}_{sample.sample_id}_"
                        f"f{frame.frame_idx:06d}_position.jpg"
                    )
                ),
                position_canvas,
            )
            cv2.imwrite(
                str(
                    diagnostics_dir
                    / (
                        f"{safe_video}_{sample.sample_id}_"
                        f"f{frame.frame_idx:06d}_support_geometry.jpg"
                    )
                ),
                support_geometry_canvas,
            )

        results.append(
            {
                "video": sample.video,
                "sample_id": sample.sample_id,
                "attempt_id": 1,
                "start_sec": sample.start_sec,
                "end_sec": sample.end_sec,
                "position": position,
                "position_confidence": position_confidence,
                "damage": damage,
                "damage_confidence": damage_confidence,
                "class_id": class_id,
                "class_name": class_name,
                "needs_review": bool(reason_codes),
                "evidence_consistency": (
                    "consistent"
                    if position_agreement == 1.0
                    else "mixed"
                ),
                "reason_codes": reason_codes,
                "frame_votes": frame_votes,
                "image_paths": [
                    str(frame.full_path) for frame in sample.frames
                ],
                "method": "sam3_only_zero_shot",
                "position_evaluated": position_enabled,
                "diagnostics": {
                    "position_agreement": position_agreement,
                    "positive_damage_frames": sum(
                        item["positive"] for item in damage_frames
                    ),
                    "strong_damage_frames": sum(
                        item["strong"] for item in damage_frames
                    ),
                    "frames": frame_diagnostics,
                },
            }
        )
    return results


def build_signature(
    samples: list[SampleInput],
    config_path: Path,
    checkpoint_path: Path,
) -> str:
    inputs: list[dict[str, Any]] = []
    for sample in samples:
        for frame in sample.frames:
            for path in (frame.full_path, frame.crop_path, frame.mask_path):
                if path is not None and path.is_file():
                    inputs.append(file_signature(path))
    payload = {
        "classifier": file_signature(Path(__file__)),
        "shared_classifier": file_signature(
            REPO_ROOT / "scripts" / "classify_samples_zero_shot.py"
        ),
        "sam3_adapter": file_signature(
            REPO_ROOT / "scripts" / "select_sample_frames.py"
        ),
        "config": file_signature(config_path),
        "checkpoint": file_signature(checkpoint_path),
        "inputs": inputs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sam3_only_classifier.yaml",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    samples = collect_samples(
        args.input,
        max_frames_per_sample=int(
            config["sampling"].get("max_frames_per_sample", 3)
        ),
        limit=args.limit,
    )
    if not samples:
        raise SystemExit("No SAM3 samples with readable frames were found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requests = [
        {
            "video": sample.video,
            "sample_id": sample.sample_id,
            "start_sec": sample.start_sec,
            "end_sec": sample.end_sec,
            "frame_ids": [frame.frame_idx for frame in sample.frames],
            "full_frame_paths": [
                str(frame.full_path) for frame in sample.frames
            ],
            "crop_paths": [
                str(frame.crop_path) for frame in sample.frames
            ],
            "mask_paths": [
                str(frame.mask_path) if frame.mask_path else None
                for frame in sample.frames
            ],
        }
        for sample in samples
    ]
    write_json(args.out_dir / "requests.json", requests)
    if args.prepare_only:
        print(
            f"Prepared {len(samples)} SAM3-only samples -> "
            f"{args.out_dir / 'requests.json'}"
        )
        return

    checkpoint_path = resolve_config_path(str(config["model"]["checkpoint"]))
    signature = build_signature(samples, config_path, checkpoint_path)
    cache_path = args.out_dir / "result_cache.json"
    results_path = args.out_dir / "results.json"
    if (
        bool(config["quality"].get("cache_results", True))
        and not args.force
        and cache_path.is_file()
        and results_path.is_file()
    ):
        cache = load_json(cache_path)
        if isinstance(cache, dict) and cache.get("signature") == signature:
            print(f"[cache] Reuse SAM3-only results -> {results_path}")
            return

    detector = SAM3PromptDetector(config["model"])
    results = classify(samples, detector, config, args.out_dir)
    write_json(results_path, results)
    write_csv_results(args.out_dir / "results.csv", results)
    write_review(args.out_dir / "review.md", results)
    write_json(
        args.out_dir / "summary.json",
        {
            "method": "sam3_only_zero_shot",
            "total": len(results),
            "classes": {
                str(class_id): sum(
                    result["class_id"] == class_id for result in results
                )
                for class_id in (0, 1, 2)
            },
            "warnings": sum(
                bool(result["reason_codes"]) for result in results
            ),
        },
    )
    write_json(
        cache_path,
        {
            "signature": signature,
            "result_count": len(results),
            "method": "sam3_only_zero_shot",
        },
    )
    print(
        f"SAM3-only classification complete: {len(results)} samples -> "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()
