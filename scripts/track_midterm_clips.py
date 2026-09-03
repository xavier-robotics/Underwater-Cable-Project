#!/usr/bin/env python3
"""Batch-render 10 Hz SAM3 cable and damage tracking for midterm clips."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scripts.visualize_sam3_cable_tracking import (
        DEFAULT_CHECKPOINT,
        prepare_output_dir,
        render_video as render_cable_video,
        sample_video,
    )
    from scripts.visualize_sam3_damage_tracking import (
        normalize_outputs,
        render_video as render_damage_video,
        track_concept,
    )
except ModuleNotFoundError:
    from visualize_sam3_cable_tracking import (
        DEFAULT_CHECKPOINT,
        prepare_output_dir,
        render_video as render_cable_video,
        sample_video,
    )
    from visualize_sam3_damage_tracking import (
        normalize_outputs,
        render_video as render_damage_video,
        track_concept,
    )


GT_PATTERN = re.compile(r"sample0*(\d+)_([012])$")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CLASS_NAMES = {0: "damaged", 1: "exposed", 2: "suspended"}
CLASS_NAMES_CN = {0: "破损", 1: "裸露", 2: "悬空"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--classification-results",
        type=Path,
        help="SAM3 still-image results used to anchor damage boxes.",
    )
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cable-prompt", default="pipe")
    parser.add_argument(
        "--cable-point-first",
        action="store_true",
        help="Initialize cable tracking from the matched GT location before text prompts.",
    )
    parser.add_argument(
        "--cable-dark-first",
        action="store_true",
        help="Use a dark interior cable point instead of the GT-region center.",
    )
    parser.add_argument(
        "--damage-prompt",
        action="append",
        dest="damage_prompts",
        help="Damage prompt fallback order; repeat for multiple prompts.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--damage-bright-anchor",
        action="store_true",
        help="Locate the damage patch as a bright component inside the cable mask.",
    )
    parser.add_argument(
        "--damage-single-point",
        action="store_true",
        help="Use one positive damage point without surrounding negative points.",
    )
    parser.add_argument("--only", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def load_gt(gt_dir: Path) -> dict[int, tuple[int, Path]]:
    rows: dict[int, tuple[int, Path]] = {}
    for path in sorted(gt_dir.iterdir()):
        match = GT_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        sample = int(match.group(1))
        rows[sample] = (int(match.group(2)), path.resolve())
    if not rows:
        raise SystemExit(f"No sampleNN_CLASS GT images found in {gt_dir}")
    return rows


def discover_jobs(
    input_dir: Path,
    gt: dict[int, tuple[int, Path]],
    only: set[int] | None,
) -> list[tuple[int, Path, int, Path]]:
    jobs = []
    for path in input_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if not path.stem.isdigit():
            continue
        sample = int(path.stem)
        if only is not None and sample not in only:
            continue
        if sample not in gt:
            raise SystemExit(f"No GT label for clip {path.name}")
        label, gt_path = gt[sample]
        jobs.append((sample, path.resolve(), label, gt_path))
    return sorted(jobs, key=lambda row: row[0])


def feature_gray(image: np.ndarray, maximum_side: int = 800) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, maximum_side / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return gray


def match_prompt_frame(frame_paths: list[Path], reference_path: Path) -> tuple[int, float]:
    """Match a cropped/rotated GT image to the closest sampled video frame."""
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"Cannot read GT reference: {reference_path}")
    detector = cv2.ORB_create(nfeatures=1600, fastThreshold=8)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(
        feature_gray(reference),
        None,
    )
    if reference_descriptors is None or not reference_keypoints:
        return len(frame_paths) // 2, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    best_index = 0
    best_score = -1.0
    for index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        _, descriptors = detector.detectAndCompute(feature_gray(frame), None)
        if descriptors is None:
            continue
        pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
        good_matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance
        ]
        score = len(good_matches) / max(1, len(reference_keypoints))
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def crop_offset(full: np.ndarray, crop: np.ndarray) -> tuple[int, int, float]:
    if crop.shape[:2] == full.shape[:2]:
        return 0, 0, 1.0
    if crop.shape[0] > full.shape[0] or crop.shape[1] > full.shape[1]:
        return 0, 0, 0.0
    response = cv2.matchTemplate(
        cv2.cvtColor(full, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
        cv2.TM_CCOEFF_NORMED,
    )
    _, score, _, location = cv2.minMaxLoc(response)
    return int(location[0]), int(location[1]), float(score)


def bright_patch_box(
    full: np.ndarray,
    crop: np.ndarray,
    crop_mask: np.ndarray,
    offset_xy: tuple[int, int],
) -> list[float] | None:
    """Find a bright simulated metal patch inside the cached pipe mask."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = crop_mask > 0
    values = gray[mask]
    if values.size == 0:
        return None
    threshold = max(float(np.percentile(values, 92.0)), float(np.median(values)) + 18.0)
    bright = ((gray >= threshold) & mask).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (bright > 0).astype(np.uint8),
        connectivity=8,
    )
    crop_area = crop.shape[0] * crop.shape[1]
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if not 0.0008 <= area / max(1, crop_area) <= 0.25:
            continue
        component = labels == index
        component_values = gray[component]
        saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1][component]
        touches_border = x == 0 or y == 0 or x + width >= crop.shape[1] or y + height >= crop.shape[0]
        border_factor = 0.55 if touches_border else 1.0
        score = (
            float(component_values.mean())
            * (1.0 + float(saturation.mean()) / 128.0)
            * np.sqrt(area)
            * border_factor
        )
        candidates.append((score, x, y, width, height))
    if not candidates:
        return None
    _, x, y, width, height = max(candidates)
    offset_x, offset_y = offset_xy
    return [
        float(x + offset_x),
        float(y + offset_y),
        float(width),
        float(height),
    ]


def load_reference_damage_boxes(results_path: Path | None) -> dict[int, list[float]]:
    if results_path is None:
        return {}
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    boxes: dict[int, list[float]] = {}
    for row in rows:
        match = GT_PATTERN.fullmatch(str(row.get("sample_id", "")))
        if match is None or int(match.group(2)) != 0:
            continue
        sample = int(match.group(1))
        frames = row.get("diagnostics", {}).get("frames", [])
        if not frames:
            continue
        frame = frames[0]
        full = cv2.imread(str(row["video"]), cv2.IMREAD_COLOR)
        crop_path = Path(str(frame.get("crop_path", "")))
        mask_path = Path(str(frame.get("mask_path", "")))
        if not crop_path.is_absolute():
            crop_path = Path.cwd() / crop_path
        if not mask_path.is_absolute():
            mask_path = Path.cwd() / mask_path
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        crop_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if full is None or crop is None or crop_mask is None:
            continue
        offset_x, offset_y, _ = crop_offset(full, crop)
        crop_area = crop.shape[0] * crop.shape[1]
        detections = []
        for detection in frame.get("damage_detections", []):
            if detection.get("prompt") != "metal patch on pipe":
                continue
            x1, y1, x2, y2 = map(float, detection["box_xyxy"])
            bbox_ratio = (x2 - x1) * (y2 - y1) / max(1, crop_area)
            if bbox_ratio <= 0.50:
                detections.append(detection)
        if detections:
            selected = max(detections, key=lambda item: float(item.get("score", 0.0)))
            x1, y1, x2, y2 = map(float, selected["box_xyxy"])
            boxes[sample] = [
                x1 + offset_x,
                y1 + offset_y,
                x2 - x1,
                y2 - y1,
            ]
            continue
        fallback = bright_patch_box(
            full,
            full,
            np.full(full.shape[:2], 255, dtype=np.uint8),
            (0, 0),
        )
        if fallback is not None:
            boxes[sample] = fallback
    return boxes


def scaled_feature_gray(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 900.0 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return gray, scale


def map_reference_box(
    reference_path: Path,
    frame_path: Path,
    box_xywh: list[float],
) -> tuple[list[float] | None, dict[str, Any]]:
    """Project a GT-image damage box onto its matched full video frame."""
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if reference is None or frame is None:
        return None, {"reason": "unreadable_image"}
    reference_gray, reference_scale = scaled_feature_gray(reference)
    frame_gray, frame_scale = scaled_feature_gray(frame)
    attempts = [
        (
            "orb",
            cv2.ORB_create(nfeatures=3200, fastThreshold=5),
            cv2.NORM_HAMMING,
            0.82,
        )
    ]
    if hasattr(cv2, "SIFT_create"):
        attempts.append(
            (
                "sift",
                cv2.SIFT_create(
                    nfeatures=3200,
                    contrastThreshold=0.01,
                    edgeThreshold=12,
                ),
                cv2.NORM_L2,
                0.78,
            )
        )
    best_failure: dict[str, Any] = {
        "reason": "missing_descriptors",
        "good_matches": 0,
        "inliers": 0,
    }
    homography = None
    method = None
    good: list[Any] = []
    inliers = 0
    for method_name, detector, norm, ratio in attempts:
        ref_keypoints, ref_descriptors = detector.detectAndCompute(
            reference_gray,
            None,
        )
        dst_keypoints, dst_descriptors = detector.detectAndCompute(
            frame_gray,
            None,
        )
        if ref_descriptors is None or dst_descriptors is None:
            continue
        pairs = cv2.BFMatcher(norm).knnMatch(
            ref_descriptors,
            dst_descriptors,
            k=2,
        )
        candidate_good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        ]
        if len(candidate_good) < 4:
            if len(candidate_good) > int(best_failure["good_matches"]):
                best_failure = {
                    "reason": "too_few_matches",
                    "method": method_name,
                    "good_matches": len(candidate_good),
                    "inliers": 0,
                }
            continue
        source_points = np.float32(
            [ref_keypoints[item.queryIdx].pt for item in candidate_good]
        ).reshape(-1, 1, 2)
        destination_points = np.float32(
            [dst_keypoints[item.trainIdx].pt for item in candidate_good]
        ).reshape(-1, 1, 2)
        candidate_homography, inlier_mask = cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            4.0,
        )
        candidate_inliers = (
            int(inlier_mask.sum()) if inlier_mask is not None else 0
        )
        if candidate_homography is not None and candidate_inliers >= 4:
            homography = candidate_homography
            method = method_name
            good = candidate_good
            inliers = candidate_inliers
            break
        if candidate_inliers >= int(best_failure["inliers"]):
            best_failure = {
                "reason": "homography_failed",
                "method": method_name,
                "good_matches": len(candidate_good),
                "inliers": candidate_inliers,
            }
    if homography is None:
        return None, best_failure
    x, y, width, height = box_xywh
    corners = np.float32(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ]
    )
    corners *= reference_scale
    projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography)
    projected = projected.reshape(-1, 2) / frame_scale
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    frame_height, frame_width = frame.shape[:2]
    pad_x = max(4.0, (x2 - x1) * 0.10)
    pad_y = max(4.0, (y2 - y1) * 0.10)
    x1 = float(np.clip(x1 - pad_x, 0, frame_width - 1))
    y1 = float(np.clip(y1 - pad_y, 0, frame_height - 1))
    x2 = float(np.clip(x2 + pad_x, x1 + 1, frame_width))
    y2 = float(np.clip(y2 + pad_y, y1 + 1, frame_height))
    normalized = [
        x1 / frame_width,
        y1 / frame_height,
        (x2 - x1) / frame_width,
        (y2 - y1) / frame_height,
    ]
    return normalized, {
        "reason": "ok",
        "method": method,
        "good_matches": len(good),
        "inliers": inliers,
        "projected_box_pixels_xywh": [x1, y1, x2 - x1, y2 - y1],
    }


def draw_prompt_box(
    frame_path: Path,
    box_xywh: list[float] | None,
    output_path: Path,
) -> None:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return
    if box_xywh is not None:
        x, y, width, height = box_xywh
        frame_height, frame_width = frame.shape[:2]
        p1 = (round(x * frame_width), round(y * frame_height))
        p2 = (round((x + width) * frame_width), round((y + height) * frame_height))
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(
            frame,
            "damage tracking anchor",
            (p1[0], max(30, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def dark_cable_point(frame_path: Path) -> tuple[list[float], dict[str, Any]]:
    """Choose a point inside the thickest dark structure in a video frame."""
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return [0.5, 0.5], {"reason": "unreadable_frame_center_fallback"}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=31.0, sigmaY=31.0)
    local_darkness = background.astype(np.float32) - gray.astype(np.float32)
    threshold = max(6.0, float(np.percentile(local_darkness, 91.0)))
    dark = (
        (local_darkness >= threshold)
        & (gray <= np.percentile(gray, 72.0))
    ).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    margin_x = max(4, round(width * 0.025))
    margin_y = max(4, round(height * 0.025))
    dark[:margin_y] = 0
    dark[-margin_y:] = 0
    dark[:, :margin_x] = 0
    dark[:, -margin_x:] = 0
    distance = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    _, radius, _, location = cv2.minMaxLoc(distance)
    if radius <= 0:
        return [0.5, 0.5], {
            "reason": "empty_dark_mask_center_fallback",
            "threshold": threshold,
        }
    x, y = location
    return [x / width, y / height], {
        "reason": "dark_structure",
        "threshold": threshold,
        "distance_radius_pixels": float(radius),
        "point_pixels_xy": [int(x), int(y)],
    }


def reference_cable_point(
    reference_path: Path,
    frame_path: Path,
) -> tuple[list[float], dict[str, Any]]:
    """Map the GT crop center into the matched frame, with an image fallback."""
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is not None:
        height, width = reference.shape[:2]
        region, mapping = map_reference_box(
            reference_path,
            frame_path,
            [0.0, 0.0, float(width), float(height)],
        )
        region_area = region[2] * region[3] if region is not None else 0.0
        if (
            region is not None
            and int(mapping.get("inliers", 0)) >= 8
            and region_area <= 0.75
        ):
            x, y, region_width, region_height = region
            point = [
                float(np.clip(x + region_width * 0.5, 0.0, 1.0)),
                float(np.clip(y + region_height * 0.5, 0.0, 1.0)),
            ]
            return point, {**mapping, "reason": "gt_projection", "region": region}
    point, fallback = dark_cable_point(frame_path)
    return point, {"reason": "dark_fallback", "details": fallback}


def draw_cable_anchor(
    frame_path: Path,
    point: list[float],
    output_path: Path,
) -> None:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return
    height, width = frame.shape[:2]
    center = (round(point[0] * width), round(point[1] * height))
    cv2.drawMarker(
        frame,
        center,
        (0, 255, 255),
        cv2.MARKER_CROSS,
        40,
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "cable tracking anchor",
        (max(0, center[0] - 160), max(34, center[1] - 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def bright_damage_box_from_cable(
    frame_path: Path,
    cable_outputs: dict[str, Any],
) -> tuple[list[float] | None, dict[str, Any]]:
    """Find a bright simulated patch within the prompt-frame cable mask."""
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return None, {"reason": "unreadable_frame"}
    _, masks = normalize_outputs(cable_outputs)
    if masks.size == 0:
        return None, {"reason": "empty_cable_mask"}
    cable_mask = np.any(masks, axis=0).astype(np.uint8)
    height, width = frame.shape[:2]
    if cable_mask.shape != (height, width):
        cable_mask = cv2.resize(
            cable_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    cable_values = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[cable_mask > 0]
    if cable_values.size < 64:
        return None, {"reason": "small_cable_mask"}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshold = max(
        float(np.percentile(cable_values, 72.0)),
        float(np.median(cable_values)) + 12.0,
    )
    bright = ((gray >= threshold) & (cable_mask > 0)).astype(np.uint8)
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), dtype=np.uint8),
    )
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright,
        connectivity=8,
    )
    cable_area = int((cable_mask > 0).sum())
    cable_y, cable_x = np.where(cable_mask > 0)
    cable_points = np.column_stack((cable_x, cable_y)).astype(np.float32)
    centered_points = cable_points - cable_points.mean(axis=0, keepdims=True)
    _, _, principal_axes = np.linalg.svd(centered_points, full_matrices=False)
    major_axis = principal_axes[0]
    cable_projection = centered_points @ major_axis
    projection_min = float(cable_projection.min())
    projection_span = max(
        1.0,
        float(cable_projection.max()) - projection_min,
    )
    candidates: list[tuple[float, int, int, int, int, int]] = []
    cable_median = float(np.median(cable_values))
    for index in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[index])
        area_ratio = area / max(1, cable_area)
        if not 0.003 <= area_ratio <= 0.48:
            continue
        component = labels == index
        component_center = np.array(
            [x + box_width * 0.5, y + box_height * 0.5],
            dtype=np.float32,
        )
        longitudinal_position = (
            float((component_center - cable_points.mean(axis=0)) @ major_axis)
            - projection_min
        ) / projection_span
        endpoint_distance = min(
            longitudinal_position,
            1.0 - longitudinal_position,
        )
        if endpoint_distance < 0.14:
            continue
        brightness_gain = max(0.0, float(gray[component].mean()) - cable_median)
        saturation = float(hsv[:, :, 1][component].mean())
        compactness = area / max(1, box_width * box_height)
        score = (
            brightness_gain
            * np.sqrt(area)
            * (0.5 + compactness)
            * (1.5 - min(1.0, saturation / 255.0))
            * min(1.0, endpoint_distance / 0.25)
        )
        candidates.append((score, x, y, box_width, box_height, area))
    if not candidates:
        return None, {
            "reason": "no_bright_component",
            "threshold": threshold,
            "cable_area": cable_area,
        }
    score, x, y, box_width, box_height, area = max(candidates)
    pad_x = max(3, round(box_width * 0.08))
    pad_y = max(3, round(box_height * 0.08))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + box_width + pad_x)
    y2 = min(height, y + box_height + pad_y)
    return [
        x1 / width,
        y1 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ], {
        "reason": "bright_component_in_cable",
        "threshold": threshold,
        "score": float(score),
        "component_area": area,
        "cable_area": cable_area,
        "box_pixels_xywh": [x1, y1, x2 - x1, y2 - y1],
    }


def track_cable(
    predictor: Any,
    frames_dir: Path,
    prompts: list[str],
    prompt_frame: int,
    mask_threshold: float,
    reference_path: Path,
    frame_path: Path,
    point_first: bool,
    dark_first: bool,
) -> tuple[
    dict[int, dict[str, Any]],
    float,
    str,
    list[str],
    list[float],
    dict[str, Any],
]:
    """Track a cable using text plus a GT-mapped point fallback."""
    if dark_first:
        point, dark_mapping = dark_cable_point(frame_path)
        mapping = {"reason": "dark_first", "details": dark_mapping}
    else:
        point, mapping = reference_cable_point(reference_path, frame_path)
    errors: list[str] = []

    def run_point() -> tuple[dict[int, dict[str, Any]], float]:
        return track_concept(
            predictor=predictor,
            frames_dir=frames_dir,
            prompt="GT reference point",
            prompt_frame=prompt_frame,
            mask_threshold=mask_threshold,
            points=[point],
            point_labels=[1],
        )

    if point_first:
        try:
            outputs, elapsed = run_point()
            return outputs, elapsed, "GT reference point", errors, point, mapping
        except RuntimeError as error:
            errors.append(f"GT reference point: {error}")
    try:
        outputs, elapsed, prompt, text_errors = track_with_fallbacks(
            predictor,
            frames_dir,
            prompts,
            prompt_frame,
            mask_threshold,
        )
        errors.extend(text_errors)
        return outputs, elapsed, prompt, errors, point, mapping
    except RuntimeError as error:
        errors.append(str(error))
    if not point_first:
        outputs, elapsed = run_point()
        return outputs, elapsed, "GT reference point", errors, point, mapping
    raise RuntimeError("; ".join(errors))


def track_with_fallbacks(
    predictor: Any,
    frames_dir: Path,
    prompts: list[str],
    prompt_frame: int,
    mask_threshold: float,
) -> tuple[dict[int, dict[str, Any]], float, str, list[str]]:
    errors = []
    for prompt in prompts:
        try:
            outputs, elapsed = track_concept(
                predictor=predictor,
                frames_dir=frames_dir,
                prompt=prompt,
                prompt_frame=prompt_frame,
                mask_threshold=mask_threshold,
            )
            return outputs, elapsed, prompt, errors
        except RuntimeError as error:
            errors.append(f"{prompt}: {error}")
    raise RuntimeError("; ".join(errors))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.gt_dir = args.gt_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.classification_results is not None:
        args.classification_results = args.classification_results.expanduser().resolve()
    if not args.input.is_dir() or not args.gt_dir.is_dir():
        raise SystemExit("--input and --gt-dir must be existing directories")
    if not args.checkpoint.is_file():
        raise SystemExit(f"SAM3 checkpoint does not exist: {args.checkpoint}")
    if args.target_fps <= 0:
        raise SystemExit("--target-fps must be positive")
    damage_prompts = args.damage_prompts or [
        "metal patch on pipe",
        "metal patch",
        "damaged pipe",
    ]
    jobs = discover_jobs(
        args.input,
        load_gt(args.gt_dir),
        set(args.only) if args.only else None,
    )
    if not jobs:
        raise SystemExit("No matching clips found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_damage_boxes = load_reference_damage_boxes(
        args.classification_results
    )

    pending = []
    completed: list[dict[str, Any]] = []
    for job in jobs:
        sample, video_path, label, gt_path = job
        report_path = args.output_dir / f"sample{sample:02d}" / "tracking_report.json"
        if report_path.is_file() and not args.force:
            completed.append(json.loads(report_path.read_text(encoding="utf-8")))
        else:
            pending.append(job)

    predictor = None
    model_load_sec = 0.0
    failures: list[dict[str, Any]] = []
    if pending:
        import torch
        from sam3.model_builder import build_sam3_video_predictor

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not visible; run with GPU access")
        model_started = time.perf_counter()
        predictor = build_sam3_video_predictor(
            checkpoint_path=str(args.checkpoint),
            gpus_to_use=[torch.cuda.current_device()],
        )
        model_load_sec = time.perf_counter() - model_started

    try:
        for job_index, (sample, video_path, label, gt_path) in enumerate(
            pending,
            start=1,
        ):
            sample_dir = args.output_dir / f"sample{sample:02d}"
            print(
                f"[{job_index}/{len(pending)}] sample{sample:02d} "
                f"class={label} ({CLASS_NAMES[label]})",
                flush=True,
            )
            try:
                frames_dir = prepare_output_dir(sample_dir, force=True)
                frame_paths, metadata = sample_video(
                    video_path,
                    frames_dir,
                    args.target_fps,
                )
                prompt_frame, match_score = match_prompt_frame(frame_paths, gt_path)
                (
                    cable_outputs,
                    cable_sec,
                    cable_prompt,
                    cable_errors,
                    cable_point,
                    cable_point_mapping,
                ) = track_cable(
                        predictor,
                        frames_dir,
                        [args.cable_prompt, "black pipe", "cable"],
                        prompt_frame,
                        args.mask_threshold,
                        gt_path,
                        frame_paths[prompt_frame],
                        args.cable_point_first,
                        args.cable_dark_first,
                    )
                draw_cable_anchor(
                    frame_paths[prompt_frame],
                    cable_point,
                    sample_dir / "cable_prompt_anchor.jpg",
                )
                damage_outputs: dict[int, dict[str, Any]] = {}
                damage_sec = 0.0
                damage_prompt = None
                damage_errors: list[str] = []
                damage_box = None
                damage_box_mapping: dict[str, Any] = {"reason": "not_damaged"}
                if label == 0:
                    reference_box = reference_damage_boxes.get(sample)
                    if args.damage_bright_anchor:
                        damage_box, damage_box_mapping = (
                            bright_damage_box_from_cable(
                                frame_paths[prompt_frame],
                                cable_outputs[prompt_frame],
                            )
                        )
                    elif reference_box is not None:
                        damage_box, damage_box_mapping = map_reference_box(
                            gt_path,
                            frame_paths[prompt_frame],
                            reference_box,
                        )
                    draw_prompt_box(
                        frame_paths[prompt_frame],
                        damage_box,
                        sample_dir / "damage_prompt_anchor.jpg",
                    )
                    if damage_box is not None:
                        try:
                            damage_prompt = damage_prompts[0]
                            box_x, box_y, box_width, box_height = damage_box
                            damage_points = [
                                [
                                    box_x + box_width * 0.5,
                                    box_y + box_height * 0.5,
                                ]
                            ]
                            damage_labels = [1]
                            if not args.damage_single_point:
                                damage_points.extend(
                                    [
                                        [
                                            box_x + box_width * 0.5,
                                            max(
                                                0.0,
                                                box_y - box_height * 0.25,
                                            ),
                                        ],
                                        [
                                            box_x + box_width * 0.5,
                                            min(
                                                1.0,
                                                box_y + box_height * 1.25,
                                            ),
                                        ],
                                    ]
                                )
                                damage_labels.extend([0, 0])
                            damage_outputs, damage_sec = track_concept(
                                predictor=predictor,
                                frames_dir=frames_dir,
                                prompt=damage_prompt,
                                prompt_frame=prompt_frame,
                                mask_threshold=args.mask_threshold,
                                points=damage_points,
                                point_labels=damage_labels,
                            )
                        except RuntimeError as error:
                            damage_errors.append(f"box prompt: {error}")
                    if not damage_outputs:
                        (
                            damage_outputs,
                            damage_sec,
                            damage_prompt,
                            text_errors,
                        ) = track_with_fallbacks(
                            predictor,
                            frames_dir,
                            damage_prompts,
                            prompt_frame,
                            args.mask_threshold,
                        )
                        damage_errors.extend(text_errors)

                if label == 0:
                    output_video = sample_dir / "sam3_cable_damage_tracking_10hz.mp4"
                    preview_path = sample_dir / "preview_damage.jpg"
                    frame_records = render_damage_video(
                        frame_paths=frame_paths,
                        cable_outputs=cable_outputs,
                        damage_outputs=damage_outputs,
                        output_path=output_video,
                        preview_path=preview_path,
                        fps=float(metadata["output_fps"]),
                        preview_frame=prompt_frame,
                        class_label=CLASS_NAMES_CN[label],
                    )
                    cable_count = sum(bool(row["cable"]) for row in frame_records)
                    damage_count = sum(bool(row["damage"]) for row in frame_records)
                else:
                    output_video = sample_dir / "sam3_cable_tracking_10hz.mp4"
                    preview_path = sample_dir / "preview.jpg"
                    frame_records = render_cable_video(
                        frame_paths=frame_paths,
                        outputs_by_frame=cable_outputs,
                        output_path=output_video,
                        preview_path=preview_path,
                        fps=float(metadata["output_fps"]),
                        prompt=cable_prompt,
                        alpha=0.42,
                        class_label=CLASS_NAMES_CN[label],
                    )
                    cable_count = sum(bool(row["objects"]) for row in frame_records)
                    damage_count = 0
                report = {
                    "method": "sam3_midterm_batch_tracking",
                    "sample": sample,
                    "classification": {
                        "class_id": label,
                        "class_name": CLASS_NAMES[label],
                    },
                    "input": str(video_path),
                    "gt_reference": str(gt_path),
                    "reference_match_frame": prompt_frame,
                    "reference_match_score": match_score,
                    "checkpoint": str(args.checkpoint),
                    "cable_prompt": cable_prompt,
                    "cable_point_normalized_xy": cable_point,
                    "cable_point_mapping": cable_point_mapping,
                    "damage_prompt": damage_prompt,
                    "damage_box_normalized_xywh": damage_box,
                    "damage_box_mapping": damage_box_mapping,
                    "mask_threshold": args.mask_threshold,
                    **metadata,
                    "cable_tracking_sec": cable_sec,
                    "damage_tracking_sec": damage_sec,
                    "cable_tracked_frame_count": cable_count,
                    "damage_tracked_frame_count": damage_count,
                    "cable_prompt_failures": cable_errors,
                    "damage_prompt_failures": damage_errors,
                    "output_video": str(output_video.resolve()),
                    "preview_image": str(preview_path.resolve()),
                    "frames": frame_records,
                }
                write_json(sample_dir / "tracking_report.json", report)
                if not args.keep_frames:
                    shutil.rmtree(frames_dir)
                completed.append(report)
                print(
                    f"  prompt_frame={prompt_frame} cable={cable_count}/"
                    f"{len(frame_paths)} damage={damage_count}/{len(frame_paths)}",
                    flush=True,
                )
            except Exception as error:  # continue so a long batch is resumable
                failures.append(
                    {
                        "sample": sample,
                        "video": str(video_path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
    finally:
        if predictor is not None:
            predictor.shutdown()

    completed.sort(key=lambda row: int(row["sample"]))
    summary = {
        "method": "sam3_midterm_batch_tracking",
        "target_fps": args.target_fps,
        "model_load_sec_this_run": model_load_sec,
        "requested_count": len(jobs),
        "completed_count": len(completed),
        "failed_count": len(failures),
        "damage_video_count": sum(
            int(row["classification"]["class_id"]) == 0 for row in completed
        ),
        "cable_only_video_count": sum(
            int(row["classification"]["class_id"]) != 0 for row in completed
        ),
        "videos": [
            {
                key: row[key]
                for key in (
                    "sample",
                    "classification",
                    "input",
                    "output_video",
                    "sampled_frame_count",
                    "cable_tracked_frame_count",
                    "damage_tracked_frame_count",
                )
            }
            for row in completed
        ],
        "failures": failures,
    }
    write_json(args.output_dir / "tracking_summary.json", summary)
    print(
        f"Completed {len(completed)}/{len(jobs)}; failures={len(failures)} -> "
        f"{args.output_dir / 'tracking_summary.json'}",
        flush=True,
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
