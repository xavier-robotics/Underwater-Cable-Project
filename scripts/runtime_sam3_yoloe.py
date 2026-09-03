#!/usr/bin/env python3
"""Run low-FPS SAM3 + YOLOE video inference with live-style overlays."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scripts.classify_contact import classify_box
    from scripts.classify_samples_zero_shot import (
        CLASS_MAP,
        Detection,
        YOLOETextDetector,
        aggregate_damage,
        aggregate_position,
        damage_frame_decision,
        load_config,
        max_prompt_score,
        position_frame_decision,
        visual_damage_features,
    )
    from scripts.select_sample_frames import SAM3Scorer
except ModuleNotFoundError:
    from classify_contact import classify_box
    from classify_samples_zero_shot import (
        CLASS_MAP,
        Detection,
        YOLOETextDetector,
        aggregate_damage,
        aggregate_position,
        damage_frame_decision,
        load_config,
        max_prompt_score,
        position_frame_decision,
        visual_damage_features,
    )
    from select_sample_frames import SAM3Scorer


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CLASS_LABELS_CN = {0: "破损", 1: "裸露", 2: "悬空"}
FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


@dataclass(frozen=True)
class FrameClassification:
    """One YOLOE classification update for a sampled video frame."""

    frame_idx: int
    timestamp_sec: float
    position_decision: dict[str, Any]
    damage_decision: dict[str, Any]
    damage_detections: tuple[Detection, ...]
    support_score: float
    evidence_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runtime_sam3_yoloe"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime_sam3_yoloe.yaml"),
    )
    parser.add_argument("--target-fps", type=float)
    parser.add_argument("--classify-interval-sec", type=float)
    parser.add_argument("--vote-window", type=int)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-duration-sec", type=float)
    parser.add_argument("--sam3-prompt", default="pipe")
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=Path("ckpts/sam3/sam3.pt"),
    )
    parser.add_argument("--sam3-conf", type=float, default=0.05)
    parser.add_argument(
        "--sam3-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-bright-support-exclusion",
        action="store_true",
        help="Do not remove bright neutral support pixels from the displayed cable mask.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> tuple[Any, ...]:
    """Return a stable numeric-aware path sort key."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def discover_videos(input_path: Path) -> list[Path]:
    """Discover input videos without descending into unrelated archives."""
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video input: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise ValueError(f"Input does not exist: {input_path}")
    return sorted(
        (
            path.resolve()
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=natural_key,
    )


def resolve_repo_path(path: Path) -> Path:
    """Resolve a project-relative command-line path."""
    path = path.expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_args(args: argparse.Namespace, videos: list[Path]) -> None:
    if not videos:
        raise SystemExit(f"No videos found in: {args.input}")
    if args.target_fps <= 0 or args.target_fps > 10:
        raise SystemExit("--target-fps must be in the range (0, 10]")
    if args.classify_interval_sec <= 0:
        raise SystemExit("--classify-interval-sec must be positive")
    if args.vote_window <= 0:
        raise SystemExit("--vote-window must be positive")
    if args.max_videos is not None and args.max_videos <= 0:
        raise SystemExit("--max-videos must be positive")
    if args.max_duration_sec is not None and args.max_duration_sec <= 0:
        raise SystemExit("--max-duration-sec must be positive")
    if not args.sam3_checkpoint.is_file():
        raise SystemExit(f"SAM3 checkpoint does not exist: {args.sam3_checkpoint}")


@lru_cache(maxsize=8)
def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a Chinese-capable font once per requested size."""
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw UTF-8 text on a BGR OpenCV image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    ImageDraw.Draw(pil_image).text(xy, text, font=load_font(size), fill=color)
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the bounding box of a non-empty binary mask."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def padded_crop(
    frame: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_ratio: float = 0.16,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop a cable region and keep its binary mask in crop coordinates."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = max(20, int((x2 - x1) * pad_ratio))
    pad_y = max(20, int((y2 - y1) * pad_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return frame[y1:y2, x1:x2], mask[y1:y2, x1:x2], (x1, y1, x2, y2)


def remove_bright_neutral_pixels(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exclude white supports and end labels from the displayed cable mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bright_neutral = (hsv[:, :, 1] < 72) & (hsv[:, :, 2] > 155)
    cleaned = mask.copy()
    cleaned[bright_neutral] = 0
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8),
        connectivity=8,
    )
    filtered = np.zeros_like(cleaned)
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < 240:
            continue
        component = (labels == component_index).astype(np.uint8)
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(
            max(contours, key=cv2.contourArea)
        )
        short_side = min(rect_width, rect_height)
        oriented_fill = area / max(rect_width * rect_height, 1.0)
        if short_side < 7 or oriented_fill < 0.18:
            continue
        filtered[labels == component_index] = 255
    return filtered


def bright_support_score(frame: np.ndarray) -> float:
    """Score long white PVC support bars without adding another model pass."""
    height, width = frame.shape[:2]
    frame_area = max(1, height * width)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    local_background = cv2.GaussianBlur(value, (0, 0), 25)
    local_contrast = cv2.subtract(value, local_background)
    bright_bar = (local_contrast > 20).astype(np.uint8)
    bright_bar = cv2.morphologyEx(
        bright_bar,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    bright_bar = cv2.morphologyEx(
        bright_bar,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    contours, _ = cv2.findContours(
        bright_bar,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area / frame_area < 0.002:
            continue
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        long_side = max(rect_width, rect_height)
        short_side = min(rect_width, rect_height)
        if short_side < 6 or long_side <= 0:
            continue
        aspect = long_side / short_side
        length_ratio = long_side / max(width, height)
        fill_ratio = area / max(rect_width * rect_height, 1.0)
        if aspect < 5.0 or length_ratio < 0.18 or fill_ratio < 0.30:
            continue
        score = (
            0.45 * min(1.0, aspect / 8.0)
            + 0.35 * min(1.0, length_ratio / 0.45)
            + 0.20 * min(1.0, fill_ratio / 0.70)
        )
        best_score = max(best_score, score)
    return round(float(best_score), 6)


def overlap_over_smaller_box(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Measure overlap relative to the smaller of two detection boxes."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    denominator = min(first_area, second_area)
    return intersection / denominator if denominator > 0 else 0.0


def suppress_exclusions(
    detections: list[Detection],
    positive_prompts: set[str],
    exclusion_prompts: set[str],
    overlap_threshold: float,
    score_ratio: float,
) -> list[Detection]:
    """Suppress damage boxes explained by labels, end caps, or ropes."""
    exclusions = [item for item in detections if item.prompt in exclusion_prompts]
    kept: list[Detection] = []
    for item in detections:
        if item.prompt not in positive_prompts:
            continue
        suppressed = any(
            overlap_over_smaller_box(item.box_xyxy, exclusion.box_xyxy)
            >= overlap_threshold
            and exclusion.score >= item.score * score_ratio
            for exclusion in exclusions
        )
        if not suppressed:
            kept.append(item)
    return kept


def offset_detection(
    detection: Detection,
    crop_box: tuple[int, int, int, int],
) -> Detection:
    """Restore a crop-space YOLOE detection to full-frame coordinates."""
    crop_x1, crop_y1, _, _ = crop_box
    x1, y1, x2, y2 = detection.box_xyxy
    return Detection(
        prompt=detection.prompt,
        score=detection.score,
        box_xyxy=(x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1),
        mask_overlap=detection.mask_overlap,
    )


def render_overlay(
    frame: np.ndarray,
    cable_mask: np.ndarray | None,
    damage_detections: tuple[Detection, ...],
    class_id: int | None,
    source_time_sec: float,
    target_fps: float,
    mask_alpha: float,
) -> np.ndarray:
    """Render cable mask, damage boxes, and current Chinese class status."""
    canvas = frame.copy()
    cable_box = None
    if cable_mask is not None and np.any(cable_mask > 0):
        active = cable_mask > 0
        color = np.zeros_like(canvas)
        color[active] = (255, 155, 20)
        canvas = cv2.addWeighted(canvas, 1.0, color, mask_alpha, 0.0)
        contours, _ = cv2.findContours(
            active.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, (255, 185, 40), 2)
        cable_box = mask_bbox(cable_mask)
        if cable_box is not None:
            x1, y1, x2, y2 = cable_box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 185, 40), 2)

    for detection in sorted(
        damage_detections,
        key=lambda item: item.score,
        reverse=True,
    )[:3]:
        x1, y1, x2, y2 = detection.box_xyxy
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 30, 245), 3)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 66), (18, 18, 18), -1)
    class_text = "等待识别" if class_id is None else CLASS_LABELS_CN[class_id]
    status = f"类别：{class_text}    时间：{source_time_sec:.1f}s    采样：{target_fps:.1f} FPS"
    canvas = draw_text(canvas, status, (16, 12), size=32, color=(245, 245, 245))
    if cable_box is not None:
        canvas = draw_text(
            canvas,
            "海缆",
            (cable_box[0], max(68, cable_box[1] - 38)),
            size=28,
            color=(40, 205, 255),
        )
    for detection in damage_detections[:1]:
        canvas = draw_text(
            canvas,
            "破损区域",
            (detection.box_xyxy[0], max(68, detection.box_xyxy[1] - 38)),
            size=28,
            color=(255, 80, 80),
        )
    return canvas


def current_class(
    damage_history: list[dict[str, Any]],
    position_history: list[dict[str, Any]],
    support_seen: bool = False,
) -> tuple[int | None, str | None, str | None, float, float, float]:
    """Fuse accumulated damage evidence and a rolling position vote."""
    if not damage_history or not position_history:
        return None, None, None, 0.0, 0.0, 0.0
    damage, damage_confidence = aggregate_damage(damage_history, {"min_positive_frames": 1})
    position, position_confidence, position_agreement = aggregate_position(position_history)
    if support_seen:
        position = "suspended"
        position_confidence = max(position_confidence, 0.88)
    class_id, class_name = CLASS_MAP[(position, damage)]
    return (
        class_id,
        class_name,
        position,
        damage_confidence,
        position_confidence,
        position_agreement,
    )


def classify_frame(
    frame: np.ndarray,
    cable_mask: np.ndarray,
    cable_area_ratio: float,
    frame_idx: int,
    timestamp_sec: float,
    evidence_path: Path,
    detector: YOLOETextDetector,
    config: dict[str, Any],
) -> FrameClassification:
    """Run the two YOLOE views needed for one classification update."""
    damage_config = config["damage"]
    position_config = config["position"]
    runtime_config = config.get("runtime", {})
    bbox = mask_bbox(cable_mask)
    if bbox is None:
        raise ValueError("Cannot classify an empty cable mask")
    crop, crop_mask, crop_box = padded_crop(frame, cable_mask, bbox)
    crop_detections = detector.predict(crop, crop_mask)

    positive_prompts = set(map(str, damage_config["positive_prompts"]))
    intact_prompts = set(map(str, damage_config["intact_prompts"]))
    exclusion_prompts = set(map(str, damage_config.get("exclusion_prompts", [])))
    positive_detections = suppress_exclusions(
        crop_detections,
        positive_prompts,
        exclusion_prompts,
        float(runtime_config.get("exclusion_overlap", 0.50)),
        float(runtime_config.get("exclusion_score_ratio", 0.65)),
    )
    positive_score = max(
        (item.score for item in positive_detections),
        default=0.0,
    )
    visual = visual_damage_features(crop, crop_mask, damage_config)
    damage_decision = damage_frame_decision(
        positive_score,
        max_prompt_score(crop_detections, intact_prompts),
        float(visual["score"]),
        damage_config,
    )

    full_detections = detector.predict(frame, cable_mask)
    exposed_prompts = set(map(str, position_config["exposed_prompts"]))
    suspended_prompts = set(map(str, position_config["suspended_prompts"]))
    support_prompts = set(map(str, position_config.get("support_prompts", [])))
    contact_result = classify_box(frame, bbox, position_config.get("contact", {}))
    position_decision = position_frame_decision(
        max_prompt_score(full_detections, exposed_prompts),
        max_prompt_score(full_detections, suspended_prompts),
        cable_area_ratio,
        contact_result,
        position_config,
    )
    support_score = max_prompt_score(full_detections, support_prompts)
    geometric_support_score = bright_support_score(frame)
    support_detected = (
        support_score >= float(position_config.get("support_score_threshold", math.inf))
        or geometric_support_score
        >= float(runtime_config.get("bright_support_score_threshold", math.inf))
    )
    if support_detected:
        position_decision.update(
            {
                "position": "suspended",
                "confidence": max(float(position_decision["confidence"]), 0.88),
                "suspended_probability": max(
                    float(position_decision["suspended_probability"]),
                    0.88,
                ),
                "support_score": round(support_score, 6),
                "bright_support_score": geometric_support_score,
            }
        )

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(evidence_path), frame)
    restored = tuple(offset_detection(item, crop_box) for item in positive_detections)
    return FrameClassification(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        position_decision=position_decision,
        damage_decision=damage_decision,
        damage_detections=restored,
        support_score=max(support_score, geometric_support_score),
        evidence_path=evidence_path,
    )


def process_video(
    video_path: Path,
    output_dir: Path,
    sam3: SAM3Scorer,
    yoloe: YOLOETextDetector,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process one video at a bounded sampled frame rate."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_stride = max(1, int(round(source_fps / args.target_fps)))
    output_fps = source_fps / sample_stride
    output_path = output_dir / "videos" / f"{video_path.stem}_sam3_yoloe_{output_fps:.2f}fps.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    runtime_config = config.get("runtime", {})
    mask_alpha = float(runtime_config.get("mask_alpha", 0.32))
    remove_support = bool(runtime_config.get("remove_bright_support_pixels", True))
    if args.no_bright_support_exclusion:
        remove_support = False

    damage_history: list[dict[str, Any]] = []
    position_history: deque[dict[str, Any]] = deque(maxlen=args.vote_window)
    classifications: list[FrameClassification] = []
    support_seen = False
    sampled_frames = 0
    decoded_frames = 0
    sam3_sec = 0.0
    yoloe_sec = 0.0
    render_sec = 0.0
    next_classification_sec = 0.0
    started = time.perf_counter()
    evidence_dir = output_dir / "evidence" / video_path.stem

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx = decoded_frames
            decoded_frames += 1
            timestamp_sec = frame_idx / source_fps
            if (
                args.max_duration_sec is not None
                and timestamp_sec >= args.max_duration_sec
            ):
                break
            if frame_idx % sample_stride != 0:
                continue
            frame_damage_detections: tuple[Detection, ...] = ()

            sam3_started = time.perf_counter()
            crop = sam3.extract_crop(frame)
            sam3_sec += time.perf_counter() - sam3_started
            raw_cable_mask = None
            display_cable_mask = None
            cable_area_ratio = 0.0
            if crop is not None:
                if crop.mask is None:
                    raw_cable_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    x1, y1, x2, y2 = crop.bbox_xyxy
                    raw_cable_mask[y1:y2, x1:x2] = 255
                else:
                    raw_cable_mask = crop.mask.copy()
                cable_area_ratio = float(
                    np.count_nonzero(raw_cable_mask) / raw_cable_mask.size
                )
                display_cable_mask = raw_cable_mask
                if remove_support:
                    display_cable_mask = remove_bright_neutral_pixels(
                        frame,
                        raw_cable_mask,
                    )

            if (
                raw_cable_mask is not None
                and np.any(raw_cable_mask > 0)
                and timestamp_sec + 1e-6 >= next_classification_sec
            ):
                yoloe_started = time.perf_counter()
                classification = classify_frame(
                    frame=frame,
                    cable_mask=raw_cable_mask,
                    cable_area_ratio=cable_area_ratio,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    evidence_path=evidence_dir / f"f{frame_idx:06d}.jpg",
                    detector=yoloe,
                    config=config,
                )
                yoloe_sec += time.perf_counter() - yoloe_started
                classifications.append(classification)
                damage_history.append(classification.damage_decision)
                position_history.append(classification.position_decision)
                support_seen = support_seen or (
                    "support_score" in classification.position_decision
                    or "bright_support_score" in classification.position_decision
                )
                frame_damage_detections = classification.damage_detections
                next_classification_sec = timestamp_sec + args.classify_interval_sec

            fused = current_class(
                damage_history,
                list(position_history),
                support_seen=support_seen,
            )
            class_id = fused[0]
            support_visible_now = (
                bright_support_score(frame)
                >= float(
                    runtime_config.get(
                        "bright_support_score_threshold",
                        math.inf,
                    )
                )
            )
            render_mask = (
                None
                if class_id == 2 and support_visible_now
                else display_cable_mask
            )
            render_started = time.perf_counter()
            canvas = render_overlay(
                frame=frame,
                cable_mask=render_mask,
                damage_detections=frame_damage_detections,
                class_id=class_id,
                source_time_sec=timestamp_sec,
                target_fps=output_fps,
                mask_alpha=mask_alpha,
            )
            writer.write(canvas)
            render_sec += time.perf_counter() - render_started
            sampled_frames += 1
    finally:
        capture.release()
        writer.release()

    elapsed_sec = time.perf_counter() - started
    source_duration_sec = (
        min(source_frames / source_fps, args.max_duration_sec)
        if args.max_duration_sec is not None and source_frames > 0
        else source_frames / source_fps if source_frames > 0 else decoded_frames / source_fps
    )
    fused = current_class(
        damage_history,
        list(position_history),
        support_seen=support_seen,
    )
    class_id, class_name, position = fused[:3]
    damage_confidence, position_confidence, position_agreement = fused[3:]
    if class_id is None:
        class_id, class_name, position = 1, "exposed_intact", "exposed"
        damage_confidence = 0.5
        position_confidence = 0.5
    damage = "damaged" if class_id == 0 else "intact"
    frame_votes = [
        {
            "frame_idx": item.frame_idx,
            "position": item.position_decision["position"],
            "position_confidence": item.position_decision["confidence"],
            "damage_evidence": (
                "clear_damage" if item.damage_decision["positive"] else "no_visible_damage"
            ),
            "damage_confidence": round(
                0.5 + 0.5 * abs(float(item.damage_decision["combined_score"]) - 0.5) * 2,
                4,
            ),
            "usable": True,
        }
        for item in classifications
    ]
    low_confidence = float(config["quality"].get("low_confidence", 0.60))
    reason_codes = []
    if not classifications:
        reason_codes.append("no_usable_cable_frame")
    if damage_confidence < low_confidence:
        reason_codes.append("low_damage_confidence")
    if position_confidence < low_confidence:
        reason_codes.append("low_position_confidence")

    result = {
        "video": str(video_path),
        "sample_id": "S001",
        "attempt_id": 1,
        "start_sec": 0.0,
        "end_sec": round(source_duration_sec, 3),
        "image_paths": [str(item.evidence_path) for item in classifications],
        "position": position,
        "position_confidence": round(float(position_confidence), 4),
        "damage": damage,
        "damage_confidence": round(float(damage_confidence), 4),
        "class_id": class_id,
        "class_name": class_name,
        "needs_review": bool(reason_codes),
        "evidence_consistency": (
            "consistent" if position_agreement >= 2 / 3 else "mixed"
        ),
        "reason_codes": reason_codes,
        "frame_votes": frame_votes,
        "method": "sam3_yoloe_realtime_low_fps",
    }
    timing = {
        "video": str(video_path),
        "output_video": str(output_path),
        "source_fps": source_fps,
        "source_frames": source_frames,
        "source_duration_sec": round(source_duration_sec, 4),
        "target_fps_requested": args.target_fps,
        "output_fps": round(output_fps, 6),
        "sampled_frames": sampled_frames,
        "classification_updates": len(classifications),
        "elapsed_sec": round(elapsed_sec, 4),
        "sam3_inference_sec": round(sam3_sec, 4),
        "yoloe_inference_sec": round(yoloe_sec, 4),
        "render_and_write_sec": round(render_sec, 4),
        "processed_sample_fps": round(sampled_frames / max(elapsed_sec, 1e-9), 4),
        "realtime_factor": round(elapsed_sec / max(source_duration_sec, 1e-9), 4),
        "meets_source_realtime": elapsed_sec <= source_duration_sec,
        "meets_sampled_fps": sampled_frames / max(elapsed_sec, 1e-9) >= output_fps,
    }
    return result, timing


def write_json(path: Path, data: Any) -> None:
    """Write stable UTF-8 JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.config = resolve_repo_path(args.config)
    args.sam3_checkpoint = resolve_repo_path(args.sam3_checkpoint)
    args.output_dir = resolve_repo_path(args.output_dir)
    config = load_config(args.config)
    runtime_config = config.get("runtime", {})
    args.target_fps = (
        float(args.target_fps)
        if args.target_fps is not None
        else float(runtime_config.get("target_fps", 1.0))
    )
    args.classify_interval_sec = (
        float(args.classify_interval_sec)
        if args.classify_interval_sec is not None
        else float(runtime_config.get("classify_interval_sec", 5.0))
    )
    args.vote_window = (
        int(args.vote_window)
        if args.vote_window is not None
        else int(runtime_config.get("vote_window", 3))
    )
    videos = discover_videos(args.input.expanduser().resolve())
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    validate_args(args, videos)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_started = time.perf_counter()
    sam3 = SAM3Scorer(
        prompt=args.sam3_prompt,
        device=args.device,
        confidence_threshold=args.sam3_conf,
        min_area_ratio=0.005,
        min_span_ratio=0.15,
        checkpoint_path=args.sam3_checkpoint,
        dtype=args.sam3_dtype,
    )
    yoloe = YOLOETextDetector(config["model"])
    prompts = (
        list(map(str, config["damage"]["positive_prompts"]))
        + list(map(str, config["damage"]["intact_prompts"]))
        + list(map(str, config["damage"].get("exclusion_prompts", [])))
        + list(map(str, config["position"]["exposed_prompts"]))
        + list(map(str, config["position"]["suspended_prompts"]))
        + list(map(str, config["position"].get("support_prompts", [])))
    )
    yoloe.set_prompts(prompts)
    model_load_sec = time.perf_counter() - model_started

    results = []
    timings = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.name}", flush=True)
        result, timing = process_video(
            video_path,
            args.output_dir,
            sam3,
            yoloe,
            config,
            args,
        )
        results.append(result)
        timings.append(timing)
        print(
            f"  class={result['class_id']} elapsed={timing['elapsed_sec']:.2f}s "
            f"sample_fps={timing['processed_sample_fps']:.2f} "
            f"realtime={timing['meets_source_realtime']}",
            flush=True,
        )

    write_json(args.output_dir / "results.json", results)
    total_source_sec = sum(float(item["source_duration_sec"]) for item in timings)
    total_elapsed_sec = sum(float(item["elapsed_sec"]) for item in timings)
    report = {
        "method": "sam3_yoloe_realtime_low_fps",
        "model_load_sec": round(model_load_sec, 4),
        "video_count": len(videos),
        "target_fps": args.target_fps,
        "classify_interval_sec": args.classify_interval_sec,
        "total_source_sec": round(total_source_sec, 4),
        "total_processing_sec_excluding_model_load": round(total_elapsed_sec, 4),
        "total_processing_sec_including_model_load": round(
            total_elapsed_sec + model_load_sec,
            4,
        ),
        "steady_state_realtime_factor": round(
            total_elapsed_sec / max(total_source_sec, 1e-9),
            4,
        ),
        "steady_state_meets_realtime": total_elapsed_sec <= total_source_sec,
        "videos": timings,
    }
    write_json(args.output_dir / "runtime_report.json", report)
    print(f"Results: {args.output_dir / 'results.json'}")
    print(f"Runtime report: {args.output_dir / 'runtime_report.json'}")


if __name__ == "__main__":
    main()
