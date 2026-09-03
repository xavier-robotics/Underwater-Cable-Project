#!/usr/bin/env python
"""Classify SAM3-selected cable samples with YOLOE prompts and fixed rules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from scripts.classify_contact import classify_box
    from scripts.classify_samples_gpt import discover_manifests, load_json, resolve_image_path, select_evenly
except ModuleNotFoundError:
    from classify_contact import classify_box
    from classify_samples_gpt import discover_manifests, load_json, resolve_image_path, select_evenly


REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_MAP = {
    ("exposed", "intact"): (1, "exposed_intact"),
    ("exposed", "damaged"): (0, "damaged"),
    ("suspended", "intact"): (2, "suspended_intact"),
    ("suspended", "damaged"): (0, "damaged"),
}


@dataclass(frozen=True)
class FrameInput:
    frame_idx: int
    full_path: Path
    crop_path: Path
    mask_path: Path | None
    crop_bbox_xyxy: tuple[int, int, int, int] | None
    sam3_score: float
    sam3_area_ratio: float


@dataclass(frozen=True)
class SampleInput:
    video: str
    sample_id: str
    start_sec: float | None
    end_sec: float | None
    frames: list[FrameInput]


@dataclass(frozen=True)
class Detection:
    prompt: str
    score: float
    box_xyxy: tuple[int, int, int, int]
    mask_overlap: float


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Zero-shot config must be a mapping: {path}")
    for section in ("model", "sampling", "damage", "position", "quality"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"Missing mapping '{section}' in {path}")
    return data


def resolve_config_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def frame_path(
    frame: dict[str, Any],
    key: str,
    manifest_path: Path,
) -> Path | None:
    raw_path = frame.get(key)
    if not raw_path:
        return None
    path = resolve_image_path(str(raw_path), manifest_path)
    return path if path.is_file() else None


def collect_samples(
    input_path: Path,
    *,
    max_frames_per_sample: int,
    limit: int | None = None,
) -> list[SampleInput]:
    samples: list[SampleInput] = []
    for manifest_path in discover_manifests(input_path):
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        video = str(manifest.get("video", ""))
        for segment in manifest.get("segments", []):
            if not isinstance(segment, dict):
                continue
            selected = select_evenly(
                list(segment.get("selected_frames", [])),
                max_frames_per_sample,
            )
            frames: list[FrameInput] = []
            for frame in selected:
                if not isinstance(frame, dict):
                    continue
                full_path = frame_path(frame, "full_frame_path", manifest_path)
                crop_path = (
                    frame_path(frame, "crop_path", manifest_path)
                    or frame_path(frame, "masked_crop_path", manifest_path)
                    or full_path
                )
                if full_path is None or crop_path is None:
                    continue
                bbox = frame.get("crop_bbox_xyxy")
                crop_bbox = None
                if isinstance(bbox, list) and len(bbox) == 4:
                    crop_bbox = tuple(int(value) for value in bbox)
                frames.append(
                    FrameInput(
                        frame_idx=int(frame.get("frame_idx", -1)),
                        full_path=full_path,
                        crop_path=crop_path,
                        mask_path=frame_path(frame, "mask_path", manifest_path),
                        crop_bbox_xyxy=crop_bbox,
                        sam3_score=float(frame.get("crop_confidence", frame.get("cable_score", 0.0)) or 0.0),
                        sam3_area_ratio=float(frame.get("crop_area_ratio", 0.0) or 0.0),
                    )
                )
            if frames:
                samples.append(
                    SampleInput(
                        video=video,
                        sample_id=str(segment.get("sample_id", f"S{len(samples) + 1:03d}")),
                        start_sec=segment.get("start_sec"),
                        end_sec=segment.get("end_sec"),
                        frames=frames,
                    )
                )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def bootstrap_yoloe_imports() -> None:
    paths = [
        REPO_ROOT / "yoloe",
        REPO_ROOT / "yoloe" / "third_party" / "CLIP",
        REPO_ROOT / "yoloe" / "third_party" / "ml-mobileclip",
    ]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/tmp/ultralytics")
    os.environ.setdefault("YOLOE_CKPT_DIR", str(REPO_ROOT / "ckpts" / "yoloe"))


class YOLOETextDetector:
    """Small adapter around the vendored YOLOE open-vocabulary model."""

    def __init__(self, config: dict[str, Any]) -> None:
        bootstrap_yoloe_imports()
        import torch
        from ultralytics import YOLOE

        device = str(config.get("device", "auto"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is not visible. Run zero-shot classification in the normal GPU shell."
            )
        self.device = device
        self.imgsz = int(config.get("imgsz", 1280))
        self.confidence = float(config.get("confidence", 0.001))
        self.iou = float(config.get("iou", 0.5))
        self.max_detections = int(config.get("max_detections", 100))
        self.min_mask_overlap = float(config.get("min_mask_overlap", 0.02))
        weights = resolve_config_path(str(config["weights"]))
        if not weights.is_file():
            raise SystemExit(f"YOLOE weights do not exist: {weights}")
        self.model = YOLOE(str(weights))
        self.model.to(device)
        self.prompts: list[str] = []
        self._has_predicted = False

    def set_prompts(self, prompts: list[str]) -> None:
        new_prompts = list(
            dict.fromkeys(str(prompt).strip() for prompt in prompts if str(prompt).strip())
        )
        if not new_prompts:
            raise ValueError("At least one YOLOE text prompt is required")
        if self._has_predicted and new_prompts != self.prompts:
            raise RuntimeError(
                "YOLOE fuses its first text vocabulary during inference; "
                "set the complete prompt vocabulary before predict()."
            )
        self.prompts = new_prompts
        self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))
        # Ultralytics caches an AutoBackend with the previous class count after
        # the first predict call. Rebuild it whenever the text vocabulary changes.
        self.model.predictor = None

    def predict(self, image: np.ndarray, target_mask: np.ndarray | None) -> list[Detection]:
        self._has_predicted = True
        result = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_detections,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        height, width = image.shape[:2]
        detections = []
        for raw_box, raw_score, raw_class in zip(
            boxes.xyxy.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
            boxes.cls.detach().cpu().numpy(),
        ):
            class_id = int(raw_class)
            if class_id < 0 or class_id >= len(self.prompts):
                continue
            x1, y1, x2, y2 = (
                max(0, int(math.floor(raw_box[0]))),
                max(0, int(math.floor(raw_box[1]))),
                min(width, int(math.ceil(raw_box[2]))),
                min(height, int(math.ceil(raw_box[3]))),
            )
            if x2 <= x1 or y2 <= y1:
                continue
            overlap = 1.0
            if target_mask is not None:
                roi = target_mask[y1:y2, x1:x2]
                overlap = float(np.count_nonzero(roi) / max(1, roi.size))
                if overlap < self.min_mask_overlap:
                    continue
            detections.append(
                Detection(
                    prompt=self.prompts[class_id],
                    score=float(raw_score),
                    box_xyxy=(x1, y1, x2, y2),
                    mask_overlap=overlap,
                )
            )
        return detections


def restore_full_mask(
    mask: np.ndarray,
    full_shape: tuple[int, ...],
    crop_bbox_xyxy: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Restore a crop-sized saved mask to its coordinates in the full frame."""
    full_height, full_width = full_shape[:2]
    if mask.shape == (full_height, full_width):
        return (mask > 0).astype(np.uint8) * 255
    if crop_bbox_xyxy is not None:
        x1, y1, x2, y2 = crop_bbox_xyxy
        x1 = max(0, min(full_width, x1))
        x2 = max(0, min(full_width, x2))
        y1 = max(0, min(full_height, y1))
        y2 = max(0, min(full_height, y2))
        if x2 > x1 and y2 > y1:
            restored = np.zeros((full_height, full_width), dtype=np.uint8)
            resized = cv2.resize(
                mask,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST,
            )
            restored[y1:y2, x1:x2] = resized
            return (restored > 0).astype(np.uint8) * 255
    resized = cv2.resize(
        mask,
        (full_width, full_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8) * 255


def read_frame_images(frame: FrameInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = cv2.imread(str(frame.full_path), cv2.IMREAD_COLOR)
    crop = cv2.imread(str(frame.crop_path), cv2.IMREAD_COLOR)
    if full is None:
        raise RuntimeError(f"Cannot read full frame: {frame.full_path}")
    if crop is None:
        raise RuntimeError(f"Cannot read cable crop: {frame.crop_path}")

    mask = None
    if frame.mask_path is not None:
        mask = cv2.imread(str(frame.mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.full(full.shape[:2], 255, dtype=np.uint8)
    else:
        mask = restore_full_mask(
            mask,
            full.shape,
            frame.crop_bbox_xyxy,
        )
    return full, crop, mask


def crop_mask_for_frame(frame: FrameInput, full_mask: np.ndarray, crop_shape: tuple[int, ...]) -> np.ndarray:
    if frame.crop_bbox_xyxy is None:
        crop_mask = full_mask
    else:
        x1, y1, x2, y2 = frame.crop_bbox_xyxy
        x1 = max(0, min(full_mask.shape[1], x1))
        x2 = max(0, min(full_mask.shape[1], x2))
        y1 = max(0, min(full_mask.shape[0], y1))
        y2 = max(0, min(full_mask.shape[0], y2))
        crop_mask = full_mask[y1:y2, x1:x2]
    if crop_mask.size == 0:
        crop_mask = np.full(crop_shape[:2], 255, dtype=np.uint8)
    if crop_mask.shape != crop_shape[:2]:
        crop_mask = cv2.resize(
            crop_mask,
            (crop_shape[1], crop_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (crop_mask > 0).astype(np.uint8) * 255


def cable_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def max_prompt_score(detections: list[Detection], prompts: set[str]) -> float:
    return max(
        (detection.score for detection in detections if detection.prompt in prompts),
        default=0.0,
    )


def top_detection_rows(detections: list[Detection], limit: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "prompt": detection.prompt,
            "score": round(detection.score, 6),
            "box_xyxy": list(detection.box_xyxy),
            "mask_overlap": round(detection.mask_overlap, 4),
        }
        for detection in sorted(detections, key=lambda item: item.score, reverse=True)[:limit]
    ]


def visual_damage_features(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    work_mask = (mask > 0).astype(np.uint8)
    if np.count_nonzero(work_mask) < 64:
        return {"score": 0.0, "component_ratio": 0.0, "total_ratio": 0.0}

    kernel = np.ones((5, 5), dtype=np.uint8)
    inner_mask = cv2.erode(work_mask, kernel, iterations=1)
    if np.count_nonzero(inner_mask) < 64:
        inner_mask = work_mask

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[inner_mask > 0]
    median = np.median(pixels, axis=0)
    luminance = lab[:, :, 0]
    lum_values = pixels[:, 0]
    lum_mad = float(np.median(np.abs(lum_values - median[0]))) + 1.0
    chroma_delta = np.linalg.norm(lab[:, :, 1:3] - median[1:3], axis=2)
    chroma_values = chroma_delta[inner_mask > 0]
    chroma_median = float(np.median(chroma_values))
    chroma_mad = float(np.median(np.abs(chroma_values - chroma_median))) + 1.0

    bright = luminance > median[0] + max(18.0, 3.0 * lum_mad)
    colorful = chroma_delta > chroma_median + max(16.0, 3.0 * chroma_mad)
    anomaly = ((bright & colorful) & (inner_mask > 0)).astype(np.uint8)
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cable_pixels = max(1, int(np.count_nonzero(inner_mask)))
    total_ratio = float(np.count_nonzero(anomaly) / cable_pixels)
    component_ratio = 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(anomaly, connectivity=8)
    for index in range(1, count):
        component_ratio = max(component_ratio, float(stats[index, cv2.CC_STAT_AREA] / cable_pixels))

    component_scale = float(config.get("visual_component_scale", 0.025))
    total_scale = float(config.get("visual_total_scale", 0.08))
    score = float(
        np.clip(
            0.65 * component_ratio / max(component_scale, 1e-6)
            + 0.35 * total_ratio / max(total_scale, 1e-6),
            0.0,
            1.0,
        )
    )
    return {
        "score": round(score, 6),
        "component_ratio": round(component_ratio, 6),
        "total_ratio": round(total_ratio, 6),
    }


def damage_frame_decision(
    positive_score: float,
    intact_score: float,
    visual_score: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    strong = float(config.get("strong_prompt_score", 0.03))
    weak = float(config.get("weak_prompt_score", 0.012))
    margin_threshold = float(config.get("prompt_margin", 0.004))
    prompt_weight = float(config.get("prompt_weight", 0.82))
    visual_weight = float(config.get("visual_weight", 0.18))
    margin = positive_score - intact_score
    prompt_strength = float(np.clip((positive_score - 0.004) / max(strong - 0.004, 1e-6), 0.0, 1.0))
    relative = positive_score / max(positive_score + intact_score, 1e-6)
    prompt_component = prompt_strength * (0.55 + 0.45 * relative)
    combined = float(
        np.clip(
            prompt_weight * prompt_component + visual_weight * visual_score,
            0.0,
            1.0,
        )
    )
    threshold = float(config.get("frame_score_threshold", 0.48))
    is_strong = positive_score >= strong
    is_positive = is_strong or (
        positive_score >= weak
        and margin >= margin_threshold
        and combined >= threshold
    )
    return {
        "positive_score": round(positive_score, 6),
        "intact_score": round(intact_score, 6),
        "margin": round(margin, 6),
        "prompt_component": round(prompt_component, 6),
        "combined_score": round(combined, 6),
        "strong": is_strong,
        "positive": is_positive,
    }


def aggregate_damage(frame_decisions: list[dict[str, Any]], config: dict[str, Any]) -> tuple[str, float]:
    strong_count = sum(bool(item["strong"]) for item in frame_decisions)
    positive_count = sum(bool(item["positive"]) for item in frame_decisions)
    min_positive = int(config.get("min_positive_frames", 2))
    damaged = strong_count > 0 or positive_count >= min_positive
    scores = [float(item["combined_score"]) for item in frame_decisions]
    vote_ratio = positive_count / max(1, len(frame_decisions))
    evidence = max(scores, default=0.0)
    if damaged:
        confidence = 0.55 + 0.45 * max(evidence, vote_ratio)
        return "damaged", round(min(confidence, 0.99), 4)
    intact_strength = max(1.0 - evidence, 1.0 - vote_ratio)
    confidence = 0.55 + 0.40 * intact_strength
    if any(
        float(item["positive_score"]) >= float(config.get("weak_prompt_score", 0.012))
        for item in frame_decisions
    ):
        confidence = min(confidence, 0.72)
    return "intact", round(min(confidence, 0.95), 4)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))


def position_frame_decision(
    exposed_score: float,
    suspended_score: float,
    sam3_area_ratio: float,
    contact_result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    cues: list[tuple[float, float, str]] = []
    prompt_source = str(config.get("prompt_source", "yoloe_prompt"))
    prompt_total = exposed_score + suspended_score
    if prompt_total >= float(config.get("minimum_prompt_total", 0.003)):
        cues.append(
            (
                suspended_score / max(prompt_total, 1e-6),
                float(config.get("prompt_weight", 0.68)),
                prompt_source,
            )
        )

    midpoint = float(config.get("suspended_area_midpoint", 0.28))
    scale = float(config.get("suspended_area_scale", 0.06))
    area_probability = sigmoid((sam3_area_ratio - midpoint) / max(scale, 1e-6))
    cues.append(
        (
            area_probability,
            float(config.get("area_prior_weight", 0.20)),
            "sam3_area_prior",
        )
    )

    contact_state = contact_result.get("state")
    if contact_state in {"exposed", "suspended"}:
        contact_probability = 0.12 if contact_state == "exposed" else 0.88
        cues.append(
            (
                contact_probability,
                float(config.get("contact_rule_weight", 0.12)),
                "bottom_gap_rule",
            )
        )

    total_weight = sum(weight for _, weight, _ in cues)
    suspended_probability = (
        sum(probability * weight for probability, weight, _ in cues) / total_weight
        if total_weight > 0
        else 0.5
    )
    position = "suspended" if suspended_probability >= 0.5 else "exposed"
    confidence = 0.5 + abs(suspended_probability - 0.5)
    return {
        "position": position,
        "confidence": round(min(confidence, 0.99), 4),
        "suspended_probability": round(suspended_probability, 6),
        "exposed_prompt_score": round(exposed_score, 6),
        "suspended_prompt_score": round(suspended_score, 6),
        "sam3_area_ratio": round(sam3_area_ratio, 6),
        "contact_rule": contact_result,
        "active_cues": [name for _, _, name in cues],
    }


def aggregate_position(frame_decisions: list[dict[str, Any]]) -> tuple[str, float, float]:
    if not frame_decisions:
        return "exposed", 0.5, 0.0
    probabilities = []
    for item in frame_decisions:
        if "suspended_probability" in item:
            probabilities.append(float(item["suspended_probability"]))
        else:
            confidence = float(item.get("confidence", 0.5))
            probabilities.append(
                0.5 + 0.5 * confidence
                if item.get("position") == "suspended"
                else 0.5 - 0.5 * confidence
            )
    mean_probability = sum(probabilities) / len(probabilities)
    position = "suspended" if mean_probability >= 0.5 else "exposed"
    agreeing = sum(
        (probability >= 0.5) == (position == "suspended")
        for probability in probabilities
    )
    agreement = agreeing / len(probabilities)
    confidence = 0.5 + abs(mean_probability - 0.5)
    return position, round(min(confidence, 0.99), 4), round(agreement, 4)


def annotate_frame(
    image: np.ndarray,
    cable_mask_image: np.ndarray,
    detections: list[Detection],
    label: str,
) -> np.ndarray:
    canvas = image.copy()
    contours, _ = cv2.findContours(
        (cable_mask_image > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, (0, 220, 255), 2)
    for detection in sorted(detections, key=lambda item: item.score, reverse=True)[:6]:
        x1, y1, x2, y2 = detection.box_xyxy
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(
            canvas,
            f"{detection.prompt}:{detection.score:.3f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 220, 80),
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
    detector: YOLOETextDetector,
    config: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    damage_config = config["damage"]
    position_config = config["position"]
    positive_prompt_list = list(map(str, damage_config["positive_prompts"]))
    intact_prompt_list = list(map(str, damage_config["intact_prompts"]))
    exposed_prompt_list = list(map(str, position_config["exposed_prompts"]))
    suspended_prompt_list = list(map(str, position_config["suspended_prompts"]))
    positive_prompts = set(positive_prompt_list)
    intact_prompts = set(intact_prompt_list)
    exposed_prompts = set(exposed_prompt_list)
    suspended_prompts = set(suspended_prompt_list)

    frame_data: dict[tuple[str, str, int], dict[str, Any]] = {}
    detector.set_prompts(
        positive_prompt_list
        + intact_prompt_list
        + exposed_prompt_list
        + suspended_prompt_list
    )
    for sample in samples:
        for frame in sample.frames:
            full, crop, full_mask = read_frame_images(frame)
            crop_mask = crop_mask_for_frame(frame, full_mask, crop.shape)
            detections = detector.predict(crop, crop_mask)
            visual = visual_damage_features(crop, crop_mask, damage_config)
            decision = damage_frame_decision(
                max_prompt_score(detections, positive_prompts),
                max_prompt_score(detections, intact_prompts),
                float(visual["score"]),
                damage_config,
            )
            frame_data[(sample.video, sample.sample_id, frame.frame_idx)] = {
                "full": full,
                "crop": crop,
                "full_mask": full_mask,
                "crop_mask": crop_mask,
                "damage_detections": detections,
                "damage_visual": visual,
                "damage_decision": decision,
            }

    for sample in samples:
        for frame in sample.frames:
            key = (sample.video, sample.sample_id, frame.frame_idx)
            item = frame_data[key]
            detections = detector.predict(item["full"], item["full_mask"])
            bbox = cable_bbox(item["full_mask"])
            contact_result = (
                classify_box(item["full"], bbox, position_config.get("contact", {}))
                if bbox is not None
                else {"state": "unknown", "distance_px": None, "contact_ratio": 0.0}
            )
            item["position_detections"] = detections
            item["position_decision"] = position_frame_decision(
                max_prompt_score(detections, exposed_prompts),
                max_prompt_score(detections, suspended_prompts),
                frame.sam3_area_ratio,
                contact_result,
                position_config,
            )

    results = []
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    low_confidence = float(config["quality"].get("low_confidence", 0.60))
    for sample in samples:
        damage_frames = [
            frame_data[(sample.video, sample.sample_id, frame.frame_idx)]["damage_decision"]
            for frame in sample.frames
        ]
        position_frames = [
            frame_data[(sample.video, sample.sample_id, frame.frame_idx)]["position_decision"]
            for frame in sample.frames
        ]
        damage, damage_confidence = aggregate_damage(damage_frames, damage_config)
        position, position_confidence, position_agreement = aggregate_position(position_frames)
        class_id, class_name = CLASS_MAP[(position, damage)]
        reason_codes = []
        if damage_confidence < low_confidence:
            reason_codes.append("low_damage_confidence")
        if position_confidence < low_confidence:
            reason_codes.append("low_position_confidence")
        if position_agreement < 2 / 3:
            reason_codes.append("mixed_position_votes")
        if damage == "intact" and any(
            float(item["positive_score"])
            >= float(damage_config.get("weak_prompt_score", 0.012))
            for item in damage_frames
        ):
            reason_codes.append("weak_damage_evidence")

        frame_votes = []
        frame_diagnostics = []
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
                        "clear_damage" if damage_decision["positive"] else "no_visible_damage"
                    ),
                    "damage_confidence": round(
                        0.5 + 0.5 * abs(float(damage_decision["combined_score"]) - 0.5) * 2,
                        4,
                    ),
                    "usable": True,
                }
            )
            diagnostic = {
                "frame_idx": frame.frame_idx,
                "full_path": str(frame.full_path),
                "crop_path": str(frame.crop_path),
                "mask_path": str(frame.mask_path) if frame.mask_path else None,
                "sam3_score": frame.sam3_score,
                "sam3_area_ratio": frame.sam3_area_ratio,
                "damage": damage_decision,
                "visual_damage": item["damage_visual"],
                "damage_detections": top_detection_rows(item["damage_detections"]),
                "position": position_decision,
                "position_detections": top_detection_rows(item["position_detections"]),
            }
            frame_diagnostics.append(diagnostic)

            safe_video = hashlib.sha1(sample.video.encode("utf-8")).hexdigest()[:10]
            damage_canvas = annotate_frame(
                item["crop"],
                item["crop_mask"],
                item["damage_detections"],
                f"{sample.sample_id} f={frame.frame_idx} damage={damage_decision['combined_score']:.3f}",
            )
            position_canvas = annotate_frame(
                item["full"],
                item["full_mask"],
                item["position_detections"],
                f"{sample.sample_id} f={frame.frame_idx} position={position_decision['position']}",
            )
            cv2.imwrite(
                str(diagnostics_dir / f"{safe_video}_{sample.sample_id}_f{frame.frame_idx:06d}_damage.jpg"),
                damage_canvas,
            )
            cv2.imwrite(
                str(diagnostics_dir / f"{safe_video}_{sample.sample_id}_f{frame.frame_idx:06d}_position.jpg"),
                position_canvas,
            )

        consistency = (
            "consistent"
            if position_agreement == 1.0
            else "mixed"
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
                "evidence_consistency": consistency,
                "reason_codes": reason_codes,
                "frame_votes": frame_votes,
                "image_paths": [str(frame.full_path) for frame in sample.frames],
                "method": "sam3_yoloe_zero_shot",
                "diagnostics": {
                    "position_agreement": position_agreement,
                    "positive_damage_frames": sum(item["positive"] for item in damage_frames),
                    "strong_damage_frames": sum(item["strong"] for item in damage_frames),
                    "frames": frame_diagnostics,
                },
            }
        )
    return results


def write_csv_results(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "video",
        "sample_id",
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
        "method",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field) for field in fields}
            for field in ("reason_codes", "frame_votes", "image_paths"):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)


def write_review(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "| video | sample | position | damage | class | pos_conf | dmg_conf | warnings |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| {video} | {sample} | {position} | {damage} | {class_id} | "
            "{position_confidence:.3f} | {damage_confidence:.3f} | {warnings} |".format(
                video=Path(result["video"]).name,
                sample=result["sample_id"],
                position=result["position"],
                damage=result["damage"],
                class_id=result["class_id"],
                position_confidence=result["position_confidence"],
                damage_confidence=result["damage_confidence"],
                warnings=", ".join(result["reason_codes"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_signature(
    samples: list[SampleInput],
    config_path: Path,
    weights_path: Path,
) -> str:
    inputs = []
    for sample in samples:
        for frame in sample.frames:
            for path in (frame.full_path, frame.crop_path, frame.mask_path):
                if path is not None and path.is_file():
                    inputs.append(file_signature(path))
    payload = {
        "classifier": file_signature(Path(__file__)),
        "config": file_signature(config_path),
        "weights": file_signature(weights_path),
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
        default=REPO_ROOT / "configs" / "zero_shot_classifier.yaml",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    max_frames = int(config["sampling"].get("max_frames_per_sample", 3))
    samples = collect_samples(
        args.input,
        max_frames_per_sample=max_frames,
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
            "full_frame_paths": [str(frame.full_path) for frame in sample.frames],
            "crop_paths": [str(frame.crop_path) for frame in sample.frames],
            "mask_paths": [str(frame.mask_path) if frame.mask_path else None for frame in sample.frames],
        }
        for sample in samples
    ]
    write_json(args.out_dir / "requests.json", requests)
    if args.prepare_only:
        print(f"Prepared {len(samples)} zero-shot samples -> {args.out_dir / 'requests.json'}")
        return

    weights_path = resolve_config_path(str(config["model"]["weights"]))
    signature = build_signature(samples, config_path, weights_path)
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
            print(f"[cache] Reuse zero-shot results -> {results_path}")
            return

    detector = YOLOETextDetector(config["model"])
    results = classify(samples, detector, config, args.out_dir)
    write_json(results_path, results)
    write_csv_results(args.out_dir / "results.csv", results)
    write_review(args.out_dir / "review.md", results)
    summary = {
        "method": "sam3_yoloe_zero_shot",
        "total": len(results),
        "classes": {
            str(class_id): sum(result["class_id"] == class_id for result in results)
            for class_id in (0, 1, 2)
        },
        "warnings": sum(bool(result["reason_codes"]) for result in results),
    }
    write_json(args.out_dir / "summary.json", summary)
    write_json(
        cache_path,
        {
            "signature": signature,
            "result_count": len(results),
            "method": "sam3_yoloe_zero_shot",
        },
    )
    print(
        f"Zero-shot classification complete: {len(results)} samples -> {results_path}"
    )


if __name__ == "__main__":
    main()
