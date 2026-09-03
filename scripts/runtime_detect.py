#!/usr/bin/env python
"""Run practical underwater cable detection on images or videos.

The script has five modes:
- opencv: no model dependency, detects long cable-like regions as a fallback.
- yolo: uses an Ultralytics YOLO detector.
- sam: uses an Ultralytics SAM model with a prompt box or point.
- sam_auto: uses OpenCV cable-like boxes as automatic SAM prompt boxes.
- sam_everything: uses Meta SAM automatic mask generation, then filters cable-like masks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]
    score: float
    cls: int = 0
    source: str = "opencv"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_args_config(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key, value in vars(args).items():
        if value is not None:
            out[key] = value
    out.setdefault("method", "opencv")
    out.setdefault("output_dir", "outputs/runtime")
    out.setdefault("sample_stride", 1)
    out.setdefault("conf", 0.25)
    out.setdefault("device", "cpu")
    out.setdefault("imgsz", 1280)
    out.setdefault("opencv", {})
    return out


def iter_sources(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = [p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]
    return sorted(files)


def parse_box(text: str | None) -> list[int] | None:
    if not text:
        return None
    values = [int(float(x)) for x in text.split(",")]
    if len(values) != 4:
        raise ValueError("--prompt-box must be x1,y1,x2,y2")
    return values


def parse_point(text: str | None) -> list[int] | None:
    if not text:
        return None
    values = [int(float(x)) for x in text.split(",")]
    if len(values) != 2:
        raise ValueError("--prompt-point must be x,y")
    return values


def detect_opencv(frame: np.ndarray, params: dict[str, Any]) -> list[Detection]:
    original_h, original_w = frame.shape[:2]
    resize_width = int(params.get("resize_width") or original_w)
    if resize_width > 0 and resize_width != original_w:
        scale = resize_width / original_w
        work = cv2.resize(frame, (resize_width, int(original_h * scale)))
    else:
        scale = 1.0
        work = frame

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = int(params.get("blur", 5))
    if blur > 1:
        blur = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    edges = cv2.Canny(gray, int(params.get("canny_low", 40)), int(params.get("canny_high", 120)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (int(params.get("morph_kernel_w", 35)), int(params.get("morph_kernel_h", 7))),
    )
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    min_area = float(params.get("min_area", 1200))
    min_aspect = float(params.get("min_aspect", 3.0))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect < min_aspect:
            continue
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + w) / scale)
        y2 = int((y + h) / scale)
        score = min(0.99, 0.35 + area / max(mask.shape[0] * mask.shape[1], 1))
        detections.append(Detection((x1, y1, x2, y2), float(score), 0, "opencv"))

    detections.sort(key=lambda d: (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]), reverse=True)
    return detections[: int(params.get("max_boxes", 5))]


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(box_iou(det.xyxy, old.xyxy) <= iou_threshold for old in kept):
            kept.append(det)
    return kept


def filter_sam_masks(masks: list[dict[str, Any]], frame_shape: tuple[int, int, int], params: dict[str, Any]) -> list[Detection]:
    height, width = frame_shape[:2]
    frame_area = height * width
    min_area_ratio = float(params.get("min_area_ratio", 0.001))
    max_area_ratio = float(params.get("max_area_ratio", 0.35))
    min_aspect = float(params.get("min_aspect", 3.0))
    max_fill_ratio = float(params.get("max_fill_ratio", 0.75))
    min_fill_ratio = float(params.get("min_fill_ratio", 0.02))
    border_margin = int(params.get("border_margin", 3))
    max_boxes = int(params.get("max_boxes", 5))
    iou_threshold = float(params.get("nms_iou", 0.5))

    detections: list[Detection] = []
    for mask_item in masks:
        bbox = mask_item.get("bbox")
        segmentation = mask_item.get("segmentation")
        if bbox is None or segmentation is None:
            continue

        x, y, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        if x1 <= border_margin or y1 <= border_margin or x2 >= width - border_margin or y2 >= height - border_margin:
            if bool(params.get("reject_border_touching", True)):
                continue

        area = float(mask_item.get("area", np.asarray(segmentation).sum()))
        area_ratio = area / max(frame_area, 1)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect < min_aspect:
            continue

        fill_ratio = area / max(w * h, 1)
        if fill_ratio < min_fill_ratio or fill_ratio > max_fill_ratio:
            continue

        predicted_iou = float(mask_item.get("predicted_iou", 0.8))
        stability_score = float(mask_item.get("stability_score", 0.8))
        shape_score = min(1.0, aspect / max(min_aspect * 2.0, 1.0))
        score = 0.45 * predicted_iou + 0.35 * stability_score + 0.20 * shape_score
        detections.append(Detection((x1, y1, x2, y2), float(score), 0, "sam_everything"))

    return nms(detections, iou_threshold)[:max_boxes]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = (0, 220, 255) if det.source == "opencv" else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det.source}:{det.score:.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return canvas


def build_model_detector(
    method: str,
    weights: str,
    conf: float,
    imgsz: int,
    device: str,
    prompt_box: list[int] | None,
    prompt_point: list[int] | None,
    opencv_params: dict[str, Any] | None = None,
    sam_everything_params: dict[str, Any] | None = None,
):
    if method == "yolo":
        from ultralytics import YOLO

        model = YOLO(weights)

        def detect(frame: np.ndarray) -> list[Detection]:
            results = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
            out: list[Detection] = []
            for box in results[0].boxes:
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(xyxy, float(box.conf[0]), int(box.cls[0]), "yolo"))
            return out

        return detect

    if method == "sam":
        from ultralytics import SAM

        model = SAM(weights)

        def detect(frame: np.ndarray) -> list[Detection]:
            kwargs: dict[str, Any] = {"device": device, "verbose": False}
            if prompt_box is not None:
                kwargs["bboxes"] = [prompt_box]
            if prompt_point is not None:
                kwargs["points"] = [prompt_point]
                kwargs["labels"] = [1]
            results = model.predict(frame, **kwargs)
            boxes = getattr(results[0], "boxes", None)
            if boxes is None:
                return []
            out: list[Detection] = []
            for box in boxes:
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(xyxy, 1.0, 0, "sam"))
            return out

        return detect

    if method == "sam_auto":
        from ultralytics import SAM

        model = SAM(weights)
        opencv_params = opencv_params or {}

        def expand_box(box: tuple[int, int, int, int], width: int, height: int, ratio: float = 0.15) -> list[int]:
            x1, y1, x2, y2 = box
            pad_x = int((x2 - x1) * ratio)
            pad_y = int((y2 - y1) * ratio)
            return [
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(width - 1, x2 + pad_x),
                min(height - 1, y2 + pad_y),
            ]

        def detect(frame: np.ndarray) -> list[Detection]:
            prompt_dets = detect_opencv(frame, opencv_params)
            if not prompt_dets:
                return []
            height, width = frame.shape[:2]
            prompt_boxes = [expand_box(det.xyxy, width, height) for det in prompt_dets]
            results = model.predict(frame, bboxes=prompt_boxes, device=device, verbose=False)
            boxes = getattr(results[0], "boxes", None)
            if boxes is None:
                return prompt_dets
            out: list[Detection] = []
            for box in boxes:
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(xyxy, 1.0, 0, "sam_auto"))
            return out or prompt_dets

        return detect

    if method == "sam_everything":
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        params = sam_everything_params or {}
        model_type = str(params.get("model_type", "vit_b"))
        sam = sam_model_registry[model_type](checkpoint=weights)
        sam.to(device=device)
        generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=int(params.get("points_per_side", 24)),
            pred_iou_thresh=float(params.get("pred_iou_thresh", 0.86)),
            stability_score_thresh=float(params.get("stability_score_thresh", 0.88)),
            crop_n_layers=int(params.get("crop_n_layers", 0)),
            crop_n_points_downscale_factor=int(params.get("crop_n_points_downscale_factor", 1)),
            min_mask_region_area=int(params.get("min_mask_region_area", 400)),
        )

        def detect(frame: np.ndarray) -> list[Detection]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                masks = generator.generate(rgb)
            return filter_sam_masks(masks, frame.shape, params)

        return detect

    raise ValueError(f"Unsupported model method: {method}")


def process_image(path: Path, out_dir: Path, detector) -> dict[str, Any]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    detections = detector(frame)
    vis = draw_detections(frame, detections)
    out_path = out_dir / "images" / path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return {"source": str(path), "frames": [{"frame": 0, "detections": [d.__dict__ for d in detections]}]}


def process_video(path: Path, out_dir: Path, detector, sample_stride: int, max_frames: int | None) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = out_dir / "videos" / f"{path.stem}_det.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    records: list[dict[str, Any]] = []
    frame_idx = 0
    sample_stride = max(1, int(sample_stride))
    last_dets: list[Detection] = []
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_stride == 0:
            last_dets = detector(frame)
        vis = draw_detections(frame, last_dets)
        writer.write(vis)
        records.append({"frame": frame_idx, "detections": [d.__dict__ for d in last_dets]})
        frame_idx += 1

    cap.release()
    writer.release()
    return {"source": str(path), "output_video": str(out_video), "frames": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", choices=["opencv", "yolo", "sam", "sam_auto", "sam_everything"], default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--prompt-box", default=None, help="SAM prompt box as x1,y1,x2,y2")
    parser.add_argument("--prompt-point", default=None, help="SAM positive point as x,y")
    args = parser.parse_args()

    cfg = merge_args_config(args, load_config(args.config))
    input_path = Path(cfg["input"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    method = cfg["method"]
    if method == "opencv":
        detector = lambda frame: detect_opencv(frame, cfg.get("opencv", {}))
    else:
        if not cfg.get("weights"):
            raise SystemExit(f"--weights is required for method={method}")
        detector = build_model_detector(
            method,
            str(cfg["weights"]),
            float(cfg["conf"]),
            int(cfg["imgsz"]),
            str(cfg["device"]),
            parse_box(args.prompt_box),
            parse_point(args.prompt_point),
            cfg.get("opencv", {}),
            cfg.get("sam_everything", {}),
        )

    summaries = []
    for source in iter_sources(input_path):
        if source.suffix.lower() in IMAGE_EXTS:
            summaries.append(process_image(source, out_dir, detector))
        elif source.suffix.lower() in VIDEO_EXTS:
            summaries.append(process_video(source, out_dir, detector, int(cfg["sample_stride"]), cfg.get("max_frames")))

    summary_path = out_dir / "detections.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
