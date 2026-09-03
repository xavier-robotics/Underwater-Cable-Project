#!/usr/bin/env python
"""Classify detected underwater cables as touching bottom or suspended."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_args(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key, value in vars(args).items():
        if value is not None:
            out[key] = value
    out.setdefault("output_dir", "outputs/contact")
    out.setdefault("contact", {})
    return out


def detection_box(det: dict[str, Any]) -> tuple[int, int, int, int] | None:
    xyxy = det.get("xyxy")
    if not xyxy or len(xyxy) != 4:
        return None
    return tuple(int(v) for v in xyxy)


def is_valid_cable_box(box: tuple[int, int, int, int], params: dict[str, Any]) -> bool:
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w * h < int(params.get("min_box_area", 800)):
        return False
    aspect = max(w / max(h, 1), h / max(w, 1))
    return aspect >= float(params.get("min_box_aspect", 2.0))


def estimate_bottom_edges(strip: np.ndarray, params: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, int(params.get("canny_low", 35)), int(params.get("canny_high", 110)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(params.get("hough_threshold", 35)),
        minLineLength=int(params.get("hough_min_line_length", 60)),
        maxLineGap=int(params.get("hough_max_line_gap", 20)),
    )
    if lines is None:
        return []

    kept: list[tuple[int, int, int, int]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = (int(v) for v in line)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 10:
            continue
        slope = abs(dy / dx)
        if slope <= 0.45:
            kept.append((x1, y1, x2, y2))
    return kept


def line_y_at_x(line: tuple[int, int, int, int], x: float) -> float | None:
    x1, y1, x2, y2 = line
    if x1 == x2:
        return None
    if x < min(x1, x2) or x > max(x1, x2):
        return None
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def classify_box(frame: np.ndarray, box: tuple[int, int, int, int], params: dict[str, Any]) -> dict[str, Any]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h - 1, y2))
    search = int(params.get("search_below_px", 160))
    threshold = float(params.get("contact_threshold_px", 18))
    y3 = min(h, y2 + search)

    if y3 <= y2 + 8 or x2 <= x1 + 8:
        return {"state": "unknown", "distance_px": None, "contact_ratio": 0.0}

    strip = frame[y2:y3, x1:x2]
    lines = estimate_bottom_edges(strip, params)
    if not lines:
        return {"state": "unknown", "distance_px": None, "contact_ratio": 0.0}

    xs = np.linspace(0, max(1, x2 - x1 - 1), num=32)
    distances = []
    for x in xs:
        ys = [line_y_at_x(line, float(x)) for line in lines]
        ys = [y for y in ys if y is not None and y >= 0]
        if ys:
            distances.append(min(ys))

    if not distances:
        return {"state": "unknown", "distance_px": None, "contact_ratio": 0.0}

    distances_arr = np.asarray(distances, dtype=np.float32)
    contact_ratio = float(np.mean(distances_arr <= threshold))
    distance_px = float(np.percentile(distances_arr, 20))
    min_ratio = float(params.get("min_contact_ratio", 0.18))
    state = "exposed" if contact_ratio >= min_ratio or distance_px <= threshold else "suspended"
    return {
        "state": state,
        "distance_px": round(distance_px, 2),
        "contact_ratio": round(contact_ratio, 3),
        "bottom_lines": len(lines),
    }


def draw(frame: np.ndarray, box: tuple[int, int, int, int], result: dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = box
    state = result["state"]
    color = (0, 220, 0) if state == "exposed" else (0, 160, 255) if state == "suspended" else (160, 160, 160)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{state}"
    if result.get("distance_px") is not None:
        label += f" d={result['distance_px']}"
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def index_detections(detections_path: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    data = json.loads(detections_path.read_text(encoding="utf-8"))
    indexed: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for item in data:
        source = Path(item["source"]).name
        per_frame: dict[int, list[dict[str, Any]]] = {}
        for frame_rec in item.get("frames", []):
            per_frame[int(frame_rec["frame"])] = frame_rec.get("detections", [])
        indexed[source] = per_frame
    return indexed


def process_video(
    path: Path,
    out_dir: Path,
    detections: dict[int, list[dict[str, Any]]],
    params: dict[str, Any],
    max_frames: int | None,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = out_dir / "videos" / f"{path.stem}_contact.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames = []
    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_results = []
        for det in detections.get(frame_idx, []):
            box = detection_box(det)
            if box is None or not is_valid_cable_box(box, params):
                continue
            result = classify_box(frame, box, params)
            result.update({"xyxy": list(box), "score": det.get("score"), "source": det.get("source")})
            frame_results.append(result)
            draw(frame, box, result)
        writer.write(frame)
        frames.append({"frame": frame_idx, "results": frame_results})
        frame_idx += 1

    cap.release()
    writer.release()
    return {"source": str(path), "output_video": str(out_video), "frames": frames}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/contact.yaml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = merge_args(args, load_config(args.config))
    input_path = Path(cfg["input"])
    out_dir = Path(cfg["output_dir"])
    params = cfg.get("contact", {})
    indexed = index_detections(Path(cfg["detections"]))

    sources = [input_path] if input_path.is_file() else sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
    summaries = []
    for source in sources:
        if source.suffix.lower() in VIDEO_EXTS:
            summaries.append(process_video(source, out_dir, indexed.get(source.name, {}), params, cfg.get("max_frames")))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "contact_states.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
