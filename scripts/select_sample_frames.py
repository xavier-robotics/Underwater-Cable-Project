#!/usr/bin/env python
"""Select 2-3 representative frames for each underwater cable sample segment.

This script is intended for long pool-scan videos where 1 m cable samples are
separated by roughly 1 m blank gaps. It supports OpenCV, YOLOE, and SAM3
detectors, segments the video by time, and exports a few full-resolution
evidence frames per sample for later vision-model inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from tqdm import tqdm


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/tmp/ultralytics")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class FrameMetric:
    frame_idx: int
    timestamp_sec: float
    cable_score: float
    smooth_score: float
    sharpness: float
    brightness: float
    dark_ratio: float
    largest_box_ratio: float
    span_ratio: float
    visible: bool


@dataclass
class Segment:
    sample_id: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    frame_count: int
    selected_frames: list[dict]


@dataclass
class CableCrop:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    area_ratio: float
    span_ratio: float
    mask: np.ndarray | None
    source: str


class FrameScorer(Protocol):
    name: str

    def score(self, frame: np.ndarray, frame_idx: int, fps: float) -> FrameMetric:
        ...


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 digest for a local model or data file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise SystemExit(f"Input is not a supported video: {input_path}")
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_")


def parse_roi(text: str) -> tuple[float, float, float, float]:
    values = [float(v) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2 as ratios in [0,1]")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("--roi values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return x1, y1, x2, y2


def crop_roi(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    return frame[int(y1 * height) : int(y2 * height), int(x1 * width) : int(x2 * width)]


def compute_metric(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    roi: tuple[float, float, float, float],
    resize_width: int,
    dark_threshold: int,
    min_aspect: float,
) -> FrameMetric:
    work = crop_roi(frame, roi)
    if resize_width > 0 and work.shape[1] > resize_width:
        scale = resize_width / work.shape[1]
        work = cv2.resize(work, (resize_width, max(1, int(work.shape[0] * scale))))

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    mask = (gray <= dark_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

    height, width = gray.shape[:2]
    frame_area = max(1, width * height)
    dark_ratio = float(np.count_nonzero(mask) / frame_area)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_box_ratio = 0.0
    span_ratio = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.003:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect < min_aspect:
            continue
        box_ratio = (w * h) / frame_area
        if box_ratio > largest_box_ratio:
            largest_box_ratio = float(box_ratio)
            span_ratio = float(max(w / max(width, 1), h / max(height, 1)))

    # A true cable frame usually has a large dark ratio and at least one
    # elongated dark component spanning a substantial part of the view.
    cable_score = min(1.0, 0.45 * dark_ratio + 0.75 * largest_box_ratio + 0.25 * span_ratio)
    return FrameMetric(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / fps if fps > 0 else 0.0,
        cable_score=float(cable_score),
        smooth_score=0.0,
        sharpness=sharpness,
        brightness=brightness,
        dark_ratio=dark_ratio,
        largest_box_ratio=largest_box_ratio,
        span_ratio=span_ratio,
        visible=False,
    )


class OpenCVScorer:
    name = "opencv"

    def __init__(
        self,
        roi: tuple[float, float, float, float],
        resize_width: int,
        dark_threshold: int,
        min_aspect: float,
    ) -> None:
        self.roi = roi
        self.resize_width = resize_width
        self.dark_threshold = dark_threshold
        self.min_aspect = min_aspect

    def score(self, frame: np.ndarray, frame_idx: int, fps: float) -> FrameMetric:
        return compute_metric(
            frame=frame,
            frame_idx=frame_idx,
            fps=fps,
            roi=self.roi,
            resize_width=self.resize_width,
            dark_threshold=self.dark_threshold,
            min_aspect=self.min_aspect,
        )


class YOLOEScorer:
    name = "yoloe"

    def __init__(
        self,
        weights: str,
        labels: list[str],
        device: str,
        imgsz: int,
        conf: float,
        iou: float,
        min_area_ratio: float,
        min_span_ratio: float,
        mobileclip_path: Path,
    ) -> None:
        self.labels = labels
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.min_area_ratio = min_area_ratio
        self.min_span_ratio = min_span_ratio

        from ultralytics import YOLOE
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._prepare_mobileclip(mobileclip_path, torch)
        self.model = YOLOE(weights)
        self.model.to(device)
        self.model.set_classes(labels, self.model.get_text_pe(labels))

    @staticmethod
    def _prepare_mobileclip(mobileclip_path: Path, torch_module) -> None:
        """Ensure YOLOE can find a valid TorchScript MobileCLIP text encoder."""
        expected_name = "mobileclip_blt.ts"
        expected_path = Path(expected_name)

        if mobileclip_path.suffix == ".pt":
            raise SystemExit(
                "--mobileclip-path must point to mobileclip_blt.ts, not mobileclip_blt.pt.\n"
                "The .pt file from Apple's MobileCLIP repo is not TorchScript-loadable by YOLOE text prompts.\n"
                "Fix with:\n"
                "  cd /home/nvidia/uw_detection\n"
                "  rm -f mobileclip_blt.ts\n"
                "  conda run -n yoloe python -c \"from ultralytics.utils.downloads import attempt_download_asset; "
                "print(attempt_download_asset('mobileclip_blt.ts'))\"\n"
                "  conda run -n yoloe python -c \"import torch; torch.jit.load('mobileclip_blt.ts', map_location='cpu'); "
                "print('mobileclip ok')\""
            )

        if mobileclip_path.exists() and mobileclip_path.name == expected_name and mobileclip_path != expected_path:
            if not expected_path.exists():
                try:
                    expected_path.symlink_to(mobileclip_path)
                except OSError:
                    pass

        try:
            torch_module.jit.load(expected_name, map_location="cpu")
        except Exception as exc:
            raise SystemExit(
                f"Invalid or missing {expected_name}: {exc}\n\n"
                "YOLOE text prompt mode needs the TorchScript MobileCLIP file named mobileclip_blt.ts in the run directory.\n"
                "Run these commands, then retry:\n"
                "  cd /home/nvidia/uw_detection\n"
                "  rm -f mobileclip_blt.ts\n"
                "  conda run -n yoloe python -c \"from ultralytics.utils.downloads import attempt_download_asset; "
                "print(attempt_download_asset('mobileclip_blt.ts'))\"\n"
                "  conda run -n yoloe python -c \"import torch; torch.jit.load('mobileclip_blt.ts', map_location='cpu'); "
                "print('mobileclip ok')\"\n\n"
                "Do not use ckpts/yoloe/mobileclip_blt.pt for this option; it is a different checkpoint format."
            ) from exc

    def _predict(self, frame: np.ndarray):
        return self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

    @staticmethod
    def _mask_from_polygon(result, mask_idx: int, shape: tuple[int, int]) -> np.ndarray | None:
        masks = getattr(result, "masks", None)
        if masks is None:
            return None
        polygons = getattr(masks, "xy", None)
        if polygons is None or mask_idx >= len(polygons):
            return None
        polygon = np.asarray(polygons[mask_idx], dtype=np.int32)
        if polygon.size == 0:
            return None
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    @staticmethod
    def _mask_from_tensor(result, mask_idx: int, shape: tuple[int, int]) -> np.ndarray | None:
        masks = getattr(result, "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            return None
        if mask_idx >= len(masks.data):
            return None
        mask = masks.data[mask_idx].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _bbox_from_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    def extract_crop(self, frame: np.ndarray) -> CableCrop | None:
        result = self._predict(frame)
        height, width = frame.shape[:2]
        frame_area = max(1, height * width)

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        best: CableCrop | None = None
        best_score = -1.0
        has_masks = getattr(result, "masks", None) is not None

        for idx, (box, conf) in enumerate(zip(xyxy, confs)):
            mask = self._mask_from_polygon(result, idx, (height, width))
            if mask is None:
                mask = self._mask_from_tensor(result, idx, (height, width))

            source = "mask"
            bbox = self._bbox_from_mask(mask) if mask is not None else None
            if bbox is None:
                bbox = self._bbox_from_box(box, width, height)
                mask = None
                source = "box_fallback" if has_masks else "box"

            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            if mask is not None:
                area_ratio = float(np.count_nonzero(mask > 0) / frame_area)
            else:
                area_ratio = float(((x2 - x1) * (y2 - y1)) / frame_area)
            span_ratio = float(max((x2 - x1) / max(width, 1), (y2 - y1) / max(height, 1)))
            score = float(conf) * (0.35 + 0.65 * min(1.0, area_ratio + span_ratio))
            if score > best_score:
                best_score = score
                best = CableCrop(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(conf),
                    area_ratio=area_ratio,
                    span_ratio=span_ratio,
                    mask=mask,
                    source=source,
                )
        return best

    def score(self, frame: np.ndarray, frame_idx: int, fps: float) -> FrameMetric:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        crop = self.extract_crop(frame)
        best_conf = crop.confidence if crop is not None else 0.0
        best_area_ratio = crop.area_ratio if crop is not None else 0.0
        best_span_ratio = crop.span_ratio if crop is not None else 0.0

        cable_score = best_conf
        visible = best_conf >= self.conf and (
            best_area_ratio >= self.min_area_ratio or best_span_ratio >= self.min_span_ratio
        )
        if not visible:
            cable_score *= 0.5

        return FrameMetric(
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / fps if fps > 0 else 0.0,
            cable_score=float(cable_score),
            smooth_score=0.0,
            sharpness=sharpness,
            brightness=brightness,
            dark_ratio=0.0,
            largest_box_ratio=best_area_ratio,
            span_ratio=best_span_ratio,
            visible=False,
        )


class SAM3Scorer:
    name = "sam3"

    def __init__(
        self,
        prompt: str,
        device: str,
        confidence_threshold: float,
        min_area_ratio: float,
        min_span_ratio: float,
        checkpoint_path: Path | None,
        dtype: str,
    ) -> None:
        self.prompt = prompt
        self.confidence_threshold = confidence_threshold
        self.min_area_ratio = min_area_ratio
        self.min_span_ratio = min_span_ratio

        try:
            import torch
            from PIL import Image
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as exc:
            if exc.name == "pkg_resources":
                raise SystemExit(
                    "SAM3 requires pkg_resources from setuptools, but it is missing in this conda env.\n"
                    "Fix with:\n"
                    "  conda run -n sam3 python -m pip install --no-cache-dir -c constraints-sam3.txt --force-reinstall setuptools==80.9.0"
                ) from exc
            raise

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise SystemExit(
                "SAM3 image model requires CUDA in this repository build; torch.cuda.is_available() is False.\n"
                "Check with:\n"
                "  conda run -n sam3 python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())\"\n"
                "If CUDA is visible in your normal shell, run the script there rather than inside a restricted session."
            )
        if device == "cuda":
            capability = torch.cuda.get_device_capability(0)
            cuda_version = torch.version.cuda or "unknown"
            if capability >= (12, 1) and cuda_version.startswith("12."):
                raise SystemExit(
                    f"Detected CUDA device capability sm_{capability[0]}{capability[1]} with PyTorch CUDA {cuda_version}.\n"
                    "This PyTorch build does not fully support this GPU architecture; runtime CUDA JIT may fail with:\n"
                    "  nvrtc: error: invalid value for --gpu-architecture (-arch)\n"
                    "Install a PyTorch build that supports this GPU/CUDA stack, then rerun this script."
                )
        self.device = device
        self._image_cls = Image
        self._torch = torch
        if dtype == "auto":
            dtype = "float32"
        self.dtype = dtype
        self._autocast_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        kwargs = {"device": device}
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = str(checkpoint_path)
            kwargs["load_from_HF"] = False
        self.model = build_sam3_image_model(**kwargs)
        if dtype not in {"float32", "bfloat16"}:
            raise SystemExit("--sam3-dtype must be one of auto, float32, bfloat16")
        if dtype == "float32":
            self._register_dtype_alignment_hooks(torch)
        self.processor = Sam3Processor(
            self.model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def _register_dtype_alignment_hooks(self, torch_module) -> None:
        linear_cls = torch_module.nn.Linear

        def align_linear_input(module, inputs):
            if not inputs:
                return inputs
            tensor = inputs[0]
            if hasattr(tensor, "dtype") and tensor.dtype != module.weight.dtype:
                return (tensor.to(dtype=module.weight.dtype), *inputs[1:])
            return inputs

        for module in self.model.modules():
            if isinstance(module, linear_cls):
                module.register_forward_pre_hook(align_linear_input)

    def _to_numpy(self, value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if value.dtype == self._torch.bfloat16:
                value = value.float()
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _bbox_from_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    @staticmethod
    def _clean_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            mask = mask.reshape(mask.shape[-2], mask.shape[-1])
        mask = (mask > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _predict(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._image_cls.fromarray(rgb)
        use_autocast = self.device == "cuda" and self.dtype == "bfloat16"
        with self._torch.autocast(device_type=self.device, dtype=self._autocast_dtype, enabled=use_autocast):
            state = self.processor.set_image(image)
            return self.processor.set_text_prompt(self.prompt, state)

    def extract_crop(self, frame: np.ndarray) -> CableCrop | None:
        output = self._predict(frame)
        height, width = frame.shape[:2]
        frame_area = max(1, height * width)

        masks = self._to_numpy(output.get("masks", []))
        boxes = self._to_numpy(output.get("boxes", []))
        scores = self._to_numpy(output.get("scores", []))
        if scores.size == 0:
            return None

        scores = scores.reshape(-1)
        best: CableCrop | None = None
        best_score = -1.0

        for idx, confidence in enumerate(scores):
            if float(confidence) < self.confidence_threshold:
                continue

            mask = None
            bbox = None
            source = "sam3_mask"
            if masks.size > 0 and idx < len(masks):
                mask = self._clean_mask(masks[idx], (height, width))
                bbox = self._bbox_from_mask(mask)

            if bbox is None and boxes.size > 0 and idx < len(boxes):
                bbox = self._bbox_from_box(boxes[idx], width, height)
                mask = None
                source = "sam3_box"
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            if mask is not None:
                area_ratio = float(np.count_nonzero(mask > 0) / frame_area)
            else:
                area_ratio = float(((x2 - x1) * (y2 - y1)) / frame_area)
            span_ratio = float(max((x2 - x1) / max(width, 1), (y2 - y1) / max(height, 1)))
            if area_ratio < self.min_area_ratio and span_ratio < self.min_span_ratio:
                continue

            shape_bonus = min(1.0, area_ratio + span_ratio)
            score = float(confidence) * (0.35 + 0.65 * shape_bonus)
            if score > best_score:
                best_score = score
                best = CableCrop(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    area_ratio=area_ratio,
                    span_ratio=span_ratio,
                    mask=mask,
                    source=source,
                )
        return best

    def score(self, frame: np.ndarray, frame_idx: int, fps: float) -> FrameMetric:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        crop = self.extract_crop(frame)
        best_conf = crop.confidence if crop is not None else 0.0
        best_area_ratio = crop.area_ratio if crop is not None else 0.0
        best_span_ratio = crop.span_ratio if crop is not None else 0.0

        cable_score = best_conf
        if crop is not None:
            cable_score *= 0.35 + 0.65 * min(1.0, best_area_ratio + best_span_ratio)

        return FrameMetric(
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / fps if fps > 0 else 0.0,
            cable_score=float(cable_score),
            smooth_score=0.0,
            sharpness=sharpness,
            brightness=brightness,
            dark_ratio=0.0,
            largest_box_ratio=best_area_ratio,
            span_ratio=best_span_ratio,
            visible=False,
        )


def smooth_scores(metrics: list[FrameMetric], window: int) -> None:
    if not metrics:
        return
    scores = np.asarray([m.cable_score for m in metrics], dtype=np.float32)
    window = max(1, int(window))
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / window
        padded = np.pad(scores, (window // 2, window - 1 - window // 2), mode="edge")
        scores = np.convolve(padded, kernel, mode="valid")
    for metric, score in zip(metrics, scores):
        metric.smooth_score = float(score)


def analyze_video(
    video_path: Path,
    scorer: FrameScorer,
    sample_every_sec: float,
    max_frames: int | None,
    smooth_window: int,
    cable_threshold: float,
) -> tuple[list[FrameMetric], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(fps * sample_every_sec)))

    metrics: list[FrameMetric] = []
    frame_idx = 0
    sampled = 0
    pbar_total = total_frames if total_frames > 0 else None
    with tqdm(total=pbar_total, desc=f"Analyze {video_path.name}", unit="frame") as pbar:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                metrics.append(scorer.score(frame, frame_idx, fps))
                sampled += 1
            frame_idx += 1
            pbar.update(1)

    cap.release()
    smooth_scores(metrics, smooth_window)
    for metric in metrics:
        metric.visible = metric.smooth_score >= cable_threshold

    info = {
        "video": str(video_path),
        "fps": fps,
        "frames": total_frames,
        "width": width,
        "height": height,
        "sample_every_sec": sample_every_sec,
        "sampled_frames": sampled,
        "detector": scorer.name,
        "cable_threshold": cable_threshold,
    }
    return metrics, info


def find_visible_fragments(metrics: list[FrameMetric]) -> list[list[FrameMetric]]:
    raw: list[list[FrameMetric]] = []
    current: list[FrameMetric] = []
    for metric in metrics:
        if metric.visible:
            current.append(metric)
        elif current:
            raw.append(current)
            current = []
    if current:
        raw.append(current)
    return raw


def merge_close_fragments(raw: list[list[FrameMetric]], merge_gap_sec: float) -> list[list[FrameMetric]]:
    if not raw:
        return []

    merged: list[list[FrameMetric]] = [list(raw[0])]
    for segment in raw[1:]:
        gap = segment[0].timestamp_sec - merged[-1][-1].timestamp_sec
        if gap <= merge_gap_sec or np.isclose(gap, merge_gap_sec, rtol=0.0, atol=1e-3):
            merged[-1].extend(segment)
        else:
            merged.append(list(segment))
    return merged


def filter_short_segments(segments: list[list[FrameMetric]], min_segment_sec: float) -> list[list[FrameMetric]]:
    return [
        seg
        for seg in segments
        if seg[-1].timestamp_sec - seg[0].timestamp_sec >= min_segment_sec
    ]


def split_by_largest_gaps(segments: list[list[FrameMetric]], expected_samples: int) -> list[list[FrameMetric]]:
    """Merge fragments into a fixed number of timeline groups using largest gaps."""
    if expected_samples <= 0 or len(segments) <= expected_samples:
        return segments

    gaps: list[tuple[float, int]] = []
    for idx in range(len(segments) - 1):
        gap = segments[idx + 1][0].timestamp_sec - segments[idx][-1].timestamp_sec
        gaps.append((gap, idx))
    split_after = {idx for _, idx in sorted(gaps, reverse=True)[: expected_samples - 1]}

    grouped: list[list[FrameMetric]] = []
    current: list[FrameMetric] = []
    for idx, segment in enumerate(segments):
        current.extend(segment)
        if idx in split_after:
            grouped.append(current)
            current = []
    if current:
        grouped.append(current)
    return grouped


def find_segments(
    metrics: list[FrameMetric],
    mode: str,
    min_segment_sec: float,
    merge_gap_sec: float,
    expected_samples: int | None,
) -> list[list[FrameMetric]]:
    raw = find_visible_fragments(metrics)
    if not raw:
        return []

    if mode == "single":
        return [[metric for segment in raw for metric in segment]]

    # Remove one-frame detector spikes before merging. Otherwise, a short false
    # positive can bridge two real cable samples across a blank interval.
    stable_fragments = filter_short_segments(raw, min_segment_sec)
    merged = merge_close_fragments(stable_fragments, merge_gap_sec)
    merged = filter_short_segments(merged, min_segment_sec)
    if expected_samples is not None and len(merged) > expected_samples:
        merged = split_by_largest_gaps(merged, expected_samples)
    return merged


def pick_representatives(segment: list[FrameMetric], frames_per_sample: int) -> list[FrameMetric]:
    if not segment:
        return []

    # A smoothed positive can surround a detector miss. Such a frame is useful
    # for temporal segmentation but cannot produce a SAM3 crop or mask.
    eligible = [idx for idx, metric in enumerate(segment) if metric.cable_score > 0.0]
    if not eligible:
        return []

    frames_per_sample = min(max(1, min(5, frames_per_sample)), len(eligible))
    if len(eligible) <= frames_per_sample:
        return [segment[idx] for idx in eligible]

    fractions_by_count = {
        1: [0.5],
        2: [0.25, 0.75],
        3: [0.2, 0.5, 0.8],
        4: [0.15, 0.38, 0.62, 0.85],
        5: [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    fractions = fractions_by_count[frames_per_sample]

    selected: list[FrameMetric] = []
    used: set[int] = set()
    raw_scores = np.asarray([m.cable_score for m in segment], dtype=np.float32)
    smooth_scores = np.asarray([m.smooth_score for m in segment], dtype=np.float32)
    sharpness = np.asarray([m.sharpness for m in segment], dtype=np.float32)
    sharp_norm = sharpness / max(float(np.percentile(sharpness, 90)), 1.0)

    for frac in fractions:
        target = int(round(frac * (len(segment) - 1)))
        radius = max(1, len(segment) // (2 * frames_per_sample))
        lo = max(0, target - radius)
        hi = min(len(segment), target + radius + 1)
        candidates = [
            idx
            for idx in eligible
            if lo <= idx < hi and segment[idx].frame_idx not in used
        ]
        if not candidates:
            candidates = [idx for idx in eligible if segment[idx].frame_idx not in used]
        best_idx = max(
            candidates,
            key=lambda idx: float(
                0.50 * raw_scores[idx]
                + 0.35 * smooth_scores[idx]
                + 0.15 * min(sharp_norm[idx], 1.0)
            ),
        )
        selected.append(segment[best_idx])
        used.add(segment[best_idx].frame_idx)

    selected.sort(key=lambda m: m.timestamp_sec)
    return selected


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Read an exact frame, falling back for codecs with unreliable seeking."""
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be non-negative, got {frame_idx}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        return frame

    # Some phone/conferencing exports use long GOPs or incomplete seek tables.
    # OpenCV may decode the file sequentially but fail when seeking near its end.
    # Reopen from the beginning so representative-frame export still works.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video for sequential read: {video_path}")
    frame = None
    for _ in range(frame_idx + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    cap.release()
    return frame


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float,
    min_pad: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = max(min_pad, int((x2 - x1) * pad_ratio))
    pad_y = max(min_pad, int((y2 - y1) * pad_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def make_masked_frame(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame.copy()
    masked = np.zeros_like(frame)
    masked[mask > 0] = frame[mask > 0]
    return masked


def draw_crop_overlay(frame: np.ndarray, crop: CableCrop) -> np.ndarray:
    canvas = frame.copy()
    if crop.mask is not None:
        overlay = canvas.copy()
        overlay[crop.mask > 0] = (0, 220, 255)
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    x1, y1, x2, y2 = crop.bbox_xyxy
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
    label = f"{crop.source}:{crop.confidence:.2f}"
    cv2.putText(canvas, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    return canvas


def save_selected_frame(
    frame: np.ndarray,
    sample_dir: Path,
    sample_id: str,
    rank: int,
    metric: FrameMetric,
    args: argparse.Namespace,
    scorer: FrameScorer,
) -> dict:
    base = f"{sample_id}_{rank:02d}_f{metric.frame_idx:06d}_t{metric.timestamp_sec:08.2f}"
    full_path = sample_dir / f"{base}_full.jpg"
    cv2.imwrite(str(full_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

    record = {
        "rank": rank,
        "frame_idx": metric.frame_idx,
        "timestamp_sec": metric.timestamp_sec,
        "path": str(full_path),
        "full_frame_path": str(full_path),
        "cable_score": metric.cable_score,
        "smooth_score": metric.smooth_score,
        "sharpness": metric.sharpness,
        "brightness": metric.brightness,
    }

    if args.crop_source == "full":
        return record

    expected_detector = {
        "yoloe-mask": "yoloe",
        "sam3-mask": "sam3",
    }.get(args.crop_source)
    if expected_detector is None:
        record["crop_error"] = f"unsupported_crop_source:{args.crop_source}"
        return record
    if scorer.name != expected_detector:
        record["crop_error"] = f"crop_source={args.crop_source} requires --detector {expected_detector}"
        return record

    extract_crop = getattr(scorer, "extract_crop", None)
    if not callable(extract_crop):
        record["crop_error"] = f"detector {scorer.name} does not support crop extraction"
        return record

    crop = extract_crop(frame)
    if crop is None:
        record["crop_error"] = f"no_{scorer.name}_detection"
        return record

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(
        crop.bbox_xyxy,
        width=width,
        height=height,
        pad_ratio=args.crop_pad_ratio,
        min_pad=args.crop_min_pad,
    )
    crop_img = frame[y1:y2, x1:x2]
    crop_path = sample_dir / f"{base}_crop.jpg"
    cv2.imwrite(str(crop_path), crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

    mask_path = None
    masked_crop_path = None
    if crop.mask is not None:
        mask_crop = crop.mask[y1:y2, x1:x2]
        masked_crop = make_masked_frame(crop_img, mask_crop)
        mask_path = sample_dir / f"{base}_mask.png"
        masked_crop_path = sample_dir / f"{base}_masked.jpg"
        cv2.imwrite(str(mask_path), mask_crop)
        cv2.imwrite(str(masked_crop_path), masked_crop, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

    annotated_path = sample_dir / f"{base}_annotated.jpg"
    cv2.imwrite(str(annotated_path), draw_crop_overlay(frame, crop), [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

    record.update(
        {
            "path": str(masked_crop_path or crop_path),
            "crop_path": str(crop_path),
            "masked_crop_path": str(masked_crop_path) if masked_crop_path else None,
            "mask_path": str(mask_path) if mask_path else None,
            "annotated_path": str(annotated_path),
            "crop_source": crop.source,
            "crop_confidence": crop.confidence,
            "crop_area_ratio": crop.area_ratio,
            "crop_span_ratio": crop.span_ratio,
            "crop_bbox_xyxy": [x1, y1, x2, y2],
            "raw_bbox_xyxy": list(crop.bbox_xyxy),
        }
    )
    return record


def save_metrics_csv(path: Path, metrics: list[FrameMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()) if metrics else [])
        if metrics:
            writer.writeheader()
            for metric in metrics:
                writer.writerow(asdict(metric))


def build_scorer(args: argparse.Namespace) -> FrameScorer:
    if args.detector == "opencv":
        return OpenCVScorer(
            roi=parse_roi(args.roi),
            resize_width=args.resize_width,
            dark_threshold=args.dark_threshold,
            min_aspect=args.min_aspect,
        )

    if args.detector == "sam3":
        return SAM3Scorer(
            prompt=args.sam3_prompt,
            device=args.device,
            confidence_threshold=args.sam3_conf,
            min_area_ratio=args.sam3_min_area_ratio,
            min_span_ratio=args.sam3_min_span_ratio,
            checkpoint_path=args.sam3_checkpoint,
            dtype=args.sam3_dtype,
        )

    labels = [label.strip() for label in args.yoloe_labels.split(",") if label.strip()]
    if not labels:
        raise SystemExit("--yoloe-labels must contain at least one text prompt")
    return YOLOEScorer(
        weights=args.yoloe_weights,
        labels=labels,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        min_area_ratio=args.yoloe_min_area_ratio,
        min_span_ratio=args.yoloe_min_span_ratio,
        mobileclip_path=args.mobileclip_path,
    )


def process_video(video_path: Path, out_dir: Path, args: argparse.Namespace, scorer: FrameScorer) -> dict:
    metrics, video_info = analyze_video(
        video_path=video_path,
        scorer=scorer,
        sample_every_sec=args.sample_every_sec,
        max_frames=args.max_frames,
        smooth_window=args.smooth_window,
        cable_threshold=args.cable_threshold,
    )
    video_out = out_dir / safe_stem(video_path)
    video_out.mkdir(parents=True, exist_ok=True)
    save_metrics_csv(video_out / "frame_metrics.csv", metrics)

    segment_metrics = find_segments(
        metrics,
        mode=args.mode,
        min_segment_sec=args.min_segment_sec,
        merge_gap_sec=args.merge_gap_sec,
        expected_samples=args.expected_samples,
    )

    segments: list[Segment] = []
    for idx, segment in enumerate(segment_metrics, start=1):
        sample_id = f"S{idx:03d}"
        sample_dir = video_out / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        selected_records: list[dict] = []
        for rank, metric in enumerate(pick_representatives(segment, args.frames_per_sample), start=1):
            frame = read_frame(video_path, metric.frame_idx)
            selected_records.append(save_selected_frame(frame, sample_dir, sample_id, rank, metric, args, scorer))

        segments.append(
            Segment(
                sample_id=sample_id,
                start_frame=segment[0].frame_idx,
                end_frame=segment[-1].frame_idx,
                start_sec=segment[0].timestamp_sec,
                end_sec=segment[-1].timestamp_sec,
                frame_count=len(segment),
                selected_frames=selected_records,
            )
        )

    manifest = {
        **video_info,
        "mode": args.mode,
        "frames_per_sample": args.frames_per_sample,
        "smooth_window": args.smooth_window,
        "min_segment_sec": args.min_segment_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "expected_samples": args.expected_samples,
        "max_frames": args.max_frames,
        "crop_source": args.crop_source,
        "crop_pad_ratio": args.crop_pad_ratio,
        "crop_min_pad": args.crop_min_pad,
        "device_requested": args.device,
        "device": getattr(scorer, "device", args.device),
        "sam3_prompt": args.sam3_prompt if args.detector == "sam3" else None,
        "sam3_conf": args.sam3_conf if args.detector == "sam3" else None,
        "sam3_min_area_ratio": args.sam3_min_area_ratio if args.detector == "sam3" else None,
        "sam3_min_span_ratio": args.sam3_min_span_ratio if args.detector == "sam3" else None,
        "sam3_checkpoint": str(args.sam3_checkpoint) if args.detector == "sam3" else None,
        "sam3_checkpoint_sha256": args.sam3_checkpoint_sha256 if args.detector == "sam3" else None,
        "sam3_dtype_requested": args.sam3_dtype if args.detector == "sam3" else None,
        "sam3_dtype": getattr(scorer, "dtype", None) if args.detector == "sam3" else None,
        "output_dir": str(video_out),
        "segments": [asdict(segment) for segment in segments],
    }
    manifest_path = video_out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{video_path.name}: found {len(segments)} sample segments, wrote {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/home/nvidia/DATA/UW/video"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sample_frames"))
    parser.add_argument("--detector", choices=["opencv", "yoloe", "sam3"], default="opencv")
    parser.add_argument(
        "--mode",
        choices=["single", "scan"],
        default="single",
        help="single: one processed sample per video. scan: split a long pool scan into sample segments.",
    )
    parser.add_argument("--sample-every-sec", type=float, default=0.25, help="Frame sampling interval for segmentation.")
    parser.add_argument("--frames-per-sample", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--cable-threshold",
        type=float,
        default=None,
        help="Threshold on smoothed cable score. Defaults to 0.04 for SAM3 and 0.22 otherwise.",
    )
    parser.add_argument("--dark-threshold", type=int, default=75, help="Gray value threshold for dark cable pixels.")
    parser.add_argument("--min-aspect", type=float, default=1.8, help="Minimum elongated dark component aspect ratio.")
    parser.add_argument("--min-segment-sec", type=float, default=2.0)
    parser.add_argument("--merge-gap-sec", type=float, default=5.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--resize-width", type=int, default=480, help="Working width for scoring; saved frames stay full resolution.")
    parser.add_argument("--roi", default="0,0,1,1", help="Analysis ROI as x1,y1,x2,y2 ratios.")
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument(
        "--crop-source",
        choices=["full", "yoloe-mask", "sam3-mask"],
        default="full",
        help="full: save full selected frames only. yoloe-mask/sam3-mask: also save detector mask crops.",
    )
    parser.add_argument("--crop-pad-ratio", type=float, default=0.18)
    parser.add_argument("--crop-min-pad", type=int, default=24)
    parser.add_argument("--yoloe-weights", default="yoloe-v8l-seg.pt")
    parser.add_argument("--yoloe-labels", default="beam")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--yoloe-min-area-ratio", type=float, default=0.01)
    parser.add_argument("--yoloe-min-span-ratio", type=float, default=0.20)
    parser.add_argument("--mobileclip-path", type=Path, default=Path("mobileclip_blt.ts"))
    parser.add_argument("--sam3-prompt", default="pipe")
    parser.add_argument("--sam3-conf", type=float, default=0.05)
    parser.add_argument("--sam3-min-area-ratio", type=float, default=0.005)
    parser.add_argument("--sam3-min-span-ratio", type=float, default=0.15)
    parser.add_argument("--sam3-checkpoint", type=Path, default=Path("ckpts/sam3/sam3.pt"))
    parser.add_argument("--sam3-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    args = parser.parse_args()

    if args.cable_threshold is None:
        args.cable_threshold = 0.04 if args.detector == "sam3" else 0.22

    args.sam3_checkpoint_sha256 = None
    if args.detector == "sam3":
        args.sam3_checkpoint = args.sam3_checkpoint.expanduser().resolve()
        if not args.sam3_checkpoint.is_file():
            raise SystemExit(f"SAM3 checkpoint does not exist: {args.sam3_checkpoint}")
        args.sam3_checkpoint_sha256 = file_sha256(args.sam3_checkpoint)

    videos = iter_videos(args.input)
    if not videos:
        raise SystemExit(f"No videos found under {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scorer = build_scorer(args)
    manifests = [process_video(video, args.out_dir, args, scorer) for video in videos]
    summary_path = args.out_dir / "manifest.json"
    summary_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
