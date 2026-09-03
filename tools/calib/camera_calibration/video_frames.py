"""Shared image-directory, video, and live-camera acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from .config import resolve_path
from .errors import ConfigurationError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass
class SourceFrame:
    """One interval-sampled frame plus audit metadata."""

    image: np.ndarray
    source: str
    frame_index: int
    timestamp_sec: float
    name: str


def _configured_paths(
    config: dict[str, Any],
    input_config: dict[str, Any],
) -> list[Path]:
    if input_config.get("paths") is not None:
        values = input_config["paths"]
        if not isinstance(values, list) or not values:
            raise ConfigurationError("input.paths must be a non-empty list")
    elif input_config.get("path") is not None:
        values = [input_config["path"]]
    else:
        raise ConfigurationError("input.path or input.paths is required")
    return [resolve_path(config, value) for value in values]


def _iter_images(paths: list[Path]) -> Iterator[SourceFrame]:
    index = 0
    for source in paths:
        if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
            files = [source]
        elif source.is_dir():
            files = sorted(
                item
                for item in source.rglob("*")
                if item.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                raise ConfigurationError(f"Image directory is empty: {source}")
        else:
            raise ConfigurationError(f"Image input does not exist: {source}")
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ConfigurationError(f"Cannot read image: {path}")
            yield SourceFrame(
                image=image,
                source=str(path),
                frame_index=index,
                timestamp_sec=0.0,
                name=f"image_{index:06d}",
            )
            index += 1


def _iter_video(path: Path, interval_sec: float) -> Iterator[SourceFrame]:
    if not path.is_file():
        raise ConfigurationError(f"Video input does not exist: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ConfigurationError(f"Unsupported video extension {path.suffix!r}: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ConfigurationError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ConfigurationError(f"Video reports an invalid frame rate: {path}")
    step = max(1, int(round(fps * interval_sec)))
    index = 0
    try:
        while True:
            ok = capture.grab()
            if not ok:
                break
            if index % step == 0:
                ok, image = capture.retrieve()
                if not ok or image is None:
                    raise ConfigurationError(
                        f"Failed to decode frame {index} from {path}"
                    )
                timestamp = index / fps
                yield SourceFrame(
                    image=image,
                    source=str(path),
                    frame_index=index,
                    timestamp_sec=timestamp,
                    name=f"{path.stem}_f{index:06d}_t{timestamp:08.2f}",
                )
            index += 1
    finally:
        capture.release()


def _iter_camera(input_config: dict[str, Any]) -> Iterator[SourceFrame]:
    device = int(input_config.get("device", 0))
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        raise ConfigurationError(f"Cannot open live camera device {device}")
    if input_config.get("image_width") is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(input_config["image_width"]))
    if input_config.get("image_height") is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(input_config["image_height"]))
    interval_sec = float(input_config.get("frame_interval_sec", 0.5))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps * interval_sec)))
    max_frames = int(input_config.get("max_capture_frames", 600))
    try:
        for index in range(max_frames):
            ok, image = capture.read()
            if not ok or image is None:
                raise ConfigurationError(
                    f"Failed to capture frame {index} from camera {device}"
                )
            if index % step == 0:
                yield SourceFrame(
                    image=image,
                    source=f"camera:{device}",
                    frame_index=index,
                    timestamp_sec=index / fps,
                    name=f"camera_{device}_f{index:06d}",
                )
    finally:
        capture.release()


def iter_input_frames(
    config: dict[str, Any],
    input_config: dict[str, Any] | None = None,
) -> Iterator[SourceFrame]:
    """Yield candidate frames without resizing or cropping."""
    input_config = input_config or config.get("input", {})
    input_type = str(input_config.get("type", "video")).lower()
    interval_sec = float(input_config.get("frame_interval_sec", 0.5))
    if interval_sec <= 0:
        raise ConfigurationError("input.frame_interval_sec must be positive")
    if input_type in {"images", "image_directory", "directory"}:
        yield from _iter_images(_configured_paths(config, input_config))
        return
    if input_type == "video":
        for path in _configured_paths(config, input_config):
            yield from _iter_video(path, interval_sec)
        return
    if input_type in {"camera", "live"}:
        yield from _iter_camera(input_config)
        return
    if input_type in {"ros", "rosbag", "topic"}:
        raise ConfigurationError(
            "ROS input was requested, but project.ros_version is none. Export the "
            "topic or bag to lossless images/video before calibration."
        )
    raise ConfigurationError(
        f"Unsupported input.type {input_type!r}; use video, images, or camera"
    )
