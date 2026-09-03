"""Reliable RTSP snapshot, frame-sequence, and video capture helpers."""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAFE_PREFIX = re.compile(r"^[A-Za-z0-9_-]+$")


class CaptureError(RuntimeError):
    """Raised when a camera stream or output cannot be captured safely."""


@dataclass(frozen=True)
class CameraConfig:
    """Connection and retry settings for one RTSP camera."""

    host: str
    password: str
    username: str = "admin"
    port: int = 554
    stream_path: str = "/h264_stream"
    transport: str = "tcp"
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    reconnect_attempts: int = 3
    reconnect_delay_sec: float = 1.0

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("Camera host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("Camera port must be between 1 and 65535")
        if not self.stream_path.startswith("/"):
            raise ValueError("RTSP stream path must start with '/'")
        if self.transport not in {"tcp", "udp"}:
            raise ValueError("RTSP transport must be tcp or udp")
        if self.open_timeout_ms <= 0 or self.read_timeout_ms <= 0:
            raise ValueError("Camera timeouts must be positive")
        if self.reconnect_attempts < 0:
            raise ValueError("Reconnect attempts must not be negative")
        if self.reconnect_delay_sec < 0:
            raise ValueError("Reconnect delay must not be negative")

    def rtsp_url(self) -> str:
        """Return the authenticated URL without logging it anywhere."""
        username = quote(self.username, safe="")
        password = quote(self.password, safe="")
        host = self.host.strip()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return (
            f"rtsp://{username}:{password}@{host}:{self.port}"
            f"{self.stream_path}"
        )

    def address(self) -> str:
        """Return a credential-free address suitable for diagnostics."""
        return f"{self.host.strip()}:{self.port}{self.stream_path}"


@dataclass(frozen=True)
class CaptureResult:
    """Summary of a completed capture operation."""

    output_path: Path
    frame_count: int
    width: int
    height: int
    duration_sec: float
    fps: float | None = None


class RtspCapture:
    """OpenCV RTSP reader with bounded reconnect behavior."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: Any | None = None

    def open(self) -> None:
        """Open the configured stream once."""
        self.close()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.config.transport}"
        )
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self.config.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self.config.read_timeout_ms,
        ]
        capture = cv2.VideoCapture(
            self.config.rtsp_url(),
            cv2.CAP_FFMPEG,
            params,
        )
        if not capture.isOpened():
            capture.release()
            raise CaptureError(
                f"Cannot open RTSP camera {self.config.address()}"
            )
        self._capture = capture

    def read(self) -> np.ndarray:
        """Read one frame, reconnecting a bounded number of times on failure."""
        last_error = "stream did not return a frame"
        attempts = self.config.reconnect_attempts + 1
        for attempt in range(attempts):
            try:
                if self._capture is None:
                    self.open()
                assert self._capture is not None
                ok, frame = self._capture.read()
                if ok and frame is not None and frame.size:
                    return frame
                last_error = "stream did not return a frame"
            except CaptureError as exc:
                last_error = str(exc)
            self.close()
            if attempt + 1 < attempts and self.config.reconnect_delay_sec:
                time.sleep(self.config.reconnect_delay_sec)
        raise CaptureError(
            f"Failed to read RTSP camera {self.config.address()} after "
            f"{attempts} attempt(s): {last_error}"
        )

    def fps(self, fallback: float = 25.0) -> float:
        """Return the stream frame rate or a validated fallback."""
        if self._capture is not None:
            value = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if np.isfinite(value) and value > 0:
                return value
        return fallback

    def close(self) -> None:
        """Release the current decoder, if any."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> RtspCapture:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def timestamp_token(moment: datetime | None = None) -> str:
    """Return an ASCII-only local timestamp suitable for filenames."""
    moment = moment or datetime.now().astimezone()
    return moment.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def default_output_path(
    output_dir: Path,
    prefix: str,
    extension: str,
    moment: datetime | None = None,
) -> Path:
    """Build a timestamped output path."""
    if not SAFE_PREFIX.fullmatch(prefix):
        raise ValueError("Output prefix must contain only ASCII letters, digits, _ or -")
    extension = extension.lower().lstrip(".")
    if not extension or not extension.isalnum():
        raise ValueError("Output extension must be alphanumeric")
    return output_dir / f"{prefix}_{timestamp_token(moment)}.{extension}"


def save_snapshot(
    config: CameraConfig,
    output_path: Path,
    *,
    quality: int = 95,
    warmup_frames: int = 2,
    overwrite: bool = False,
) -> CaptureResult:
    """Capture and atomically save one JPEG or PNG image."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    if warmup_frames < 0:
        raise ValueError("Warm-up frame count must not be negative")
    output_path = _prepare_output_file(
        output_path,
        SUPPORTED_IMAGE_EXTENSIONS,
        overwrite,
    )
    started = time.monotonic()
    with RtspCapture(config) as stream:
        frame = stream.read()
        for _ in range(warmup_frames):
            frame = stream.read()
    _write_image_atomic(output_path, frame, quality)
    height, width = frame.shape[:2]
    return CaptureResult(
        output_path,
        1,
        width,
        height,
        time.monotonic() - started,
    )


def record_video(
    config: CameraConfig,
    output_path: Path,
    *,
    seconds: float,
    codec: str = "mp4v",
    fps: float | None = None,
    overwrite: bool = False,
) -> CaptureResult:
    """Decode and record a fixed-duration silent video at native resolution."""
    if seconds <= 0:
        raise ValueError("Recording duration must be positive")
    if len(codec) != 4 or not codec.isascii():
        raise ValueError("Video codec must be a four-character ASCII code")
    if fps is not None and fps <= 0:
        raise ValueError("Video frame rate must be positive")
    output_path = _prepare_output_file(
        output_path,
        {".avi", ".mkv", ".mp4"},
        overwrite,
    )
    temporary = _temporary_output(output_path)
    writer: Any | None = None
    frame_count = 0
    try:
        with RtspCapture(config) as stream:
            frame = stream.read()
            height, width = frame.shape[:2]
            output_fps = fps or stream.fps()
            writer = cv2.VideoWriter(
                str(temporary),
                cv2.VideoWriter_fourcc(*codec),
                output_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise CaptureError(
                    f"Cannot create video {output_path} with codec {codec!r}"
                )
            target_frames = max(1, int(round(seconds * output_fps)))
            recording_started = time.monotonic()
            while frame_count < target_frames:
                if frame.shape[:2] != (height, width):
                    raise CaptureError(
                        "Camera resolution changed during recording: "
                        f"expected {width}x{height}, got "
                        f"{frame.shape[1]}x{frame.shape[0]}"
                    )
                elapsed = time.monotonic() - recording_started
                expected_frames = min(
                    target_frames,
                    int(elapsed * output_fps) + 1,
                )
                while frame_count < expected_frames:
                    writer.write(frame)
                    frame_count += 1
                if frame_count >= target_frames:
                    break
                frame = stream.read()
        writer.release()
        writer = None
        temporary.replace(output_path)
    except Exception:
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
        raise
    return CaptureResult(
        output_path,
        frame_count,
        width,
        height,
        frame_count / output_fps,
        output_fps,
    )


def capture_frames(
    config: CameraConfig,
    output_dir: Path,
    *,
    interval_sec: float,
    count: int,
    image_format: str = "jpg",
    prefix: str = "frame",
    quality: int = 95,
    overwrite: bool = False,
) -> CaptureResult:
    """Save a timestamped image sequence at a wall-clock interval."""
    if interval_sec <= 0:
        raise ValueError("Frame interval must be positive")
    if count <= 0:
        raise ValueError("Frame count must be positive")
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    if not SAFE_PREFIX.fullmatch(prefix):
        raise ValueError("Frame prefix must contain only ASCII letters, digits, _ or -")
    image_format = image_format.lower().lstrip(".")
    if f".{image_format}" not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Frame format must be jpg, jpeg, or png")
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = timestamp_token()
    started = time.monotonic()
    next_due = started
    saved = 0
    width = height = 0
    with RtspCapture(config) as stream:
        while saved < count:
            frame = stream.read()
            now = time.monotonic()
            if now < next_due:
                continue
            path = output_dir / (
                f"{prefix}_{batch}_{saved + 1:06d}.{image_format}"
            )
            path = _prepare_output_file(
                path,
                SUPPORTED_IMAGE_EXTENSIONS,
                overwrite,
            )
            _write_image_atomic(path, frame, quality)
            height, width = frame.shape[:2]
            saved += 1
            next_due = now + interval_sec
    return CaptureResult(
        output_dir,
        saved,
        width,
        height,
        time.monotonic() - started,
    )


def _prepare_output_file(
    path: Path,
    allowed_extensions: set[str],
    overwrite: bool,
) -> Path:
    path = path.expanduser()
    if path.suffix.lower() not in allowed_extensions:
        values = ", ".join(sorted(allowed_extensions))
        raise CaptureError(f"Unsupported output extension {path.suffix!r}; use {values}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise CaptureError(f"Output already exists: {path}; pass --overwrite to replace it")
    return path


def _temporary_output(path: Path) -> Path:
    return path.with_name(
        f".{path.stem}.{uuid.uuid4().hex}.part{path.suffix}"
    )


def _write_image_atomic(path: Path, frame: np.ndarray, quality: int) -> None:
    temporary = _temporary_output(path)
    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, quality]
        if path.suffix.lower() in {".jpg", ".jpeg"}
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    try:
        if not cv2.imwrite(str(temporary), frame, parameters):
            raise CaptureError(f"OpenCV failed to write image: {path}")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
