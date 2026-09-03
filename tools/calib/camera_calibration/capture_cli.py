"""Command-line interface for direct RTSP camera capture."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Sequence

from .capture import (
    CameraConfig,
    CaptureError,
    CaptureResult,
    capture_frames,
    default_output_path,
    record_video,
    save_snapshot,
)
from .preview import browser_preview


DEFAULT_OUTPUT_DIR = Path("outputs/camera_capture")
DEFAULT_PASSWORD_FILE = Path(__file__).resolve().parents[1] / ".camera_password"


def build_parser() -> argparse.ArgumentParser:
    """Create the camera-capture CLI parser."""
    parser = argparse.ArgumentParser(
        prog="camera_capture",
        description=(
            "Preview or capture snapshots, videos, and sampled frames "
            "from an RTSP camera"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="save one image")
    _add_connection_arguments(snapshot)
    snapshot.add_argument("--output", type=Path)
    snapshot.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    snapshot.add_argument("--format", choices=("jpg", "png"), default="jpg")
    snapshot.add_argument("--quality", type=_quality, default=95)
    snapshot.add_argument("--warmup-frames", type=_non_negative_int, default=2)
    snapshot.add_argument("--overwrite", action="store_true")

    record = subparsers.add_parser("record", help="record a fixed-duration video")
    _add_connection_arguments(record)
    record.add_argument("--seconds", type=_positive_float, required=True)
    record.add_argument("--output", type=Path)
    record.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    record.add_argument("--codec", default="mp4v")
    record.add_argument("--fps", type=_positive_float)
    record.add_argument("--overwrite", action="store_true")

    frames = subparsers.add_parser(
        "frames",
        help="save images at a fixed wall-clock interval",
    )
    _add_connection_arguments(frames)
    frames.add_argument("--interval", type=_positive_float, default=1.0)
    frames.add_argument("--count", type=_positive_int, default=30)
    frames.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    frames.add_argument("--format", choices=("jpg", "jpeg", "png"), default="jpg")
    frames.add_argument("--prefix", default="frame")
    frames.add_argument("--quality", type=_quality, default=95)
    frames.add_argument("--overwrite", action="store_true")

    preview = subparsers.add_parser(
        "preview",
        help="serve a live camera preview in a web browser",
    )
    _add_connection_arguments(preview)
    preview.add_argument(
        "--bind",
        default="127.0.0.1",
        help="HTTP bind address (default: local machine only)",
    )
    preview.add_argument("--web-port", type=_positive_int, default=8765)
    preview.add_argument("--fps", type=_positive_float, default=10.0)
    preview.add_argument("--quality", type=_quality, default=80)
    preview.add_argument("--max-width", type=_positive_int, default=1280)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one capture command and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _camera_config(args)
        if args.command == "preview":
            browser_preview(
                config,
                bind_host=args.bind,
                port=args.web_port,
                fps=args.fps,
                quality=args.quality,
                max_width=args.max_width,
            )
            return 0
        result = _dispatch(args, config)
    except (CaptureError, ValueError) as exc:
        parser.exit(2, f"camera_capture: error: {exc}\n")
    _print_result(args.command, result)
    return 0


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        default=os.environ.get("CAMERA_HOST"),
        help="camera IP/hostname (or CAMERA_HOST)",
    )
    parser.add_argument("--port", type=int, default=554)
    parser.add_argument(
        "--stream-path",
        default=os.environ.get("CAMERA_STREAM_PATH", "/h264_stream"),
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("CAMERA_USERNAME", "admin"),
    )
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--open-timeout-ms", type=_positive_int, default=5000)
    parser.add_argument("--read-timeout-ms", type=_positive_int, default=5000)
    parser.add_argument("--reconnect-attempts", type=_non_negative_int, default=3)
    parser.add_argument("--reconnect-delay", type=_non_negative_float, default=1.0)


def _camera_config(args: argparse.Namespace) -> CameraConfig:
    if not args.host:
        raise CaptureError("Pass --host or set CAMERA_HOST")
    password = _camera_password()
    return CameraConfig(
        host=args.host,
        port=args.port,
        stream_path=args.stream_path,
        username=args.username,
        password=password,
        transport=args.transport,
        open_timeout_ms=args.open_timeout_ms,
        read_timeout_ms=args.read_timeout_ms,
        reconnect_attempts=args.reconnect_attempts,
        reconnect_delay_sec=args.reconnect_delay,
    )


def _camera_password() -> str:
    password = os.environ.get("CAMERA_PASSWORD")
    if password is not None:
        return password

    configured_path = os.environ.get("CAMERA_PASSWORD_FILE")
    password_path = Path(configured_path) if configured_path else DEFAULT_PASSWORD_FILE
    try:
        password = password_path.read_text(encoding="utf-8").rstrip("\r\n")
    except FileNotFoundError:
        password = None
    except OSError as exc:
        raise CaptureError(f"Cannot read camera password file {password_path}: {exc}") from exc

    if password:
        return password
    if password == "":
        raise CaptureError(f"Camera password file is empty: {password_path}")
    if not sys.stdin.isatty():
        raise CaptureError(
            "Set CAMERA_PASSWORD or create the local .camera_password file; "
            "passwords are intentionally not accepted as command-line arguments"
        )
    return getpass.getpass("Camera password: ")


def _dispatch(args: argparse.Namespace, config: CameraConfig) -> CaptureResult:
    if args.command == "snapshot":
        output = args.output or default_output_path(
            args.output_dir,
            "snapshot",
            args.format,
        )
        return save_snapshot(
            config,
            output,
            quality=args.quality,
            warmup_frames=args.warmup_frames,
            overwrite=args.overwrite,
        )
    if args.command == "record":
        output = args.output or default_output_path(
            args.output_dir,
            "video",
            "mp4",
        )
        return record_video(
            config,
            output,
            seconds=args.seconds,
            codec=args.codec,
            fps=args.fps,
            overwrite=args.overwrite,
        )
    if args.command == "frames":
        return capture_frames(
            config,
            args.output_dir,
            interval_sec=args.interval,
            count=args.count,
            image_format=args.format,
            prefix=args.prefix,
            quality=args.quality,
            overwrite=args.overwrite,
        )
    raise CaptureError(f"Unknown command: {args.command}")


def _print_result(command: str, result: CaptureResult) -> None:
    dimensions = f"{result.width}x{result.height}"
    if command == "record":
        print(
            f"Saved video: {result.output_path} "
            f"({dimensions}, {result.frame_count} frames, "
            f"{result.duration_sec:.2f}s, {result.fps:.3f} fps)"
        )
    elif command == "frames":
        print(
            f"Saved {result.frame_count} frames in {result.output_path} "
            f"({dimensions}, {result.duration_sec:.2f}s)"
        )
    else:
        print(
            f"Saved snapshot: {result.output_path} "
            f"({dimensions}, {result.duration_sec:.2f}s)"
        )


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _quality(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
