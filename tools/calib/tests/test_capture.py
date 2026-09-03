from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from camera_calibration.capture import (
    CameraConfig,
    CaptureError,
    default_output_path,
    timestamp_token,
)
from camera_calibration.capture_cli import build_parser, main
from camera_calibration.preview import encode_preview_frame

import cv2
import numpy as np


class CameraCaptureTests(unittest.TestCase):
    def test_rtsp_url_escapes_credentials(self) -> None:
        config = CameraConfig(
            host="camera.local",
            username="user@example.com",
            password="p@ss:/word",
        )
        self.assertEqual(
            config.rtsp_url(),
            "rtsp://user%40example.com:p%40ss%3A%2Fword@"
            "camera.local:554/h264_stream",
        )
        self.assertNotIn(config.password, config.address())

    def test_camera_config_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "host"):
            CameraConfig(host="", password="secret")
        with self.assertRaisesRegex(ValueError, "transport"):
            CameraConfig(host="camera", password="secret", transport="invalid")
        with self.assertRaisesRegex(ValueError, "port"):
            CameraConfig(host="camera", password="secret", port=0)

    def test_timestamped_output_is_ascii_and_deterministic(self) -> None:
        moment = datetime(2026, 8, 4, 16, 30, 12, 345000, tzinfo=timezone.utc)
        self.assertEqual(timestamp_token(moment), "20260804_163012_345")
        self.assertEqual(
            default_output_path(Path("captures"), "snapshot", "jpg", moment),
            Path("captures/snapshot_20260804_163012_345.jpg"),
        )

    def test_parser_accepts_all_four_commands(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["snapshot", "--host", "camera"]).command, "snapshot")
        self.assertEqual(
            parser.parse_args(["record", "--host", "camera", "--seconds", "2"]).seconds,
            2.0,
        )
        self.assertEqual(
            parser.parse_args(["frames", "--host", "camera", "--count", "3"]).count,
            3,
        )
        preview = parser.parse_args(["preview", "--host", "camera"])
        self.assertEqual(preview.command, "preview")
        self.assertEqual(preview.bind, "127.0.0.1")
        self.assertEqual(preview.web_port, 8765)

    def test_preview_frame_is_resized_and_jpeg_encoded(self) -> None:
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        encoded = encode_preview_frame(frame, quality=75, max_width=100)
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[:2], (50, 100))

    def test_preview_frame_rejects_invalid_options(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "quality"):
            encode_preview_frame(frame, quality=0)
        with self.assertRaisesRegex(ValueError, "width"):
            encode_preview_frame(frame, max_width=0)

    def test_local_password_file_avoids_interactive_prompt(self) -> None:
        from camera_calibration import capture_cli

        with tempfile.TemporaryDirectory() as directory:
            password_path = Path(directory) / "camera_password"
            password_path.write_text("123456\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                capture_cli,
                "DEFAULT_PASSWORD_FILE",
                password_path,
            ), mock.patch("sys.stdin.isatty", return_value=False):
                self.assertEqual(capture_cli._camera_password(), "123456")

    def test_non_interactive_capture_requires_password_configuration(self) -> None:
        from camera_calibration import capture_cli

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "sys.stdin.isatty",
            return_value=False,
        ), mock.patch.object(
            capture_cli,
            "DEFAULT_PASSWORD_FILE",
            Path("/does/not/exist/camera_password"),
        ), self.assertRaises(SystemExit) as raised:
            main(["snapshot", "--host", "camera"])
        self.assertEqual(raised.exception.code, 2)

    def test_existing_output_is_not_overwritten_by_default(self) -> None:
        from camera_calibration.capture import save_snapshot

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.jpg"
            path.write_bytes(b"existing")
            with self.assertRaisesRegex(CaptureError, "already exists"):
                save_snapshot(
                    CameraConfig(host="camera", password="secret"),
                    path,
                )


if __name__ == "__main__":
    unittest.main()
