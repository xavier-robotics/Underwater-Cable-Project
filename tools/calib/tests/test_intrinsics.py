from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_calibration.camera_models import CameraParameters
from camera_calibration.errors import ConfigurationError, DetectionError
from camera_calibration.intrinsic_calibrator import (
    calibrate_from_observations,
    calibrate_intrinsics,
    reprojection_errors,
)

from tests.helpers import projected_observations


class IntrinsicCalibrationTests(unittest.TestCase):
    def test_known_intrinsics_recovery(self) -> None:
        expected, spec, observations = projected_observations()
        rms, actual, rotations, translations = calibrate_from_observations(
            "pinhole",
            spec,
            observations,
            (expected.width, expected.height),
        )
        errors = reprojection_errors(
            actual,
            spec,
            observations,
            rotations,
            translations,
        )
        self.assertLess(rms, 0.2)
        self.assertLess(float(np.mean(errors)), 0.2)
        self.assertLess(
            abs(actual.camera_matrix[0, 0] - expected.camera_matrix[0, 0]),
            8.0,
        )
        self.assertLess(
            abs(actual.camera_matrix[1, 1] - expected.camera_matrix[1, 1]),
            8.0,
        )
        self.assertLess(
            np.linalg.norm(
                actual.camera_matrix[:2, 2] - expected.camera_matrix[:2, 2]
            ),
            8.0,
        )

    def test_pinhole_fisheye_coefficients_cannot_mix(self) -> None:
        matrix = np.eye(3)
        matrix[0, 0] = matrix[1, 1] = 500
        with self.assertRaisesRegex(ConfigurationError, "cannot be mixed"):
            CameraParameters("bad", 640, 480, "fisheye", matrix, np.zeros(5))
        with self.assertRaisesRegex(ConfigurationError, "cannot be mixed"):
            CameraParameters("bad", 640, 480, "pinhole", matrix, np.zeros(4))

    def test_insufficient_valid_frames_fails_without_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            cv2.imwrite(
                str(image_dir / "blank.png"),
                np.full((480, 640, 3), 128, np.uint8),
            )
            config = {
                "_config_dir": str(root),
                "input": {
                    "type": "images",
                    "path": "images",
                    "frame_interval_sec": 0.5,
                },
                "camera": {
                    "model": "pinhole",
                    "image_width": 640,
                    "image_height": 480,
                },
                "target": {
                    "type": "checkerboard",
                    "inner_corners_cols": 8,
                    "inner_corners_rows": 6,
                    "square_size_m": 0.05,
                },
                "quality": {
                    "min_valid_frames": 2,
                    "max_valid_frames": 3,
                },
                "output": {"directory": "result"},
            }
            with self.assertRaisesRegex(DetectionError, "requires"):
                calibrate_intrinsics(config)
            self.assertFalse((root / "result" / "intrinsics.yaml").exists())


if __name__ == "__main__":
    unittest.main()
