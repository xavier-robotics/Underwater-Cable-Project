from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from camera_calibration.config import load_yaml, save_yaml
from camera_calibration.errors import ConfigurationError
from camera_calibration.validation import validate_calibration


class ValidationGateTests(unittest.TestCase):
    def test_implausible_intrinsics_fail_overall_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intrinsics = root / "intrinsics.yaml"
            extrinsics = root / "extrinsics.yaml"
            matrix = np.array(
                [[20.0, 0.0, 320.0], [0.0, 20.0, 240.0], [0.0, 0.0, 1.0]]
            )
            save_yaml(
                intrinsics,
                {
                    "camera_name": "synthetic",
                    "image_width": 640,
                    "image_height": 480,
                    "camera_model": "pinhole",
                    "units": {"length": "metre"},
                    "camera_matrix": {
                        "rows": 3,
                        "cols": 3,
                        "data": matrix.reshape(-1).tolist(),
                    },
                    "distortion_coefficients": {
                        "rows": 1,
                        "cols": 5,
                        "data": [0.0] * 5,
                    },
                },
            )
            save_yaml(
                extrinsics,
                {
                    "parent_frame": "base_link",
                    "child_frame": "camera_optical_frame",
                    "transform_convention": (
                        "p_parent = T_parent_child * p_child"
                    ),
                    "units": {"translation": "metre"},
                    "coordinate_frames": {
                        "camera_optical_frame": (
                            "optical: +x right, +y down, +z forward"
                        )
                    },
                    "T_base_camera": np.eye(4).tolist(),
                    "T_camera_base": np.eye(4).tolist(),
                },
            )
            config = {
                "_config_dir": str(root),
                "mode": "validation",
                "intrinsics_file": intrinsics.name,
                "extrinsics_file": extrinsics.name,
                "output": {"directory": "validation"},
            }
            with self.assertRaisesRegex(ConfigurationError, "validation failed"):
                validate_calibration(config)
            result = load_yaml(root / "validation" / "validation.yaml")
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["intrinsics"]["passed"])
            self.assertTrue(result["extrinsics"]["passed"])
            self.assertNotIn("housing", result)


if __name__ == "__main__":
    unittest.main()
