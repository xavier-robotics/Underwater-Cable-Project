from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from camera_calibration.calibration import UnderwaterCameraCalibration
from camera_calibration.config import save_yaml
from camera_calibration.errors import ConfigurationError

from tests.helpers import camera


class RuntimeCompositionTests(unittest.TestCase):
    def _files(
        self,
        root: Path,
        *,
        camera_frame: str = "camera_optical_frame",
        inverse_ok: bool = True,
        length_unit: str = "metre",
        environment: str = "underwater",
    ) -> tuple[Path, Path]:
        cam = camera()
        intrinsics = root / "intrinsics.yaml"
        extrinsics = root / "extrinsics.yaml"
        save_yaml(
            intrinsics,
            {
                "camera_name": cam.camera_name,
                "calibration_environment": environment,
                "image_width": cam.width,
                "image_height": cam.height,
                "camera_model": cam.model,
                "units": {"length": length_unit},
                "camera_matrix": {
                    "rows": 3,
                    "cols": 3,
                    "data": cam.camera_matrix.reshape(-1).tolist(),
                },
                "distortion_coefficients": {
                    "rows": 1,
                    "cols": 5,
                    "data": cam.distortion.tolist(),
                },
            },
        )
        t_base_camera = np.eye(4)
        t_base_camera[:3, 3] = [0.2, -0.1, 0.05]
        t_camera_base = np.linalg.inv(t_base_camera)
        if not inverse_ok:
            t_camera_base = np.eye(4)
        save_yaml(
            extrinsics,
            {
                "parent_frame": "base_link",
                "child_frame": camera_frame,
                "transform_convention": (
                    "p_parent = T_parent_child * p_child"
                ),
                "units": {"translation": "metre"},
                "T_base_camera": t_base_camera.tolist(),
                "T_camera_base": t_camera_base.tolist(),
            },
        )
        return intrinsics, extrinsics

    def test_load_two_files_and_generate_base_ray(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = self._files(Path(directory))
            calibration = UnderwaterCameraCalibration.load(*files)
            ray = calibration.pixel_to_base_ray(320.0, 240.0)
            np.testing.assert_allclose(
                ray.origin,
                [0.2, -0.1, 0.05],
                atol=1e-9,
            )
            np.testing.assert_allclose(ray.direction, [0, 0, 1], atol=1e-9)
            self.assertEqual(
                calibration.summary()["projection_assumption"],
                "central_camera_in_calibration_medium",
            )

    def test_non_optical_child_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = self._files(Path(directory), camera_frame="camera_link")
            with self.assertRaisesRegex(ConfigurationError, "optical frame"):
                UnderwaterCameraCalibration.load(*files)

    def test_bad_inverse_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = self._files(Path(directory), inverse_ok=False)
            with self.assertRaisesRegex(ConfigurationError, "not the inverse"):
                UnderwaterCameraCalibration.load(*files)

    def test_non_metric_unit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = self._files(Path(directory), length_unit="millimetre")
            with self.assertRaisesRegex(ConfigurationError, "length unit"):
                UnderwaterCameraCalibration.load(*files)

    def test_invalid_calibration_environment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = self._files(Path(directory), environment="vacuum")
            with self.assertRaisesRegex(
                ConfigurationError,
                "calibration_environment",
            ):
                UnderwaterCameraCalibration.load(*files)


if __name__ == "__main__":
    unittest.main()
