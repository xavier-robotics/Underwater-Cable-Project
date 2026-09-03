"""Validated runtime composition of intrinsics and mechanical extrinsics."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .camera_models import CameraParameters
from .config import (
    load_yaml,
    validate_format_version,
    validate_length_units,
)
from .errors import ConfigurationError
from .transforms import invert_transform, validate_transform


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ray:
    """A normalized central-camera ray expressed in one coordinate frame."""

    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        direction = np.asarray(self.direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
            raise ConfigurationError("Ray contains non-finite values")
        if norm < 1e-12:
            raise ConfigurationError("Ray direction must be nonzero")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", direction / norm)


def transform_ray(transform: np.ndarray, ray: Ray) -> Ray:
    """Transform a ray with ``p_parent = T_parent_child * p_child``."""
    value = validate_transform(transform)
    return Ray(
        value[:3, :3] @ ray.origin + value[:3, 3],
        value[:3, :3] @ ray.direction,
    )


class UnderwaterCameraCalibration:
    """Effective in-medium intrinsics plus a fixed mechanical transform.

    A dedicated sealed camera is treated as one calibrated central imaging
    unit.  No user-visible housing or refractive-port parameters are required.
    """

    def __init__(
        self,
        camera: CameraParameters,
        t_base_camera: np.ndarray,
        base_frame: str,
        camera_frame: str,
        calibration_environment: str,
    ) -> None:
        self.camera = camera
        self.t_base_camera = validate_transform(t_base_camera)
        self.base_frame = str(base_frame)
        self.camera_frame = str(camera_frame)
        self.calibration_environment = str(calibration_environment)

    @classmethod
    def load(
        cls,
        intrinsics_file: Path | str,
        extrinsics_file: Path | str,
    ) -> "UnderwaterCameraCalibration":
        """Load two versioned result files and reject transform mismatches."""
        camera = CameraParameters.load(intrinsics_file)
        intrinsics = load_yaml(intrinsics_file)
        extrinsics = load_yaml(extrinsics_file)
        validate_format_version(extrinsics, extrinsics_file)
        validate_length_units(extrinsics, extrinsics_file)
        if extrinsics.get("transform_convention") != (
            "p_parent = T_parent_child * p_child"
        ):
            raise ConfigurationError(
                "Unsupported or missing extrinsics transform_convention"
            )
        try:
            t_base_camera = validate_transform(
                np.asarray(extrinsics["T_base_camera"], dtype=np.float64)
            )
            t_camera_base = validate_transform(
                np.asarray(extrinsics["T_camera_base"], dtype=np.float64)
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Missing transform in {extrinsics_file}: {exc}"
            ) from exc
        if not np.allclose(
            t_camera_base,
            invert_transform(t_base_camera),
            atol=1e-7,
        ):
            raise ConfigurationError(
                "T_camera_base is not the inverse of T_base_camera"
            )
        base_frame = str(extrinsics.get("parent_frame", ""))
        camera_frame = str(extrinsics.get("child_frame", ""))
        if not base_frame or not camera_frame:
            raise ConfigurationError(
                "Extrinsics parent_frame and child_frame are required"
            )
        if "optical" not in camera_frame.lower():
            raise ConfigurationError(
                f"Pixel rays require an optical frame, got {camera_frame!r}"
            )
        environment = str(
            intrinsics.get("calibration_environment", "underwater")
        ).lower()
        if environment not in {"underwater", "air"}:
            raise ConfigurationError(
                "intrinsics calibration_environment must be underwater or air"
            )
        result = cls(
            camera,
            t_base_camera,
            base_frame,
            camera_frame,
            environment,
        )
        LOGGER.info(
            "Loaded %s %dx%d %s (%s), %s -> %s",
            camera.camera_name,
            camera.width,
            camera.height,
            camera.model,
            environment,
            camera_frame,
            base_frame,
        )
        return result

    def pixel_to_camera_ray(self, u: float, v: float) -> Ray:
        """Back-project a raw pixel using the calibrated in-medium model."""
        pixel = np.asarray([u, v], dtype=np.float64)
        if not np.all(np.isfinite(pixel)):
            raise ConfigurationError("Pixel coordinates must be finite")
        normalized = self.camera.undistort_pixels(pixel.reshape(1, 2))[0]
        return Ray(np.zeros(3), np.r_[normalized, 1.0])

    def pixel_to_base_ray(self, u: float, v: float) -> Ray:
        """Back-project a pixel and apply the fixed ``T_base_camera``."""
        return transform_ray(
            self.t_base_camera,
            self.pixel_to_camera_ray(u, v),
        )

    def summary(self) -> dict[str, Any]:
        """Return runtime metadata suitable for reports and logs."""
        return {
            "camera_name": self.camera.camera_name,
            "resolution": [self.camera.width, self.camera.height],
            "camera_model": self.camera.model,
            "calibration_environment": self.calibration_environment,
            "projection_assumption": "central_camera_in_calibration_medium",
            "base_frame": self.base_frame,
            "camera_frame": self.camera_frame,
            "length_unit": "metre",
        }


# Shorter name for new integrations; retain the established public name.
CameraCalibration = UnderwaterCameraCalibration
