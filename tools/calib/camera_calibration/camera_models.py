"""Strictly separated pinhole and fisheye camera models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import (
    load_yaml,
    validate_format_version,
    validate_length_units,
)
from .errors import ConfigurationError


@dataclass(frozen=True)
class CameraParameters:
    """Fixed effective camera parameters in the calibrated medium."""

    camera_name: str
    width: int
    height: int
    model: str
    camera_matrix: np.ndarray
    distortion: np.ndarray

    def __post_init__(self) -> None:
        model = str(self.model).lower()
        if model not in {"pinhole", "fisheye"}:
            raise ConfigurationError(f"Unsupported camera model: {model!r}")
        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError(
                f"Invalid camera resolution {self.width}x{self.height}"
            )
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ConfigurationError("camera_matrix must be a finite 3x3 matrix")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ConfigurationError("Camera focal lengths fx and fy must be positive")
        expected = 4 if model == "fisheye" else 5
        if distortion.size != expected:
            raise ConfigurationError(
                f"{model} requires {expected} distortion coefficients, got "
                f"{distortion.size}; pinhole and fisheye formats cannot be mixed"
            )
        if not np.all(np.isfinite(distortion)):
            raise ConfigurationError("Distortion coefficients must be finite")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion", distortion)

    @classmethod
    def load(cls, path: Path | str) -> "CameraParameters":
        """Load and validate an intrinsic result file."""
        data = load_yaml(path)
        validate_format_version(data, path)
        validate_length_units(data, path)
        matrix = data.get("camera_matrix", {}).get("data")
        distortion = data.get("distortion_coefficients", {}).get("data")
        if matrix is None or distortion is None:
            raise ConfigurationError(
                f"Missing camera_matrix or distortion_coefficients in {path}"
            )
        try:
            camera_matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        except ValueError as exc:
            raise ConfigurationError(
                f"camera_matrix.data in {path} must contain 9 values"
            ) from exc
        return cls(
            camera_name=str(data.get("camera_name", "underwater_camera")),
            width=int(data.get("image_width", 0)),
            height=int(data.get("image_height", 0)),
            model=str(data.get("camera_model", "")),
            camera_matrix=camera_matrix,
            distortion=np.asarray(distortion, dtype=np.float64),
        )

    def project(
        self,
        points: np.ndarray,
        rotation_vector: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        """Project 3-D points through the matching OpenCV camera API."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rotation = np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1)
        translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
        if self.model == "fisheye":
            image_points, _ = cv2.fisheye.projectPoints(
                points.reshape(-1, 1, 3),
                rotation,
                translation,
                self.camera_matrix,
                self.distortion.reshape(4, 1),
            )
        else:
            image_points, _ = cv2.projectPoints(
                points,
                rotation,
                translation,
                self.camera_matrix,
                self.distortion.reshape(5, 1),
            )
        return image_points.reshape(-1, 2)

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Convert raw pixels to normalized camera coordinates."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == "fisheye":
            normalized = cv2.fisheye.undistortPoints(
                pixels,
                self.camera_matrix,
                self.distortion.reshape(4, 1),
            )
        else:
            normalized = cv2.undistortPoints(
                pixels,
                self.camera_matrix,
                self.distortion.reshape(5, 1),
            )
        return normalized.reshape(-1, 2)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image without crossing camera-model APIs."""
        if self.model == "fisheye":
            return cv2.fisheye.undistortImage(
                image,
                self.camera_matrix,
                self.distortion.reshape(4, 1),
                Knew=self.camera_matrix,
            )
        return cv2.undistort(
            image,
            self.camera_matrix,
            self.distortion.reshape(5, 1),
        )


def calibrate_camera_model(
    model: str,
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Calibrate with conservative coefficients for the selected model."""
    if len(object_points) != len(image_points) or not object_points:
        raise ConfigurationError("Calibration point lists must be non-empty and aligned")
    model = str(model).lower()
    if model == "pinhole":
        return cv2.calibrateCamera(
            [np.asarray(item, np.float32) for item in object_points],
            [np.asarray(item, np.float32) for item in image_points],
            image_size,
            None,
            None,
            flags=0,
        )
    if model == "fisheye":
        objects = [
            np.asarray(item, np.float64).reshape(1, -1, 3)
            for item in object_points
        ]
        images = [
            np.asarray(item, np.float64).reshape(1, -1, 2)
            for item in image_points
        ]
        matrix = np.array(
            [
                [max(image_size), 0.0, image_size[0] / 2],
                [0.0, max(image_size), image_size[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1), dtype=np.float64)
        result = cv2.fisheye.calibrate(
            objects,
            images,
            image_size,
            matrix,
            distortion,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-7,
            ),
        )
        return result
    raise ConfigurationError(f"Unsupported camera model: {model!r}")
