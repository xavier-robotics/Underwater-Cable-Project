"""Synthetic fixtures shared by calibration tests."""

from __future__ import annotations

import cv2
import numpy as np

from camera_calibration.camera_models import CameraParameters
from camera_calibration.target_detector import TargetObservation, TargetSpec
from camera_calibration.video_frames import SourceFrame


def camera() -> CameraParameters:
    """Return a small deterministic pinhole camera."""
    return CameraParameters(
        "synthetic",
        640,
        480,
        "pinhole",
        np.array(
            [[520.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        np.array([-0.04, 0.012, 0.0005, -0.0003, 0.001]),
    )


def spec(cols: int = 8, rows: int = 6) -> TargetSpec:
    """Return a checkerboard target."""
    return TargetSpec("checkerboard", cols, rows, 0.04)


def observation(
    name: str,
    image_points: np.ndarray,
    *,
    center_xy: tuple[float, float] = (0.5, 0.5),
    area_ratio: float = 0.1,
    tilt_score: float = 0.1,
) -> TargetObservation:
    """Construct an accepted synthetic corner observation."""
    frame = SourceFrame(
        np.full((480, 640, 3), 128, dtype=np.uint8),
        "synthetic",
        int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else 0,
        0.0,
        name,
    )
    value = TargetObservation(
        frame,
        np.asarray(image_points, dtype=np.float64),
        True,
        500.0,
        128.0,
        0.0,
        0.0,
        center_xy=center_xy,
        area_ratio=area_ratio,
        angle_deg=0.0,
        tilt_score=tilt_score,
        coverage_cells=[4],
        selected=True,
        accepted=True,
    )
    return value


def projected_observations(
    count: int = 24,
) -> tuple[CameraParameters, TargetSpec, list[TargetObservation]]:
    """Generate non-degenerate synthetic pinhole corner observations."""
    cam = camera()
    target = spec()
    rng = np.random.default_rng(13)
    observations: list[TargetObservation] = []
    for index in range(count):
        rotation = np.array(
            [
                rng.uniform(-0.35, 0.35),
                rng.uniform(-0.30, 0.30),
                rng.uniform(-0.20, 0.20),
            ]
        )
        translation = np.array(
            [
                rng.uniform(-0.18, 0.12),
                rng.uniform(-0.13, 0.10),
                rng.uniform(0.75, 1.5),
            ]
        )
        points = cam.project(target.object_points(), rotation, translation)
        points += rng.normal(0.0, 0.03, points.shape)
        center = np.mean(points, axis=0) / [cam.width, cam.height]
        observations.append(
            observation(
                f"view_{index}",
                points,
                center_xy=(float(center[0]), float(center[1])),
                area_ratio=float(
                    cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
                    / (cam.width * cam.height)
                ),
                tilt_score=abs(rotation[0]) + abs(rotation[1]),
            )
        )
    return cam, target, observations
