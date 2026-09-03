"""Checkerboard detection, image quality, and diverse-frame selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .errors import ConfigurationError
from .video_frames import SourceFrame


@dataclass(frozen=True)
class TargetSpec:
    """Physical target; cols are horizontal and rows are vertical."""

    target_type: str
    cols: int
    rows: int
    square_size_m: float
    target_origin: str = "first_inner_corner"

    @classmethod
    def from_config(cls, target: dict[str, Any]) -> "TargetSpec":
        """Validate checkerboard/ChArUco configuration."""
        target_type = str(target.get("type", "checkerboard")).lower()
        if target_type not in {"checkerboard", "charuco"}:
            raise ConfigurationError(f"Unsupported target.type: {target_type}")
        cols = int(target.get("inner_corners_cols", 0))
        rows = int(target.get("inner_corners_rows", 0))
        size = float(target.get("square_size_m", 0.0))
        if cols < 2 or rows < 2:
            raise ConfigurationError(
                "target.inner_corners_cols (horizontal) and "
                "inner_corners_rows (vertical) must both be at least 2"
            )
        if size <= 0:
            raise ConfigurationError("target.square_size_m must be positive metres")
        origin = str(target.get("target_origin", "first_inner_corner"))
        if origin != "first_inner_corner":
            raise ConfigurationError(
                "Only target_origin: first_inner_corner is supported; +x follows "
                "columns, +y follows rows, and +z completes the right-handed frame"
            )
        return cls(target_type, cols, rows, size, origin)

    def object_points(self) -> np.ndarray:
        """Return row-major inner corners in target coordinates."""
        points = np.zeros((self.rows * self.cols, 3), dtype=np.float64)
        points[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        points[:, :2] *= self.square_size_m
        return points


@dataclass
class TargetObservation:
    """Target detection plus quality/diversity metadata."""

    frame: SourceFrame
    image_points: np.ndarray | None
    detected: bool
    blur_score: float
    mean_intensity: float
    overexposed_ratio: float
    underexposed_ratio: float
    center_xy: tuple[float, float] = (0.0, 0.0)
    area_ratio: float = 0.0
    angle_deg: float = 0.0
    tilt_score: float = 0.0
    coverage_cells: list[int] = field(default_factory=list)
    selected: bool = False
    accepted: bool = False
    rejection_reason: str = ""
    reprojection_error_px: float | None = None
    pose_rvec: np.ndarray | None = None
    pose_tvec: np.ndarray | None = None

    def feature_vector(self) -> np.ndarray:
        """Dimensionless pose/coverage proxy used for de-duplication."""
        return np.array(
            [
                self.center_xy[0],
                self.center_xy[1],
                np.sqrt(max(self.area_ratio, 0.0)),
                np.sin(np.radians(self.angle_deg)),
                np.cos(np.radians(self.angle_deg)),
                self.tilt_score,
            ],
            dtype=np.float64,
        )


def image_quality(image: np.ndarray) -> tuple[float, float, float, float]:
    """Return Laplacian sharpness, mean intensity, and clipping ratios."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ConfigurationError("Input image must be a BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        float(np.mean(gray)),
        float(np.mean(gray >= 250)),
        float(np.mean(gray <= 5)),
    )


def _checkerboard_points(gray: np.ndarray, spec: TargetSpec) -> np.ndarray | None:
    pattern = (spec.cols, spec.rows)
    corners: np.ndarray | None = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, value = cv2.findChessboardCornersSB(gray, pattern, flags)
        if found:
            corners = value
    if corners is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, value = cv2.findChessboardCorners(gray, pattern, flags)
        if found:
            corners = cv2.cornerSubPix(
                gray,
                value,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    40,
                    1e-3,
                ),
            )
    return None if corners is None else corners.reshape(-1, 2).astype(np.float64)


def _charuco_points(gray: np.ndarray, spec: TargetSpec) -> np.ndarray | None:
    del gray, spec
    if not hasattr(cv2, "aruco"):
        raise ConfigurationError(
            "target.type=charuco requires cv2.aruco; install a matching "
            "opencv-contrib-python-headless build"
        )
    raise ConfigurationError(
        "ChArUco is reserved but requires dictionary, marker_size_m, and board "
        "dimensions in a future configuration version"
    )


def detect_target(frame: SourceFrame, spec: TargetSpec) -> TargetObservation:
    """Detect one target and calculate coverage and tilt proxies."""
    height, width = frame.image.shape[:2]
    blur, mean, over, under = image_quality(frame.image)
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    points = (
        _checkerboard_points(gray, spec)
        if spec.target_type == "checkerboard"
        else _charuco_points(gray, spec)
    )
    observation = TargetObservation(
        frame,
        points,
        points is not None,
        blur,
        mean,
        over,
        under,
    )
    if points is None:
        observation.rejection_reason = "target_not_detected"
        return observation
    center = np.mean(points, axis=0)
    hull = cv2.convexHull(points.astype(np.float32))
    top = np.linalg.norm(points[spec.cols - 1] - points[0])
    bottom = np.linalg.norm(points[-1] - points[-spec.cols])
    left = np.linalg.norm(points[-spec.cols] - points[0])
    right = np.linalg.norm(points[-1] - points[spec.cols - 1])
    row_vector = points[spec.cols - 1] - points[0]
    observation.center_xy = (float(center[0] / width), float(center[1] / height))
    observation.area_ratio = float(cv2.contourArea(hull) / (width * height))
    observation.angle_deg = float(
        np.degrees(np.arctan2(row_vector[1], row_vector[0]))
    )
    observation.tilt_score = float(
        max(
            abs(top - bottom) / max(top, bottom, 1e-9),
            abs(left - right) / max(left, right, 1e-9),
        )
    )
    observation.coverage_cells = sorted(
        {
            min(2, int(point[0] / width * 3))
            + 3 * min(2, int(point[1] / height * 3))
            for point in points
        }
    )
    return observation


def apply_quality_rules(
    observation: TargetObservation,
    quality: dict[str, Any],
) -> None:
    """Record every quality rejection reason."""
    if not observation.detected:
        return
    if observation.blur_score < float(quality.get("blur_threshold", 80.0)):
        observation.rejection_reason = "blur"
        return
    if bool(quality.get("reject_overexposed", True)) and (
        observation.overexposed_ratio
        > float(quality.get("max_overexposed_ratio", 0.20))
    ):
        observation.rejection_reason = "overexposed"
        return
    if bool(quality.get("reject_underexposed", True)) and (
        observation.underexposed_ratio
        > float(quality.get("max_underexposed_ratio", 0.20))
    ):
        observation.rejection_reason = "underexposed"
        return
    if observation.area_ratio < float(quality.get("min_board_area_ratio", 0.005)):
        observation.rejection_reason = "insufficient_coverage"
        return
    observation.accepted = True


def _feature_distance(
    first: TargetObservation,
    second: TargetObservation,
) -> float:
    weights = np.array([2.2, 2.2, 1.8, 0.35, 0.35, 1.2])
    return float(
        np.linalg.norm((first.feature_vector() - second.feature_vector()) * weights)
    )


def select_diverse_observations(
    observations: list[TargetObservation],
    minimum: int,
    maximum: int,
    duplicate_distance: float = 0.12,
) -> list[TargetObservation]:
    """Reject duplicate poses then greedily maximize feature/cell coverage."""
    if minimum <= 0 or maximum < minimum:
        raise ConfigurationError(
            f"Invalid valid-frame range: min={minimum}, max={maximum}"
        )
    candidates = [item for item in observations if item.accepted]
    candidates.sort(
        key=lambda item: (len(item.coverage_cells), item.tilt_score, item.blur_score),
        reverse=True,
    )
    unique: list[TargetObservation] = []
    for candidate in candidates:
        if unique and min(
            _feature_distance(candidate, previous) for previous in unique
        ) < duplicate_distance:
            candidate.accepted = False
            candidate.rejection_reason = "duplicate_pose"
        else:
            unique.append(candidate)
    if len(unique) < minimum:
        return []
    selected = [max(unique, key=lambda item: item.blur_score)]
    remaining = [item for item in unique if item is not selected[0]]
    cells = set(selected[0].coverage_cells)
    while remaining and len(selected) < maximum:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                min(
                    _feature_distance(remaining[index], old)
                    for old in selected
                )
                + 0.08
                * len(set(remaining[index].coverage_cells) - cells)
                + 0.03
                * min(remaining[index].blur_score / 500.0, 1.0)
            ),
        )
        best = remaining.pop(best_index)
        selected.append(best)
        cells.update(best.coverage_cells)
    for item in selected:
        item.selected = True
    for item in remaining:
        item.accepted = False
        item.rejection_reason = "diversity_limit"
    return selected


def collect_target_observations(
    frames: list[SourceFrame],
    spec: TargetSpec,
    quality: dict[str, Any],
) -> list[TargetObservation]:
    """Detect and quality-check all interval-sampled candidates."""
    observations: list[TargetObservation] = []
    for frame in frames:
        observation = detect_target(frame, spec)
        apply_quality_rules(observation, quality)
        observations.append(observation)
    return observations
