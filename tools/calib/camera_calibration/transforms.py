"""Rigid transforms using the convention ``p_A = T_A_B p_B``."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .errors import ConfigurationError


def validate_rotation_matrix(rotation: np.ndarray, tolerance: float = 1e-5) -> None:
    """Require a finite, orthogonal matrix with determinant +1."""
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ConfigurationError(f"Rotation matrix must be 3x3, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ConfigurationError("Rotation matrix contains non-finite values")
    if not np.allclose(value.T @ value, np.eye(3), atol=tolerance):
        raise ConfigurationError("Rotation matrix is not orthogonal")
    determinant = float(np.linalg.det(value))
    if not np.isclose(determinant, 1.0, atol=tolerance):
        raise ConfigurationError(
            f"Rotation matrix determinant must be +1, got {determinant:.8f}"
        )


def validate_transform(transform: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    """Validate a rigid 4x4 homogeneous transform."""
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ConfigurationError(f"Homogeneous transform must be 4x4, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ConfigurationError("Homogeneous transform contains non-finite values")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=tolerance):
        raise ConfigurationError(
            "Homogeneous transform last row must be [0, 0, 0, 1]"
        )
    validate_rotation_matrix(value[:3, :3], tolerance)
    return value


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a checked homogeneous transform."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    validate_rotation_matrix(rotation)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    value = validate_transform(transform)
    rotation = value[:3, :3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ value[:3, 3]
    return result


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert an explicitly xyzw-ordered normalized quaternion to SO(3)."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ConfigurationError(
            "quaternion_xyzw must contain exactly 4 values in xyzw order"
        )
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ConfigurationError("Quaternion xyzw has zero or invalid norm")
    if abs(norm - 1.0) > 1e-3:
        raise ConfigurationError(
            f"Quaternion xyzw must be normalized; norm is {norm:.8f}"
        )
    x, y, z, w = value / norm
    result = np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )
    validate_rotation_matrix(result)
    return result


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert SO(3) to a normalized quaternion in xyzw order."""
    validate_rotation_matrix(rotation)
    rvec, _ = cv2.Rodrigues(np.asarray(rotation, np.float64))
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-14:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rvec.reshape(3) / theta
    result = np.r_[axis * np.sin(theta / 2), np.cos(theta / 2)]
    if result[3] < 0:
        result *= -1
    return result / np.linalg.norm(result)


def pose_vector_to_transform(pose: np.ndarray) -> np.ndarray:
    """Convert ``[rx, ry, rz, tx, ty, tz]`` to a transform."""
    value = np.asarray(pose, dtype=np.float64).reshape(6)
    rotation, _ = cv2.Rodrigues(value[:3])
    return make_transform(rotation, value[3:])


def transform_to_pose_vector(transform: np.ndarray) -> np.ndarray:
    """Convert a transform to Rodrigues rotation plus translation."""
    value = validate_transform(transform)
    rotation, _ = cv2.Rodrigues(value[:3, :3])
    return np.r_[rotation.reshape(3), value[:3, 3]]


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform N points from child to parent coordinates."""
    value = validate_transform(transform)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (value[:3, :3] @ points.T).T + value[:3, 3]


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the geodesic SO(3) distance in degrees."""
    relative = np.asarray(first).reshape(3, 3).T @ np.asarray(second).reshape(3, 3)
    cosine = float(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    return float(np.degrees(np.arccos(cosine)))


def average_quaternions_xyzw(
    quaternions: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Markley mean with quaternion sign disambiguation."""
    values = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    if len(values) == 0:
        raise ConfigurationError("Cannot average an empty quaternion set")
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    reference = values[0]
    values[np.sum(values * reference, axis=1) < 0] *= -1
    if weights is None:
        weights = np.ones(len(values))
    weights = np.asarray(weights, dtype=np.float64).reshape(len(values))
    if np.any(weights < 0) or np.sum(weights) <= 0:
        raise ConfigurationError("Quaternion weights must be non-negative and nonzero")
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quaternion, weight in zip(values, weights):
        accumulator += weight * np.outer(quaternion, quaternion)
    _, vectors = np.linalg.eigh(accumulator)
    result = vectors[:, -1]
    if np.dot(result, reference) < 0:
        result *= -1
    return result / np.linalg.norm(result)


def transform_from_config(value: dict[str, Any], name: str) -> np.ndarray:
    """Parse matrix or translation_m + quaternion_xyzw."""
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping with explicit metre units")
    if "matrix" in value:
        if value.get("length_unit", "metre") not in {"metre", "meter", "m"}:
            raise ConfigurationError(f"{name}.length_unit must be metre")
        return validate_transform(np.asarray(value["matrix"], dtype=np.float64))
    if "translation_m" not in value or "quaternion_xyzw" not in value:
        raise ConfigurationError(
            f"{name} requires matrix or translation_m plus quaternion_xyzw"
        )
    translation = np.asarray(value["translation_m"], dtype=np.float64)
    quaternion = np.asarray(value["quaternion_xyzw"], dtype=np.float64)
    if translation.shape != (3,):
        raise ConfigurationError(f"{name}.translation_m must contain 3 values")
    return make_transform(quaternion_xyzw_to_matrix(quaternion), translation)

