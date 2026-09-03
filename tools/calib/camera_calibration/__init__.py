"""Underwater camera calibration toolkit.

The transform convention used everywhere is ``p_A = T_A_B @ p_B``.  Lengths
are metres unless the field name explicitly contains another unit suffix.
"""

from .calibration import (
    CameraCalibration,
    Ray,
    UnderwaterCameraCalibration,
    transform_ray,
)

__all__ = [
    "CameraCalibration",
    "Ray",
    "UnderwaterCameraCalibration",
    "transform_ray",
]

__version__ = "1.1.0"
