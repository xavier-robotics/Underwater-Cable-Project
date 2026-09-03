"""ROS-compatible text/YAML exports without a ROS dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera_models import CameraParameters
from .config import save_yaml
from .errors import ConfigurationError


def save_camera_info_yaml(path: Path, camera: CameraParameters) -> Path:
    """Write a sensor_msgs/CameraInfo-compatible YAML file."""
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:, :3] = camera.camera_matrix
    return save_yaml(
        path,
        {
            "image_width": camera.width,
            "image_height": camera.height,
            "camera_name": camera.camera_name,
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": camera.camera_matrix.reshape(-1).tolist(),
            },
            "distortion_model": (
                "equidistant" if camera.model == "fisheye" else "plumb_bob"
            ),
            "distortion_coefficients": {
                "rows": 1,
                "cols": int(camera.distortion.size),
                "data": camera.distortion.tolist(),
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": np.eye(3).reshape(-1).tolist(),
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": projection.reshape(-1).tolist(),
            },
        },
    )


def static_tf_command(
    translation: np.ndarray,
    quaternion_xyzw: np.ndarray,
    parent_frame: str,
    child_frame: str,
) -> str:
    """Create a directionally explicit ROS 2 static TF command."""
    if not parent_frame or not child_frame or parent_frame == child_frame:
        raise ConfigurationError("Static TF parent and child must be distinct names")
    x, y, z = np.asarray(translation, dtype=float).reshape(3)
    qx, qy, qz, qw = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    return (
        "ros2 run tf2_ros static_transform_publisher "
        f"--x {x:.10g} --y {y:.10g} --z {z:.10g} "
        f"--qx {qx:.10g} --qy {qy:.10g} --qz {qz:.10g} --qw {qw:.10g} "
        f"--frame-id {parent_frame} --child-frame-id {child_frame}"
    )
