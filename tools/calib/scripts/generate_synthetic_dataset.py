#!/usr/bin/env python
"""Generate deterministic central-camera calibration videos.

The data exercise the dedicated-camera workflow and are not production
parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from camera_calibration.camera_models import CameraParameters
from camera_calibration.config import save_yaml
from camera_calibration.target_detector import TargetSpec
from camera_calibration.transforms import make_transform


WIDTH = 960
HEIGHT = 540
FPS = 5.0
TARGET = TargetSpec("checkerboard", 8, 6, 0.04)


def synthetic_camera() -> CameraParameters:
    """Return the known effective in-medium camera used for rendering."""
    return CameraParameters(
        "underwater_camera",
        WIDTH,
        HEIGHT,
        "pinhole",
        np.array(
            [[760.0, 0.0, WIDTH / 2], [0.0, 755.0, HEIGHT / 2], [0, 0, 1]]
        ),
        np.array([-0.035, 0.012, 0.0004, -0.0002, 0.001]),
    )


def pose_for_image_center(
    camera: CameraParameters,
    center_uv: tuple[float, float],
    distance_m: float,
    rotation_vector: np.ndarray,
) -> np.ndarray:
    """Build T_camera_target with its inner-corner centroid near a pixel."""
    rotation, _ = cv2.Rodrigues(
        np.asarray(rotation_vector, dtype=np.float64)
    )
    target_center = np.mean(TARGET.object_points(), axis=0)
    normalized = np.array(
        [
            (center_uv[0] - camera.camera_matrix[0, 2])
            / camera.camera_matrix[0, 0],
            (center_uv[1] - camera.camera_matrix[1, 2])
            / camera.camera_matrix[1, 1],
            1.0,
        ]
    )
    desired_center = normalized * distance_m
    translation = desired_center - rotation @ target_center
    return make_transform(rotation, translation)


def _board_grid(spec: TargetSpec) -> tuple[np.ndarray, int, int]:
    x_values = np.arange(-1, spec.cols + 1, dtype=float) * spec.square_size_m
    y_values = np.arange(-1, spec.rows + 1, dtype=float) * spec.square_size_m
    points = np.array(
        [[x, y, 0.0] for y in y_values for x in x_values],
        dtype=np.float64,
    )
    return points, len(x_values), len(y_values)


def _draw_board(
    projected: np.ndarray,
    grid_width: int,
    grid_height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scale = 2
    canvas = np.full((HEIGHT * scale, WIDTH * scale, 3), 105, np.uint8)
    points = projected.reshape(grid_height, grid_width, 2) * scale
    for row in range(grid_height - 1):
        for col in range(grid_width - 1):
            polygon = np.array(
                [
                    points[row, col],
                    points[row, col + 1],
                    points[row + 1, col + 1],
                    points[row + 1, col],
                ],
                dtype=np.int32,
            )
            color = 232 if (row + col) % 2 == 0 else 18
            cv2.fillConvexPoly(
                canvas,
                polygon,
                (color, color, color),
                lineType=cv2.LINE_AA,
            )
    canvas = cv2.resize(
        canvas,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    noise = rng.normal(0.0, 1.2, canvas.shape)
    return np.clip(canvas.astype(float) + noise, 0, 255).astype(np.uint8)


def render_camera(
    camera: CameraParameters,
    transform: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render a checkerboard using the effective camera model."""
    grid, grid_width, grid_height = _board_grid(TARGET)
    rotation, _ = cv2.Rodrigues(transform[:3, :3])
    pixels = camera.project(grid, rotation, transform[:3, 3])
    return _draw_board(pixels, grid_width, grid_height, rng)


def write_video(path: Path, images: list[np.ndarray]) -> None:
    """Write an MJPG AVI that OpenCV can reliably decode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create synthetic video: {path}")
    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()


def diverse_poses(
    camera: CameraParameters,
    count: int,
    *,
    distances: tuple[float, ...],
    phase: float = 0.0,
) -> list[np.ndarray]:
    """Create visible center/edge/corner, distance, and tilt coverage."""
    centers = [
        (0.24, 0.27),
        (0.50, 0.22),
        (0.76, 0.27),
        (0.22, 0.50),
        (0.50, 0.50),
        (0.78, 0.50),
        (0.25, 0.73),
        (0.50, 0.75),
        (0.75, 0.73),
    ]
    poses: list[np.ndarray] = []
    for index in range(count):
        center = centers[index % len(centers)]
        distance = distances[index % len(distances)]
        angle = phase + index * 0.73
        rotation = np.array(
            [
                0.22 * np.sin(angle),
                0.20 * np.cos(angle * 1.17),
                0.16 * np.sin(angle * 0.61),
            ]
        )
        poses.append(
            pose_for_image_center(
                camera,
                (center[0] * WIDTH, center[1] * HEIGHT),
                distance,
                rotation,
            )
        )
    return poses


def _input_quality(minimum: int) -> dict[str, object]:
    return {
        "min_valid_frames": minimum,
        "max_valid_frames": 60,
        "blur_threshold": 40,
        "reject_overexposed": True,
        "reject_underexposed": True,
        "min_board_area_ratio": 0.004,
        "duplicate_distance": 0.008,
    }


def generate(root: Path) -> dict[str, Path]:
    """Generate videos, two calibration configs, validation, and truth."""
    root = root.resolve()
    data = root / "data"
    configs = root / "config"
    outputs = root / "output"
    configs.mkdir(parents=True, exist_ok=True)
    camera = synthetic_camera()
    rng = np.random.default_rng(20260730)

    intrinsic_poses = diverse_poses(
        camera,
        34,
        distances=(0.78, 0.95, 1.2, 1.45),
    )
    intrinsic_video = data / "underwater_intrinsics.avi"
    write_video(
        intrinsic_video,
        [render_camera(camera, pose, rng) for pose in intrinsic_poses],
    )

    base_camera_rotation, _ = cv2.Rodrigues(
        np.array([0.05, -0.08, 0.04])
    )
    t_base_camera = make_transform(
        base_camera_rotation,
        np.array([0.15, -0.06, 0.10]),
    )
    fixture_camera_poses = [
        pose_for_image_center(
            camera,
            (WIDTH * 0.38, HEIGHT * 0.42),
            0.92,
            np.array([0.10, -0.12, 0.03]),
        ),
        pose_for_image_center(
            camera,
            (WIDTH * 0.63, HEIGHT * 0.43),
            1.15,
            np.array([-0.12, 0.16, -0.05]),
        ),
        pose_for_image_center(
            camera,
            (WIDTH * 0.49, HEIGHT * 0.62),
            1.38,
            np.array([0.15, 0.08, 0.10]),
        ),
    ]
    fixture_datasets = []
    for index, camera_target in enumerate(fixture_camera_poses, start=1):
        name = f"pose_{index:02d}"
        video = data / "extrinsics" / f"{name}.avi"
        base_image = render_camera(camera, camera_target, rng)
        images = [
            np.clip(
                base_image.astype(float) + rng.normal(0, 0.6, base_image.shape),
                0,
                255,
            ).astype(np.uint8)
            for _ in range(7)
        ]
        write_video(video, images)
        t_base_target = t_base_camera @ camera_target
        fixture_datasets.append(
            {
                "name": name,
                "video": f"../data/extrinsics/{name}.avi",
                "frame_interval_sec": 0.2,
                "T_base_target": {
                    "length_unit": "metre",
                    "matrix": t_base_target.tolist(),
                },
            }
        )

    validation_video = data / "validation" / "underwater_intrinsics.avi"
    write_video(
        validation_video,
        [
            render_camera(camera, pose, rng)
            for pose in diverse_poses(
                camera,
                7,
                distances=(0.85, 1.1, 1.35),
                phase=4.2,
            )
        ],
    )

    target_config = {
        "type": "checkerboard",
        "inner_corners_cols": TARGET.cols,
        "inner_corners_rows": TARGET.rows,
        "square_size_m": TARGET.square_size_m,
        "target_origin": TARGET.target_origin,
    }
    save_yaml(
        configs / "intrinsics.yaml",
        {
            "mode": "intrinsic",
            "input": {
                "type": "video",
                "path": "../data/underwater_intrinsics.avi",
                "frame_interval_sec": 0.2,
            },
            "camera": {
                "camera_name": camera.camera_name,
                "model": camera.model,
                "image_width": WIDTH,
                "image_height": HEIGHT,
                "fixed_focus": True,
                "calibration_environment": "underwater",
            },
            "target": target_config,
            "quality": _input_quality(20),
            "outlier_rejection": {
                "max_reprojection_error_px": 1.5,
                "mad_factor": 3.5,
                "max_rounds": 2,
            },
            "output": {"directory": "../output/intrinsics"},
        },
    )
    save_yaml(
        configs / "extrinsics.yaml",
        {
            "mode": "extrinsic",
            "intrinsics_file": "../output/intrinsics/intrinsics.yaml",
            "frames": {
                "base_frame": "base_link",
                "camera_mechanical_frame": "camera_link",
                "camera_frame": "camera_optical_frame",
                "target_frame": "calibration_target",
            },
            "target": target_config,
            "datasets": fixture_datasets,
            "quality": {
                "min_frames_per_dataset": 5,
                "max_frames_per_dataset": 7,
                "min_total_observations": 6,
                "blur_threshold": 40,
                "min_board_area_ratio": 0.004,
                "duplicate_distance": 0.0,
            },
            "outlier_rejection": {
                "max_reprojection_error_px": 2.0,
                "max_translation_residual_mm": 20.0,
                "max_rotation_residual_deg": 2.0,
            },
            "optimization": {
                "enabled": True,
                "robust_loss": "huber",
                "f_scale_px": 1.0,
                "max_iterations": 200,
            },
            "physical_bounds": {
                "translation_min_m": [-0.5, -0.5, -0.5],
                "translation_max_m": [0.5, 0.5, 0.5],
            },
            "output": {"directory": "../output/extrinsics"},
        },
    )
    save_yaml(
        configs / "validation.yaml",
        {
            "mode": "validation",
            "intrinsics_file": "../output/intrinsics/intrinsics.yaml",
            "extrinsics_file": "../output/extrinsics/extrinsics.yaml",
            "target": target_config,
            "intrinsic_validation": {
                "input": {
                    "type": "video",
                    "path": "../data/validation/underwater_intrinsics.avi",
                    "frame_interval_sec": 0.2,
                },
                "quality": {
                    **_input_quality(3),
                    "duplicate_distance": 0.0,
                },
                "max_mean_reprojection_error_px": 1.0,
                "max_corner_reprojection_error_px": 1.5,
            },
            "extrinsic_validation": {
                "max_mean_reprojection_error_px": 1.0,
                "max_mean_translation_residual_mm": 10.0,
                "max_mean_rotation_residual_deg": 1.0,
                "physical_bounds": {
                    "translation_min_m": [-0.5, -0.5, -0.5],
                    "translation_max_m": [0.5, 0.5, 0.5],
                },
                "datasets": [
                    {
                        **fixture_datasets[0],
                        "min_valid_frames": 1,
                        "max_valid_frames": 3,
                    }
                ],
                "quality": {
                    "blur_threshold": 40,
                    "min_board_area_ratio": 0.004,
                    "duplicate_distance": 0.0,
                },
            },
            "output": {"directory": "../output/validation"},
        },
    )
    save_yaml(
        root / "truth.yaml",
        {
            "units": {"length": "metre"},
            "camera_matrix": camera.camera_matrix.tolist(),
            "distortion": camera.distortion.tolist(),
            "T_base_camera": t_base_camera.tolist(),
        },
    )
    return {
        "root": root,
        "intrinsics_config": configs / "intrinsics.yaml",
        "extrinsics_config": configs / "extrinsics.yaml",
        "validation_config": configs / "validation.yaml",
        "output": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_calibration"),
    )
    args = parser.parse_args()
    paths = generate(args.output)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
