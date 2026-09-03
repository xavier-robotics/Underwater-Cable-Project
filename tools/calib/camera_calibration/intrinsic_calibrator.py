"""In-medium intrinsic calibration with quality/diversity/outlier filtering."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .camera_models import CameraParameters, calibrate_camera_model
from .config import (
    output_path,
    public_config,
    require_keys,
    save_yaml,
    validate_resolution,
)
from .errors import ConfigurationError, DetectionError
from .reporting import (
    prepare_output_directories,
    save_coverage_plot,
    save_observation_images,
    save_reprojection_overlay,
    write_markdown_report,
    write_observations_csv,
)
from .ros_export import save_camera_info_yaml
from .target_detector import (
    TargetObservation,
    TargetSpec,
    collect_target_observations,
    select_diverse_observations,
)
from .video_frames import iter_input_frames


def calibrate_from_observations(
    model: str,
    spec: TargetSpec,
    selected: list[TargetObservation],
    image_size: tuple[int, int],
    camera_name: str = "underwater_camera",
) -> tuple[float, CameraParameters, list[np.ndarray], list[np.ndarray]]:
    """Calibrate a camera from already detected corners (also used by tests)."""
    result = calibrate_camera_model(
        model,
        [spec.object_points() for _ in selected],
        [
            np.asarray(item.image_points, dtype=np.float64).reshape(-1, 2)
            for item in selected
        ],
        image_size,
    )
    rms, matrix, distortion, rotations, translations = result
    return (
        float(rms),
        CameraParameters(
            camera_name,
            image_size[0],
            image_size[1],
            model,
            matrix,
            np.asarray(distortion).reshape(-1),
        ),
        rotations,
        translations,
    )


def reprojection_errors(
    camera: CameraParameters,
    spec: TargetSpec,
    selected: list[TargetObservation],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
) -> list[float]:
    """Calculate and attach per-frame corner RMSE."""
    errors: list[float] = []
    for observation, rotation, translation in zip(
        selected,
        rotations,
        translations,
    ):
        projected = camera.project(spec.object_points(), rotation, translation)
        delta = projected - np.asarray(observation.image_points).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        observation.reprojection_error_px = error
        observation.pose_rvec = np.asarray(rotation, dtype=np.float64).reshape(3)
        observation.pose_tvec = np.asarray(translation, dtype=np.float64).reshape(3)
        errors.append(error)
    return errors


def _outlier_indices(
    errors: list[float],
    config: dict[str, Any],
) -> set[int]:
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        return set()
    fixed = float(config.get("max_reprojection_error_px", np.inf))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    factor = float(config.get("mad_factor", 3.5))
    robust = median + factor * 1.4826 * mad if mad > 1e-12 else np.inf
    threshold = min(fixed, robust)
    return {index for index, value in enumerate(values) if value > threshold}


def calibrate_intrinsics(config: dict[str, Any]) -> dict[str, Any]:
    """Run the complete intrinsic workflow and write auditable output."""
    require_keys(
        config,
        [
            "input.type",
            "camera.model",
            "camera.image_width",
            "camera.image_height",
            "target.inner_corners_cols",
            "target.inner_corners_rows",
            "target.square_size_m",
        ],
    )
    image_size = validate_resolution(config["camera"])
    model = str(config["camera"]["model"]).lower()
    spec = TargetSpec.from_config(config["target"])
    quality = config.get("quality", {})
    minimum = int(quality.get("min_valid_frames", 20))
    maximum = int(quality.get("max_valid_frames", 60))
    frames = list(iter_input_frames(config))
    if not frames:
        raise DetectionError("Input produced no candidate frames")
    for frame in frames:
        actual = (frame.image.shape[1], frame.image.shape[0])
        if actual != image_size:
            raise DetectionError(
                f"Frame {frame.name} is {actual[0]}x{actual[1]}, but config is "
                f"{image_size[0]}x{image_size[1]}; calibration never resizes/crops"
            )
    observations = collect_target_observations(frames, spec, quality)
    selected = select_diverse_observations(
        observations,
        minimum,
        maximum,
        float(quality.get("duplicate_distance", 0.06)),
    )
    if len(selected) < minimum:
        raise DetectionError(
            f"Intrinsic calibration requires {minimum}-{maximum} diverse valid "
            f"frames; got {len(selected)} from {len(observations)} candidates "
            f"({sum(item.detected for item in observations)} target detections). "
            "No result was written."
        )

    rejection = config.get("outlier_rejection", quality)
    rounds = min(2, max(0, int(rejection.get("max_rounds", 2))))
    camera: CameraParameters | None = None
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    rms = 0.0
    for round_index in range(rounds + 1):
        rms, camera, rotations, translations = calibrate_from_observations(
            model,
            spec,
            selected,
            image_size,
            str(config["camera"].get("camera_name", "underwater_camera")),
        )
        errors = reprojection_errors(
            camera,
            spec,
            selected,
            rotations,
            translations,
        )
        if round_index == rounds:
            break
        outliers = _outlier_indices(errors, rejection)
        if not outliers:
            break
        if len(selected) - len(outliers) < minimum:
            raise DetectionError(
                f"Outlier rejection would leave {len(selected) - len(outliers)} "
                f"frames, below min_valid_frames={minimum}; no result was written"
            )
        kept: list[TargetObservation] = []
        for index, observation in enumerate(selected):
            if index in outliers:
                observation.selected = False
                observation.accepted = False
                observation.rejection_reason = (
                    f"reprojection_outlier_round_{round_index + 1}"
                )
            else:
                kept.append(observation)
        selected = kept
    assert camera is not None
    errors = reprojection_errors(
        camera,
        spec,
        selected,
        rotations,
        translations,
    )

    output_dir = output_path(config, "output/intrinsics")
    directories = prepare_output_directories(
        output_dir,
        ["detected", "rejected", "undistorted", "reprojection"],
    )
    save_observation_images(observations, spec, directories)
    for item, rotation, translation in zip(selected, rotations, translations):
        projected = camera.project(spec.object_points(), rotation, translation)
        save_reprojection_overlay(
            directories["reprojection"] / f"{item.frame.name}.jpg",
            item.frame.image,
            np.asarray(item.image_points),
            projected,
        )
        if not cv2.imwrite(
            str(directories["undistorted"] / f"{item.frame.name}.jpg"),
            camera.undistort_image(item.frame.image),
        ):
            raise DetectionError(f"Failed to save undistorted frame {item.frame.name}")
    save_coverage_plot(output_dir / "coverage.png", observations, image_size)
    write_observations_csv(output_dir / "observations.csv", observations)

    rejected_count = len(observations) - len(selected)
    distortion_model = "equidistant" if model == "fisheye" else "plumb_bob"
    environment = str(
        config["camera"].get("calibration_environment", "underwater")
    ).lower()
    if environment not in {"underwater", "air"}:
        raise ConfigurationError(
            "camera.calibration_environment must be underwater or air"
        )
    payload = {
        "camera_name": camera.camera_name,
        "calibration_environment": environment,
        "units": {"length": "metre", "image_error": "pixel"},
        "image_width": camera.width,
        "image_height": camera.height,
        "camera_model": model,
        "distortion_model": distortion_model,
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": camera.camera_matrix.reshape(-1).tolist(),
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(camera.distortion.size),
            "data": camera.distortion.tolist(),
        },
        "statistics": {
            "rms_reprojection_error_px": rms,
            "mean_reprojection_error_px": float(np.mean(errors)),
            "median_reprojection_error_px": float(np.median(errors)),
            "max_reprojection_error_px": float(np.max(errors)),
            "valid_frame_count": len(selected),
            "rejected_frame_count": rejected_count,
        },
        "target": {
            "type": spec.target_type,
            "inner_corners_cols": spec.cols,
            "inner_corners_rows": spec.rows,
            "square_size_m": spec.square_size_m,
            "target_origin": spec.target_origin,
            "axes": "+x across columns, +y down rows, +z right-handed",
        },
        "configuration": {
            "fixed_focus": bool(config["camera"].get("fixed_focus", True)),
            "image_scaling": "none",
            "electronic_crop": "none",
            "source_config": public_config(config),
        },
    }
    result_file = save_yaml(output_dir / "intrinsics.yaml", payload)
    save_camera_info_yaml(output_dir / "camera_info.yaml", camera)
    write_markdown_report(
        output_dir / "report.md",
        f"{environment.title()} Intrinsic Calibration",
        [
            (
                "Result",
                f"- Camera: `{camera.camera_name}`\n"
                f"- Calibration environment: `{environment}`\n"
                f"- Model: `{model}` / `{distortion_model}`\n"
                f"- Resolution: {camera.width} × {camera.height} (no resize/crop)\n"
                f"- Valid frames: {len(selected)}\n"
                f"- Rejected candidates: {rejected_count}",
            ),
            (
                "Reprojection error",
                f"- OpenCV RMS: {rms:.4f} px\n"
                f"- Mean per-frame RMSE: {np.mean(errors):.4f} px\n"
                f"- Median per-frame RMSE: {np.median(errors):.4f} px\n"
                f"- Maximum per-frame RMSE: {np.max(errors):.4f} px",
            ),
            (
                "Use constraints",
                "Valid only in the recorded calibration environment while lens, "
                "focus, resolution, crop, electronic zoom, and camera assembly "
                "remain unchanged. Inspect "
                "`coverage.png`, `reprojection/`, and `undistorted/`.",
            ),
        ],
    )
    return {
        "result_file": str(result_file),
        "output_dir": str(output_dir),
        "camera": camera,
        "statistics": payload["statistics"],
    }
