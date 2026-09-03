"""Independent validation for intrinsic and mechanical extrinsic results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import UnderwaterCameraCalibration
from .camera_models import CameraParameters
from .config import (
    load_yaml,
    output_path,
    require,
    resolve_path,
    save_yaml,
    validate_format_version,
)
from .errors import ConfigurationError, DetectionError
from .extrinsic_calibrator import solve_planar_pnp
from .reporting import (
    prepare_output_directories,
    save_observation_images,
    write_markdown_report,
    write_observations_csv,
)
from .target_detector import (
    TargetObservation,
    TargetSpec,
    collect_target_observations,
    select_diverse_observations,
)
from .transforms import (
    invert_transform,
    rotation_distance_deg,
    transform_from_config,
    validate_transform,
)
from .video_frames import iter_input_frames


def _validation_observations(
    root_config: dict[str, Any],
    section: dict[str, Any],
    spec: TargetSpec,
    image_size: tuple[int, int],
) -> tuple[list[TargetObservation], list[TargetObservation]]:
    merged = {**root_config, "input": section.get("input", section)}
    quality = section.get("quality", root_config.get("quality", {}))
    frames = list(iter_input_frames(merged))
    for frame in frames:
        actual = (frame.image.shape[1], frame.image.shape[0])
        if actual != image_size:
            raise ConfigurationError(
                f"Validation frame {frame.name} is {actual[0]}x{actual[1]}, "
                f"expected {image_size[0]}x{image_size[1]}"
            )
    observations = collect_target_observations(frames, spec, quality)
    minimum = int(quality.get("min_valid_frames", 3))
    maximum = int(quality.get("max_valid_frames", 30))
    selected = select_diverse_observations(
        observations,
        minimum,
        maximum,
        float(quality.get("duplicate_distance", 0.02)),
    )
    if len(selected) < minimum:
        raise DetectionError(
            f"Validation needs {minimum} diverse target frames; got {len(selected)}"
        )
    return observations, selected


def _intrinsic_validation(
    config: dict[str, Any],
    section: dict[str, Any] | None,
    camera: CameraParameters,
    spec: TargetSpec,
    output_dir: Path,
) -> dict[str, Any]:
    fx = float(camera.camera_matrix[0, 0])
    fy = float(camera.camera_matrix[1, 1])
    cx = float(camera.camera_matrix[0, 2])
    cy = float(camera.camera_matrix[1, 2])
    plausible = bool(
        0.1 * camera.width < fx < 10.0 * camera.width
        and 0.1 * camera.height < fy < 10.0 * camera.height
        and 0 <= cx < camera.width
        and 0 <= cy < camera.height
        and 0.2 < fx / fy < 5.0
    )
    result: dict[str, Any] = {
        "parameter_plausibility_passed": plausible,
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": cx,
        "cy_px": cy,
        "data_validation_performed": section is not None,
    }
    if section is None:
        result["passed"] = plausible
        return result
    observations, selected = _validation_observations(
        config,
        section,
        spec,
        (camera.width, camera.height),
    )
    directories = prepare_output_directories(
        output_dir / "intrinsics",
        ["detected", "rejected", "undistorted"],
    )
    save_observation_images(observations, spec, directories)
    errors: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    for observation in selected:
        transform, _ = solve_planar_pnp(
            camera,
            spec.object_points(),
            np.asarray(observation.image_points),
        )
        rotation, _ = cv2.Rodrigues(transform[:3, :3])
        projected = camera.project(
            spec.object_points(),
            rotation,
            transform[:3, 3],
        )
        errors.append(
            np.linalg.norm(
                projected - np.asarray(observation.image_points),
                axis=1,
            )
        )
        points = np.asarray(observation.image_points)
        radii.append(
            np.linalg.norm(
                (points - [camera.width / 2, camera.height / 2])
                / [camera.width / 2, camera.height / 2],
                axis=1,
            )
        )
        if not cv2.imwrite(
            str(directories["undistorted"] / f"{observation.frame.name}.jpg"),
            camera.undistort_image(observation.frame.image),
        ):
            raise DetectionError(
                f"Cannot write intrinsic preview {observation.frame.name}"
            )
    values = np.concatenate(errors)
    radius_values = np.concatenate(radii)
    center = values[radius_values < 0.5]
    corners = values[radius_values >= 0.8]
    write_observations_csv(
        output_dir / "intrinsics" / "observations.csv",
        observations,
    )
    result.update(
        {
            "valid_frame_count": len(selected),
            "mean_reprojection_error_px": float(np.mean(values)),
            "median_reprojection_error_px": float(np.median(values)),
            "max_reprojection_error_px": float(np.max(values)),
            "center_mean_error_px": (
                float(np.mean(center)) if len(center) else None
            ),
            "corner_mean_error_px": (
                float(np.mean(corners)) if len(corners) else None
            ),
        }
    )
    maximum_mean = float(section.get("max_mean_reprojection_error_px", 2.0))
    maximum_corner = float(
        section.get("max_corner_reprojection_error_px", 3.0)
    )
    result["mean_reprojection_error_passed"] = bool(
        result["mean_reprojection_error_px"] <= maximum_mean
    )
    result["corner_reprojection_error_passed"] = bool(
        result["corner_mean_error_px"] is None
        or result["corner_mean_error_px"] <= maximum_corner
    )
    result["thresholds_px"] = {
        "max_mean_reprojection_error": maximum_mean,
        "max_corner_reprojection_error": maximum_corner,
    }
    result["passed"] = bool(
        plausible
        and result["mean_reprojection_error_passed"]
        and result["corner_reprojection_error_passed"]
    )
    return result


def _extrinsic_validation(
    config: dict[str, Any],
    section: dict[str, Any] | None,
    calibration: UnderwaterCameraCalibration,
    spec: TargetSpec,
    extrinsics_result: dict[str, Any],
) -> dict[str, Any]:
    axis_description = str(
        extrinsics_result.get("coordinate_frames", {}).get(
            calibration.camera_frame,
            "",
        )
    ).lower()
    optical_axis_passed = bool(
        "optical" in calibration.camera_frame.lower()
        and "+x right" in axis_description
        and "+y down" in axis_description
        and "+z forward" in axis_description
    )
    chain = extrinsics_result.get("mechanical_frame_chain")
    chain_passed = True
    if chain is not None:
        try:
            base_link = chain["base_to_camera_link"]
            link_optical = chain["camera_link_to_camera_optical"]
            t_base_link = validate_transform(
                np.asarray(base_link["matrix"], dtype=float)
            )
            t_link_optical = validate_transform(
                np.asarray(link_optical["matrix"], dtype=float)
            )
            chain_passed = bool(
                base_link["parent_frame"] == calibration.base_frame
                and link_optical["child_frame"] == calibration.camera_frame
                and base_link["child_frame"] == link_optical["parent_frame"]
                and np.allclose(
                    t_base_link @ t_link_optical,
                    calibration.t_base_camera,
                    atol=1e-7,
                )
            )
        except (KeyError, TypeError, ValueError, ConfigurationError):
            chain_passed = False
    result: dict[str, Any] = {
        "transform_inverse_passed": True,
        "parent_frame": calibration.base_frame,
        "child_frame": calibration.camera_frame,
        "optical_axis_definition": axis_description,
        "optical_axis_definition_passed": optical_axis_passed,
        "mechanical_to_optical_chain_present": chain is not None,
        "mechanical_to_optical_chain_passed": chain_passed,
        "transform_convention": "p_parent = T_parent_child * p_child",
        "data_validation_performed": section is not None,
    }
    bounds = (section or {}).get("physical_bounds", {})
    if bounds:
        translation = calibration.t_base_camera[:3, 3]
        minimum = np.asarray(
            bounds.get("translation_min_m", [-np.inf] * 3)
        )
        maximum = np.asarray(
            bounds.get("translation_max_m", [np.inf] * 3)
        )
        result["physical_position_passed"] = bool(
            np.all(translation >= minimum)
            and np.all(translation <= maximum)
        )
    if section is None:
        result["passed"] = bool(
            result["transform_inverse_passed"]
            and result["optical_axis_definition_passed"]
            and result["mechanical_to_optical_chain_passed"]
            and result.get("physical_position_passed", True)
        )
        return result
    datasets = section.get("datasets", [])
    if not datasets:
        raise ConfigurationError(
            "extrinsic_validation exists but contains no datasets"
        )
    image_errors: list[float] = []
    translation_residuals: list[float] = []
    rotation_residuals: list[float] = []
    for dataset in datasets:
        name = str(dataset.get("name", "<unnamed>"))
        t_base_target = transform_from_config(
            require(dataset, "T_base_target"),
            f"extrinsic_validation.datasets[{name}].T_base_target",
        )
        input_section = {
            "input": {
                "type": dataset.get(
                    "type",
                    "video" if "video" in dataset else "images",
                ),
                "path": dataset.get("video", dataset.get("path")),
                "paths": dataset.get("paths"),
                "frame_interval_sec": dataset.get(
                    "frame_interval_sec",
                    0.5,
                ),
            },
            "quality": {
                **section.get("quality", {}),
                "min_valid_frames": int(dataset.get("min_valid_frames", 1)),
                "max_valid_frames": int(dataset.get("max_valid_frames", 10)),
            },
        }
        _, selected = _validation_observations(
            config,
            input_section,
            spec,
            (calibration.camera.width, calibration.camera.height),
        )
        predicted = invert_transform(calibration.t_base_camera) @ t_base_target
        rotation, _ = cv2.Rodrigues(predicted[:3, :3])
        for observation in selected:
            projected = calibration.camera.project(
                spec.object_points(),
                rotation,
                predicted[:3, 3],
            )
            image_errors.extend(
                np.linalg.norm(
                    projected - np.asarray(observation.image_points),
                    axis=1,
                ).tolist()
            )
            measured_camera_target, _ = solve_planar_pnp(
                calibration.camera,
                spec.object_points(),
                np.asarray(observation.image_points),
            )
            measured_base_camera = t_base_target @ invert_transform(
                measured_camera_target
            )
            translation_residuals.append(
                float(
                    np.linalg.norm(
                        measured_base_camera[:3, 3]
                        - calibration.t_base_camera[:3, 3]
                    )
                    * 1000.0
                )
            )
            rotation_residuals.append(
                rotation_distance_deg(
                    measured_base_camera[:3, :3],
                    calibration.t_base_camera[:3, :3],
                )
            )
    result.update(
        {
            "mean_reprojection_error_px": float(np.mean(image_errors)),
            "max_reprojection_error_px": float(np.max(image_errors)),
            "mean_translation_residual_mm": float(
                np.mean(translation_residuals)
            ),
            "max_translation_residual_mm": float(
                np.max(translation_residuals)
            ),
            "mean_rotation_residual_deg": float(
                np.mean(rotation_residuals)
            ),
            "max_rotation_residual_deg": float(np.max(rotation_residuals)),
        }
    )
    maximum_reprojection = float(
        section.get("max_mean_reprojection_error_px", 2.0)
    )
    maximum_translation = float(
        section.get("max_mean_translation_residual_mm", 15.0)
    )
    maximum_rotation = float(
        section.get("max_mean_rotation_residual_deg", 1.5)
    )
    result["reprojection_error_passed"] = bool(
        result["mean_reprojection_error_px"] <= maximum_reprojection
    )
    result["translation_residual_passed"] = bool(
        result["mean_translation_residual_mm"] <= maximum_translation
    )
    result["rotation_residual_passed"] = bool(
        result["mean_rotation_residual_deg"] <= maximum_rotation
    )
    result["thresholds"] = {
        "max_mean_reprojection_error_px": maximum_reprojection,
        "max_mean_translation_residual_mm": maximum_translation,
        "max_mean_rotation_residual_deg": maximum_rotation,
    }
    result["passed"] = bool(
        result["transform_inverse_passed"]
        and result["optical_axis_definition_passed"]
        and result["mechanical_to_optical_chain_passed"]
        and result.get("physical_position_passed", True)
        and result["reprojection_error_passed"]
        and result["translation_residual_passed"]
        and result["rotation_residual_passed"]
    )
    return result


def validate_calibration(config: dict[str, Any]) -> dict[str, Any]:
    """Validate two-file compatibility and optional independent datasets."""
    intrinsics_file = resolve_path(config, require(config, "intrinsics_file"))
    extrinsics_file = resolve_path(config, require(config, "extrinsics_file"))
    calibration = UnderwaterCameraCalibration.load(
        intrinsics_file,
        extrinsics_file,
    )
    extrinsics_result = load_yaml(extrinsics_file)
    validate_format_version(extrinsics_result, extrinsics_file)
    spec = TargetSpec.from_config(
        config.get(
            "target",
            {
                "type": "checkerboard",
                "inner_corners_cols": 8,
                "inner_corners_rows": 6,
                "square_size_m": 0.05,
            },
        )
    )
    output_dir = output_path(config, "output/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsic = _intrinsic_validation(
        config,
        config.get("intrinsic_validation"),
        calibration.camera,
        spec,
        output_dir,
    )
    extrinsic = _extrinsic_validation(
        config,
        config.get("extrinsic_validation"),
        calibration,
        spec,
        extrinsics_result,
    )
    test_pixels = [
        (calibration.camera.width / 2, calibration.camera.height / 2),
        (0.0, 0.0),
        (calibration.camera.width - 1.0, 0.0),
        (0.0, calibration.camera.height - 1.0),
        (
            calibration.camera.width - 1.0,
            calibration.camera.height - 1.0,
        ),
    ]
    rays = []
    for u, v in test_pixels:
        ray = calibration.pixel_to_base_ray(u, v)
        rays.append(
            {
                "pixel_uv": [u, v],
                "origin_base_m": ray.origin.tolist(),
                "direction_base_unit": ray.direction.tolist(),
            }
        )
    status = (
        "passed"
        if bool(intrinsic["passed"]) and bool(extrinsic["passed"])
        else "failed"
    )
    payload = {
        "status": status,
        "runtime_summary": calibration.summary(),
        "intrinsics": intrinsic,
        "extrinsics": extrinsic,
        "sample_base_rays": rays,
    }
    result_file = save_yaml(output_dir / "validation.yaml", payload)
    write_markdown_report(
        output_dir / "report.md",
        "Underwater Camera Calibration Validation",
        [
            (
                "Runtime composition",
                f"- Camera: `{calibration.camera.camera_name}`\n"
                f"- Resolution/model: {calibration.camera.width}×"
                f"{calibration.camera.height} / `{calibration.camera.model}`\n"
                f"- Calibration environment: "
                f"`{calibration.calibration_environment}`\n"
                f"- Rays: `{calibration.camera_frame}` → "
                f"`{calibration.base_frame}`",
            ),
            ("Intrinsic validation", _markdown_mapping(intrinsic)),
            ("Mechanical extrinsic validation", _markdown_mapping(extrinsic)),
            (
                "Runtime test",
                "Center and four corner pixels were converted to normalized "
                "central-camera rays in the robot base frame and saved in "
                "`validation.yaml`.",
            ),
        ],
    )
    if status != "passed":
        raise ConfigurationError(
            f"Calibration validation failed; inspect {result_file}"
        )
    return {
        "result_file": str(result_file),
        "output_dir": str(output_dir),
        "results": payload,
    }


def _markdown_mapping(values: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: `{value}`" for key, value in values.items())
