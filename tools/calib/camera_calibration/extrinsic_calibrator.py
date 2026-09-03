"""Robot-base to optical-camera mechanical extrinsic calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .camera_models import CameraParameters
from .config import output_path, require, resolve_path, save_yaml
from .errors import ConfigurationError, DetectionError
from .optimization import scipy_least_squares
from .reporting import (
    prepare_output_directories,
    save_observation_images,
    save_reprojection_overlay,
    write_markdown_report,
    write_observations_csv,
)
from .ros_export import static_tf_command
from .target_detector import (
    TargetObservation,
    TargetSpec,
    collect_target_observations,
    select_diverse_observations,
)
from .transforms import (
    average_quaternions_xyzw,
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
    pose_vector_to_transform,
    quaternion_xyzw_to_matrix,
    rotation_distance_deg,
    transform_from_config,
    transform_to_pose_vector,
)
from .video_frames import iter_input_frames


@dataclass
class ExtrinsicCandidate:
    """One observation's estimate of T_base_camera."""

    dataset_name: str
    observation: TargetObservation
    t_base_target: np.ndarray
    t_camera_target: np.ndarray
    t_base_camera: np.ndarray
    reprojection_error_px: float
    solution_index: int = 0
    solution_count: int = 1
    accepted: bool = True
    translation_residual_mm: float = 0.0
    rotation_residual_deg: float = 0.0


def _pnp_input(
    camera: CameraParameters,
    image_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera.model == "pinhole":
        return (
            np.asarray(image_points, np.float64),
            camera.camera_matrix,
            camera.distortion.reshape(-1, 1),
        )
    return (
        camera.undistort_pixels(image_points),
        np.eye(3, dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
    )


def solve_planar_pnp_candidates(
    camera: CameraParameters,
    object_points: np.ndarray,
    image_points: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """Return cheirality-valid IPPE/iterative T_camera_target candidates."""
    points, matrix, distortion = _pnp_input(camera, image_points)
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            result = cv2.solvePnPGeneric(
                np.asarray(object_points, np.float64),
                np.asarray(points, np.float64),
                matrix,
                distortion,
                flags=flag,
            )
        except cv2.error:
            continue
        if result and bool(result[0]):
            raw.extend(
                (
                    np.asarray(rotation, np.float64).reshape(3),
                    np.asarray(translation, np.float64).reshape(3),
                )
                for rotation, translation in zip(result[1], result[2])
            )
    if not raw:
        try:
            ok, rotation, translation = cv2.solvePnP(
                np.asarray(object_points, np.float64),
                np.asarray(points, np.float64),
                matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as exc:
            raise DetectionError(f"OpenCV PnP failed: {exc}") from exc
        if ok:
            raw.append((rotation.reshape(3), translation.reshape(3)))
    valid: list[tuple[np.ndarray, float]] = []
    for rotation, translation in raw:
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        camera_points = (
            rotation_matrix @ np.asarray(object_points).reshape(-1, 3).T
        ).T + translation
        if np.any(camera_points[:, 2] <= 1e-6):
            continue
        projected = camera.project(object_points, rotation, translation)
        delta = projected - np.asarray(image_points).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        candidate = make_transform(rotation_matrix, translation)
        if not any(np.allclose(candidate, previous[0], atol=1e-9) for previous in valid):
            valid.append((candidate, error))
    valid.sort(key=lambda item: item[1])
    return valid


def solve_planar_pnp(
    camera: CameraParameters,
    object_points: np.ndarray,
    image_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Choose the lowest-error physically valid planar PnP solution."""
    valid = solve_planar_pnp_candidates(camera, object_points, image_points)
    if not valid:
        raise DetectionError(
            "PnP produced no solution with all checkerboard corners in front of "
            "the optical camera"
        )
    return valid[0]


def robust_fuse_transforms(
    transforms: list[np.ndarray],
    iterations: int = 5,
) -> np.ndarray:
    """Huber-fuse translation and quaternion/SO(3) rotation."""
    if not transforms:
        raise DetectionError("Cannot fuse an empty extrinsic candidate set")
    translations = np.asarray([item[:3, 3] for item in transforms])
    quaternions = np.asarray(
        [matrix_to_quaternion_xyzw(item[:3, :3]) for item in transforms]
    )
    translation = np.median(translations, axis=0)
    quaternion = average_quaternions_xyzw(quaternions)
    for _ in range(iterations):
        rotation = quaternion_xyzw_to_matrix(quaternion)
        translation_residuals = np.linalg.norm(
            translations - translation,
            axis=1,
        )
        rotation_residuals = np.asarray(
            [
                rotation_distance_deg(rotation, transform[:3, :3])
                for transform in transforms
            ]
        )
        translation_scale = max(
            float(np.median(translation_residuals)) * 1.4826,
            1e-6,
        )
        rotation_scale = max(
            float(np.median(rotation_residuals)) * 1.4826,
            1e-3,
        )
        combined = np.sqrt(
            (translation_residuals / translation_scale) ** 2
            + (rotation_residuals / rotation_scale) ** 2
        )
        weights = np.where(
            combined <= 1.5,
            1.0,
            1.5 / np.maximum(combined, 1e-9),
        )
        translation = np.sum(
            translations * weights[:, None],
            axis=0,
        ) / np.sum(weights)
        quaternion = average_quaternions_xyzw(quaternions, weights)
    return make_transform(quaternion_xyzw_to_matrix(quaternion), translation)


def _candidate_residuals(
    candidates: list[ExtrinsicCandidate],
    estimate: np.ndarray,
) -> None:
    for candidate in candidates:
        candidate.translation_residual_mm = float(
            np.linalg.norm(candidate.t_base_camera[:3, 3] - estimate[:3, 3])
            * 1000.0
        )
        candidate.rotation_residual_deg = rotation_distance_deg(
            candidate.t_base_camera[:3, :3],
            estimate[:3, :3],
        )


def _translation_within_bounds(
    transform: np.ndarray,
    bounds: dict[str, Any],
) -> bool:
    if not bounds:
        return True
    translation = transform[:3, 3]
    lower = np.asarray(bounds.get("translation_min_m", [-np.inf] * 3), float)
    upper = np.asarray(bounds.get("translation_max_m", [np.inf] * 3), float)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ConfigurationError(
            "physical_bounds translation_min_m/max_m must contain 3 values"
        )
    return bool(np.all(translation >= lower) and np.all(translation <= upper))


def select_consistent_pnp_candidates(
    candidate_groups: list[list[ExtrinsicCandidate]],
    *,
    physical_bounds: dict[str, Any] | None = None,
    translation_scale_mm: float = 10.0,
    rotation_scale_deg: float = 1.0,
    iterations: int = 8,
) -> list[ExtrinsicCandidate]:
    """Resolve planar ambiguity using reprojection, consensus, and mechanics."""
    if not candidate_groups or any(not group for group in candidate_groups):
        raise DetectionError("PnP candidate groups must all be non-empty")
    if translation_scale_mm <= 0 or rotation_scale_deg <= 0 or iterations <= 0:
        raise ConfigurationError(
            "PnP consistency scales and iteration count must be positive"
        )
    bounds = physical_bounds or {}
    def choose(
        consensus: np.ndarray,
    ) -> tuple[list[ExtrinsicCandidate], float]:
        chosen: list[ExtrinsicCandidate] = []
        total = 0.0
        for group in candidate_groups:
            mechanically_valid = [
                item
                for item in group
                if _translation_within_bounds(item.t_base_camera, bounds)
            ]
            pool = mechanically_valid or group

            def score(item: ExtrinsicCandidate) -> float:
                translation_mm = (
                    np.linalg.norm(
                        item.t_base_camera[:3, 3] - consensus[:3, 3]
                    )
                    * 1000.0
                )
                rotation_deg = rotation_distance_deg(
                    item.t_base_camera[:3, :3],
                    consensus[:3, :3],
                )
                return float(
                    item.reprojection_error_px
                    + translation_mm / translation_scale_mm
                    + rotation_deg / rotation_scale_deg
                )

            best = min(pool, key=score)
            chosen.append(best)
            total += score(best)
        return chosen, total

    best_selected: list[ExtrinsicCandidate] | None = None
    best_score = np.inf
    # Planar ambiguity can make the lowest-reprojection candidates form a poor
    # local consensus.  Seed from every mechanically plausible candidate and
    # retain the cross-observation mode with the smallest total robust score.
    seeds = [
        item
        for group in candidate_groups
        for item in group
        if _translation_within_bounds(item.t_base_camera, bounds)
    ]
    if not seeds:
        seeds = [item for group in candidate_groups for item in group]
    for seed in seeds:
        consensus = seed.t_base_camera
        selected: list[ExtrinsicCandidate] = []
        for _ in range(iterations):
            updated, _ = choose(consensus)
            new_consensus = robust_fuse_transforms(
                [item.t_base_camera for item in updated]
            )
            if selected and all(
                first is second for first, second in zip(selected, updated)
            ):
                selected = updated
                consensus = new_consensus
                break
            selected = updated
            consensus = new_consensus
        selected, total = choose(consensus)
        if total < best_score:
            best_score = total
            best_selected = selected
    assert best_selected is not None
    return best_selected


def global_reprojection_residual(
    pose: np.ndarray,
    candidates: list[ExtrinsicCandidate],
    camera: CameraParameters,
    object_points: np.ndarray,
) -> np.ndarray:
    """Residual for the sole global variable T_base_camera."""
    t_camera_base = invert_transform(pose_vector_to_transform(pose))
    residuals: list[np.ndarray] = []
    for candidate in candidates:
        # T_camera_target = T_camera_base @ T_base_target
        t_camera_target = t_camera_base @ candidate.t_base_target
        rotation, _ = cv2.Rodrigues(t_camera_target[:3, :3])
        projected = camera.project(
            object_points,
            rotation.reshape(3),
            t_camera_target[:3, 3],
        )
        residuals.append(
            (
                projected
                - np.asarray(candidate.observation.image_points).reshape(-1, 2)
            ).reshape(-1)
        )
    return np.concatenate(residuals)


def _dataset_input(dataset: dict[str, Any]) -> dict[str, Any]:
    if "video" in dataset:
        return {
            "type": "video",
            "path": dataset["video"],
            "frame_interval_sec": dataset.get("frame_interval_sec", 0.5),
        }
    if "path" in dataset or "paths" in dataset:
        return {
            "type": dataset.get("type", "images"),
            "path": dataset.get("path"),
            "paths": dataset.get("paths"),
            "frame_interval_sec": dataset.get("frame_interval_sec", 0.5),
        }
    raise ConfigurationError(
        f"Dataset {dataset.get('name', '<unnamed>')} requires video/path/paths"
    )


def _physical_bounds_check(
    transform: np.ndarray,
    bounds: dict[str, Any],
) -> None:
    if not bounds:
        return
    translation = transform[:3, 3]
    lower = np.asarray(bounds.get("translation_min_m", [-np.inf] * 3), float)
    upper = np.asarray(bounds.get("translation_max_m", [np.inf] * 3), float)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ConfigurationError(
            "physical_bounds translation_min_m/max_m must contain 3 values"
        )
    if np.any(translation < lower) or np.any(translation > upper):
        raise DetectionError(
            f"Estimated camera translation {translation.tolist()} m is outside "
            "configured mechanical bounds"
        )


def calibrate_extrinsics(config: dict[str, Any]) -> dict[str, Any]:
    """Estimate, reject outliers, fuse, and optionally refine T_base_camera."""
    camera = CameraParameters.load(
        resolve_path(config, require(config, "intrinsics_file"))
    )
    spec = TargetSpec.from_config(config.get("target", {}))
    frames = config.get("frames", {})
    base_frame = str(frames.get("base_frame", "base_link"))
    camera_frame = str(frames.get("camera_frame", "camera_optical_frame"))
    target_frame = str(frames.get("target_frame", "calibration_target"))
    if "optical" not in camera_frame:
        raise ConfigurationError(
            "frames.camera_frame used by PnP must be camera_optical_frame, not "
            "the mechanical camera_link frame"
        )
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ConfigurationError(
            "datasets must contain at least one fixture pose with full T_base_target"
        )
    names = [str(item.get("name", "")) for item in datasets]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ConfigurationError("Every dataset must have a unique non-empty name")

    all_observations: list[TargetObservation] = []
    candidate_groups: list[list[ExtrinsicCandidate]] = []
    default_quality = config.get("quality", {})
    for dataset in datasets:
        name = str(dataset["name"])
        t_base_target = transform_from_config(
            require(dataset, "T_base_target"),
            f"datasets[{name}].T_base_target",
        )
        quality = {**default_quality, **dataset.get("quality", {})}
        minimum = int(quality.get("min_frames_per_dataset", 5))
        maximum = min(10, int(quality.get("max_frames_per_dataset", 10)))
        source_frames = list(iter_input_frames(config, _dataset_input(dataset)))
        for frame in source_frames:
            frame.name = f"{name}_{frame.name}"
            actual = (frame.image.shape[1], frame.image.shape[0])
            if actual != (camera.width, camera.height):
                raise ConfigurationError(
                    f"Dataset {name} frame is {actual[0]}x{actual[1]}, but "
                    f"intrinsics are {camera.width}x{camera.height}"
                )
        observations = collect_target_observations(source_frames, spec, quality)
        selected = select_diverse_observations(
            observations,
            minimum,
            maximum,
            float(quality.get("duplicate_distance", 0.02)),
        )
        all_observations.extend(observations)
        if len(selected) < minimum:
            raise DetectionError(
                f"Dataset {name} needs {minimum}-{maximum} diverse frames; "
                f"only {len(selected)} passed"
            )
        for observation in selected:
            solutions = solve_planar_pnp_candidates(
                camera,
                spec.object_points(),
                np.asarray(observation.image_points),
            )
            if not solutions:
                observation.selected = False
                observation.accepted = False
                observation.rejection_reason = "pnp_no_positive_depth_solution"
                continue
            group: list[ExtrinsicCandidate] = []
            for solution_index, (t_camera_target, error) in enumerate(solutions):
                # Required convention: T_base_camera =
                # T_base_target @ inverse(T_camera_target).
                t_base_camera = t_base_target @ invert_transform(t_camera_target)
                group.append(
                    ExtrinsicCandidate(
                        name,
                        observation,
                        t_base_target,
                        t_camera_target,
                        t_base_camera,
                        error,
                        solution_index=solution_index,
                        solution_count=len(solutions),
                    )
                )
            candidate_groups.append(group)
    if not candidate_groups:
        raise DetectionError("No valid PnP observations were produced")

    disambiguation = config.get("pnp_disambiguation", {})
    candidates = select_consistent_pnp_candidates(
        candidate_groups,
        physical_bounds=config.get("physical_bounds", {}),
        translation_scale_mm=float(
            disambiguation.get("translation_scale_mm", 10.0)
        ),
        rotation_scale_deg=float(
            disambiguation.get("rotation_scale_deg", 1.0)
        ),
        iterations=int(disambiguation.get("iterations", 8)),
    )
    for candidate in candidates:
        candidate.observation.reprojection_error_px = (
            candidate.reprojection_error_px
        )
        pose = transform_to_pose_vector(candidate.t_camera_target)
        candidate.observation.pose_rvec = pose[:3]
        candidate.observation.pose_tvec = pose[3:]
    fused = robust_fuse_transforms([item.t_base_camera for item in candidates])
    _candidate_residuals(candidates, fused)
    rejection = config.get("outlier_rejection", {})
    limits = (
        float(rejection.get("max_reprojection_error_px", 1.5)),
        float(rejection.get("max_translation_residual_mm", 10.0)),
        float(rejection.get("max_rotation_residual_deg", 1.0)),
    )
    for candidate in candidates:
        candidate.accepted = (
            candidate.reprojection_error_px <= limits[0]
            and candidate.translation_residual_mm <= limits[1]
            and candidate.rotation_residual_deg <= limits[2]
        )
        if not candidate.accepted:
            candidate.observation.selected = False
            candidate.observation.accepted = False
            candidate.observation.rejection_reason = "extrinsic_outlier"
    accepted = [item for item in candidates if item.accepted]
    minimum_total = int(default_quality.get("min_total_observations", 3))
    if len(accepted) < minimum_total:
        raise DetectionError(
            f"Extrinsic rejection left {len(accepted)} observations, below "
            f"min_total_observations={minimum_total}; no result was written"
        )
    fused = robust_fuse_transforms([item.t_base_camera for item in accepted])
    initial_pose = transform_to_pose_vector(fused)
    before = global_reprojection_residual(
        initial_pose,
        accepted,
        camera,
        spec.object_points(),
    )
    optimization = config.get("optimization", {})
    if bool(optimization.get("enabled", True)):
        fit = scipy_least_squares(
            lambda pose: global_reprojection_residual(
                pose,
                accepted,
                camera,
                spec.object_points(),
            ),
            initial_pose,
            loss=str(optimization.get("robust_loss", "huber")),
            f_scale=float(optimization.get("f_scale_px", 1.0)),
            max_iterations=int(optimization.get("max_iterations", 200)),
        )
        final = pose_vector_to_transform(fit.x)
    else:
        final = fused
    after = global_reprojection_residual(
        transform_to_pose_vector(final),
        accepted,
        camera,
        spec.object_points(),
    )
    before_rmse = float(np.sqrt(np.mean(before * before)))
    after_rmse = float(np.sqrt(np.mean(after * after)))
    if bool(optimization.get("enabled", True)) and after_rmse > before_rmse * 1.001:
        raise DetectionError(
            "Global extrinsic optimization increased reprojection error; no result "
            "was written"
        )
    _physical_bounds_check(final, config.get("physical_bounds", {}))
    _candidate_residuals(candidates, final)
    t_camera_base = invert_transform(final)
    quaternion = matrix_to_quaternion_xyzw(final[:3, :3])

    output_dir = output_path(config, "output/extrinsics")
    directories = prepare_output_directories(
        output_dir,
        ["detected", "rejected", "reprojection"],
    )
    save_observation_images(all_observations, spec, directories)
    extra_rows: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        extra_rows[candidate.observation.frame.name] = {
            "dataset_name": candidate.dataset_name,
            "pnp_solution_index": candidate.solution_index,
            "pnp_solution_count": candidate.solution_count,
            "translation_residual_mm": f"{candidate.translation_residual_mm:.8f}",
            "rotation_residual_deg": f"{candidate.rotation_residual_deg:.8f}",
        }
        predicted_camera_target = t_camera_base @ candidate.t_base_target
        rotation_vector, _ = cv2.Rodrigues(predicted_camera_target[:3, :3])
        save_reprojection_overlay(
            directories["reprojection"]
            / f"{candidate.observation.frame.name}.jpg",
            candidate.observation.frame.image,
            np.asarray(candidate.observation.image_points),
            camera.project(
                spec.object_points(),
                rotation_vector,
                predicted_camera_target[:3, 3],
            ),
        )
    write_observations_csv(
        output_dir / "observations.csv",
        all_observations,
        extra_rows,
    )

    translation_residuals = np.asarray(
        [item.translation_residual_mm for item in accepted]
    )
    rotation_residuals = np.asarray(
        [item.rotation_residual_deg for item in accepted]
    )
    payload: dict[str, Any] = {
        "parent_frame": base_frame,
        "child_frame": camera_frame,
        "target_frame": target_frame,
        "coordinate_frames": {
            camera_frame: "optical: +x right, +y down, +z forward",
            target_frame: (
                "origin at first inner corner; +x columns, +y rows, +z "
                "right-handed"
            ),
        },
        "transform_convention": "p_parent = T_parent_child * p_child",
        "units": {"translation": "metre", "rotation": "quaternion_xyzw"},
        "translation_m": dict(
            zip(("x", "y", "z"), map(float, final[:3, 3]))
        ),
        "quaternion_xyzw": dict(
            zip(("x", "y", "z", "w"), map(float, quaternion))
        ),
        "rotation_matrix": final[:3, :3].tolist(),
        "T_base_camera": final.tolist(),
        "T_camera_base": t_camera_base.tolist(),
        "statistics": {
            "mean_reprojection_error_px": after_rmse,
            "optimization_before_rmse_px": before_rmse,
            "optimization_after_rmse_px": after_rmse,
            "translation_std_mm": float(np.std(translation_residuals)),
            "rotation_std_deg": float(np.std(rotation_residuals)),
            "valid_observation_count": len(accepted),
            "rejected_observation_count": len(candidates) - len(accepted),
        },
        "ros2_static_tf_command": static_tf_command(
            final[:3, 3],
            quaternion,
            base_frame,
            camera_frame,
        ),
        "camera_link_note": (
            "This transform ends at camera_optical_frame. camera_link is never "
            "silently treated as the optical frame."
        ),
    }
    link_transform_config = frames.get("T_camera_link_camera_optical")
    if link_transform_config is not None:
        link_frame = str(frames.get("camera_mechanical_frame", "camera_link"))
        t_link_optical = transform_from_config(
            link_transform_config,
            "frames.T_camera_link_camera_optical",
        )
        t_base_link = final @ invert_transform(t_link_optical)
        payload["mechanical_frame_chain"] = {
            "base_to_camera_link": {
                "parent_frame": base_frame,
                "child_frame": link_frame,
                "matrix": t_base_link.tolist(),
            },
            "camera_link_to_camera_optical": {
                "parent_frame": link_frame,
                "child_frame": camera_frame,
                "matrix": t_link_optical.tolist(),
            },
        }
    result_file = save_yaml(output_dir / "extrinsics.yaml", payload)
    write_markdown_report(
        output_dir / "report.md",
        "Mechanical Extrinsic Calibration",
        [
            (
                "Transform direction",
                f"`p_{base_frame} = T_{base_frame}_{camera_frame} "
                f"p_{camera_frame}`\n\n"
                f"Translation: {final[:3, 3].tolist()} m\n\n"
                f"Quaternion xyzw: {quaternion.tolist()}",
            ),
            (
                "Observations",
                f"- Valid: {len(accepted)}\n"
                f"- Rejected: {len(candidates) - len(accepted)}\n"
                f"- Optimization RMSE: {before_rmse:.4f} → "
                f"{after_rmse:.4f} px\n"
                f"- Translation residual std: "
                f"{np.std(translation_residuals):.3f} mm\n"
                f"- Rotation residual std: "
                f"{np.std(rotation_residuals):.4f} deg",
            ),
            (
                "Direction warning",
                "OpenCV PnP returns `T_camera_target`; every candidate is "
                "`T_base_target @ inverse(T_camera_target)`. The inverse final "
                "transform is separately saved as `T_camera_base`.",
            ),
        ],
    )
    return {
        "result_file": str(result_file),
        "output_dir": str(output_dir),
        "T_base_camera": final,
        "statistics": payload["statistics"],
    }
