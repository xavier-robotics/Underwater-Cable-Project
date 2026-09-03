#!/usr/bin/env python
"""Run the deterministic two-stage calibration acceptance workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from camera_calibration.calibration import UnderwaterCameraCalibration
from camera_calibration.config import load_yaml, save_yaml
from camera_calibration.transforms import rotation_distance_deg
from generate_synthetic_dataset import generate


def _run(command: str, config: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "camera_calibration",
            command,
            "--config",
            str(config),
        ],
        check=True,
    )


def _require_artifacts(root: Path, names: list[str]) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"Missing acceptance artifacts in {root}: {missing}")


def _require_images(path: Path) -> None:
    if not path.is_dir() or not any(
        item.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for item in path.iterdir()
    ):
        raise RuntimeError(f"No debug images were written to {path}")


def run_acceptance(root: Path) -> dict[str, Any]:
    """Generate videos, execute three commands, and check truth/artifacts."""
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"Acceptance output must be absent or empty: {root}. "
            "Choose a new --output directory."
        )
    paths = generate(root)
    _run("calibrate-intrinsics", paths["intrinsics_config"])
    _run("calibrate-extrinsics", paths["extrinsics_config"])
    _run("validate", paths["validation_config"])

    output = paths["output"]
    intrinsics_dir = output / "intrinsics"
    extrinsics_dir = output / "extrinsics"
    validation_dir = output / "validation"
    _require_artifacts(
        intrinsics_dir,
        [
            "intrinsics.yaml",
            "camera_info.yaml",
            "report.md",
            "observations.csv",
            "coverage.png",
            "detected",
            "rejected",
            "undistorted",
            "reprojection",
        ],
    )
    _require_artifacts(
        extrinsics_dir,
        [
            "extrinsics.yaml",
            "report.md",
            "observations.csv",
            "detected",
            "rejected",
            "reprojection",
        ],
    )
    _require_artifacts(
        validation_dir,
        ["validation.yaml", "report.md"],
    )
    for directory in (
        intrinsics_dir / "detected",
        intrinsics_dir / "undistorted",
        intrinsics_dir / "reprojection",
        extrinsics_dir / "detected",
        extrinsics_dir / "reprojection",
    ):
        _require_images(directory)

    truth = load_yaml(root / "truth.yaml")
    intrinsics = load_yaml(intrinsics_dir / "intrinsics.yaml")
    extrinsics = load_yaml(extrinsics_dir / "extrinsics.yaml")
    validation = load_yaml(validation_dir / "validation.yaml")

    actual_matrix = np.asarray(
        intrinsics["camera_matrix"]["data"],
        dtype=float,
    ).reshape(3, 3)
    true_matrix = np.asarray(truth["camera_matrix"], dtype=float)
    focal_error_px = float(
        np.max(np.abs(np.diag(actual_matrix)[:2] - np.diag(true_matrix)[:2]))
    )
    principal_error_px = float(
        np.linalg.norm(actual_matrix[:2, 2] - true_matrix[:2, 2])
    )
    if focal_error_px > 5.0 or principal_error_px > 5.0:
        raise RuntimeError(
            "Synthetic intrinsic recovery exceeded 5 px tolerance: "
            f"focal={focal_error_px:.4f}, principal={principal_error_px:.4f}"
        )
    if intrinsics.get("calibration_environment") != "underwater":
        raise RuntimeError("Synthetic intrinsics lost calibration environment")

    actual_transform = np.asarray(extrinsics["T_base_camera"], dtype=float)
    true_transform = np.asarray(truth["T_base_camera"], dtype=float)
    translation_error_mm = float(
        np.linalg.norm(actual_transform[:3, 3] - true_transform[:3, 3])
        * 1000.0
    )
    rotation_error_deg = rotation_distance_deg(
        actual_transform[:3, :3],
        true_transform[:3, :3],
    )
    if translation_error_mm > 10.0 or rotation_error_deg > 1.0:
        raise RuntimeError(
            "Synthetic extrinsic recovery exceeded tolerance: "
            f"{translation_error_mm:.4f} mm, {rotation_error_deg:.4f} deg"
        )
    if validation.get("status") != "passed":
        raise RuntimeError("Synthetic validation did not report status: passed")

    calibration = UnderwaterCameraCalibration.load(
        intrinsics_dir / "intrinsics.yaml",
        extrinsics_dir / "extrinsics.yaml",
    )
    ray = calibration.pixel_to_base_ray(
        calibration.camera.width / 2.0,
        calibration.camera.height / 2.0,
    )
    if not np.isclose(np.linalg.norm(ray.direction), 1.0, atol=1e-10):
        raise RuntimeError("Runtime base ray direction is not normalized")

    summary = {
        "status": "passed",
        "commands": [
            "calibrate-intrinsics",
            "calibrate-extrinsics",
            "validate",
        ],
        "synthetic_truth_errors": {
            "intrinsic_focal_max_error_px": focal_error_px,
            "intrinsic_principal_point_error_px": principal_error_px,
            "extrinsic_translation_error_mm": translation_error_mm,
            "extrinsic_rotation_error_deg": rotation_error_deg,
        },
        "sample_base_ray": {
            "pixel_uv": [
                calibration.camera.width / 2.0,
                calibration.camera.height / 2.0,
            ],
            "origin_base_m": ray.origin.tolist(),
            "direction_base_unit": ray.direction.tolist(),
        },
        "result_files": {
            "intrinsics": str(intrinsics_dir / "intrinsics.yaml"),
            "extrinsics": str(extrinsics_dir / "extrinsics.yaml"),
            "validation": str(validation_dir / "validation.yaml"),
        },
    }
    save_yaml(root / "acceptance_summary.yaml", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or empty directory for generated videos and results",
    )
    args = parser.parse_args()
    summary = run_acceptance(args.output.resolve())
    print(f"Synthetic acceptance passed: {args.output.resolve()}")
    print(summary["sample_base_ray"])


if __name__ == "__main__":
    main()
