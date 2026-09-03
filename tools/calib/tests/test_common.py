from __future__ import annotations

import tempfile
import unittest
import csv
import ast
from pathlib import Path

import cv2
import numpy as np

from camera_calibration.config import (
    load_yaml,
    metres_to_millimetres,
    millimetres_to_metres,
    require,
    save_yaml,
    validate_resolution,
)
from camera_calibration.errors import ConfigurationError
from camera_calibration.reporting import save_coverage_plot, write_observations_csv
from camera_calibration.target_detector import (
    TargetObservation,
    TargetSpec,
    detect_target,
    select_diverse_observations,
)
from camera_calibration.video_frames import SourceFrame, iter_input_frames


class ConfigurationAndInputTests(unittest.TestCase):
    def test_public_functions_have_type_annotations(self) -> None:
        package = Path(__file__).resolve().parents[1] / "camera_calibration"
        missing: list[str] = []
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                arguments = (
                    node.args.posonlyargs
                    + node.args.args
                    + node.args.kwonlyargs
                )
                untyped = [
                    item.arg
                    for item in arguments
                    if item.arg not in {"self", "cls"} and item.annotation is None
                ]
                if node.returns is None or untyped:
                    missing.append(f"{path.name}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [])

    def test_yaml_round_trip_is_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.yaml"
            save_yaml(path, {"array": np.array([1.0, 2.0])})
            loaded = load_yaml(path)
            self.assertEqual(loaded["format_version"], "1.0")
            self.assertEqual(loaded["array"], [1.0, 2.0])
            self.assertIn("generated_at", loaded)

    def test_units(self) -> None:
        self.assertAlmostEqual(metres_to_millimetres(0.123), 123.0)
        self.assertAlmostEqual(millimetres_to_metres(123.0), 0.123)

    def test_missing_configuration(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "Missing required"):
            require({"camera": {}}, "camera.model")

    def test_invalid_resolution(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "Invalid image resolution"):
            validate_resolution({"image_width": 0, "image_height": 480})

    def test_unreadable_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.mp4"
            path.write_bytes(b"not a video")
            config = {
                "_config_dir": directory,
                "input": {
                    "type": "video",
                    "path": path.name,
                    "frame_interval_sec": 0.5,
                },
            }
            with self.assertRaisesRegex(ConfigurationError, "Cannot open video"):
                list(iter_input_frames(config))

    def test_empty_image_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "_config_dir": directory,
                "input": {
                    "type": "images",
                    "path": ".",
                    "frame_interval_sec": 0.5,
                },
            }
            with self.assertRaisesRegex(ConfigurationError, "empty"):
                list(iter_input_frames(config))

    def test_checkerboard_detection_failure_is_explicit(self) -> None:
        frame = SourceFrame(
            np.full((240, 320, 3), 127, np.uint8),
            "blank",
            0,
            0.0,
            "blank",
        )
        result = detect_target(
            frame,
            TargetSpec("checkerboard", 8, 6, 0.05),
        )
        self.assertFalse(result.detected)
        self.assertEqual(result.rejection_reason, "target_not_detected")

    def test_duplicate_pose_filter(self) -> None:
        observations = []
        for index, x in enumerate((0.50, 0.501, 0.80)):
            frame = SourceFrame(
                np.zeros((20, 20, 3), np.uint8),
                "synthetic",
                index,
                0.0,
                f"f{index}",
            )
            observations.append(
                TargetObservation(
                    frame,
                    np.zeros((4, 2)),
                    True,
                    100.0 + index,
                    100.0,
                    0.0,
                    0.0,
                    center_xy=(x, 0.5),
                    area_ratio=0.1,
                    tilt_score=0.1,
                    accepted=True,
                )
            )
        selected = select_diverse_observations(
            observations,
            minimum=2,
            maximum=3,
            duplicate_distance=0.05,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            sum(item.rejection_reason == "duplicate_pose" for item in observations),
            1,
        )

    def test_coverage_plot_and_distance_audit_columns(self) -> None:
        frame = SourceFrame(
            np.zeros((480, 640, 3), np.uint8),
            "synthetic",
            0,
            0.0,
            "coverage",
        )
        item = TargetObservation(
            frame,
            np.zeros((4, 2)),
            True,
            120.0,
            100.0,
            0.0,
            0.0,
            center_xy=(0.2, 0.8),
            area_ratio=0.08,
            tilt_score=0.2,
            coverage_cells=[3, 6, 7],
            selected=True,
            accepted=True,
            pose_tvec=np.array([0.1, -0.1, 1.2]),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot = root / "coverage.png"
            table = root / "observations.csv"
            save_coverage_plot(plot, [item], (640, 480))
            write_observations_csv(table, [item])
            image = cv2.imread(str(plot))
            self.assertIsNotNone(image)
            assert image is not None
            self.assertEqual(image.shape[:2], (800, 1200))
            with table.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertIn("estimated_distance_m", row)
            self.assertAlmostEqual(
                float(row["estimated_distance_m"]),
                float(np.linalg.norm(item.pose_tvec)),
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
