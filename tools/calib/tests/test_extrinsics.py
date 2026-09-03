from __future__ import annotations

import unittest

import cv2
import numpy as np

from camera_calibration.extrinsic_calibrator import (
    ExtrinsicCandidate,
    robust_fuse_transforms,
    select_consistent_pnp_candidates,
    solve_planar_pnp,
    solve_planar_pnp_candidates,
)
from camera_calibration.ros_export import static_tf_command
from camera_calibration.transforms import (
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    rotation_distance_deg,
    transform_points,
)

from tests.helpers import camera, observation, spec


class ExtrinsicCalibrationTests(unittest.TestCase):
    def test_transform_multiply_inverse_and_direction(self) -> None:
        rotation, _ = cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))
        transform = make_transform(rotation, np.array([0.3, -0.1, 0.2]))
        np.testing.assert_allclose(
            transform @ invert_transform(transform),
            np.eye(4),
            atol=1e-12,
        )
        point_child = np.array([[0.2, 0.1, 1.0]])
        point_parent = transform_points(transform, point_child)
        restored = transform_points(invert_transform(transform), point_parent)
        np.testing.assert_allclose(restored, point_child, atol=1e-12)

    def test_quaternion_matrix_round_trip_and_sign(self) -> None:
        quaternion = np.array([0.1, -0.2, 0.3, 0.9273618495])
        quaternion /= np.linalg.norm(quaternion)
        rotation = quaternion_xyzw_to_matrix(quaternion)
        actual = matrix_to_quaternion_xyzw(rotation)
        self.assertAlmostEqual(abs(float(np.dot(actual, quaternion))), 1.0, 10)

    def test_planar_pnp_candidates_have_positive_depth(self) -> None:
        cam = camera()
        target = spec()
        rotation = np.array([0.12, -0.18, 0.04])
        translation = np.array([-0.08, -0.05, 1.1])
        image_points = cam.project(target.object_points(), rotation, translation)
        candidates = solve_planar_pnp_candidates(
            cam,
            target.object_points(),
            image_points,
        )
        self.assertGreaterEqual(len(candidates), 1)
        for transform, _ in candidates:
            self.assertTrue(
                np.all(
                    transform_points(transform, target.object_points())[:, 2] > 0
                )
            )

    def test_known_base_camera_recovery(self) -> None:
        cam = camera()
        target = spec()
        base_camera_rotation, _ = cv2.Rodrigues(
            np.array([0.03, -0.10, 0.08])
        )
        expected = make_transform(
            base_camera_rotation,
            np.array([0.2, -0.12, 0.08]),
        )
        candidates = []
        for index, pose in enumerate(
            (
                np.array([0.1, 0.0, 0.0, 0.05, -0.05, 1.0]),
                np.array([-0.1, 0.15, 0.05, -0.15, 0.02, 1.2]),
                np.array([0.08, -0.12, -0.05, 0.12, 0.05, 1.4]),
            )
        ):
            target_rotation, _ = cv2.Rodrigues(pose[:3])
            t_base_target = make_transform(target_rotation, pose[3:])
            t_camera_target = invert_transform(expected) @ t_base_target
            rotation, _ = cv2.Rodrigues(t_camera_target[:3, :3])
            image_points = cam.project(
                target.object_points(),
                rotation,
                t_camera_target[:3, 3],
            )
            solved, error = solve_planar_pnp(
                cam,
                target.object_points(),
                image_points,
            )
            estimate = t_base_target @ invert_transform(solved)
            candidates.append(estimate)
            self.assertLess(error, 1e-5)
        actual = robust_fuse_transforms(candidates)
        np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=1e-5)
        self.assertLess(
            rotation_distance_deg(actual[:3, :3], expected[:3, :3]),
            1e-3,
        )

    def test_robust_fusion_rejects_large_outlier_influence(self) -> None:
        transforms = [
            make_transform(np.eye(3), np.array([0.1 + 0.001 * i, 0.0, 0.0]))
            for i in range(6)
        ]
        bad_rotation, _ = cv2.Rodrigues(np.array([0.0, 0.0, 1.5]))
        transforms.append(
            make_transform(bad_rotation, np.array([2.0, -1.0, 0.5]))
        )
        fused = robust_fuse_transforms(transforms)
        self.assertLess(np.linalg.norm(fused[:3, 3] - [0.1025, 0, 0]), 0.01)
        self.assertLess(rotation_distance_deg(fused[:3, :3], np.eye(3)), 2.0)

    def test_planar_candidates_are_disambiguated_by_consistency(self) -> None:
        expected = make_transform(np.eye(3), np.array([0.12, -0.03, 0.08]))
        groups: list[list[ExtrinsicCandidate]] = []
        wrong_translations = (
            np.array([0.8, 0.4, -0.1]),
            np.array([-0.7, 0.5, 0.3]),
            np.array([0.4, -0.8, 0.6]),
            np.array([-0.5, -0.6, -0.4]),
        )
        for index, wrong_translation in enumerate(wrong_translations):
            observed = observation(f"candidate_{index}", np.zeros((4, 2)))
            correct_transform = expected.copy()
            correct_transform[:3, 3] += [index * 0.0002, 0.0, 0.0]
            wrong_rotation, _ = cv2.Rodrigues(
                np.array([0.3 * index, -0.2 * index, 0.4])
            )
            wrong_transform = make_transform(
                wrong_rotation,
                wrong_translation,
            )
            groups.append(
                [
                    ExtrinsicCandidate(
                        "fixture",
                        observed,
                        np.eye(4),
                        np.eye(4),
                        wrong_transform,
                        0.05,
                        solution_index=0,
                        solution_count=2,
                    ),
                    ExtrinsicCandidate(
                        "fixture",
                        observed,
                        np.eye(4),
                        np.eye(4),
                        correct_transform,
                        0.15,
                        solution_index=1,
                        solution_count=2,
                    ),
                ]
            )
        selected = select_consistent_pnp_candidates(
            groups,
            translation_scale_mm=10.0,
            rotation_scale_deg=1.0,
        )
        self.assertTrue(all(item.solution_index == 1 for item in selected))
        fused = robust_fuse_transforms(
            [item.t_base_camera for item in selected]
        )
        np.testing.assert_allclose(
            fused[:3, 3],
            expected[:3, 3],
            atol=1e-3,
        )

    def test_ros_static_tf_parent_child(self) -> None:
        command = static_tf_command(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            "base_link",
            "camera_optical_frame",
        )
        self.assertIn("--frame-id base_link", command)
        self.assertIn("--child-frame-id camera_optical_frame", command)


if __name__ == "__main__":
    unittest.main()
