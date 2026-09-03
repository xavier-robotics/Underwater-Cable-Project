from __future__ import annotations

import unittest

import cv2
import numpy as np

from scripts.classify_samples_sam3_only import (
    PromptDetection,
    aggregate_sam3_damage,
    damage_frame_decision,
    damage_prompt_groups,
    damage_support_mask,
    aggregate_support_position,
    filter_excluded_damage_detections,
    filter_endpoint_damage_detections,
    mask_overlap_coefficient,
    mask_axial_position,
    position_support_mask,
    prompt_best_scores,
    project_crop_detections,
    resolve_sam3_class,
    strongest_detections_by_prompt,
    support_geometry_features,
    support_position_frame_decision,
)


def detection(
    prompt: str,
    score: float,
    *,
    area_ratio: float = 0.03,
    reference_coverage: float = 0.1,
    axial_position: float | None = None,
    center_inside_reference: bool = True,
    mask: np.ndarray | None = None,
) -> PromptDetection:
    if mask is None:
        mask = np.ones((4, 4), dtype=np.uint8) * 255
    return PromptDetection(
        prompt=prompt,
        score=score,
        box_xyxy=(0, 0, 4, 4),
        mask=mask,
        area_ratio=area_ratio,
        mask_overlap=1.0,
        reference_coverage=reference_coverage,
        axial_position=axial_position,
        center_inside_reference=center_inside_reference,
    )


class SAM3OnlyClassifierTests(unittest.TestCase):
    config = {
        "positive_prompts": ["damaged cable", "cable with missing sheath"],
        "intact_prompts": ["smooth cable surface"],
        "local_max_area_ratio": 0.25,
        "consensus_min_score": 0.020,
        "consensus_bonus": 0.004,
        "strong_prompt_score": 0.045,
        "weak_prompt_score": 0.020,
        "prompt_margin": 0.003,
        "global_strong_score": 0.060,
        "global_margin": 0.010,
        "global_consensus_min_score": 0.020,
        "min_global_supporting_prompts": 2,
        "enable_global_strong": True,
        "frame_score_threshold": 0.40,
        "prompt_weight": 0.92,
        "visual_weight": 0.08,
    }

    def test_suspended_priority_overrides_damage_for_final_class(self) -> None:
        class_id, class_name = resolve_sam3_class(
            "suspended",
            "damaged",
            {"classification": {"priority": "suspended"}},
        )

        self.assertEqual(class_id, 2)
        self.assertEqual(class_name, "suspended_intact")

    def test_suspended_priority_keeps_exposed_damage_as_class_zero(self) -> None:
        class_id, class_name = resolve_sam3_class(
            "exposed",
            "damaged",
            {"classification": {"priority": "suspended"}},
        )

        self.assertEqual(class_id, 0)
        self.assertEqual(class_name, "damaged")

    def test_mask_axial_position_finds_ends_and_middle(self) -> None:
        reference = np.zeros((20, 100), dtype=np.uint8)
        reference[8:12, 5:95] = 255
        left = np.zeros_like(reference)
        left[8:12, 8:18] = 255
        middle = np.zeros_like(reference)
        middle[8:12, 45:55] = 255
        right = np.zeros_like(reference)
        right[8:12, 82:92] = 255

        positions = [
            mask_axial_position(mask, reference)
            for mask in (left, middle, right)
        ]

        self.assertTrue(
            (positions[0] < 0.2 and positions[2] > 0.8)
            or (positions[2] < 0.2 and positions[0] > 0.8)
        )
        self.assertAlmostEqual(positions[1], 0.5, delta=0.03)

    def test_terminal_positive_is_ignored_but_middle_positive_remains(self) -> None:
        kept, ignored = filter_endpoint_damage_detections(
            [
                detection("metal patch", 0.9, axial_position=0.05),
                detection("metal patch", 0.8, axial_position=0.55),
                detection("intact pipe", 0.7, axial_position=0.95),
            ],
            ["metal patch"],
            {
                "middle_axis_min_fraction": 0.20,
                "middle_axis_max_fraction": 0.80,
            },
        )

        self.assertEqual([item.score for item in kept], [0.8, 0.7])
        self.assertEqual(len(ignored), 1)
        self.assertEqual(ignored[0]["terminal_side"], "end_a")

    def test_endpoint_only_damage_cannot_make_frame_positive(self) -> None:
        decision = damage_frame_decision(
            [
                detection(
                    "damaged cable",
                    0.90,
                    axial_position=0.05,
                ),
                detection(
                    "smooth cable surface",
                    0.20,
                    axial_position=0.50,
                ),
            ],
            visual_score=0.0,
            config={
                **self.config,
                "middle_axis_min_fraction": 0.20,
                "middle_axis_max_fraction": 0.80,
            },
        )

        self.assertFalse(decision["positive"])
        self.assertEqual(decision["positive_score"], 0.0)
        self.assertEqual(decision["ignored_endpoint_region_count"], 1)

    def test_prompt_scores_can_be_limited_to_local_masks(self) -> None:
        detections = [
            detection("damaged cable", 0.08, area_ratio=0.60),
            detection("damaged cable", 0.03, area_ratio=0.04),
        ]

        scores = prompt_best_scores(
            detections,
            ["damaged cable"],
            max_area_ratio=0.25,
        )

        self.assertEqual(scores["damaged cable"], 0.03)

    def test_positive_score_can_reject_whole_pipe_response(self) -> None:
        scores = prompt_best_scores(
            [
                detection(
                    "torn pipe",
                    0.80,
                    area_ratio=0.10,
                    reference_coverage=0.90,
                ),
                detection(
                    "torn pipe",
                    0.45,
                    area_ratio=0.02,
                    reference_coverage=0.12,
                ),
            ],
            ["torn pipe"],
            max_area_ratio=0.25,
            max_reference_coverage=0.35,
        )

        self.assertEqual(scores["torn pipe"], 0.45)

    def test_prompt_specific_floor_rejects_weak_foil_response(self) -> None:
        config = {
            **self.config,
            "positive_prompts": ["torn silver foil"],
            "positive_prompt_min_scores": {"torn silver foil": 0.70},
        }
        decision = damage_frame_decision(
            [detection("torn silver foil", 0.56)],
            visual_score=0.0,
            config=config,
        )

        self.assertFalse(decision["positive"])
        self.assertEqual(decision["positive_score"], 0.0)

    def test_strong_foil_breach_can_bypass_label_and_coverage_filters(self) -> None:
        config = {
            **self.config,
            "positive_prompts": ["torn silver foil"],
            "positive_prompt_min_scores": {"torn silver foil": 0.70},
            "positive_max_reference_coverage": 0.35,
            "positive_prompt_max_reference_coverage": {
                "torn silver foil": 1.0,
            },
            "positive_prompt_exclusion_bypass_scores": {
                "torn silver foil": 0.70,
            },
            "positive_prompt_exclusion_bypass_margins": {
                "torn silver foil": 0.10,
            },
            "exclusion_prompts": ["white label"],
            "exclusion_min_score": 0.12,
            "exclusion_score_ratio": 0.55,
            "exclusion_overlap_coefficient": 0.30,
        }
        decision = damage_frame_decision(
            [
                detection(
                    "torn silver foil",
                    0.82,
                    reference_coverage=0.93,
                ),
                detection("white label", 0.65),
            ],
            visual_score=0.0,
            config=config,
        )

        self.assertTrue(decision["positive"])
        self.assertEqual(decision["positive_score"], 0.82)

    def test_near_tied_label_blocks_foil_breach_bypass(self) -> None:
        config = {
            **self.config,
            "positive_prompts": ["torn silver foil"],
            "positive_prompt_min_scores": {"torn silver foil": 0.70},
            "positive_prompt_exclusion_bypass_scores": {
                "torn silver foil": 0.70,
            },
            "positive_prompt_exclusion_bypass_margins": {
                "torn silver foil": 0.10,
            },
            "exclusion_prompts": ["white label"],
            "exclusion_min_score": 0.12,
            "exclusion_score_ratio": 0.55,
            "exclusion_overlap_coefficient": 0.30,
        }
        decision = damage_frame_decision(
            [
                detection("torn silver foil", 0.77),
                detection("white label", 0.70),
            ],
            visual_score=0.0,
            config=config,
        )

        self.assertFalse(decision["positive"])
        self.assertEqual(decision["positive_score"], 0.0)

    def test_gallery_keeps_one_strong_local_detection_per_prompt(self) -> None:
        selected = strongest_detections_by_prompt(
            [
                detection("metal patch", 0.04, area_ratio=0.03),
                detection("metal patch", 0.08, area_ratio=0.04),
                detection("metal patch", 0.20, area_ratio=0.60),
                detection(
                    "metal patch",
                    0.30,
                    area_ratio=0.08,
                    reference_coverage=0.90,
                ),
                detection("broken jacket", 0.01, area_ratio=0.02),
                detection("intact cable", 0.50, area_ratio=0.02),
            ],
            ["metal patch", "broken jacket"],
            max_area_ratio=0.25,
            max_reference_coverage=0.35,
            min_score=0.02,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].prompt, "metal patch")
        self.assertEqual(selected[0].score, 0.08)

    def test_crop_detection_projects_to_original_frame(self) -> None:
        projected = project_crop_detections(
            [detection("metal patch", 0.08, area_ratio=0.25)],
            crop_shape=(4, 4, 3),
            full_shape=(10, 12, 3),
            crop_bbox_xyxy=(2, 1, 10, 9),
        )

        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0].box_xyxy, (2, 1, 10, 9))
        self.assertEqual(projected[0].mask.shape, (10, 12))
        self.assertEqual(np.count_nonzero(projected[0].mask), 64)

    def test_damage_support_mask_fills_gap_in_cable_silhouette(self) -> None:
        reference = np.zeros((20, 40), dtype=np.uint8)
        reference[6:14, 2:17] = 255
        reference[6:14, 23:38] = 255

        support = damage_support_mask(reference)

        self.assertTrue(np.all(support[8:12, 17:23] == 255))
        self.assertEqual(np.count_nonzero(support[:4]), 0)

    def test_multiple_damage_prompts_add_consensus_evidence(self) -> None:
        decision = damage_frame_decision(
            [
                detection("damaged cable", 0.029),
                detection("cable with missing sheath", 0.021),
                detection("smooth cable surface", 0.027),
            ],
            visual_score=0.0,
            config=self.config,
        )

        self.assertEqual(decision["supporting_prompts"], 2)
        self.assertAlmostEqual(decision["positive_score"], 0.033)
        self.assertTrue(decision["positive"])

    def test_synonyms_in_one_group_do_not_duplicate_consensus(self) -> None:
        config = {
            **self.config,
            "positive_prompt_groups": {
                "metal_exposure": [
                    "silver patch",
                    "metallic patch",
                ],
                "sheath_damage": ["damaged cable sheath"],
            },
        }
        decision = damage_frame_decision(
            [
                detection("silver patch", 0.029),
                detection("metallic patch", 0.027),
                detection("smooth cable surface", 0.020),
            ],
            visual_score=0.0,
            config=config,
        )

        self.assertEqual(decision["supporting_groups"], 1)
        self.assertAlmostEqual(decision["positive_score"], 0.029)

    def test_independent_groups_add_consensus_evidence(self) -> None:
        config = {
            **self.config,
            "positive_prompt_groups": {
                "metal_exposure": ["silver patch"],
                "sheath_damage": ["damaged cable sheath"],
            },
        }
        decision = damage_frame_decision(
            [
                detection("silver patch", 0.029),
                detection("damaged cable sheath", 0.021),
                detection("smooth cable surface", 0.027),
            ],
            visual_score=0.0,
            config=config,
        )

        self.assertEqual(decision["supporting_groups"], 2)
        self.assertAlmostEqual(decision["positive_score"], 0.033)
        self.assertTrue(decision["positive"])

    def test_subthreshold_groups_cannot_accumulate_into_damage(self) -> None:
        config = {
            **self.config,
            "positive_prompt_groups": {
                "semantic_damage": ["surface damage"],
                "metal_exposure": ["metal patch"],
                "structural_damage": ["broken jacket"],
            },
        }
        decision = damage_frame_decision(
            [
                detection("surface damage", 0.014),
                detection("metal patch", 0.019),
                detection("broken jacket", 0.010),
            ],
            visual_score=0.03,
            config=config,
        )

        self.assertEqual(decision["supporting_groups"], 0)
        self.assertAlmostEqual(decision["positive_score"], 0.019)
        self.assertFalse(decision["positive"])

    def test_flat_prompt_config_remains_backward_compatible(self) -> None:
        groups = damage_prompt_groups(self.config)

        self.assertEqual(
            groups,
            {
                "damaged cable": ["damaged cable"],
                "cable with missing sheath": [
                    "cable with missing sheath"
                ],
            },
        )

    def test_single_near_tie_is_not_damage(self) -> None:
        decision = damage_frame_decision(
            [
                detection("damaged cable", 0.029),
                detection("smooth cable surface", 0.027),
            ],
            visual_score=0.0,
            config=self.config,
        )

        self.assertFalse(decision["positive"])

    def test_direct_prompt_mode_uses_configured_marker_threshold(self) -> None:
        config = {
            **self.config,
            "decision_mode": "direct_prompt_threshold",
            "direct_positive_prompt_thresholds": {
                "damaged cable": 0.20,
            },
        }
        positive = damage_frame_decision(
            [detection("damaged cable", 0.21)],
            visual_score=0.0,
            config=config,
        )
        negative = damage_frame_decision(
            [detection("damaged cable", 0.19)],
            visual_score=1.0,
            config=config,
        )

        self.assertTrue(positive["positive"])
        self.assertEqual(positive["direct_positive_prompts"], ["damaged cable"])
        self.assertFalse(negative["positive"])

    def test_single_frame_direct_prompt_uses_still_image_floor(self) -> None:
        config = {
            **self.config,
            "decision_mode": "direct_prompt_threshold",
            "direct_positive_prompt_thresholds": {"metal patch": 0.026},
            "single_frame_direct_prompt_thresholds": {
                "metal patch": 0.0275,
            },
            "direct_require_center_inside_reference": True,
        }
        rejected = damage_frame_decision(
            [detection("metal patch", 0.0271)],
            visual_score=0.0,
            config=config,
        )
        accepted = damage_frame_decision(
            [detection("metal patch", 0.0279)],
            visual_score=0.0,
            config=config,
        )

        self.assertEqual(
            aggregate_sam3_damage([rejected], config)[0],
            "intact",
        )
        self.assertEqual(
            aggregate_sam3_damage([accepted], config)[0],
            "damaged",
        )

    def test_borderline_damage_requires_two_matching_video_frames(self) -> None:
        config = {
            **self.config,
            "decision_mode": "direct_prompt_threshold",
            "direct_positive_prompt_thresholds": {"metal patch": 0.026},
            "single_frame_direct_prompt_thresholds": {
                "metal patch": 0.0275,
            },
            "direct_strong_prompt_thresholds": {"metal patch": 0.05},
            "direct_require_center_inside_reference": True,
            "direct_borderline_min_matching_frames": 2,
            "direct_borderline_axial_tolerance": 0.10,
        }
        first = damage_frame_decision(
            [detection("metal patch", 0.030, axial_position=0.20)],
            visual_score=0.0,
            config=config,
        )
        matching = damage_frame_decision(
            [detection("metal patch", 0.031, axial_position=0.78)],
            visual_score=0.0,
            config=config,
        )
        distant = damage_frame_decision(
            [detection("metal patch", 0.032, axial_position=0.50)],
            visual_score=0.0,
            config=config,
        )

        self.assertEqual(
            aggregate_sam3_damage([first, matching], config)[0],
            "damaged",
        )
        self.assertEqual(
            aggregate_sam3_damage([first, distant], config)[0],
            "intact",
        )

    def test_off_cable_direct_candidate_does_not_confirm_damage(self) -> None:
        config = {
            **self.config,
            "decision_mode": "direct_prompt_threshold",
            "direct_positive_prompt_thresholds": {"metal patch": 0.026},
            "single_frame_direct_prompt_thresholds": {
                "metal patch": 0.0275,
            },
            "direct_require_center_inside_reference": True,
        }
        decision = damage_frame_decision(
            [
                detection(
                    "metal patch",
                    0.20,
                    center_inside_reference=False,
                )
            ],
            visual_score=0.0,
            config=config,
        )

        self.assertEqual(
            aggregate_sam3_damage([decision], config)[0],
            "intact",
        )

    def test_overlapping_label_suppresses_false_damage_region(self) -> None:
        config = {
            **self.config,
            "exclusion_prompts": ["white paper label"],
            "exclusion_min_score": 0.12,
            "exclusion_score_ratio": 0.72,
            "exclusion_overlap_coefficient": 0.30,
        }
        decision = damage_frame_decision(
            [
                detection("damaged cable", 0.80),
                detection("white paper label", 0.75),
            ],
            visual_score=0.0,
            config=config,
        )

        self.assertFalse(decision["positive"])
        self.assertEqual(len(decision["suppressed_positive_regions"]), 1)

    def test_near_tied_number_label_suppresses_false_metal_marker(self) -> None:
        config = {
            **self.config,
            "exclusion_prompts": ["white numbered label"],
            "exclusion_min_score": 0.12,
            "exclusion_score_ratio": 0.55,
            "exclusion_overlap_coefficient": 0.30,
        }
        decision = damage_frame_decision(
            [
                detection("damaged cable", 0.95),
                detection("white numbered label", 0.54),
            ],
            visual_score=0.0,
            config=config,
        )

        self.assertFalse(decision["positive"])

    def test_separate_label_does_not_suppress_damage_region(self) -> None:
        positive_mask = np.zeros((4, 4), dtype=np.uint8)
        positive_mask[2:, 2:] = 255
        label_mask = np.zeros((4, 4), dtype=np.uint8)
        label_mask[0, 0] = 255
        positive = detection("damaged cable", 0.80, mask=positive_mask)
        label = detection("white paper label", 0.75, mask=label_mask)
        filtered, suppressed = filter_excluded_damage_detections(
            [positive, label],
            ["damaged cable"],
            ["white paper label"],
            {
                **self.config,
                "exclusion_min_score": 0.12,
                "exclusion_score_ratio": 0.72,
                "exclusion_overlap_coefficient": 0.30,
            },
        )

        self.assertIn(positive, filtered)
        self.assertEqual(suppressed, [])

    def test_mask_overlap_coefficient_uses_smaller_region(self) -> None:
        large = np.ones((6, 6), dtype=np.uint8)
        small = np.zeros((6, 6), dtype=np.uint8)
        small[2:4, 2:4] = 1

        self.assertEqual(mask_overlap_coefficient(large, small), 1.0)

    def test_support_search_mask_expands_around_cable(self) -> None:
        reference = np.zeros((20, 30), dtype=np.uint8)
        reference[8:12, 12:18] = 255

        search = position_support_mask(reference, margin_px=3)

        self.assertTrue(np.all(search[5:15, 9:21] == 255))
        self.assertEqual(np.count_nonzero(search[:4]), 0)

    def test_white_support_is_direct_suspension_evidence(self) -> None:
        config = {
            "support_prompt_thresholds": {
                "white PVC support": 0.50,
                "white T-shaped stand": 0.25,
            },
            "minimum_support_reference_overlap": 0.10,
        }
        frame = support_position_frame_decision(
            [
                detection("white PVC support", 0.82),
                detection("white T-shaped stand", 0.40),
            ],
            ["white PVC support", "white T-shaped stand"],
            config,
            np.ones((4, 4), dtype=np.uint8),
        )
        position, confidence, agreement = aggregate_support_position(
            [
                frame,
                support_position_frame_decision(
                    [],
                    ["white PVC support", "white T-shaped stand"],
                    config,
                    np.ones((4, 4), dtype=np.uint8),
                ),
            ]
        )

        self.assertTrue(frame["support_positive"])
        self.assertEqual(position, "suspended")
        self.assertGreater(confidence, 0.8)
        self.assertEqual(agreement, 0.5)

    def test_support_requires_shape_consensus_and_target_overlap(self) -> None:
        config = {
            "support_prompt_thresholds": {
                "white PVC support": 0.50,
                "white T-shaped stand": 0.25,
            },
            "minimum_support_reference_overlap": 0.10,
        }
        reference = np.zeros((4, 4), dtype=np.uint8)
        reference[2:, 2:] = 255
        distant_mask = np.zeros((4, 4), dtype=np.uint8)
        distant_mask[:2, :2] = 255
        decision = support_position_frame_decision(
            [
                detection("white PVC support", 0.90, mask=distant_mask),
                detection("white T-shaped stand", 0.80, mask=distant_mask),
            ],
            ["white PVC support", "white T-shaped stand"],
            config,
            reference,
        )

        self.assertFalse(decision["support_positive"])
        self.assertEqual(decision["eligible_support_detections"], 0)

    def test_endpoint_perpendicular_line_is_support_geometry(self) -> None:
        image = np.full((240, 240, 3), 60, dtype=np.uint8)
        reference = np.zeros((240, 240), dtype=np.uint8)
        reference[20:205, 100:140] = 255
        image[reference > 0] = 35
        cv2.line(image, (45, 205), (195, 205), (225, 225, 225), 8)

        geometry = support_geometry_features(image, reference, {})

        self.assertGreaterEqual(geometry["score"], 0.11)
        self.assertGreaterEqual(geometry["outside_fraction"], 0.55)

    def test_perpendicular_line_at_cable_middle_is_not_support(self) -> None:
        image = np.full((240, 240, 3), 60, dtype=np.uint8)
        reference = np.zeros((240, 240), dtype=np.uint8)
        reference[20:205, 100:140] = 255
        image[reference > 0] = 35
        cv2.line(image, (45, 112), (195, 112), (225, 225, 225), 8)

        geometry = support_geometry_features(image, reference, {})

        self.assertEqual(geometry["score"], 0.0)

    def test_geometry_only_mode_needs_no_support_prompt(self) -> None:
        decision = support_position_frame_decision(
            [],
            [],
            {
                "support_decision_mode": "geometry_only",
                "support_geometry_score_threshold": 0.11,
            },
            geometry={"score": 0.13, "line_xyxy": [1, 2, 3, 4]},
        )

        self.assertTrue(decision["support_positive"])
        self.assertEqual(decision["position"], "suspended")
        self.assertIn("perpendicular_endpoint_support", decision["active_cues"])

    def test_global_damage_requires_high_score_and_margin(self) -> None:
        decision = damage_frame_decision(
            [
                detection("damaged cable", 0.08, area_ratio=0.50),
                detection(
                    "cable with missing sheath",
                    0.07,
                    area_ratio=0.50,
                ),
                detection("smooth cable surface", 0.05, area_ratio=0.50),
            ],
            visual_score=0.0,
            config=self.config,
        )

        self.assertTrue(decision["global_strong"])
        self.assertTrue(decision["strong"])
        self.assertTrue(decision["positive"])


if __name__ == "__main__":
    unittest.main()
