from __future__ import annotations

import unittest

import numpy as np

from scripts.classify_samples_zero_shot import (
    aggregate_damage,
    aggregate_position,
    damage_frame_decision,
    position_frame_decision,
    restore_full_mask,
    visual_damage_features,
)
from scripts.compare_classification_results import compare


class ZeroShotClassifierTests(unittest.TestCase):
    damage_config = {
        "strong_prompt_score": 0.03,
        "weak_prompt_score": 0.012,
        "prompt_margin": 0.004,
        "frame_score_threshold": 0.48,
        "min_positive_frames": 2,
        "prompt_weight": 0.82,
        "visual_weight": 0.18,
        "visual_component_scale": 0.025,
        "visual_total_scale": 0.08,
    }

    def test_strong_damage_prompt_has_priority(self) -> None:
        frames = [
            damage_frame_decision(0.04, 0.01, 0.1, self.damage_config),
            damage_frame_decision(0.004, 0.02, 0.0, self.damage_config),
            damage_frame_decision(0.003, 0.02, 0.0, self.damage_config),
        ]

        damage, confidence = aggregate_damage(frames, self.damage_config)

        self.assertEqual(damage, "damaged")
        self.assertGreater(confidence, 0.55)

    def test_weak_damage_requires_multiple_frames(self) -> None:
        weak = damage_frame_decision(0.02, 0.005, 0.8, self.damage_config)
        clean = damage_frame_decision(0.004, 0.02, 0.0, self.damage_config)

        one_vote, _ = aggregate_damage([weak, clean, clean], self.damage_config)
        two_votes, _ = aggregate_damage([weak, weak, clean], self.damage_config)

        self.assertEqual(one_vote, "intact")
        self.assertEqual(two_votes, "damaged")

    def test_position_uses_prompt_and_sam3_area_cues(self) -> None:
        config = {
            "prompt_weight": 0.68,
            "area_prior_weight": 0.20,
            "contact_rule_weight": 0.12,
            "minimum_prompt_total": 0.003,
            "suspended_area_midpoint": 0.28,
            "suspended_area_scale": 0.06,
        }
        decision = position_frame_decision(
            exposed_score=0.005,
            suspended_score=0.03,
            sam3_area_ratio=0.35,
            contact_result={"state": "suspended"},
            config=config,
        )

        self.assertEqual(decision["position"], "suspended")
        self.assertGreater(decision["confidence"], 0.7)

    def test_position_aggregation_uses_majority(self) -> None:
        position, confidence, agreement = aggregate_position(
            [
                {"position": "exposed", "confidence": 0.8},
                {"position": "suspended", "confidence": 0.7},
                {"position": "suspended", "confidence": 0.9},
            ]
        )

        self.assertEqual(position, "suspended")
        self.assertEqual(agreement, 0.6667)
        self.assertGreater(confidence, 0.6)

    def test_visual_damage_feature_reacts_to_colored_patch(self) -> None:
        image = np.full((160, 320, 3), (45, 75, 55), dtype=np.uint8)
        mask = np.full((160, 320), 255, dtype=np.uint8)
        clean = visual_damage_features(image, mask, self.damage_config)
        image[60:100, 130:190] = (20, 180, 245)
        patched = visual_damage_features(image, mask, self.damage_config)

        self.assertEqual(clean["score"], 0.0)
        self.assertGreater(patched["score"], clean["score"])

    def test_crop_mask_is_restored_at_its_full_frame_coordinates(self) -> None:
        crop_mask = np.full((2, 3), 255, dtype=np.uint8)

        restored = restore_full_mask(
            crop_mask,
            (6, 8, 3),
            (2, 1, 5, 3),
        )

        self.assertEqual(restored.shape, (6, 8))
        self.assertEqual(np.count_nonzero(restored), 6)
        self.assertTrue(np.all(restored[1:3, 2:5] == 255))
        self.assertEqual(np.count_nonzero(restored[:1]), 0)

    def test_comparison_normalizes_old_damaged_class_names(self) -> None:
        baseline = [
            {
                "video": "/data/pool.mp4",
                "sample_id": "S001",
                "position": "suspended",
                "damage": "damaged",
                "class_id": 3,
                "class_name": "suspended_damaged",
            }
        ]
        candidate = [
            {
                "video": "/data/pool.mp4",
                "sample_id": "S001",
                "position": "exposed",
                "damage": "damaged",
                "class_id": 0,
                "class_name": "damaged",
            }
        ]

        report = compare(baseline, candidate)

        self.assertEqual(report["summary"]["agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
