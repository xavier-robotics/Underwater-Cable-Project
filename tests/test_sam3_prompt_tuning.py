from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from scripts.classify_samples_sam3_only import PromptDetection
from scripts.tune_sam3_prompts import (
    evaluate_damage,
    prompt_catalog,
    summarize_prompt,
)


class SAM3PromptTuningTests(unittest.TestCase):
    def test_project_search_catalog_expands_all_noun_families(self) -> None:
        config = yaml.safe_load(
            Path("configs/sam3_prompt_search.yaml").read_text(
                encoding="utf-8"
            )
        )

        catalog = prompt_catalog(config)

        self.assertEqual(len(catalog["localization"]), 14)
        self.assertEqual(len(catalog["damage"]), 66)
        self.assertEqual(len(catalog["intact"]), 20)
        self.assertIn("damaged beam", catalog["damage_families"]["beam"])
        self.assertIn(
            "pipe touching pool floor",
            catalog["exposed_families"]["pipe"],
        )
        self.assertIn(
            "silver metallic patch on black cable",
            catalog["damage_candidate_groups"]["metal_exposure"],
        )

    def test_selected_prompts_are_part_of_the_sweep_catalog(self) -> None:
        config = yaml.safe_load(
            Path("configs/sam3_prompt_search.yaml").read_text(
                encoding="utf-8"
            )
        )
        catalog = prompt_catalog(config)
        selected = config["selected"]

        self.assertIn(
            selected["localization_prompt"],
            catalog["localization"],
        )
        self.assertTrue(
            set(selected["damage_positive_prompts"])
            <= set(catalog["damage"])
        )
        self.assertTrue(
            set(selected["intact_prompts"]) <= set(catalog["intact"])
        )

    def test_prompt_summary_computes_reference_iou(self) -> None:
        mask = np.ones((2, 2), dtype=np.uint8)
        detection = PromptDetection(
            prompt="pipe",
            score=0.8,
            box_xyxy=(0, 0, 2, 2),
            mask=mask,
            area_ratio=0.1,
            mask_overlap=1.0,
            reference_coverage=0.5,
        )

        summary = summarize_prompt(
            [detection],
            reference_area_ratio=0.2,
            local_max_area_ratio=0.25,
        )

        self.assertAlmostEqual(summary["best_iou"], 0.5)
        self.assertEqual(summary["local_score"], 0.8)

    def test_damage_evaluation_uses_cached_prompt_scores(self) -> None:
        config = yaml.safe_load(
            Path("configs/sam3_only_classifier.yaml").read_text(
                encoding="utf-8"
            )
        )["damage"]

        def frame(positive_score: float) -> dict:
            return {
                "visual_damage": {"score": 0.0},
                "crop_prompts": {
                    "damaged cable": {
                        "local_score": positive_score,
                        "global_score": positive_score,
                    },
                    "smooth cable surface": {
                        "local_score": 0.01,
                        "global_score": 0.01,
                    },
                },
            }

        features = [
            {
                "video": "damaged.mp4",
                "sample_id": "S001",
                "reference": {"damage": "damaged"},
                "frames": [frame(0.50), frame(0.50), frame(0.0)],
            },
            {
                "video": "intact.mp4",
                "sample_id": "S001",
                "reference": {"damage": "intact"},
                "frames": [frame(0.0), frame(0.0), frame(0.0)],
            },
        ]

        report = evaluate_damage(
            features,
            ["damaged cable"],
            ["smooth cable surface"],
            config,
        )

        self.assertEqual(report["correct"], 2)
        self.assertEqual(report["false_positive"], 0)
        self.assertEqual(report["false_negative"], 0)


if __name__ == "__main__":
    unittest.main()
