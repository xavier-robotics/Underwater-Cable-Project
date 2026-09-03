from __future__ import annotations

import unittest

from scripts.reaggregate_sam3_any_hit import reaggregate


class ReaggregateAnyHitTests(unittest.TestCase):
    def test_one_positive_frame_marks_whole_sample_damaged(self) -> None:
        clean = {
            "positive": False,
            "strong": False,
            "positive_score": 0.0,
            "combined_score": 0.1,
        }
        hit = {
            "positive": True,
            "strong": False,
            "positive_score": 0.3,
            "combined_score": 0.6,
        }
        source = [
            {
                "video": "/clip/0.mp4",
                "frame_votes": [{}, {}, {}],
                "diagnostics": {
                    "frames": [
                        {"damage": clean},
                        {"damage": hit},
                        {"damage": clean},
                    ]
                },
            }
        ]
        config = {
            "damage": {
                "min_positive_frames": 2,
                "weak_prompt_score": 0.18,
            }
        }

        result = reaggregate(source, config)[0]

        self.assertEqual(result["damage"], "damaged")
        self.assertEqual(result["class_name"], "damaged")
        self.assertFalse(result["position_evaluated"])


if __name__ == "__main__":
    unittest.main()
