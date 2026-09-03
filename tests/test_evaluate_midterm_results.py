from __future__ import annotations

import unittest

from scripts.evaluate_midterm_results import evaluate, sample_number


class EvaluateMidtermResultsTests(unittest.TestCase):
    def test_sample_number_accepts_filename_labeled_sample_id(self) -> None:
        number = sample_number(
            {
                "video": "/gt/sample26_2.jpg",
                "sample_id": "sample26_2",
            }
        )

        self.assertEqual(number, 26)

    def test_evaluate_builds_truth_by_prediction_confusion_matrix(self) -> None:
        report = evaluate(
            [
                {
                    "video": "/clips/0.mp4",
                    "class_id": 0,
                    "class_name": "damaged",
                },
                {
                    "video": "/clips/1.mp4",
                    "class_id": 1,
                    "class_name": "exposed_intact",
                },
                {
                    "video": "/clips/2.mp4",
                    "class_id": 2,
                    "class_name": "suspended_intact",
                },
            ],
            {0: 0, 1: 1, 2: 2},
        )

        self.assertEqual(report["correct"], 3)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(
            report["confusion_matrix_rows_truth_columns_prediction"],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )


if __name__ == "__main__":
    unittest.main()
