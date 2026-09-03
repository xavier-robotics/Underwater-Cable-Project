from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from scripts.select_sample_frames import (
    FrameMetric,
    SAM3Scorer,
    file_sha256,
    find_segments,
    merge_close_fragments,
    pick_representatives,
    read_frame,
)


def make_metric(
    timestamp_sec: float,
    *,
    visible: bool = True,
    cable_score: float = 0.5,
    smooth_score: float = 0.5,
    sharpness: float = 10.0,
) -> FrameMetric:
    return FrameMetric(
        frame_idx=int(round(timestamp_sec * 30)),
        timestamp_sec=timestamp_sec,
        cable_score=cable_score,
        smooth_score=smooth_score,
        sharpness=sharpness,
        brightness=100.0,
        dark_ratio=0.1,
        largest_box_ratio=0.1,
        span_ratio=0.5,
        visible=visible,
    )


class SegmentSelectionTests(unittest.TestCase):
    def test_read_frame_falls_back_to_sequential_decode(self) -> None:
        target = np.full((2, 3, 3), 7, dtype=np.uint8)

        class FakeCapture:
            def __init__(self, frames: list[np.ndarray], seek_fails: bool) -> None:
                self.frames = frames
                self.seek_fails = seek_fails
                self.index = 0

            def isOpened(self) -> bool:
                return True

            def set(self, _property: int, value: float) -> bool:
                self.index = int(value)
                return True

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.seek_fails:
                    return False, None
                if self.index >= len(self.frames):
                    return False, None
                frame = self.frames[self.index]
                self.index += 1
                return True, frame

            def release(self) -> None:
                pass

        direct = FakeCapture([], seek_fails=True)
        sequential = FakeCapture(
            [
                np.zeros_like(target),
                np.ones_like(target),
                target,
            ],
            seek_fails=False,
        )
        with patch(
            "scripts.select_sample_frames.cv2.VideoCapture",
            side_effect=[direct, sequential],
        ):
            frame = read_frame(Path("long-gop.mp4"), 2)

        np.testing.assert_array_equal(frame, target)

    def test_file_sha256(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.pt"
            path.write_bytes(b"sam3")

            digest = file_sha256(path, chunk_size=2)

        self.assertEqual(
            digest,
            "12ef11821e043e83112d476c8ea182a90528ec78626dc941e48754d0b879cd1f",
        )

    def test_short_spike_does_not_bridge_real_segments(self) -> None:
        metrics = []
        for second in range(81, 113):
            visible = 81 <= second <= 89 or second == 91 or 93 <= second <= 112
            metrics.append(make_metric(float(second), visible=visible))

        segments = find_segments(
            metrics,
            mode="scan",
            min_segment_sec=0.5,
            merge_gap_sec=2.0,
            expected_samples=None,
        )

        self.assertEqual(
            [(segment[0].timestamp_sec, segment[-1].timestamp_sec) for segment in segments],
            [(81.0, 89.0), (93.0, 112.0)],
        )

    def test_merge_gap_is_stable_at_floating_point_boundary(self) -> None:
        fragments = [
            [make_metric(0.0), make_metric(1.0)],
            [make_metric(3.000006), make_metric(4.000006)],
        ]

        merged = merge_close_fragments(fragments, merge_gap_sec=2.0)

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]), 4)

    def test_representatives_require_a_raw_detection(self) -> None:
        segment = [
            make_metric(0.0, cable_score=0.7, smooth_score=0.6, sharpness=10.0),
            make_metric(1.0, cable_score=0.6, smooth_score=0.7, sharpness=12.0),
            make_metric(2.0, cable_score=0.0, smooth_score=1.0, sharpness=1000.0),
            make_metric(3.0, cable_score=0.8, smooth_score=0.8, sharpness=11.0),
            make_metric(4.0, cable_score=0.5, smooth_score=0.6, sharpness=9.0),
        ]

        selected = pick_representatives(segment, frames_per_sample=3)

        self.assertEqual(len(selected), 3)
        self.assertTrue(all(metric.cable_score > 0.0 for metric in selected))
        self.assertNotIn(2.0, [metric.timestamp_sec for metric in selected])

    def test_can_select_five_temporally_distributed_representatives(self) -> None:
        segment = [
            make_metric(float(idx), cable_score=0.5 + idx / 100, smooth_score=0.6)
            for idx in range(20)
        ]

        selected = pick_representatives(segment, frames_per_sample=5)

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, sorted(selected, key=lambda metric: metric.timestamp_sec))
        self.assertLessEqual(selected[0].timestamp_sec, 4.0)
        self.assertGreaterEqual(selected[-1].timestamp_sec, 15.0)

    def test_sam3_bfloat16_tensor_conversion(self) -> None:
        scorer = SAM3Scorer.__new__(SAM3Scorer)
        scorer._torch = torch
        value = torch.tensor([0.25, 0.5], dtype=torch.bfloat16)

        converted = scorer._to_numpy(value)

        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_allclose(converted, [0.25, 0.5])


if __name__ == "__main__":
    unittest.main()
