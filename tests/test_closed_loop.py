from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from scripts.classify_samples_gpt import build_contact_sheet, collect_samples
from scripts.run_closed_loop import main as run_closed_loop_main
from scripts.validate_agent_results import validate_results


class ClosedLoopTests(unittest.TestCase):
    def make_sample_manifest(self, root: Path) -> Path:
        frames = []
        for frame_idx in range(5):
            full_path = root / f"full_{frame_idx}.jpg"
            detail_path = root / f"detail_{frame_idx}.jpg"
            full_image = np.full((180, 320, 3), (20 + frame_idx, 80, 140), dtype=np.uint8)
            detail_image = np.full((240, 120, 3), (140, 80, 20 + frame_idx), dtype=np.uint8)
            cv2.imwrite(str(full_path), full_image)
            cv2.imwrite(str(detail_path), detail_image)
            frames.append(
                {
                    "frame_idx": frame_idx,
                    "full_frame_path": str(full_path),
                    "masked_crop_path": str(detail_path),
                }
            )

        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "video": "/data/pool.mp4",
                    "segments": [
                        {
                            "sample_id": "S001",
                            "start_sec": 0.0,
                            "end_sec": 4.0,
                            "selected_frames": frames,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

    def test_first_attempt_uses_three_frames_with_context_and_detail(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manifest_path = self.make_sample_manifest(Path(tmp_dir))

            samples = collect_samples(
                [manifest_path],
                "full_frame_path",
                evidence_mode="full-and-crop",
                max_frames_per_sample=3,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual([image.frame_idx for image in samples[0].images], [0, 0, 2, 2, 4, 4])
        self.assertEqual([image.kind for image in samples[0].images], ["context", "detail"] * 3)

    def test_contact_sheet_reduces_sample_to_one_image(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = self.make_sample_manifest(root)
            samples = collect_samples(
                [manifest_path],
                "full_frame_path",
                evidence_mode="contact-sheet",
                max_frames_per_sample=3,
            )

            sheet_path = build_contact_sheet(samples[0], root / "out", attempt_id=1)
            sheet = cv2.imread(str(sheet_path), cv2.IMREAD_COLOR)

            self.assertIsNotNone(sheet)
            self.assertEqual(sheet.shape[:2], (1002, 1280))
            self.assertEqual(samples[0].frame_ids, [0, 2, 4])
            self.assertEqual(len(samples[0].images), 1)
            self.assertEqual(samples[0].images[0].kind, "contact_sheet")
            self.assertEqual(len(samples[0].source_images or []), 6)

    def test_validator_accepts_consistent_high_confidence_result(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_paths = []
            for idx in range(3):
                path = root / f"{idx}.jpg"
                path.write_bytes(b"image")
                image_paths.append(str(path))
            request = {
                "video": "/data/pool.mp4",
                "sample_id": "S001",
                "attempt_id": 1,
                "selected_frame_count": 3,
                "image_paths": image_paths,
            }
            result = {
                **request,
                "position": "exposed",
                "position_confidence": 0.9,
                "damage": "intact",
                "damage_confidence": 0.8,
                "class_id": 1,
                "class_name": "exposed_intact",
                "needs_review": False,
                "evidence_consistency": "consistent",
                "reason_codes": [],
            }

            report = validate_results([request], [result], min_frames=3)

        self.assertEqual(report["summary"]["pass"], 1)
        self.assertEqual(report["items"][0]["status"], "pass")

    def test_validator_accepts_complete_contact_sheet_request(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            sheet_path = Path(tmp_dir) / "sheet.jpg"
            sheet_path.write_bytes(b"image")
            request = {
                "video": "/data/pool.mp4",
                "sample_id": "S001",
                "attempt_id": 1,
                "selected_frame_count": 3,
                "frame_ids": [10, 20, 30],
                "evidence_mode": "contact-sheet",
                "source_frames": [
                    {"frame_idx": frame_idx, "has_context": True, "has_detail": True}
                    for frame_idx in (10, 20, 30)
                ],
                "images": [
                    {"path": str(sheet_path), "kind": "contact_sheet", "frame_idx": -1}
                ],
                "image_paths": [str(sheet_path)],
            }
            result = {
                **request,
                "position": "exposed",
                "position_confidence": 0.9,
                "damage": "intact",
                "damage_confidence": 0.9,
                "class_id": 1,
                "class_name": "exposed_intact",
                "needs_review": False,
                "evidence_consistency": "consistent",
                "reason_codes": [],
                "frame_votes": [
                    {
                        "frame_idx": frame_idx,
                        "position": "exposed",
                        "position_confidence": 0.9,
                        "damage_evidence": "no_visible_damage",
                        "damage_confidence": 0.9,
                        "usable": True,
                    }
                    for frame_idx in (10, 20, 30)
                ],
            }

            report = validate_results([request], [result], min_frames=3)

        self.assertEqual(report["items"][0]["status"], "pass")

    def test_validator_retries_low_confidence_or_mixed_evidence(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.jpg"
            image_path.write_bytes(b"image")
            request = {
                "video": "/data/pool.mp4",
                "sample_id": "S001",
                "attempt_id": 1,
                "selected_frame_count": 3,
                "image_paths": [str(image_path)],
            }
            result = {
                **request,
                "position": "suspended",
                "position_confidence": 0.6,
                "damage": "damaged",
                "damage_confidence": 0.9,
                "class_id": 0,
                "class_name": "damaged",
                "needs_review": True,
                "evidence_consistency": "mixed",
                "reason_codes": ["conflicting_frames"],
            }

            report = validate_results([request], [result], min_frames=3)

        item = report["items"][0]
        self.assertEqual(item["status"], "retry")
        self.assertIn("low_position_confidence", item["reason_codes"])
        self.assertIn("agent_requested_review", item["reason_codes"])
        self.assertNotIn("class_mapping_mismatch", item["reason_codes"])

    def test_validator_accepts_position_majority_and_clear_damage(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            images = []
            for frame_idx in range(3):
                for kind in ("context", "detail"):
                    path = root / f"{frame_idx}_{kind}.jpg"
                    path.write_bytes(b"image")
                    images.append({"path": str(path), "kind": kind, "frame_idx": frame_idx})
            request = {
                "video": "/data/pool.mp4",
                "sample_id": "S001",
                "attempt_id": 1,
                "selected_frame_count": 3,
                "evidence_mode": "full-and-crop",
                "images": images,
                "image_paths": [image["path"] for image in images],
            }
            result = {
                **request,
                "position": "exposed",
                "position_confidence": 0.9,
                "damage": "damaged",
                "damage_confidence": 0.9,
                "class_id": 0,
                "class_name": "damaged",
                "needs_review": False,
                "evidence_consistency": "mixed",
                "reason_codes": [],
                "frame_votes": [
                    {
                        "frame_idx": 0,
                        "position": "exposed",
                        "position_confidence": 0.9,
                        "damage_evidence": "no_visible_damage",
                        "damage_confidence": 0.85,
                        "usable": True,
                    },
                    {
                        "frame_idx": 1,
                        "position": "suspended",
                        "position_confidence": 0.7,
                        "damage_evidence": "clear_damage",
                        "damage_confidence": 0.9,
                        "usable": True,
                    },
                    {
                        "frame_idx": 2,
                        "position": "exposed",
                        "position_confidence": 0.88,
                        "damage_evidence": "no_visible_damage",
                        "damage_confidence": 0.8,
                        "usable": True,
                    },
                ],
            }

            report = validate_results([request], [result], min_frames=3)

        self.assertEqual(report["items"][0]["status"], "pass")
        self.assertEqual(report["items"][0]["diagnostics"]["position_agreement"], 0.6667)
        self.assertEqual(report["items"][0]["diagnostics"]["clear_damage_frames"], 1)

    def test_damage_class_is_independent_of_position(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.jpg"
            image_path.write_bytes(b"image")
            requests = [
                {
                    "video": "/data/pool.mp4",
                    "sample_id": f"S00{index}",
                    "attempt_id": 1,
                    "selected_frame_count": 3,
                    "image_paths": [str(image_path)],
                }
                for index in (1, 2)
            ]
            results = [
                {
                    **request,
                    "position": position,
                    "position_confidence": 0.9,
                    "damage": "damaged",
                    "damage_confidence": 0.9,
                    "class_id": 0,
                    "class_name": "damaged",
                    "needs_review": False,
                    "evidence_consistency": "consistent",
                    "reason_codes": [],
                }
                for request, position in zip(requests, ("exposed", "suspended"), strict=True)
            ]

            report = validate_results(requests, results, min_frames=3)

        self.assertEqual(report["summary"]["pass"], 2)
        self.assertTrue(all(item["status"] == "pass" for item in report["items"]))

    def test_strict_controller_retries_only_failed_sample_and_merges_results(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_paths = []
            for idx in range(5):
                path = root / f"{idx}.jpg"
                path.write_bytes(b"image")
                image_paths.append(str(path))
            work_dir = root / "work"

            def fake_agent(command: list[str], **_: object) -> None:
                attempt_dir = Path(command[command.index("--out-dir") + 1])
                attempt_id = int(command[command.index("--attempt-id") + 1])
                attempt_dir.mkdir(parents=True, exist_ok=True)
                if "--prepare-only" in command:
                    sample_ids = ["S001", "S002"] if attempt_id == 1 else ["S002"]
                    requests = [
                        {
                            "video": "/data/pool.mp4",
                            "sample_id": sample_id,
                            "attempt_id": attempt_id,
                            "selected_frame_count": 3,
                            "image_paths": image_paths[:3],
                        }
                        for sample_id in sample_ids
                    ]
                    (attempt_dir / "requests.json").write_text(json.dumps(requests), encoding="utf-8")
                    (attempt_dir / "codex_classification_task.md").write_text("task", encoding="utf-8")
                    return

                requests = json.loads((attempt_dir / "requests.json").read_text(encoding="utf-8"))
                if attempt_id == 1:
                    requests = [request for request in requests if request["sample_id"] == "S001"]
                results = [
                    {
                        **request,
                        "position": "exposed",
                        "position_confidence": 0.9,
                        "damage": "intact",
                        "damage_confidence": 0.9,
                        "class_id": 1,
                        "class_name": "exposed_intact",
                        "needs_review": False,
                        "evidence_consistency": "consistent",
                        "reason_codes": [],
                    }
                    for request in requests
                ]
                (attempt_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

            argv = [
                "run_closed_loop.py",
                "--input",
                str(root / "input"),
                "--work-dir",
                str(work_dir),
                "--config",
                str(Path("configs/closed_loop_full.yaml").resolve()),
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.run_closed_loop.subprocess.run",
                side_effect=fake_agent,
            ) as subprocess_mock:
                run_closed_loop_main()

            state = json.loads((work_dir / "closed_loop_state.json").read_text(encoding="utf-8"))
            final_results = json.loads((work_dir / "results.json").read_text(encoding="utf-8"))

        self.assertEqual(subprocess_mock.call_count, 6)
        execute_calls = [
            call
            for call in subprocess_mock.call_args_list
            if "--execute-only" in call.args[0]
        ]
        self.assertEqual(len(execute_calls), 2)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["summary"], {"total": 2, "pass": 2, "human_review": 0})
        self.assertEqual(len(final_results), 2)
        self.assertEqual({result["attempt_id"] for result in final_results}, {1, 2})

    def test_default_controller_accepts_low_confidence_without_retry(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            work_dir = root / "work"

            def fake_agent(command: list[str], **_: object) -> None:
                attempt_dir = Path(command[command.index("--out-dir") + 1])
                attempt_dir.mkdir(parents=True, exist_ok=True)
                request = {
                    "video": "/data/pool.mp4",
                    "sample_id": "S001",
                    "attempt_id": 1,
                    "selected_frame_count": 3,
                    "image_paths": [str(image_path)],
                }
                if "--prepare-only" in command:
                    (attempt_dir / "requests.json").write_text(
                        json.dumps([request]),
                        encoding="utf-8",
                    )
                    (attempt_dir / "codex_classification_task.md").write_text(
                        "task",
                        encoding="utf-8",
                    )
                    return
                result = {
                    **request,
                    "position": "suspended",
                    "position_confidence": 0.55,
                    "damage": "intact",
                    "damage_confidence": 0.6,
                    "class_id": 2,
                    "class_name": "suspended_intact",
                    "needs_review": True,
                    "evidence_consistency": "insufficient",
                    "reason_codes": ["low_visibility"],
                }
                (attempt_dir / "results.json").write_text(
                    json.dumps([result]),
                    encoding="utf-8",
                )

            argv = [
                "run_closed_loop.py",
                "--input",
                str(root / "input"),
                "--work-dir",
                str(work_dir),
                "--config",
                str(Path("configs/closed_loop.yaml").resolve()),
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.run_closed_loop.subprocess.run",
                side_effect=fake_agent,
            ) as subprocess_mock:
                run_closed_loop_main()

            state = json.loads(
                (work_dir / "closed_loop_state.json").read_text(encoding="utf-8")
            )
            warnings = json.loads(
                (work_dir / "warnings.json").read_text(encoding="utf-8")
            )
            human_review_exists = (work_dir / "human_review.json").exists()

        self.assertEqual(subprocess_mock.call_count, 3)
        self.assertEqual(len(state["attempts"]), 1)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(
            state["summary"],
            {
                "total": 1,
                "classified": 1,
                "clean": 0,
                "with_warnings": 1,
                "unclassified": 0,
            },
        )
        self.assertEqual(len(warnings), 1)
        self.assertFalse(human_review_exists)

    def test_controller_reuses_cached_agent_result(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            work_dir = root / "work"

            def fake_agent(command: list[str], **_: object) -> None:
                attempt_dir = Path(command[command.index("--out-dir") + 1])
                attempt_dir.mkdir(parents=True, exist_ok=True)
                request = {
                    "video": "/data/pool.mp4",
                    "sample_id": "S001",
                    "attempt_id": 1,
                    "selected_frame_count": 3,
                    "image_paths": [str(image_path)],
                }
                if "--prepare-only" in command:
                    (attempt_dir / "requests.json").write_text(json.dumps([request]), encoding="utf-8")
                    (attempt_dir / "codex_classification_task.md").write_text("task", encoding="utf-8")
                    return
                result = {
                    **request,
                    "position": "exposed",
                    "position_confidence": 0.9,
                    "damage": "intact",
                    "damage_confidence": 0.9,
                    "class_id": 1,
                    "class_name": "exposed_intact",
                    "needs_review": False,
                    "evidence_consistency": "consistent",
                    "reason_codes": [],
                }
                (attempt_dir / "results.json").write_text(json.dumps([result]), encoding="utf-8")

            argv = [
                "run_closed_loop.py",
                "--input",
                str(root / "input"),
                "--work-dir",
                str(work_dir),
                "--config",
                str(Path("configs/closed_loop.yaml").resolve()),
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.run_closed_loop.subprocess.run",
                side_effect=fake_agent,
            ) as subprocess_mock:
                run_closed_loop_main()
                run_closed_loop_main()

            state = json.loads((work_dir / "closed_loop_state.json").read_text(encoding="utf-8"))

        self.assertEqual(subprocess_mock.call_count, 5)
        self.assertTrue(state["attempts"][0]["batches"][0]["cache_hit"])

    def test_controller_stops_and_preserves_state_on_agent_failure(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"image")
            work_dir = root / "work"

            def fake_agent(command: list[str], **_: object) -> None:
                attempt_dir = Path(command[command.index("--out-dir") + 1])
                attempt_dir.mkdir(parents=True, exist_ok=True)
                if "--execute-only" in command:
                    raise subprocess.CalledProcessError(9, command)
                request = {
                    "video": "/data/pool.mp4",
                    "sample_id": "S001",
                    "attempt_id": 1,
                    "selected_frame_count": 3,
                    "image_paths": [str(image_path)],
                }
                (attempt_dir / "requests.json").write_text(json.dumps([request]), encoding="utf-8")
                (attempt_dir / "codex_classification_task.md").write_text("task", encoding="utf-8")

            argv = [
                "run_closed_loop.py",
                "--input",
                str(root / "input"),
                "--work-dir",
                str(work_dir),
                "--config",
                str(Path("configs/closed_loop.yaml").resolve()),
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.run_closed_loop.subprocess.run",
                side_effect=fake_agent,
            ):
                with self.assertRaises(SystemExit):
                    run_closed_loop_main()

            state = json.loads((work_dir / "closed_loop_state.json").read_text(encoding="utf-8"))

        self.assertEqual(state["status"], "agent_interrupted")
        self.assertEqual(state["failed_batch"]["batch_id"], 1)


if __name__ == "__main__":
    unittest.main()
