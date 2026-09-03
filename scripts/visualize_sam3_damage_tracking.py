#!/usr/bin/env python3
"""Render combined SAM3 cable and silver damage-patch tracking."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scripts.visualize_sam3_cable_tracking import (
        CLASS_COLORS_BGR,
        DEFAULT_CHECKPOINT,
        draw_unicode_label,
        normalize_outputs,
        prepare_output_dir,
        resize_mask,
        sample_video,
    )
except ModuleNotFoundError:
    from visualize_sam3_cable_tracking import (
        CLASS_COLORS_BGR,
        DEFAULT_CHECKPOINT,
        draw_unicode_label,
        normalize_outputs,
        prepare_output_dir,
        resize_mask,
        sample_video,
    )


CABLE_COLOR = (255, 170, 30)
DAMAGE_COLOR = (20, 35, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sam3_cable_damage_tracking"),
    )
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cable-prompt", default="black pipe")
    parser.add_argument("--cable-prompt-frame", type=int, default=50)
    parser.add_argument(
        "--damage-prompt",
        default="metal patch on pipe",
    )
    parser.add_argument("--damage-prompt-frame", type=int, default=50)
    parser.add_argument(
        "--damage-box",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Normalized xywh box around the silver damage marker.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise SystemExit(f"Input video does not exist: {args.input}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"SAM3 checkpoint does not exist: {args.checkpoint}")
    if args.target_fps <= 0:
        raise SystemExit("--target-fps must be positive")
    if args.cable_prompt_frame < 0 or args.damage_prompt_frame < 0:
        raise SystemExit("Prompt frame indices must be non-negative")
    x, y, width, height = args.damage_box
    if min(x, y, width, height) < 0 or x + width > 1 or y + height > 1:
        raise SystemExit("--damage-box must be normalized xywh coordinates within [0, 1]")


def track_concept(
    predictor: Any,
    frames_dir: Path,
    prompt: str,
    prompt_frame: int,
    mask_threshold: float,
    box_xywh: list[float] | None = None,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
) -> tuple[dict[int, dict[str, Any]], float]:
    """Track one semantic concept in its own SAM3 session."""
    started = time.perf_counter()
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frames_dir),
            "offload_video_to_cpu": True,
        }
    )
    session_id = str(response["session_id"])
    try:
        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": prompt_frame,
            "output_prob_thresh": mask_threshold,
        }
        if points is not None:
            if point_labels is None or len(points) != len(point_labels):
                raise ValueError("points and point_labels must have the same length")
            request.update(
                {
                    "points": points,
                    "point_labels": point_labels,
                    "obj_id": 0,
                    "rel_coordinates": True,
                }
            )
        else:
            request["text"] = prompt
        if box_xywh is not None and points is None:
            request["bounding_boxes"] = [box_xywh]
            request["bounding_box_labels"] = [1]
        prompt_response = predictor.handle_request(request)
        prompt_outputs = prompt_response["outputs"]
        object_ids, masks = normalize_outputs(prompt_outputs)
        if len(object_ids) == 0 or not masks.any():
            raise RuntimeError(
                f"SAM3 found no object on frame {prompt_frame} with prompt {prompt!r}"
            )

        outputs_by_frame: dict[int, dict[str, Any]] = {
            prompt_frame: prompt_outputs
        }
        propagation_request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both" if prompt_frame > 0 else "forward",
            "start_frame_index": prompt_frame,
            "output_prob_thresh": mask_threshold,
        }
        for result in predictor.handle_stream_request(propagation_request):
            result_frame = int(result["frame_index"])
            result_outputs = result["outputs"]
            result_ids, result_masks = normalize_outputs(result_outputs)
            if (
                result_frame == prompt_frame
                and (len(result_ids) == 0 or not result_masks.any())
            ):
                continue
            outputs_by_frame[result_frame] = result_outputs
        return outputs_by_frame, time.perf_counter() - started
    finally:
        predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": True,
            }
        )


def run_dual_tracking(
    frames_dir: Path,
    checkpoint: Path,
    cable_prompt: str,
    cable_prompt_frame: int,
    damage_prompt: str,
    damage_prompt_frame: int,
    damage_box: list[float],
    mask_threshold: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, float]]:
    """Load SAM3 once, then track cable and damage in independent sessions."""
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible; run with the sam3 environment and GPU access")

    total_started = time.perf_counter()
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint),
        gpus_to_use=[torch.cuda.current_device()],
    )
    try:
        cable_outputs, cable_sec = track_concept(
            predictor=predictor,
            frames_dir=frames_dir,
            prompt=cable_prompt,
            prompt_frame=cable_prompt_frame,
            mask_threshold=mask_threshold,
        )
        box_x, box_y, box_width, box_height = damage_box
        damage_points = [
            [box_x + box_width * 0.5, box_y + box_height * 0.5],
            [box_x + box_width * 0.5, max(0.0, box_y - box_height * 0.25)],
            [
                box_x + box_width * 0.5,
                min(1.0, box_y + box_height * 1.35),
            ],
        ]
        damage_outputs, damage_sec = track_concept(
            predictor=predictor,
            frames_dir=frames_dir,
            prompt=damage_prompt,
            prompt_frame=damage_prompt_frame,
            mask_threshold=mask_threshold,
            points=damage_points,
            point_labels=[1, 0, 0],
        )
        timing = {
            "cable_tracking_sec": cable_sec,
            "damage_tracking_sec": damage_sec,
            "total_inference_sec_including_model_load": time.perf_counter()
            - total_started,
        }
        return cable_outputs, damage_outputs, timing
    finally:
        predictor.shutdown()


def draw_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    top = max(56, y - 48)
    draw_unicode_label(
        canvas,
        text,
        x,
        top,
        color,
        font_size=27,
    )


def draw_concept(
    canvas: np.ndarray,
    outputs: dict[str, Any] | None,
    color: tuple[int, int, int],
    label: str,
    alpha: float,
    draw_box: bool,
) -> list[dict[str, Any]]:
    """Draw one concept and return its per-instance records."""
    records: list[dict[str, Any]] = []
    if outputs is None:
        return records
    height, width = canvas.shape[:2]
    object_ids, masks = normalize_outputs(outputs)
    for object_id, raw_mask in zip(object_ids.tolist(), masks):
        mask = resize_mask(raw_mask, width, height)
        if not mask.any():
            continue
        color_layer = np.empty_like(canvas)
        color_layer[:] = color
        canvas[mask] = cv2.addWeighted(
            canvas[mask], 1.0 - alpha, color_layer[mask], alpha, 0.0
        )
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, 3, cv2.LINE_AA)
        x, y, box_width, box_height = cv2.boundingRect(mask_u8)
        if draw_box:
            cv2.rectangle(
                canvas,
                (x, y),
                (x + box_width, y + box_height),
                color,
                3,
                cv2.LINE_AA,
            )
        draw_label(canvas, label, x, y, color)
        records.append(
            {
                "raw_object_id": int(object_id),
                "box_xywh": [int(x), int(y), int(box_width), int(box_height)],
                "mask_area_pixels": int(mask.sum()),
                "mask_area_ratio": float(mask.mean()),
            }
        )
    return records


def render_video(
    frame_paths: list[Path],
    cable_outputs: dict[int, dict[str, Any]],
    damage_outputs: dict[int, dict[str, Any]],
    output_path: Path,
    preview_path: Path,
    fps: float,
    preview_frame: int,
    class_label: str = "破损",
) -> list[dict[str, Any]]:
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise SystemExit(f"Cannot read sampled frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot create output video: {output_path}")

    frame_records: list[dict[str, Any]] = []
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise SystemExit(f"Cannot read sampled frame: {frame_path}")
            canvas = frame.copy()
            cable_records = draw_concept(
                canvas=canvas,
                outputs=cable_outputs.get(frame_index),
                color=CABLE_COLOR,
                label=f"海缆｜类别：{class_label}",
                alpha=0.30,
                draw_box=False,
            )
            damage_records = draw_concept(
                canvas=canvas,
                outputs=damage_outputs.get(frame_index),
                color=DAMAGE_COLOR,
                label=f"破损区域｜类别：{class_label}",
                alpha=0.68,
                draw_box=True,
            )

            cv2.rectangle(canvas, (0, 0), (width, 56), (12, 12, 12), -1)
            draw_unicode_label(
                canvas,
                f"{fps:g} Hz",
                14,
                8,
                (12, 12, 12),
                font_size=28,
            )
            class_color = CLASS_COLORS_BGR.get(
                class_label,
                CLASS_COLORS_BGR["未分类"],
            )
            draw_unicode_label(
                canvas,
                f"类别：{class_label}",
                width - 14,
                8,
                class_color,
                font_size=28,
                align_right=True,
            )
            writer.write(canvas)
            if frame_index == preview_frame:
                cv2.imwrite(str(preview_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                    "cable": cable_records,
                    "damage": damage_records,
                }
            )
    finally:
        writer.release()
    return frame_records


def main() -> None:
    args = parse_args()
    validate_args(args)
    frames_dir = prepare_output_dir(args.output_dir, args.force)
    frame_paths, video_metadata = sample_video(args.input, frames_dir, args.target_fps)
    for name, index in (
        ("cable prompt", args.cable_prompt_frame),
        ("damage prompt", args.damage_prompt_frame),
    ):
        if index >= len(frame_paths):
            raise SystemExit(f"{name} frame {index} is outside {len(frame_paths)} frames")

    print(
        f"Sampled {len(frame_paths)} frames. Loading SAM3 for cable and damage tracking ...",
        flush=True,
    )
    cable_outputs, damage_outputs, timing = run_dual_tracking(
        frames_dir=frames_dir,
        checkpoint=args.checkpoint,
        cable_prompt=args.cable_prompt,
        cable_prompt_frame=args.cable_prompt_frame,
        damage_prompt=args.damage_prompt,
        damage_prompt_frame=args.damage_prompt_frame,
        damage_box=list(args.damage_box),
        mask_threshold=args.mask_threshold,
    )

    output_video = args.output_dir / "sam3_cable_damage_tracking_10hz.mp4"
    preview_path = args.output_dir / "preview_damage.jpg"
    frame_records = render_video(
        frame_paths=frame_paths,
        cable_outputs=cable_outputs,
        damage_outputs=damage_outputs,
        output_path=output_video,
        preview_path=preview_path,
        fps=float(video_metadata["output_fps"]),
        preview_frame=args.damage_prompt_frame,
    )
    cable_frame_count = sum(bool(frame["cable"]) for frame in frame_records)
    damage_frame_count = sum(bool(frame["damage"]) for frame in frame_records)
    report = {
        "method": "sam3_dual_concept_video_tracking",
        "classification": {"class_id": 0, "class_name": "damaged"},
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "cable_prompt": args.cable_prompt,
        "cable_prompt_frame": args.cable_prompt_frame,
        "damage_prompt": args.damage_prompt,
        "damage_prompt_frame": args.damage_prompt_frame,
        "damage_box_normalized_xywh": list(args.damage_box),
        "damage_tracking_prompt_type": "one_positive_and_two_negative_points",
        "mask_threshold": args.mask_threshold,
        **video_metadata,
        **timing,
        "cable_tracked_frame_count": cable_frame_count,
        "damage_tracked_frame_count": damage_frame_count,
        "output_video": str(output_video.resolve()),
        "preview_image": str(preview_path.resolve()),
        "frames": frame_records,
    }
    report_path = args.output_dir / "tracking_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    print(f"Done: {output_video}")
    print(
        f"Cable tracked {cable_frame_count}/{len(frame_paths)} frames; "
        f"damage tracked {damage_frame_count}/{len(frame_paths)} frames; "
        f"elapsed {timing['total_inference_sec_including_model_load']:.1f}s"
    )
    print(f"Preview: {preview_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
