#!/usr/bin/env python3
"""Track cable instances with SAM3 and render a 10 Hz overlay video."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpts" / "sam3" / "sam3.pt"
COLORS_BGR = (
    (0, 215, 255),
    (255, 110, 40),
    (80, 220, 80),
    (220, 80, 220),
    (40, 140, 255),
    (255, 210, 70),
)
CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
CLASS_COLORS_BGR = {
    "破损": (20, 35, 255),
    "裸露": (0, 155, 255),
    "悬空": (180, 95, 40),
    "未分类": (90, 90, 90),
}


@lru_cache(maxsize=8)
def cjk_font(size: int) -> ImageFont.FreeTypeFont:
    """Load and cache the project CJK font."""
    return ImageFont.truetype(str(CJK_FONT_PATH), size=size)


def draw_unicode_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    background_bgr: tuple[int, int, int],
    font_size: int = 28,
    align_right: bool = False,
    foreground_bgr: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int]:
    """Draw a filled UTF-8 label on a BGR OpenCV image in place."""
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = cjk_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pad_x = max(8, round(font_size * 0.35))
    pad_y = max(5, round(font_size * 0.20))
    box_width = text_width + pad_x * 2
    box_height = text_height + pad_y * 2
    left = x - box_width if align_right else x
    left = max(0, min(left, canvas.shape[1] - box_width))
    top = max(0, min(y, canvas.shape[0] - box_height))
    background_rgb = (
        background_bgr[2],
        background_bgr[1],
        background_bgr[0],
    )
    foreground_rgb = (
        foreground_bgr[2],
        foreground_bgr[1],
        foreground_bgr[0],
    )
    draw.rectangle(
        (left, top, left + box_width, top + box_height),
        fill=background_rgb,
    )
    draw.text(
        (left + pad_x, top + pad_y - bbox[1]),
        text,
        font=font,
        fill=foreground_rgb,
    )
    canvas[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return box_width, box_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input MP4 video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sam3_cable_tracking"),
        help="Directory for sampled frames, visualization, and metadata.",
    )
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument(
        "--prompt",
        default="black pipe",
        help="SAM3 text prompt used to initialize dense video tracking.",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Sampled frame index used to initialize tracking; propagation runs both ways.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="SAM3 output probability threshold.",
    )
    parser.add_argument("--mask-alpha", type=float, default=0.42)
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the intermediate 10 Hz JPEG frames after inference.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise SystemExit(f"Input video does not exist: {args.input}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"SAM3 checkpoint does not exist: {args.checkpoint}")
    if args.target_fps <= 0:
        raise SystemExit("--target-fps must be positive")
    if args.prompt_frame < 0:
        raise SystemExit("--prompt-frame must be non-negative")
    if not 0.0 < args.mask_threshold < 1.0:
        raise SystemExit("--mask-threshold must be between 0 and 1")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise SystemExit("--mask-alpha must be between 0 and 1")


def prepare_output_dir(output_dir: Path, force: bool) -> Path:
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. Use --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    frames_dir = output_dir / "frames_10hz"
    frames_dir.mkdir()
    return frames_dir


def sample_video(
    input_path: Path, frames_dir: Path, target_fps: float
) -> tuple[list[Path], dict[str, Any]]:
    """Decode a video and write frames nearest to a uniform target-FPS timeline."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV cannot open video: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or source_frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise SystemExit(f"Invalid video metadata: {input_path}")

    effective_fps = min(target_fps, source_fps)
    frame_paths: list[Path] = []
    source_indices: list[int] = []
    next_sample = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        wanted_index = round(next_sample * source_fps / effective_fps)
        if frame_index >= wanted_index:
            frame_path = frames_dir / f"{next_sample:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                capture.release()
                raise SystemExit(f"Failed to write sampled frame: {frame_path}")
            frame_paths.append(frame_path)
            source_indices.append(frame_index)
            next_sample += 1
        frame_index += 1
    capture.release()

    if not frame_paths:
        raise SystemExit(f"No frames decoded from: {input_path}")
    metadata = {
        "source_fps": source_fps,
        "source_frame_count_reported": source_frame_count,
        "source_frame_count_decoded": frame_index,
        "source_duration_sec": frame_index / source_fps,
        "target_fps_requested": target_fps,
        "output_fps": effective_fps,
        "sampled_frame_count": len(frame_paths),
        "sampled_source_indices": source_indices,
        "width": width,
        "height": height,
    }
    return frame_paths, metadata


def normalize_outputs(outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    object_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.size == 0:
        masks = np.empty((0, 0, 0), dtype=bool)
    if masks.ndim != 3:
        raise RuntimeError(f"Unexpected SAM3 mask shape: {masks.shape}")
    if len(object_ids) != len(masks):
        raise RuntimeError(
            f"SAM3 returned {len(object_ids)} IDs but {len(masks)} masks"
        )
    return object_ids, masks


def run_tracking(
    frames_dir: Path,
    checkpoint: Path,
    prompt: str,
    prompt_frame: int,
    mask_threshold: float,
) -> tuple[dict[int, dict[str, Any]], float]:
    """Run text-prompted SAM3 dense tracking on an ordered JPEG directory."""
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not visible. Run this script with the sam3 environment and GPU access."
        )

    started = time.perf_counter()
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint),
        gpus_to_use=[torch.cuda.current_device()],
    )
    session_id: str | None = None
    try:
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frames_dir),
                "offload_video_to_cpu": True,
            }
        )
        session_id = str(response["session_id"])
        prompt_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame,
                "text": prompt,
                "output_prob_thresh": mask_threshold,
            }
        )
        first_outputs = prompt_response["outputs"]
        first_ids, first_masks = normalize_outputs(first_outputs)
        if len(first_ids) == 0 or not first_masks.any():
            raise SystemExit(
                f"SAM3 found no object on frame {prompt_frame} with prompt {prompt!r}. "
                "Try --prompt pipe or select a clearer --prompt-frame."
            )

        outputs_by_frame: dict[int, dict[str, Any]] = {prompt_frame: first_outputs}
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both" if prompt_frame > 0 else "forward",
            "start_frame_index": prompt_frame,
            "output_prob_thresh": mask_threshold,
        }
        for propagation_response in predictor.handle_stream_request(request):
            outputs_by_frame[int(propagation_response["frame_index"])] = (
                propagation_response["outputs"]
            )
        elapsed_sec = time.perf_counter() - started
        return outputs_by_frame, elapsed_sec
    finally:
        if session_id is not None:
            predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": True,
                }
            )
        predictor.shutdown()


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


def draw_tracking_overlay(
    frame: np.ndarray,
    outputs: dict[str, Any] | None,
    frame_index: int,
    fps: float,
    prompt: str,
    alpha: float,
    class_label: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Render masks, contours, boxes, and stable object IDs on one frame."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    object_records: list[dict[str, Any]] = []
    if outputs is not None:
        object_ids, masks = normalize_outputs(outputs)
        for object_id, raw_mask in zip(object_ids.tolist(), masks):
            mask = resize_mask(raw_mask, width, height)
            if not mask.any():
                continue
            color = COLORS_BGR[object_id % len(COLORS_BGR)]
            colored = np.empty_like(canvas)
            colored[:] = color
            canvas[mask] = cv2.addWeighted(
                canvas[mask], 1.0 - alpha, colored[mask], alpha, 0.0
            )
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, color, 3, cv2.LINE_AA)
            x, y, box_width, box_height = cv2.boundingRect(mask_u8)
            cv2.rectangle(
                canvas,
                (x, y),
                (x + box_width, y + box_height),
                color,
                3,
                cv2.LINE_AA,
            )
            label = f"海缆｜类别：{class_label}"
            label_top = max(56, y - 48)
            draw_unicode_label(
                canvas,
                label,
                x,
                label_top,
                color,
                font_size=27,
                foreground_bgr=(15, 15, 15),
            )
            object_records.append(
                {
                    "object_id": int(object_id),
                    "box_xywh": [int(x), int(y), int(box_width), int(box_height)],
                    "mask_area_pixels": int(mask.sum()),
                    "mask_area_ratio": float(mask.mean()),
                }
            )

    header_height = 56
    cv2.rectangle(canvas, (0, 0), (width, header_height), (12, 12, 12), -1)
    draw_unicode_label(
        canvas,
        f"{fps:g} Hz",
        14,
        8,
        (12, 12, 12),
        font_size=28,
    )
    class_color = CLASS_COLORS_BGR.get(class_label, CLASS_COLORS_BGR["未分类"])
    draw_unicode_label(
        canvas,
        f"类别：{class_label}",
        width - 14,
        8,
        class_color,
        font_size=28,
        align_right=True,
    )
    return canvas, object_records


def render_video(
    frame_paths: list[Path],
    outputs_by_frame: dict[int, dict[str, Any]],
    output_path: Path,
    preview_path: Path,
    fps: float,
    prompt: str,
    alpha: float,
    class_label: str = "未分类",
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
    preview_index = len(frame_paths) // 2
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise SystemExit(f"Cannot read sampled frame: {frame_path}")
            rendered, objects = draw_tracking_overlay(
                frame=frame,
                outputs=outputs_by_frame.get(frame_index),
                frame_index=frame_index,
                fps=fps,
                prompt=prompt,
                alpha=alpha,
                class_label=class_label,
            )
            writer.write(rendered)
            if frame_index == preview_index:
                cv2.imwrite(str(preview_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                    "objects": objects,
                }
            )
    finally:
        writer.release()
    return frame_records


def main() -> None:
    args = parse_args()
    validate_args(args)
    frames_dir = prepare_output_dir(args.output_dir, args.force)
    print(f"Sampling {args.input} at {args.target_fps:g} Hz ...", flush=True)
    frame_paths, video_metadata = sample_video(
        args.input, frames_dir, args.target_fps
    )
    if args.prompt_frame >= len(frame_paths):
        raise SystemExit(
            f"--prompt-frame {args.prompt_frame} is outside the sampled video "
            f"({len(frame_paths)} frames)"
        )
    print(f"Sampled {len(frame_paths)} frames. Loading SAM3 ...", flush=True)
    outputs_by_frame, inference_sec = run_tracking(
        frames_dir=frames_dir,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        prompt_frame=args.prompt_frame,
        mask_threshold=args.mask_threshold,
    )

    output_video = args.output_dir / "sam3_cable_tracking_10hz.mp4"
    preview_path = args.output_dir / "preview.jpg"
    frame_records = render_video(
        frame_paths=frame_paths,
        outputs_by_frame=outputs_by_frame,
        output_path=output_video,
        preview_path=preview_path,
        fps=float(video_metadata["output_fps"]),
        prompt=args.prompt,
        alpha=args.mask_alpha,
    )
    tracked_frame_count = sum(bool(record["objects"]) for record in frame_records)
    all_object_ids = sorted(
        {
            obj["object_id"]
            for record in frame_records
            for obj in record["objects"]
        }
    )
    report = {
        "method": "sam3_text_prompt_dense_video_tracking",
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt": args.prompt,
        "prompt_frame": args.prompt_frame,
        "mask_threshold": args.mask_threshold,
        **video_metadata,
        "inference_sec_including_model_load": inference_sec,
        "sampled_inference_fps_including_model_load": len(frame_paths) / inference_sec,
        "tracked_frame_count": tracked_frame_count,
        "tracked_frame_ratio": tracked_frame_count / len(frame_paths),
        "object_ids_seen": all_object_ids,
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
        f"Tracked {tracked_frame_count}/{len(frame_paths)} frames; "
        f"object IDs: {all_object_ids}; elapsed: {inference_sec:.1f}s"
    )
    print(f"Preview: {preview_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
