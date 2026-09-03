#!/usr/bin/env python
"""Extract frames from underwater cable videos for annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def iter_videos(video_dir: Path) -> list[Path]:
    return sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_")


def extract_video(video_path: Path, out_dir: Path, every_sec: float, max_frames: int | None) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(fps * every_sec))) if fps > 0 else 30

    video_out = out_dir / safe_stem(video_path)
    video_out.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0
    pbar_total = total_frames if total_frames > 0 else None
    with tqdm(total=pbar_total, desc=video_path.name, unit="frame") as pbar:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                timestamp_sec = frame_idx / fps if fps > 0 else 0.0
                name = f"{safe_stem(video_path)}_f{frame_idx:06d}_t{timestamp_sec:08.2f}.jpg"
                cv2.imwrite(str(video_out / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            frame_idx += 1
            pbar.update(1)

    cap.release()
    return {
        "video": str(video_path),
        "fps": fps,
        "frames": total_frames,
        "width": width,
        "height": height,
        "sample_every_sec": every_sec,
        "saved_frames": saved,
        "output_dir": str(video_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=Path("/home/nvidia/DATA/UW/video"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--every-sec", type=float, default=1.0)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    args = parser.parse_args()

    videos = iter_videos(args.video_dir)
    if not videos:
        raise SystemExit(f"No videos found under {args.video_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [extract_video(v, args.out_dir, args.every_sec, args.max_frames_per_video) for v in videos]
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
