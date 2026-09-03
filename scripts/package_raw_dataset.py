#!/usr/bin/env python3
"""Collect original and midterm underwater data into one documented package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_PATTERN = re.compile(r"_([012])$")
CLASS_NAMES = {0: "破损", 1: "裸露", 2: "悬空"}

DATASETS = (
    {
        "role": "original_video",
        "source": "video",
        "destination": "original/videos",
        "suffixes": VIDEO_SUFFIXES,
        "expected_count": 4,
        "description": "原有阶段采集视频",
    },
    {
        "role": "original_image",
        "source": "images",
        "destination": "original/images",
        "suffixes": IMAGE_SUFFIXES,
        "expected_count": 1,
        "description": "原有阶段采集照片",
    },
    {
        "role": "midterm_raw_video",
        "source": "mid/raw",
        "destination": "midterm/raw_videos",
        "suffixes": VIDEO_SUFFIXES,
        "expected_count": 4,
        "description": "中期测试完整原始视频",
    },
    {
        "role": "midterm_sample_clip",
        "source": "mid/clip",
        "destination": "midterm/sample_clips",
        "suffixes": VIDEO_SUFFIXES,
        "expected_count": 30,
        "description": "中期测试30段样本视频",
    },
    {
        "role": "midterm_ground_truth",
        "source": "mid/gt",
        "destination": "midterm/ground_truth",
        "suffixes": IMAGE_SUFFIXES,
        "expected_count": 30,
        "description": "中期测试当前真值图像",
    },
    {
        "role": "midterm_selected_frame",
        "source": "mid/single_frame",
        "destination": "midterm/selected_frames",
        "suffixes": IMAGE_SUFFIXES,
        "expected_count": 30,
        "description": "中期测试原始选帧",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/nvidia/DATA/UW"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use independent copies by default; hardlink is useful for local staging.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, mode: str) -> None:
    """Copy or hard-link one source file into the package."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
        shutil.copystat(source, destination)
    else:
        shutil.copy2(source, destination)


def media_metadata(path: Path) -> dict[str, Any]:
    """Read basic dimensions and verify that OpenCV can decode the media."""
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None or frame.size == 0:
            raise ValueError(f"Cannot decode video: {path}")
        return {
            "media_type": "video",
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": round(fps, 6),
            "duration_seconds": round(frame_count / fps, 3) if fps > 0 else None,
        }
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise ValueError(f"Cannot decode image: {path}")
    height, width = image.shape[:2]
    return {
        "media_type": "image",
        "width": width,
        "height": height,
        "frame_count": None,
        "fps": None,
        "duration_seconds": None,
    }


def label_from_name(path: Path) -> int | None:
    """Read a 0/1/2 class suffix when a labeled image has one."""
    match = LABEL_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def write_readme(
    out_dir: Path,
    manifest: dict[str, Any],
    section_rows: list[dict[str, Any]],
) -> None:
    """Write a concise Chinese data-package guide."""
    summary = manifest["summary"]
    lines = [
        "# 水下海缆原始数据汇总包",
        "",
        "本包汇总原有阶段与中期测试阶段的数据，并保持原文件名不变。",
        "",
        "## 数据规模",
        "",
        f"- 文件总数：{summary['file_count']}",
        f"- 视频：{summary['video_count']}，图像：{summary['image_count']}",
        f"- 数据总量：{summary['total_size_bytes'] / 1024 ** 3:.3f} GiB",
        "",
        "| 目录 | 内容 | 文件数 | 大小 |",
        "|---|---|---:|---:|",
    ]
    for row in section_rows:
        lines.append(
            f"| `{row['destination']}/` | {row['description']} | "
            f"{row['file_count']} | {row['size_bytes'] / 1024 ** 2:.2f} MiB |"
        )
    lines.extend(
        [
            "",
            "## 中期标签",
            "",
            "中期真值图文件名末尾数字为类别：`0=破损`、`1=裸露`、`2=悬空`。",
            "当前真值以 `midterm/ground_truth/` 为准。`selected_frames/` 保存的是原始",
            "选帧历史，其中sample09仍带旧标签1；人工复核后的当前标签为0。",
            "",
            "## 完整性校验",
            "",
            "- `manifest.json`：完整结构化清单、媒体参数及SHA-256；",
            "- `manifest.csv`：便于Excel查看的逐文件清单。",
            "",
            "## 未纳入内容",
            "",
            "色彩增强图属于派生数据，`data.zip` 与 `clip_new.zip` 属于重复压缩副本，",
            "因此未收入本原始数据包。算法结果、模型权重和tracking视频也不属于原始",
            "数据范围。",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Source root does not exist: {source_root}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        source_dir = source_root / dataset["source"]
        if not source_dir.is_dir():
            raise ValueError(f"Missing source directory: {source_dir}")
        files = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in dataset["suffixes"]
        )
        if len(files) != dataset["expected_count"]:
            raise ValueError(
                f"{source_dir}: expected {dataset['expected_count']} files, "
                f"found {len(files)}"
            )

        section_size = 0
        for source in files:
            destination = out_dir / dataset["destination"] / source.name
            copy_file(source, destination, args.copy_mode)
            metadata = media_metadata(destination)
            label = (
                label_from_name(destination)
                if dataset["role"]
                in {"midterm_ground_truth", "midterm_selected_frame"}
                else None
            )
            size = destination.stat().st_size
            section_size += size
            records.append(
                {
                    "role": dataset["role"],
                    "source_path": str(source.resolve()),
                    "packaged_path": str(destination.relative_to(out_dir)),
                    "filename": destination.name,
                    "size_bytes": size,
                    "sha256": sha256(destination),
                    "media_type": metadata["media_type"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "frame_count": metadata["frame_count"],
                    "fps": metadata["fps"],
                    "duration_seconds": metadata["duration_seconds"],
                    "label": label,
                    "class_zh": CLASS_NAMES[label] if label is not None else None,
                }
            )
        section_rows.append(
            {
                "role": dataset["role"],
                "source": dataset["source"],
                "destination": dataset["destination"],
                "description": dataset["description"],
                "file_count": len(files),
                "size_bytes": section_size,
            }
        )

    type_counts = Counter(row["media_type"] for row in records)
    role_counts = Counter(row["role"] for row in records)
    current_labels = Counter(
        row["label"]
        for row in records
        if row["role"] == "midterm_ground_truth"
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": "水下海缆原有及中期原始数据汇总",
        "source_root": str(source_root),
        "copy_mode": args.copy_mode,
        "class_mapping": {str(key): value for key, value in CLASS_NAMES.items()},
        "summary": {
            "file_count": len(records),
            "video_count": type_counts["video"],
            "image_count": type_counts["image"],
            "total_size_bytes": sum(row["size_bytes"] for row in records),
            "role_counts": dict(role_counts),
            "current_ground_truth_counts": {
                str(key): current_labels[key] for key in sorted(CLASS_NAMES)
            },
        },
        "included_sections": section_rows,
        "excluded": [
            "mid/gt_enhanced_wb_clahe (derived color-enhanced images)",
            "mid/data.zip and mid/clip_new.zip (duplicate archives)",
            "algorithm outputs, tracking results, and model checkpoints",
        ],
        "label_note": (
            "Current ground truth labels sample09 as damaged (0); the historical "
            "selected frame filename still carries the previous exposed label (1)."
        ),
        "files": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_readme(out_dir, manifest, section_rows)

    summary = manifest["summary"]
    print(
        f"Packaged {summary['file_count']} files: {summary['video_count']} videos, "
        f"{summary['image_count']} images, "
        f"{summary['total_size_bytes'] / 1024 ** 3:.3f} GiB -> {out_dir}"
    )


if __name__ == "__main__":
    main()
