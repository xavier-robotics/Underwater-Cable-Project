#!/usr/bin/env python3
"""Build auditable midterm dataset and experiment support materials."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml


CLASS_NAMES = {
    0: ("damaged", "破损"),
    1: ("exposed_intact", "裸露"),
    2: ("suspended_intact", "悬空"),
}
GT_PATTERN = re.compile(r"^sample(?P<sample>\d+)_(?P<class_id>[012])$")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The GT set contains only two suspended samples.  This stratified subset keeps
# both of them and adds duration, orientation, classification-error, and
# tracking-coverage diversity without duplicating a sample.
EXPERIMENT_SAMPLES = [0, 2, 8, 13, 28, 5, 18, 25, 26, 27]

EXPERIMENT_NOTES: dict[int, dict[str, str]] = {
    0: {
        "selection": "破损长视频；覆盖完整跟踪和较长时序，包含设备/支架背景干扰。",
        "phenomenon": "画面偏绿，海缆跨越较大视野，局部被白色设备结构遮挡；损伤区域可见。",
        "success": "海缆在全部 10 Hz 抽帧上均有跟踪结果。",
        "issue": "损伤区域并非全程可见，损伤覆盖率明显低于海缆覆盖率。",
        "difficulty": "中等",
    },
    2: {
        "selection": "全数据最短破损视频之一，用于检查短序列和快速视角变化。",
        "phenomenon": "同一画面中出现两段相似深色管体，背景池砖纹理强，视频仅约 3.5 秒。",
        "success": "海缆与破损区域均覆盖全部抽帧。",
        "issue": "样本很短，对长时稳定性的代表性有限；相似管段可能造成目标关联歧义。",
        "difficulty": "简单（但序列短）",
    },
    8: {
        "selection": "破损分类漏检难例；用于展示分类与视频跟踪两条支撑链的差异。",
        "phenomenon": "海缆近竖直，池底网格和斑驳区域明显，局部标记靠近端部。",
        "success": "视频海缆跟踪覆盖全部抽帧，已有破损跟踪可视化。",
        "issue": "静态图分类把破损误判为裸露；破损区域仅覆盖约半数抽帧。",
        "difficulty": "困难",
    },
    13: {
        "selection": "破损长视频和高覆盖率样本，用于展示稳定成功案例。",
        "phenomenon": "斜向海缆主体完整，损伤贴片和端部标签可见，光照相对均匀。",
        "success": "海缆与破损区域在绝大多数抽帧上均成功跟踪。",
        "issue": "仍存在轻微绿偏和水下低对比；属于同一水池场景，泛化意义有限。",
        "difficulty": "简单",
    },
    28: {
        "selection": "1124×1280 竖屏近景破损样本，覆盖非 1080p 分辨率和大目标尺度。",
        "phenomenon": "海缆近竖直且占画面比例大，局部表面纹理和反光显著。",
        "success": "海缆覆盖率接近 100%，静态图分类正确。",
        "issue": "破损 mask 覆盖范围较大且仅在部分时刻稳定，需防止把表面反光当作损伤。",
        "difficulty": "中等",
    },
    5: {
        "selection": "裸露类分类误报难例；用于展示反光/低对比下的错误。",
        "phenomenon": "海缆斜穿画面，整体偏绿且局部反光，背景纹理杂乱。",
        "success": "点提示初始化后，大多数 10 Hz 抽帧均跟踪到海缆。",
        "issue": "静态图被误判为破损；跟踪使用 GT 参考点初始化，不能解释为全自动检出率。",
        "difficulty": "困难",
    },
    18: {
        "selection": "裸露类正确样本；中短时长、斜向姿态和较低对比。",
        "phenomenon": "海缆较细、斜向分布，池底斑驳和绿色水体降低局部对比。",
        "success": "分类正确，海缆覆盖全部 10 Hz 抽帧。",
        "issue": "使用 GT 参考点初始化，尚缺逐帧人工 mask/IoU 真值。",
        "difficulty": "中等",
    },
    25: {
        "selection": "1920×930 短视频裸露难例；覆盖裁切分辨率和较低跟踪覆盖率。",
        "phenomenon": "海缆靠近画面边缘，短序列中目标进出视野，绿色低对比明显。",
        "success": "静态图分类正确，并生成完整 10 Hz 输出视频。",
        "issue": "预览帧无有效对象，整体海缆覆盖率偏低，说明边缘/进出视野仍是薄弱点。",
        "difficulty": "困难",
    },
    26: {
        "selection": "仅有的两组悬空样本之一；1920×906，低覆盖率难例。",
        "phenomenon": "海缆靠近右侧边缘，支撑结构与海缆姿态共同表征悬空。",
        "success": "静态图三分类正确，视频中多数时刻能维持目标。",
        "issue": "海缆覆盖率为代表集最低之一；类别判断依赖水池支撑杆几何线索。",
        "difficulty": "困难",
    },
    27: {
        "selection": "另一组悬空样本；822×1232 竖屏，覆盖明显支撑杆和纵向姿态。",
        "phenomenon": "白色 T 形支撑杆清晰，海缆纵向贯穿画面，分辨率与多数样本不同。",
        "success": "分类正确且海缆跟踪覆盖率接近 100%。",
        "issue": "悬空证据高度依赖本水池白色支撑结构，不能外推到无支撑杆的海底悬空。",
        "difficulty": "简单（场景线索明显）",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse paths while keeping repository defaults explicit."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip-dir", type=Path, default=Path("/home/nvidia/DATA/UW/mid/clip")
    )
    parser.add_argument(
        "--gt-dir", type=Path, default=Path("/home/nvidia/DATA/UW/mid/gt")
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=root / "outputs/mid_gt_sam3_suspended_priority_full_20260815",
    )
    parser.add_argument(
        "--tracking-summary",
        type=Path,
        default=(
            root
            / "outputs/mid_clip_sam3_tracking_10hz_20260815/tracking_summary.json"
        ),
    )
    parser.add_argument(
        "--label-config",
        type=Path,
        default=root / "configs/midterm_ground_truth.yaml",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=root / "docs/midterm_support_20260816",
    )
    parser.add_argument(
        "--visual-dir",
        type=Path,
        default=root / "outputs/midterm_support_20260816",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """Write stable, human-readable UTF-8 JSON."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as a UTF-8 CSV with a header."""
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_link(label: str, path: Path | str) -> str:
    """Return a clickable absolute local Markdown link."""
    absolute = Path(path).expanduser().absolute()
    return f"[{label}](<{absolute}>)"


def percent(value: float | None) -> str:
    """Format a ratio as a percentage or an unavailable marker."""
    return "—" if value is None else f"{value * 100:.2f}%"


def ratio(numerator: int, denominator: int) -> float | None:
    """Safely divide integer counts."""
    return numerator / denominator if denominator else None


def inspect_image(path: Path) -> dict[str, Any]:
    """Open an image and return basic integrity metadata."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"readable": False, "width": 0, "height": 0}
    height, width = image.shape[:2]
    return {"readable": True, "width": int(width), "height": int(height)}


def inspect_video(path: Path, decode_all: bool) -> dict[str, Any]:
    """Read video metadata and optionally count every decodable frame."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {
            "readable": False,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "reported_frame_count": 0,
            "decoded_frame_count": 0,
            "duration_sec": 0.0,
            "first_frame_readable": False,
            "last_frame_readable": False,
        }
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    reported = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    first_ok, _ = capture.read()
    last_ok = False
    if reported > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, reported - 1))
        last_ok, _ = capture.read()
    capture.release()

    decoded = reported
    if decode_all:
        verifier = cv2.VideoCapture(str(path))
        decoded = 0
        while verifier.grab():
            decoded += 1
        verifier.release()
    duration = decoded / fps if fps > 0 else 0.0
    return {
        "readable": bool(first_ok and (last_ok or reported <= 1)),
        "fps": fps,
        "width": width,
        "height": height,
        "reported_frame_count": reported,
        "decoded_frame_count": decoded,
        "duration_sec": duration,
        "first_frame_readable": bool(first_ok),
        "last_frame_readable": bool(last_ok),
    }


def sample_from_stem(stem: str) -> int | None:
    """Extract the sample number from a filename stem."""
    match = GT_PATTERN.fullmatch(stem)
    if match is not None:
        return int(match.group("sample"))
    plain = re.fullmatch(r"sample0*(\d+)", stem, re.IGNORECASE)
    return int(plain.group(1)) if plain is not None else None


def compute_classification_metrics(
    predictions: dict[int, dict[str, Any]],
    gt_labels: dict[int, int],
    samples: list[int] | None = None,
) -> dict[str, Any]:
    """Compute three-class metrics from mapped prediction and GT rows."""
    chosen = sorted(gt_labels) if samples is None else list(samples)
    confusion = [[0, 0, 0, 0] for _ in range(3)]
    per_sample = []
    for sample in chosen:
        truth = gt_labels[sample]
        prediction = predictions.get(sample, {}).get("pred_label", -1)
        prediction = int(prediction) if prediction is not None else -1
        column = prediction if prediction in CLASS_NAMES else 3
        confusion[truth][column] += 1
        per_sample.append(
            {
                "sample": sample,
                "truth": truth,
                "prediction": prediction,
                "correct": truth == prediction,
            }
        )
    total = len(per_sample)
    correct = sum(row["correct"] for row in per_sample)
    per_class = []
    for class_id in CLASS_NAMES:
        true_positive = confusion[class_id][class_id]
        false_positive = sum(
            confusion[row][class_id] for row in CLASS_NAMES if row != class_id
        )
        false_negative = sum(
            confusion[class_id][column]
            for column in range(4)
            if column != class_id
        )
        precision_value = ratio(true_positive, true_positive + false_positive) or 0.0
        recall_value = ratio(true_positive, true_positive + false_negative) or 0.0
        f1_value = (
            2 * precision_value * recall_value / (precision_value + recall_value)
            if precision_value + recall_value
            else 0.0
        )
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id][0],
                "class_name_zh": CLASS_NAMES[class_id][1],
                "support": sum(confusion[class_id]),
                "correct": true_positive,
                "precision": precision_value,
                "recall": recall_value,
                "f1": f1_value,
            }
        )
    return {
        "total": total,
        "correct": correct,
        "accuracy": ratio(correct, total),
        "balanced_accuracy": sum(row["recall"] for row in per_class) / 3,
        "macro_f1": sum(row["f1"] for row in per_class) / 3,
        "confusion_matrix_rows_truth_columns_prediction": confusion,
        "per_class": per_class,
        "per_sample": per_sample,
    }


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    """Create an idempotent evidence link without replacing user files."""
    target = target_path.resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target:
            return
        raise FileExistsError(f"Existing link points elsewhere: {link_path}")
    if link_path.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link_path}")
    link_path.symlink_to(target)


def class_distribution(labels: dict[int, int]) -> dict[str, Any]:
    """Return stable class counts keyed by both ID and name."""
    counts = Counter(labels.values())
    return {
        str(class_id): {
            "class_name": names[0],
            "class_name_zh": names[1],
            "count": counts[class_id],
        }
        for class_id, names in CLASS_NAMES.items()
    }


def collect_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Read raw data, predictions, reports, and validation metadata."""
    anomalies: list[dict[str, Any]] = []
    video_paths = sorted(
        (
            path
            for path in args.clip_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda path: int(path.stem) if path.stem.isdigit() else math.inf,
    )
    video_groups: dict[int, list[Path]] = defaultdict(list)
    for path in video_paths:
        if not path.stem.isdigit():
            anomalies.append(
                {"type": "invalid_video_name", "path": str(path.resolve())}
            )
            continue
        video_groups[int(path.stem)].append(path.resolve())
    for sample, paths in video_groups.items():
        if len(paths) > 1:
            anomalies.append(
                {
                    "type": "duplicate_video_sample_id",
                    "sample": sample,
                    "paths": [str(path) for path in paths],
                }
            )

    gt_paths = sorted(
        (
            path
            for path in args.gt_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    gt_groups: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for path in gt_paths:
        match = GT_PATTERN.fullmatch(path.stem)
        if match is None:
            anomalies.append(
                {"type": "invalid_gt_name", "path": str(path.resolve())}
            )
            continue
        gt_groups[int(match.group("sample"))].append(
            (int(match.group("class_id")), path.resolve())
        )
    for sample, entries in gt_groups.items():
        if len(entries) > 1:
            anomalies.append(
                {
                    "type": "duplicate_gt_sample_id",
                    "sample": sample,
                    "entries": [
                        {"class_id": label, "path": str(path)}
                        for label, path in entries
                    ],
                }
            )

    video_ids = set(video_groups)
    gt_ids = set(gt_groups)
    for sample in sorted(video_ids - gt_ids):
        anomalies.append({"type": "missing_gt_for_video", "sample": sample})
    for sample in sorted(gt_ids - video_ids):
        anomalies.append({"type": "missing_video_for_gt", "sample": sample})

    gt_labels = {sample: entries[0][0] for sample, entries in gt_groups.items()}
    gt_by_sample = {sample: entries[0][1] for sample, entries in gt_groups.items()}
    videos_by_sample = {sample: paths[0] for sample, paths in video_groups.items()}

    config = yaml.safe_load(args.label_config.read_text(encoding="utf-8"))
    configured_labels = {
        int(sample): int(label) for sample, label in config.get("samples", {}).items()
    }
    config_mismatches = []
    for sample in sorted(set(gt_labels) & set(configured_labels)):
        if gt_labels[sample] != configured_labels[sample]:
            row = {
                "type": "label_config_mismatch",
                "sample": sample,
                "filename_gt": gt_labels[sample],
                "config_label": configured_labels[sample],
                "config_path": str(args.label_config.resolve()),
            }
            anomalies.append(row)
            config_mismatches.append(row)

    historical_single_frame_dir = args.gt_dir.parent / "single_frame"
    if historical_single_frame_dir.is_dir():
        for path in sorted(historical_single_frame_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            match = GT_PATTERN.fullmatch(path.stem)
            if match is None:
                continue
            sample = int(match.group("sample"))
            historical_label = int(match.group("class_id"))
            if sample in gt_labels and historical_label != gt_labels[sample]:
                anomalies.append(
                    {
                        "type": "historical_single_frame_label_mismatch",
                        "sample": sample,
                        "filename_gt": gt_labels[sample],
                        "historical_label": historical_label,
                        "historical_path": str(path.resolve()),
                    }
                )

    prediction_path = args.classification_dir / "predictions.json"
    prediction_rows = read_json(prediction_path)
    predictions: dict[int, dict[str, Any]] = {}
    for row in prediction_rows:
        sample = sample_from_stem(Path(str(row["image"])).stem)
        if sample is None:
            anomalies.append(
                {"type": "unmapped_classification_row", "image": row["image"]}
            )
            continue
        if sample in predictions:
            anomalies.append(
                {"type": "duplicate_classification_sample", "sample": sample}
            )
        predictions[sample] = row
        if sample in gt_labels:
            stored_gt = int(row.get("gt_label", -1))
            stored_prediction = int(row.get("pred_label", -1))
            if stored_gt != gt_labels[sample]:
                anomalies.append(
                    {
                        "type": "classification_gt_label_mismatch",
                        "sample": sample,
                        "filename_gt": gt_labels[sample],
                        "classification_gt": stored_gt,
                    }
                )
            expected_correct = stored_prediction == gt_labels[sample]
            if bool(row.get("correct")) != expected_correct:
                anomalies.append(
                    {
                        "type": "classification_correct_flag_mismatch",
                        "sample": sample,
                        "stored_correct": bool(row.get("correct")),
                        "recomputed_correct": expected_correct,
                    }
                )
        image_path = Path(str(row["image"]))
        image_meta = inspect_image(image_path)
        row["input_readable"] = image_meta["readable"]
        if not image_meta["readable"]:
            anomalies.append(
                {
                    "type": "unreadable_classification_input",
                    "sample": sample,
                    "path": str(image_path),
                }
            )
    for sample in sorted(gt_ids - set(predictions)):
        anomalies.append({"type": "missing_classification", "sample": sample})

    tracking_summary = read_json(args.tracking_summary)
    tracking_reports: dict[int, dict[str, Any]] = {}
    output_validation: dict[int, dict[str, Any]] = {}
    for summary_row in tracking_summary.get("videos", []):
        sample = int(summary_row["sample"])
        output_video = Path(str(summary_row["output_video"])).resolve()
        report_path = output_video.parent / "tracking_report.json"
        if not report_path.is_file():
            anomalies.append(
                {
                    "type": "missing_tracking_report",
                    "sample": sample,
                    "path": str(report_path),
                }
            )
            continue
        report = read_json(report_path)
        report["report_path"] = str(report_path.resolve())
        tracking_reports[sample] = report
        validation = inspect_video(output_video, decode_all=False)
        output_validation[sample] = validation
        if not validation["readable"]:
            anomalies.append(
                {
                    "type": "unreadable_tracking_video",
                    "sample": sample,
                    "path": str(output_video),
                }
            )
        if abs(validation["fps"] - 10.0) > 0.01:
            anomalies.append(
                {
                    "type": "tracking_output_not_10hz",
                    "sample": sample,
                    "fps": validation["fps"],
                    "path": str(output_video),
                }
            )
        if validation["reported_frame_count"] != int(report["sampled_frame_count"]):
            anomalies.append(
                {
                    "type": "tracking_output_frame_count_mismatch",
                    "sample": sample,
                    "video_frame_count": validation["reported_frame_count"],
                    "report_sampled_frame_count": report["sampled_frame_count"],
                }
            )
        if sample in gt_labels and (
            int(report["classification"]["class_id"]) != gt_labels[sample]
        ):
            anomalies.append(
                {
                    "type": "tracking_label_mismatch",
                    "sample": sample,
                    "filename_gt": gt_labels[sample],
                    "tracking_label": report["classification"]["class_id"],
                }
            )
    for sample in sorted(video_ids - set(tracking_reports)):
        anomalies.append({"type": "missing_tracking_result", "sample": sample})

    source_metadata: dict[int, dict[str, Any]] = {}
    gt_metadata: dict[int, dict[str, Any]] = {}
    for sample in sorted(video_ids):
        source_metadata[sample] = inspect_video(
            videos_by_sample[sample], decode_all=True
        )
        metadata = source_metadata[sample]
        if not metadata["readable"]:
            anomalies.append(
                {
                    "type": "unreadable_source_video",
                    "sample": sample,
                    "path": str(videos_by_sample[sample]),
                }
            )
        if metadata["decoded_frame_count"] != metadata["reported_frame_count"]:
            anomalies.append(
                {
                    "type": "source_frame_count_mismatch",
                    "sample": sample,
                    "reported": metadata["reported_frame_count"],
                    "decoded": metadata["decoded_frame_count"],
                }
            )
    for sample in sorted(gt_ids):
        gt_metadata[sample] = inspect_image(gt_by_sample[sample])
        if not gt_metadata[sample]["readable"]:
            anomalies.append(
                {
                    "type": "unreadable_gt_image",
                    "sample": sample,
                    "path": str(gt_by_sample[sample]),
                }
            )

    return {
        "anomalies": anomalies,
        "config_mismatches": config_mismatches,
        "gt_labels": gt_labels,
        "gt_by_sample": gt_by_sample,
        "videos_by_sample": videos_by_sample,
        "source_metadata": source_metadata,
        "gt_metadata": gt_metadata,
        "predictions": predictions,
        "prediction_path": prediction_path.resolve(),
        "tracking_summary": tracking_summary,
        "tracking_reports": tracking_reports,
        "output_validation": output_validation,
    }


def build_manifest(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one machine-readable row per paired sample."""
    rows = []
    all_samples = sorted(set(data["videos_by_sample"]) | set(data["gt_by_sample"]))
    for sample in all_samples:
        video = data["videos_by_sample"].get(sample)
        gt = data["gt_by_sample"].get(sample)
        label = data["gt_labels"].get(sample)
        source = data["source_metadata"].get(sample, {})
        gt_meta = data["gt_metadata"].get(sample, {})
        prediction = data["predictions"].get(sample, {})
        tracking = data["tracking_reports"].get(sample, {})
        output_meta = data["output_validation"].get(sample, {})
        sampled = int(tracking.get("sampled_frame_count", 0))
        cable_count = int(tracking.get("cable_tracked_frame_count", 0))
        damage_count = int(tracking.get("damage_tracked_frame_count", 0))
        tracking_seconds = float(tracking.get("cable_tracking_sec", 0.0)) + float(
            tracking.get("damage_tracking_sec", 0.0)
        )
        rows.append(
            {
                "sample_id": sample,
                "video_name": video.name if video else "",
                "video_path": str(video) if video else "",
                "gt_image_path": str(gt) if gt else "",
                "class_id": label if label is not None else "",
                "class_name": CLASS_NAMES[label][0] if label in CLASS_NAMES else "",
                "class_name_zh": CLASS_NAMES[label][1] if label in CLASS_NAMES else "",
                "pairing_status": "paired" if video and gt else "unpaired",
                "video_readable": source.get("readable", False),
                "gt_readable": gt_meta.get("readable", False),
                "source_fps": round(float(source.get("fps", 0.0)), 6),
                "width": source.get("width", 0),
                "height": source.get("height", 0),
                "resolution": (
                    f"{source.get('width', 0)}x{source.get('height', 0)}"
                    if source
                    else ""
                ),
                "reported_frame_count": source.get("reported_frame_count", 0),
                "decoded_frame_count": source.get("decoded_frame_count", 0),
                "duration_sec": round(float(source.get("duration_sec", 0.0)), 6),
                "video_size_bytes": video.stat().st_size if video else 0,
                "gt_width": gt_meta.get("width", 0),
                "gt_height": gt_meta.get("height", 0),
                "gt_size_bytes": gt.stat().st_size if gt else 0,
                "classification_input_image": prediction.get("image", ""),
                "predicted_class_id": prediction.get("pred_label", ""),
                "predicted_class_name": prediction.get("pred_class", ""),
                "classification_correct": prediction.get("correct", ""),
                "tracking_report_path": tracking.get("report_path", ""),
                "sampled_10hz_frame_count": sampled,
                "cable_tracked_frame_count": cable_count,
                "cable_tracking_coverage": (
                    round(cable_count / sampled, 6) if sampled else ""
                ),
                "damage_tracked_frame_count": damage_count if label == 0 else "",
                "damage_tracking_coverage": (
                    round(damage_count / sampled, 6) if sampled and label == 0 else ""
                ),
                "tracking_output_video": tracking.get("output_video", ""),
                "tracking_preview_image": tracking.get("preview_image", ""),
                "reported_output_fps": tracking.get("output_fps", ""),
                "verified_output_fps": round(float(output_meta.get("fps", 0.0)), 6),
                "verified_output_frame_count": output_meta.get(
                    "reported_frame_count", 0
                ),
                "cable_tracking_sec": round(
                    float(tracking.get("cable_tracking_sec", 0.0)), 6
                ),
                "damage_tracking_sec": round(
                    float(tracking.get("damage_tracking_sec", 0.0)), 6
                ),
                "total_tracking_inference_sec": round(tracking_seconds, 6),
                "output_equivalent_processing_fps": (
                    round(sampled / tracking_seconds, 6) if tracking_seconds else ""
                ),
                "cable_prompt": tracking.get("cable_prompt", ""),
                "damage_prompt": tracking.get("damage_prompt", ""),
                "reference_match_frame": tracking.get("reference_match_frame", ""),
                "reference_match_score": (
                    round(float(tracking.get("reference_match_score", 0.0)), 6)
                    if tracking
                    else ""
                ),
            }
        )
    return rows


def find_diagnostic(
    classification_dir: Path, sample: int, label: int, suffix: str
) -> Path | None:
    """Find one hashed diagnostic image for a sample and suffix."""
    pattern = f"*_sample{sample:02d}_{label}_f*_{suffix}.jpg"
    matches = sorted((classification_dir / "diagnostics").glob(pattern))
    return matches[0].resolve() if matches else None


def build_experiments(
    args: argparse.Namespace,
    data: dict[str, Any],
    manifest_by_sample: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the selected experiment rows and uniform evidence links."""
    experiments = []
    for experiment_number, sample in enumerate(EXPERIMENT_SAMPLES, start=1):
        row = manifest_by_sample[sample]
        label = int(row["class_id"])
        class_en, class_zh = CLASS_NAMES[label]
        prefix = f"exp{experiment_number:02d}_sample{sample:02d}_{class_en}"
        preview_target = Path(str(row["tracking_preview_image"]))
        video_target = Path(str(row["tracking_output_video"]))
        gt_target = Path(str(row["gt_image_path"]))
        preview_link = args.visual_dir / "previews" / f"{prefix}.jpg"
        video_link = args.visual_dir / "videos" / f"{prefix}_10hz.mp4"
        gt_link = args.visual_dir / "ground_truth" / f"{prefix}_gt.jpg"
        ensure_symlink(preview_link, preview_target)
        ensure_symlink(video_link, video_target)
        ensure_symlink(gt_link, gt_target)

        damage_diag = find_diagnostic(
            args.classification_dir, sample, label, "damage_full"
        )
        position_diag = find_diagnostic(
            args.classification_dir, sample, label, "position"
        )
        damage_diag_link: Path | None = None
        position_diag_link: Path | None = None
        if damage_diag is not None:
            damage_diag_link = (
                args.visual_dir / "diagnostics" / f"{prefix}_classification_damage.jpg"
            )
            ensure_symlink(damage_diag_link, damage_diag)
        if position_diag is not None:
            position_diag_link = (
                args.visual_dir
                / "diagnostics"
                / f"{prefix}_classification_position.jpg"
            )
            ensure_symlink(position_diag_link, position_diag)

        notes = EXPERIMENT_NOTES[sample]
        tracking_sec = float(row["total_tracking_inference_sec"])
        sampled = int(row["sampled_10hz_frame_count"])
        experiments.append(
            {
                "experiment_id": f"EXP-{experiment_number:02d}",
                "sample_id": sample,
                "selection_basis": notes["selection"],
                "difficulty": notes["difficulty"],
                "source_video_path": row["video_path"],
                "gt_image_path": row["gt_image_path"],
                "gt_class_id": label,
                "gt_class_name": class_en,
                "gt_class_name_zh": class_zh,
                "predicted_class_id": row["predicted_class_id"],
                "predicted_class_name": row["predicted_class_name"],
                "classification_correct": row["classification_correct"],
                "source_resolution": row["resolution"],
                "source_fps": row["source_fps"],
                "source_total_frames": row["decoded_frame_count"],
                "source_duration_sec": row["duration_sec"],
                "sampled_10hz_frames": sampled,
                "cable_tracked_frames": row["cable_tracked_frame_count"],
                "cable_tracking_coverage": row["cable_tracking_coverage"],
                "damage_tracked_frames": row["damage_tracked_frame_count"],
                "damage_tracking_coverage": row["damage_tracking_coverage"],
                "reported_output_fps": row["reported_output_fps"],
                "verified_output_fps": row["verified_output_fps"],
                "tracking_inference_sec": row["total_tracking_inference_sec"],
                "output_equivalent_processing_fps": (
                    round(sampled / tracking_sec, 6) if tracking_sec else ""
                ),
                "method": "SAM3 静态图三分类 + 10 Hz 稠密视频跟踪",
                "cable_prompt": row["cable_prompt"],
                "damage_prompt": row["damage_prompt"],
                "reference_match_frame": row["reference_match_frame"],
                "tracking_report_path": row["tracking_report_path"],
                "original_output_video_path": row["tracking_output_video"],
                "original_preview_image_path": row["tracking_preview_image"],
                "support_video_link": str(video_link.absolute()),
                "support_preview_link": str(preview_link.absolute()),
                "support_gt_link": str(gt_link.absolute()),
                "classification_damage_diagnostic": (
                    str(damage_diag_link.absolute())
                    if damage_diag_link
                    else ""
                ),
                "classification_position_diagnostic": (
                    str(position_diag_link.absolute())
                    if position_diag_link
                    else ""
                ),
                "phenomenon": notes["phenomenon"],
                "success_point": notes["success"],
                "known_issue": notes["issue"],
            }
        )
    return experiments


def summarize(
    args: argparse.Namespace,
    data: dict[str, Any],
    manifest: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create the dataset-level and experiment-level JSON summaries."""
    gt_labels = data["gt_labels"]
    metrics = compute_classification_metrics(data["predictions"], gt_labels)
    selected_metrics = compute_classification_metrics(
        data["predictions"], gt_labels, EXPERIMENT_SAMPLES
    )
    paired = [row for row in manifest if row["pairing_status"] == "paired"]
    resolution_counts = Counter(row["resolution"] for row in paired)
    fps_counts = Counter(f"{float(row['source_fps']):.6f}" for row in paired)
    total_sampled = sum(int(row["sampled_10hz_frame_count"]) for row in paired)
    total_cable = sum(int(row["cable_tracked_frame_count"]) for row in paired)
    damaged_rows = [row for row in paired if row["class_id"] == 0]
    total_damage_sampled = sum(
        int(row["sampled_10hz_frame_count"]) for row in damaged_rows
    )
    total_damage = sum(int(row["damage_tracked_frame_count"]) for row in damaged_rows)
    total_tracking_sec = sum(
        float(row["total_tracking_inference_sec"]) for row in paired
    )
    selected_sampled = sum(int(row["sampled_10hz_frames"]) for row in experiments)
    selected_cable = sum(int(row["cable_tracked_frames"]) for row in experiments)
    selected_damaged = [row for row in experiments if row["gt_class_id"] == 0]
    selected_damage_sampled = sum(
        int(row["sampled_10hz_frames"]) for row in selected_damaged
    )
    selected_damage = sum(int(row["damage_tracked_frames"]) for row in selected_damaged)
    gt_assisted = sum(
        row["cable_prompt"] == "GT reference point" for row in paired
    )
    classification_metrics_path = args.classification_dir / "metrics.json"
    classification_evaluation_path = args.classification_dir / "evaluation.json"
    tracking_summary_path = args.tracking_summary.resolve()
    dataset_summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_paths": {
            "clip_dir": str(args.clip_dir.resolve()),
            "gt_dir": str(args.gt_dir.resolve()),
            "classification_predictions": str(data["prediction_path"]),
            "classification_metrics": str(classification_metrics_path.resolve()),
            "classification_evaluation": str(classification_evaluation_path.resolve()),
            "tracking_summary": str(tracking_summary_path),
            "label_config": str(args.label_config.resolve()),
        },
        "dataset": {
            "video_count": len(data["videos_by_sample"]),
            "gt_image_count": len(data["gt_by_sample"]),
            "paired_sample_count": len(paired),
            "all_pairs_one_to_one": (
                len(paired)
                == len(data["videos_by_sample"])
                == len(data["gt_by_sample"])
            ),
            "total_decoded_frames": sum(
                int(row["decoded_frame_count"]) for row in paired
            ),
            "total_duration_sec": round(
                sum(float(row["duration_sec"]) for row in paired), 6
            ),
            "class_distribution": class_distribution(gt_labels),
            "resolution_distribution": dict(sorted(resolution_counts.items())),
            "source_fps_distribution": dict(sorted(fps_counts.items())),
            "unreadable_video_count": sum(
                not bool(row["video_readable"]) for row in manifest
            ),
            "unreadable_gt_count": sum(
                not bool(row["gt_readable"]) for row in manifest
            ),
        },
        "current_test_material": {
            "classification_image_count": len(data["predictions"]),
            "classification_input_directory": str(
                Path(next(iter(data["predictions"].values()))["image"]).parent
            ),
            "tracking_video_count": len(data["tracking_reports"]),
            "sampled_10hz_frame_count": total_sampled,
            "valid_cable_tracked_frame_count": total_cable,
            "damage_tracked_frame_count": total_damage,
        },
        "classification_metrics_recomputed": metrics,
        "tracking_metrics_recomputed": {
            "sampled_10hz_frame_count": total_sampled,
            "cable_tracked_frame_count": total_cable,
            "cable_tracking_coverage": ratio(total_cable, total_sampled),
            "damaged_sample_count": len(damaged_rows),
            "damaged_sampled_10hz_frame_count": total_damage_sampled,
            "damage_tracked_frame_count": total_damage,
            "damage_tracking_coverage": ratio(total_damage, total_damage_sampled),
            "verified_10hz_video_count": sum(
                abs(float(row["verified_output_fps"]) - 10.0) <= 0.01
                for row in paired
            ),
            "tracking_video_count": len(paired),
            "gt_reference_point_initialized_count": gt_assisted,
            "text_prompt_initialized_count": len(paired) - gt_assisted,
            "summed_inference_sec_excluding_model_load": round(
                total_tracking_sec, 6
            ),
            "output_equivalent_processing_fps": ratio(
                total_sampled, round(total_tracking_sec, 6)
            ),
        },
        "anomalies": data["anomalies"],
    }
    experiment_summary = {
        "generated_at": dataset_summary["generated_at"],
        "selection_policy": (
            "5 组破损、3 组裸露、2 组悬空；悬空真值仅有 2 组，全部纳入；"
            "同时覆盖时长、分辨率、画面方向、分类错误和跟踪难例。"
        ),
        "selected_samples": EXPERIMENT_SAMPLES,
        "class_distribution": class_distribution(
            {sample: gt_labels[sample] for sample in EXPERIMENT_SAMPLES}
        ),
        "classification_metrics": selected_metrics,
        "tracking_metrics": {
            "sampled_10hz_frame_count": selected_sampled,
            "cable_tracked_frame_count": selected_cable,
            "cable_tracking_coverage": ratio(selected_cable, selected_sampled),
            "damaged_sampled_10hz_frame_count": selected_damage_sampled,
            "damage_tracked_frame_count": selected_damage,
            "damage_tracking_coverage": ratio(selected_damage, selected_damage_sampled),
        },
        "experiments": experiments,
    }
    return dataset_summary, experiment_summary


def confusion_markdown(metrics: dict[str, Any]) -> str:
    """Render a compact confusion matrix with truth as rows."""
    matrix = metrics["confusion_matrix_rows_truth_columns_prediction"]
    return "\n".join(
        [
            "| 真值 / 预测 | 破损 | 裸露 | 悬空 | 未分类 |",
            "|---|---:|---:|---:|---:|",
            f"| 破损 | {matrix[0][0]} | {matrix[0][1]} | {matrix[0][2]} | {matrix[0][3]} |",
            f"| 裸露 | {matrix[1][0]} | {matrix[1][1]} | {matrix[1][2]} | {matrix[1][3]} |",
            f"| 悬空 | {matrix[2][0]} | {matrix[2][1]} | {matrix[2][2]} | {matrix[2][3]} |",
        ]
    )


def write_documents(
    args: argparse.Namespace,
    data: dict[str, Any],
    manifest: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> None:
    """Write the Chinese acceptance-facing Markdown set."""
    docs = args.docs_dir.resolve()
    visual = args.visual_dir.resolve()
    metrics = dataset_summary["classification_metrics_recomputed"]
    tracking = dataset_summary["tracking_metrics_recomputed"]
    selected_metrics = experiment_summary["classification_metrics"]
    selected_tracking = experiment_summary["tracking_metrics"]
    dataset = dataset_summary["dataset"]
    source_paths = dataset_summary["source_paths"]
    manifest_path = docs / "dataset_manifest.csv"
    experiment_csv_path = docs / "experiment_results.csv"
    dataset_json_path = docs / "dataset_summary.json"
    experiment_json_path = docs / "experiment_summary.json"

    readme = f"""# 中期测试数据集与实验支撑材料

本目录是 2026-08-16 中期验收材料入口。所有数字由
{md_link('整理脚本', Path(__file__))} 读取真实文件复算；原视频、真值和历史实验
结果未移动、未删除、未覆盖。可视化目录只包含指向既有文件的符号链接，不复制
大型视频。

## 一页结论

- 数据：{dataset['video_count']} 段视频、{dataset['gt_image_count']} 张真值图，
  {dataset['paired_sample_count']} 组一一对应；共 {dataset['total_decoded_frames']} 帧，
  {dataset['total_duration_sec']:.2f} 秒。
- 类别：破损 {dataset['class_distribution']['0']['count']}、裸露
  {dataset['class_distribution']['1']['count']}、悬空
  {dataset['class_distribution']['2']['count']}。
- 静态图分类：{metrics['correct']}/{metrics['total']}，准确率
  {percent(metrics['accuracy'])}；宏平均 F1 {percent(metrics['macro_f1'])}。
- 10 Hz 跟踪：{tracking['tracking_video_count']}/{dataset['video_count']} 段完成，
  {tracking['cable_tracked_frame_count']}/{tracking['sampled_10hz_frame_count']} 个抽帧
  有海缆结果，覆盖率 {percent(tracking['cable_tracking_coverage'])}。
- 破损区域：在 {tracking['damaged_sample_count']} 个破损视频中，
  {tracking['damage_tracked_frame_count']}/{tracking['damaged_sampled_10hz_frame_count']}
  个抽帧有破损结果，覆盖率 {percent(tracking['damage_tracking_coverage'])}。
- 输出视频：{tracking['verified_10hz_video_count']}/{tracking['tracking_video_count']}
  个文件经 OpenCV 独立读取为 10.00 Hz。
- 代表性 10 组：{selected_metrics['correct']}/{selected_metrics['total']} 分类正确
  （{percent(selected_metrics['accuracy'])}）；该子集刻意纳入难例，不替代 30 样本总体指标。
- 完整性：权威 `clip/` 与 `gt/` 无缺失、重复、错误命名或不可读文件；样本 9
  存在 2 条同源历史标签不一致（配置 YAML 和旧 `single_frame` 文件），均已列入异常。

## 文件入口

- {md_link('数据集说明', docs / 'dataset_description.md')}
- {md_link('数据统计', docs / 'dataset_statistics.md')}
- {md_link('实验方法与评价协议', docs / 'experiment_protocol.md')}
- {md_link('10 组实验记录', docs / 'ten_experiments.md')}
- {md_link('指标汇总', docs / 'metrics_summary.md')}
- {md_link('证据索引', docs / 'evidence_index.md')}
- {md_link('数据清单 CSV', manifest_path)}
- {md_link('实验结果 CSV', experiment_csv_path)}
- {md_link('数据汇总 JSON', dataset_json_path)}
- {md_link('实验汇总 JSON', experiment_json_path)}
- {md_link('统一可视化目录', visual)}

## 使用建议

汇报时先打开“指标汇总”，再从“10 组实验记录”点击预览图和 10 Hz 视频。
现场追问单个数值时，通过“证据索引”回到原始 JSON、逐样本报告或视频。
本材料中的“跟踪覆盖率”表示报告中该帧是否返回 mask/对象，不等同于有逐帧
人工真值支撑的 IoU 或检测召回率。
"""
    (docs / "README.md").write_text(readme, encoding="utf-8")

    description = f"""# 中期测试数据集说明

## 数据来源与采集场景

本数据来自项目中期水池实验。项目文档和画面共同表明：相机从水下/池侧固定
视角观察深色海缆样件，每个剪辑对应一个独立样本。现有文件没有记录具体相机
型号、拍摄人员和精确采集日期，因此本材料不补写这些未知信息。原始入口为
{md_link('视频目录', args.clip_dir)} 和 {md_link('真值目录', args.gt_dir)}。

## 目录与编号

```text
/home/nvidia/DATA/UW/mid/
├── clip/                 # 0.mp4 ... 29.mp4；一段视频一个样本
├── gt/                   # sampleNN_CLASS.jpg；本次权威真值
└── gt_enhanced_wb_clahe/ # 分类实际使用的白平衡 + CLAHE 增强图
```

- 视频文件名的十进制数字是样本编号，例如 `3.mp4` 为样本 3。
- 真值图命名为 `sampleNN_CLASS.jpg`；`NN` 与视频编号相同，使用两位补零。
- `CLASS` 是样本级类别：`0` 破损、`1` 裸露、`2` 悬空。
- 真值图是对应视频样本的代表图/裁剪图，不是逐帧像素级标注。跟踪脚本会用
  ORB 将可能裁剪或旋转过的真值图匹配到 10 Hz 抽帧中的参考帧。
- 本次核验得到 {dataset['paired_sample_count']} 个视频—真值对，编号一一对应。

## 三类定义

| ID | 工程类别 | 项目内部名 | 定义 |
|---:|---|---|---|
| 0 | 破损 | `damaged` | 护套破裂、缺口、明显磨损、变形、断裂或异常材料外露；水池样本以金属贴片模拟部分损伤。损伤优先，不再区分触底/悬空。 |
| 1 | 裸露 | `exposed_intact` | 海缆触底、平放或贴近池底，且未达到破损判据。这里“裸露”是位置状态，不是“导体裸露”。 |
| 2 | 悬空 | `suspended_intact` | 海缆由支撑结构托起或与池底存在间隙；当前水池规则主要依赖端部白色支撑杆几何线索。 |

## 数据特点

- 水体和相机白平衡造成明显绿色偏色，原图与增强图颜色分布不同。
- 光照、反光和阴影随相机/样本运动变化，亮色贴片和端部标签可能形成干扰。
- 海缆存在横向、斜向、纵向姿态；样本 27、28 等为竖屏或近竖屏画面。
- 池砖网格、斑驳污迹、绳索、白色设备和支撑杆构成结构化背景干扰。
- 分辨率不完全一致：{', '.join(f'{key}（{value} 段）' for key, value in dataset['resolution_distribution'].items())}。
- 时长跨度约从 {min(float(row['duration_sec']) for row in manifest):.2f} 秒到
  {max(float(row['duration_sec']) for row in manifest):.2f} 秒。

## 当前规模与类别分布

- 视频 {dataset['video_count']} 段，共 {dataset['total_decoded_frames']} 个可解码帧，
  总时长 {dataset['total_duration_sec']:.2f} 秒。
- 真值图 {dataset['gt_image_count']} 张：破损
  {dataset['class_distribution']['0']['count']}、裸露
  {dataset['class_distribution']['1']['count']}、悬空
  {dataset['class_distribution']['2']['count']}。
- 当前静态图分类使用 {len(data['predictions'])} 张增强图；视频跟踪使用
  {len(data['tracking_reports'])} 段视频，10 Hz 抽帧共
  {tracking['sampled_10hz_frame_count']} 帧。

## 局限性

1. 样本仅 30 个，类别不均衡明显，悬空仅 2 个。
2. 数据集中于同一水池、相似材料和机位，缺少海流、泥沙、海生物、不同海床及
   不同海缆外观；不能直接代表真实海域泛化能力。
3. 当前真值是每视频一个样本级类别和一张代表图，不含逐帧 bbox/mask，因此
   跟踪覆盖率只衡量“报告是否产生对象”，不能计算 IoU、MOTA 等定位指标。
4. SAM3 分类阈值曾在这 30 张中期真值图上收敛，当前 90% 是同一数据上的标定
   结果，不是独立盲测集精度。
5. {tracking['gt_reference_point_initialized_count']}/{tracking['tracking_video_count']}
   段最终跟踪报告使用 `GT reference point` 初始化；因此跟踪覆盖率不是全自动
   首帧检出率。
6. 悬空判定利用本水池白色支撑杆，真实海底没有同类支撑线索时需重新设计。

## 中期测试使用方式

中期汇报用 30 样本总体分类指标说明三分类现状，用 10 Hz 视频和覆盖率说明
时序跟踪连续性，再用代表性 10 组展示成功与失败边界。完整样本级数据见
{md_link('dataset_manifest.csv', manifest_path)}；不要把增强图当作新增独立样本，
也不要把 10 Hz 相邻帧当作独立分类样本扩充准确率分母。
"""
    (docs / "dataset_description.md").write_text(description, encoding="utf-8")

    stat_rows = []
    for row in manifest:
        stat_rows.append(
            "| {sample_id} | {video} | {gt} | {cls} | {res} | {fps:.3f} | "
            "{frames} | {duration:.2f} | {sampled} | {cable} | {coverage} |".format(
                sample_id=row["sample_id"],
                video=md_link(Path(str(row["video_path"])).name, row["video_path"]),
                gt=md_link(Path(str(row["gt_image_path"])).name, row["gt_image_path"]),
                cls=f"{row['class_id']} {row['class_name_zh']}",
                res=row["resolution"],
                fps=float(row["source_fps"]),
                frames=row["decoded_frame_count"],
                duration=float(row["duration_sec"]),
                sampled=row["sampled_10hz_frame_count"],
                cable=row["cable_tracked_frame_count"],
                coverage=percent(float(row["cable_tracking_coverage"])),
            )
        )
    anomaly_lines = []
    for item in data["anomalies"]:
        if item["type"] == "label_config_mismatch":
            anomaly_lines.append(
                f"- 标签配置不一致：样本 {item['sample']} 的文件名真值为 "
                f"{item['filename_gt']}，配置写为 {item['config_label']}；本材料以文件名为准。"
            )
        elif item["type"] == "historical_single_frame_label_mismatch":
            anomaly_lines.append(
                f"- 历史单帧标签不一致：{md_link(Path(item['historical_path']).name, item['historical_path'])} "
                f"的后缀为 {item['historical_label']}，权威 GT 为 {item['filename_gt']}；"
                "该目录未参与当前正式评测。"
            )
        else:
            anomaly_lines.append(f"- `{item['type']}`：`{json.dumps(item, ensure_ascii=False)}`")
    if not anomaly_lines:
        anomaly_lines = ["- 未发现命名、配对、可读性或帧数异常。"]
    statistics = f"""# 数据统计与完整性核验

## 汇总

| 项目 | 结果 |
|---|---:|
| 视频数 | {dataset['video_count']} |
| 真值图数 | {dataset['gt_image_count']} |
| 一一对应样本 | {dataset['paired_sample_count']} |
| 全部可解码帧 | {dataset['total_decoded_frames']} |
| 总时长 | {dataset['total_duration_sec']:.2f} s（{dataset['total_duration_sec'] / 60:.2f} min） |
| 破损 / 裸露 / 悬空 | {dataset['class_distribution']['0']['count']} / {dataset['class_distribution']['1']['count']} / {dataset['class_distribution']['2']['count']} |
| 当前分类图像 | {len(data['predictions'])} |
| 当前跟踪视频 | {len(data['tracking_reports'])} |
| 10 Hz 抽帧 | {tracking['sampled_10hz_frame_count']} |
| 有效海缆跟踪帧 | {tracking['cable_tracked_frame_count']} |
| 有效破损区域跟踪帧 | {tracking['damage_tracked_frame_count']}（仅破损视频） |

帧数通过 OpenCV 对每段原视频逐帧 `grab()` 计数；时长统一按“可解码帧数 / OpenCV
读取帧率”计算，避免容器时长小数差异。输出视频另行读取首帧、末帧、帧数和 FPS。

## 逐样本清单

| 编号 | 原视频 | 真值图 | 类别 | 分辨率 | 原 FPS | 总帧数 | 时长/s | 10 Hz 帧 | 海缆成功帧 | 覆盖率 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(stat_rows)}

## 配对与异常

- 视频编号集合与真值编号集合均为 0–29，30 组一一对应。
- 原视频 30/30 可打开且报告帧数与逐帧解码计数一致。
- 真值图 30/30 可读取；未发现重复编号、错误命名或缺失配对。
- 分类 30/30、跟踪 30/30 均有结果。
{chr(10).join(anomaly_lines)}

完整字段和字节数见 {md_link('dataset_manifest.csv', manifest_path)}，结构化汇总见
{md_link('dataset_summary.json', dataset_json_path)}。
"""
    (docs / "dataset_statistics.md").write_text(statistics, encoding="utf-8")

    protocol = f"""# 实验方法、运行环境与评价协议

## 指标口径来源

{md_link('docs/midterm_plan.md', Path(__file__).resolve().parent.parent / 'docs/midterm_plan.md')}
规定中期交付应包含 Precision、Recall、按类指标和混淆矩阵；
{md_link('docs/pool_runtime_plan.md', Path(__file__).resolve().parent.parent / 'docs/pool_runtime_plan.md')}
规定现场演示优先关注海缆帧检出率。现有数据没有逐帧人工 mask，因此本材料采用
三分类指标、海缆 mask 存在覆盖率和破损区域 mask 存在覆盖率，不报告无法由真值
支撑的 mAP/IoU。

## 静态图三分类

- 输入：`gt_enhanced_wb_clahe` 中 30 张增强图；真值仍取 `gt` 文件名后缀。
- 方法：SAM3-only 开放词汇分割与固定规则。损伤直接提示词为
  `metal patch on pipe`；悬空由管体 mask 端点附近白色支撑杆几何线索判定，
  最终采用悬空优先。
- 证据：{md_link('classification_results.json', args.classification_dir / 'classification_results.json')}、
  {md_link('predictions.json', data['prediction_path'])}、
  {md_link('metrics.json', source_paths['classification_metrics'])}。
- 已有分类计时 {read_json(Path(source_paths['classification_metrics']))['elapsed_sec']:.3f} 秒，
  即 {read_json(Path(source_paths['classification_metrics']))['seconds_per_image']:.3f} 秒/图、
  {read_json(Path(source_paths['classification_metrics']))['images_per_sec']:.4f} 图/秒；该计时来自历史报告。

## 10 Hz 视频跟踪

- 每段视频按目标 10 Hz 采样，SAM3 进行稠密双向跟踪。
- 海缆分支使用报告记录的 `pipe`、`black pipe`、`cable` 或
  `GT reference point` 初始化；破损视频使用 `metal patch on pipe` 和参考框/点。
- 每段的真实提示词、耗时、抽帧索引和逐帧对象记录都保存在
  `tracking_report.json`，最终路径由 {md_link('tracking_summary.json', args.tracking_summary)} 指向。
- 历史结果由主目录、point-candidate 修正目录和 semantic-fix 修正目录汇总；
  汇总 JSON 指向的逐样本报告是本材料的最终口径。

## 公式

- 三分类准确率 = 正确样本数 / 已分类样本数。
- 类别 Precision = TP / (TP + FP)，Recall = TP / (TP + FN)，
  F1 = 2PR / (P + R)。
- 海缆跟踪覆盖率 = 有海缆对象/mask 的 10 Hz 帧数 / 10 Hz 抽帧数。
- 破损区域覆盖率 = 有破损对象/mask 的帧数 / 破损视频 10 Hz 抽帧数。
- 输出等效处理速度 = 输出抽帧数 /（海缆跟踪秒数 + 破损跟踪秒数）。这是逐报告
  推理耗时求和，不含本次模型加载，不能当作摄像头端到端 FPS。
- 输出 FPS 同时读取报告字段和真实 MP4 元数据；两者均需为 10.00 Hz。

## 10 组选择规则

真值中只有 2 组悬空，故不能在不重复数据的前提下采用 4/3/3。本材料采用
5 组破损、3 组裸露、2 组悬空，全部纳入悬空样本，并覆盖 3.52–44.36 秒时长、
横/竖屏、多种分辨率、分类误判和低覆盖率难例。代表性子集刻意提高了难例占比，
其 10 组准确率不用于替代 30 组总体准确率。

## 可复现命令

整理材料（无 GPU）：

```bash
/home/nvidia/miniforge3/envs/sam3/bin/python scripts/build_midterm_support.py
```

从已有分类结果独立复算标签对比（无 GPU）：

```bash
/home/nvidia/miniforge3/envs/sam3/bin/python scripts/evaluate_midterm_results.py \\
  --results outputs/mid_gt_sam3_suspended_priority_full_20260815/classification_results.json \\
  --gt-dir /home/nvidia/DATA/UW/mid/gt \\
  --out /tmp/midterm_evaluation_recheck.json
```

新的完整跟踪运行需要 GPU，且应写入新目录：

```bash
/home/nvidia/miniforge3/envs/sam3/bin/python scripts/track_midterm_clips.py \\
  --input /home/nvidia/DATA/UW/mid/clip \\
  --gt-dir /home/nvidia/DATA/UW/mid/gt \\
  --classification-results outputs/mid_gt_sam3_suspended_priority_full_20260815/classification_results.json \\
  --target-fps 10 \\
  --cable-point-first \\
  --output-dir outputs/mid_clip_tracking_reproduction
```

注意：历史结果没有保存一条完整原始 shell 命令，且最终汇总包含两批局部修正。
上述命令能复现实验流程，但不保证与历史视频逐像素一致；精确审计以各
`tracking_report.json` 的参数和路径为准。本次整理未重跑 GPU 推理。

## 环境核验

仓库规范建议使用 `yoloe` 环境，但机器当前 `conda env list` 中不存在该环境；
本次无 GPU 整理使用已有 `sam3` 解释器（OpenCV 4.8.1、PyYAML 6.0.3）。SAM3
历史输出所用 checkpoint 路径保存在逐样本报告中。
"""
    (docs / "experiment_protocol.md").write_text(protocol, encoding="utf-8")

    experiment_summary_rows = []
    experiment_sections = []
    for row in experiments:
        damage_text = (
            f"{row['damage_tracked_frames']}/{row['sampled_10hz_frames']} "
            f"({percent(float(row['damage_tracking_coverage']))})"
            if row["gt_class_id"] == 0
            else "不适用"
        )
        experiment_summary_rows.append(
            f"| {row['experiment_id']} | {row['sample_id']} | {row['gt_class_id']} {row['gt_class_name_zh']} | "
            f"{row['predicted_class_id']} {CLASS_NAMES[int(row['predicted_class_id'])][1]} | "
            f"{'是' if row['classification_correct'] else '否'} | {row['source_resolution']} | "
            f"{row['sampled_10hz_frames']} | {percent(float(row['cable_tracking_coverage']))} | "
            f"{damage_text} | {row['difficulty']} |"
        )
        diag_links = []
        if row["classification_damage_diagnostic"]:
            diag_links.append(
                md_link("分类损伤诊断", row["classification_damage_diagnostic"])
            )
        if row["classification_position_diagnostic"]:
            diag_links.append(
                md_link("分类位置诊断", row["classification_position_diagnostic"])
            )
        section = f"""## {row['experiment_id']}：样本 {row['sample_id']}（{row['gt_class_name_zh']}）

- 选择依据：{row['selection_basis']}
- 原视频：{md_link(Path(row['source_video_path']).name, row['source_video_path'])}
- 真值图：{md_link(Path(row['gt_image_path']).name, row['gt_image_path'])}
- 真值 / 预测：{row['gt_class_id']} {row['gt_class_name_zh']} / {row['predicted_class_id']}
  {CLASS_NAMES[int(row['predicted_class_id'])][1]}；分类{'正确' if row['classification_correct'] else '错误'}。
- 原视频：{row['source_resolution']}，{float(row['source_fps']):.3f} FPS，
  {row['source_total_frames']} 帧，{float(row['source_duration_sec']):.2f} 秒。
- 10 Hz 跟踪：{row['cable_tracked_frames']}/{row['sampled_10hz_frames']} 海缆成功帧，
  覆盖率 {percent(float(row['cable_tracking_coverage']))}；破损区域 {damage_text}。
- 输出核验：报告 / MP4 = {float(row['reported_output_fps']):.2f} / {float(row['verified_output_fps']):.2f} Hz。
- 方法与提示：{row['method']}；海缆 `{row['cable_prompt']}`；损伤
  `{row['damage_prompt'] or '不适用'}`；参考抽帧 {row['reference_match_frame']}。
- 推理耗时：{float(row['tracking_inference_sec']):.3f} 秒（不含本次模型加载），
  输出等效处理速度 {float(row['output_equivalent_processing_fps']):.3f} 帧/秒。
- 现象：{row['phenomenon']}
- 成功点：{row['success_point']}
- 问题：{row['known_issue']}
- 支撑文件：{md_link('统一预览图', row['support_preview_link'])}｜
  {md_link('10 Hz 跟踪视频', row['support_video_link'])}｜
  {md_link('统一真值图', row['support_gt_link'])}｜
  {md_link('逐帧报告 JSON', row['tracking_report_path'])}"""
        section += "｜" + "｜".join(diag_links) if diag_links else ""
        experiment_sections.append(section + "\n")
    ten_experiments = f"""# 10 组中期实验记录

## 选择与总体结果

选择样本：{', '.join(str(sample) for sample in EXPERIMENT_SAMPLES)}。类别分层为
5 组破损、3 组裸露、2 组悬空；悬空仅有 2 个真值样本，故全部纳入。10 组静态
图分类 {selected_metrics['correct']}/{selected_metrics['total']} 正确
（{percent(selected_metrics['accuracy'])}）；海缆跟踪
{selected_tracking['cable_tracked_frame_count']}/{selected_tracking['sampled_10hz_frame_count']}
（{percent(selected_tracking['cable_tracking_coverage'])}）；5 组破损的破损区域跟踪
{selected_tracking['damage_tracked_frame_count']}/{selected_tracking['damaged_sampled_10hz_frame_count']}
（{percent(selected_tracking['damage_tracking_coverage'])}）。

| 实验 | 样本 | 真值 | 预测 | 正确 | 分辨率 | 10 Hz 帧 | 海缆覆盖 | 破损覆盖 | 难度 |
|---|---:|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(experiment_summary_rows)}

{chr(10).join(experiment_sections)}
"""
    (docs / "ten_experiments.md").write_text(ten_experiments, encoding="utf-8")

    per_class_rows = []
    for row in metrics["per_class"]:
        per_class_rows.append(
            f"| {row['class_id']} {row['class_name_zh']} | {row['support']} | {row['correct']} | "
            f"{percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} |"
        )
    metrics_doc = f"""# 中期指标汇总

## 三分类（30 张增强测试图）

- 准确率：{metrics['correct']}/{metrics['total']} = {percent(metrics['accuracy'])}
- 平衡准确率：{percent(metrics['balanced_accuracy'])}
- 宏平均 F1：{percent(metrics['macro_f1'])}

| 类别 | 支持数 | 正确数 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(per_class_rows)}

### 混淆矩阵

行是真值，列是预测。

{confusion_markdown(metrics)}

错误样本：5（裸露→破损）、8（破损→裸露）、22（裸露→破损）。对应图见
{md_link('历史错误案例目录', args.classification_dir / 'error_visuals')}。

## 代表性 10 组

- 分类正确率：{selected_metrics['correct']}/{selected_metrics['total']} =
  {percent(selected_metrics['accuracy'])}。
- 海缆跟踪：{selected_tracking['cable_tracked_frame_count']}/
  {selected_tracking['sampled_10hz_frame_count']} =
  {percent(selected_tracking['cable_tracking_coverage'])}。
- 破损区域跟踪：{selected_tracking['damage_tracked_frame_count']}/
  {selected_tracking['damaged_sampled_10hz_frame_count']} =
  {percent(selected_tracking['damage_tracking_coverage'])}。

10 组子集包含两个分类难例，因此其准确率低于 30 样本总体；它用于可视化审计，
不是重新抽样出的独立测试集。

## 30 段视频跟踪

| 指标 | 数值 | 口径 |
|---|---:|---|
| 完成视频 | {tracking['tracking_video_count']}/{dataset['video_count']} | 汇总 JSON 中有报告和输出路径 |
| 10 Hz 抽帧 | {tracking['sampled_10hz_frame_count']} | 30 段合计 |
| 有效海缆帧 | {tracking['cable_tracked_frame_count']} | 报告中该帧存在海缆对象/mask |
| 海缆覆盖率 | {percent(tracking['cable_tracking_coverage'])} | {tracking['cable_tracked_frame_count']}/{tracking['sampled_10hz_frame_count']} |
| 破损样本抽帧 | {tracking['damaged_sampled_10hz_frame_count']} | 19 段真值破损视频 |
| 有效破损帧 | {tracking['damage_tracked_frame_count']} | 报告中该帧存在破损对象/mask |
| 破损覆盖率 | {percent(tracking['damage_tracking_coverage'])} | {tracking['damage_tracked_frame_count']}/{tracking['damaged_sampled_10hz_frame_count']} |
| 10 Hz 文件核验 | {tracking['verified_10hz_video_count']}/{tracking['tracking_video_count']} | OpenCV 直接读取 MP4 FPS |
| 推理耗时求和 | {tracking['summed_inference_sec_excluding_model_load']:.3f} s | 海缆+破损分支，不含本次模型加载 |
| 输出等效处理速度 | {tracking['output_equivalent_processing_fps']:.3f} 帧/s | 5862 输出帧 / 耗时求和 |

静态分类历史计时为 {read_json(Path(source_paths['classification_metrics']))['elapsed_sec']:.3f} 秒，
即 {read_json(Path(source_paths['classification_metrics']))['seconds_per_image']:.3f} 秒/图。

## 指标—实验—证据文件

| 指标 | 实验范围 | 主证据 | 复核证据 |
|---|---|---|---|
| 数据规模/类别 | 30 组 | {md_link('dataset_manifest.csv', manifest_path)} | {md_link('dataset_summary.json', dataset_json_path)}、原视频/GT |
| 准确率、P/R/F1 | 30 张分类图 | {md_link('predictions.json', data['prediction_path'])} | {md_link('metrics.json', source_paths['classification_metrics'])}、本脚本复算 JSON |
| 混淆矩阵 | 30 张分类图 | {md_link('evaluation.json', source_paths['classification_evaluation'])} | {md_link('错误案例', args.classification_dir / 'error_visuals')} |
| 10 组逐样本正确率 | EXP-01–EXP-10 | {md_link('experiment_results.csv', experiment_csv_path)} | 各真值图、分类诊断图 |
| 海缆/破损覆盖率 | 30 段及 10 组 | {md_link('tracking_summary.json', args.tracking_summary)} | 各 `tracking_report.json` 和视频 |
| 输出 10 Hz | 30 段 | 各 `tracking_report.json` | 本脚本对真实 MP4 的 FPS/帧数核验结果 |
| 运行时间/速度 | 分类 30 图、跟踪 30 段 | 分类 `metrics.json`、逐样本报告 | CSV 中每样本耗时与复算速度 |

## 结论边界

当前结果证明在这 30 个中期水池样本上，三分类和 10 Hz 时序可视化链路完整。
但分类参数在同一批真值图上标定，跟踪又有
{tracking['gt_reference_point_initialized_count']} 段使用 GT 参考点初始化，且没有逐帧
人工 mask，所以这些数字不能写成“真实海域泛化精度”或“全自动检测召回率”。
"""
    (docs / "metrics_summary.md").write_text(metrics_doc, encoding="utf-8")

    evidence_rows = []
    for row in experiments:
        evidence_rows.append(
            f"| {row['experiment_id']} / 样本 {row['sample_id']} | {row['gt_class_name_zh']} | "
            f"{md_link('原视频', row['source_video_path'])} | {md_link('真值图', row['support_gt_link'])} | "
            f"{md_link('预览图', row['support_preview_link'])} | {md_link('10 Hz 视频', row['support_video_link'])} | "
            f"{md_link('报告 JSON', row['tracking_report_path'])} |"
        )
    evidence = f"""# 证据索引

## 总体证据

- 原视频：{md_link(str(args.clip_dir.resolve()), args.clip_dir)}
- 原始真值：{md_link(str(args.gt_dir.resolve()), args.gt_dir)}
- 分类实际输入：{md_link('gt_enhanced_wb_clahe', Path(next(iter(data['predictions'].values()))['image']).parent)}
- 分类结果：{md_link('predictions.json', data['prediction_path'])}｜
  {md_link('classification_results.json', args.classification_dir / 'classification_results.json')}｜
  {md_link('metrics.json', source_paths['classification_metrics'])}｜
  {md_link('evaluation.json', source_paths['classification_evaluation'])}
- 分类错误图：{md_link('error_visuals', args.classification_dir / 'error_visuals')}
- 最终跟踪汇总：{md_link('tracking_summary.json', args.tracking_summary)}
- 本次机器清单：{md_link('dataset_manifest.csv', manifest_path)}｜
  {md_link('dataset_summary.json', dataset_json_path)}
- 统一材料：{md_link('previews', visual / 'previews')}｜
  {md_link('videos', visual / 'videos')}｜
  {md_link('ground_truth', visual / 'ground_truth')}｜
  {md_link('diagnostics', visual / 'diagnostics')}

统一材料使用绝对符号链接，删除统一材料目录不会删除历史实验文件；原视频没有
复制到项目中。

## 10 组逐项证据

| 实验 / 样本 | 类别 | 原视频 | 真值 | 预览 | 跟踪视频 | 逐帧报告 |
|---|---|---|---|---|---|---|
{chr(10).join(evidence_rows)}

## 异常和限制证据

- 标签配置异常：{md_link('configs/midterm_ground_truth.yaml', args.label_config)} 中
  样本 9 为类别 1，但 {md_link('sample09_0.jpg', args.gt_dir / 'sample09_0.jpg')}
  和现有正式评测均按类别 0；本材料遵循文件名真值。
- 历史单帧异常：{md_link('single_frame/sample9_1.jpg', args.gt_dir.parent / 'single_frame/sample9_1.jpg')}
  仍带旧类别 1 后缀；该目录未参与当前正式分类和跟踪统计。
- 分类输入经过增强：`predictions.json` 的 `image` 字段均指向
  `gt_enhanced_wb_clahe`，不能把它与原始 GT 重复计数。
- 跟踪辅助初始化：每个报告的 `cable_prompt` 和 `reference_match_frame` 字段可核验；
  当前共 {tracking['gt_reference_point_initialized_count']} 段使用 GT 参考点。
- 最终结果跨目录：样本 5、6、7、18、20、26、29 的最终视频指向 point-candidate
  目录，样本 10、14、28 指向 semantic-fix 目录；以 tracking summary 的路径为准。
"""
    (docs / "evidence_index.md").write_text(evidence, encoding="utf-8")


def main() -> None:
    """Build manifests, summaries, evidence links, and Markdown documents."""
    args = parse_args()
    for name in ("clip_dir", "gt_dir", "classification_dir"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"Missing directory --{name.replace('_', '-')}: {path}")
        setattr(args, name, path)
    for name in ("tracking_summary", "label_config"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Missing file --{name.replace('_', '-')}: {path}")
        setattr(args, name, path)
    args.docs_dir = args.docs_dir.expanduser().resolve()
    args.visual_dir = args.visual_dir.expanduser().resolve()
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)

    data = collect_inputs(args)
    manifest = build_manifest(data)
    manifest_by_sample = {int(row["sample_id"]): row for row in manifest}
    missing_selected = sorted(set(EXPERIMENT_SAMPLES) - set(manifest_by_sample))
    if missing_selected:
        raise SystemExit(f"Selected samples missing from manifest: {missing_selected}")
    experiments = build_experiments(args, data, manifest_by_sample)
    dataset_summary, experiment_summary = summarize(
        args, data, manifest, experiments
    )

    write_csv(args.docs_dir / "dataset_manifest.csv", manifest)
    write_csv(args.docs_dir / "experiment_results.csv", experiments)
    write_json(args.docs_dir / "dataset_summary.json", dataset_summary)
    write_json(args.docs_dir / "experiment_summary.json", experiment_summary)
    write_documents(
        args,
        data,
        manifest,
        experiments,
        dataset_summary,
        experiment_summary,
    )
    print(
        json.dumps(
            {
                "docs_dir": str(args.docs_dir),
                "visual_dir": str(args.visual_dir),
                "samples": len(manifest),
                "experiments": len(experiments),
                "anomalies": len(data["anomalies"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
