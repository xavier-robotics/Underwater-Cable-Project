#!/usr/bin/env python
"""Classify selected underwater cable samples with an OpenAI vision model."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_MAP = {
    ("exposed", "intact"): (1, "exposed_intact"),
    ("exposed", "damaged"): (0, "damaged"),
    ("suspended", "intact"): (2, "suspended_intact"),
    ("suspended", "damaged"): (0, "damaged"),
}

POSITION_VALUES = {"exposed", "suspended"}
DAMAGE_VALUES = {"intact", "damaged"}
EVIDENCE_CONSISTENCY_VALUES = {"consistent", "mixed", "insufficient"}


@dataclass(frozen=True)
class EvidenceImage:
    path: Path
    kind: str
    frame_idx: int


@dataclass
class SampleInput:
    video: str
    video_dir: Path
    sample_id: str
    segment: dict[str, Any]
    images: list[EvidenceImage]
    frame_ids: list[int]
    source_images: list[EvidenceImage] | None = None

    @property
    def image_paths(self) -> list[Path]:
        return [image.path for image in self.images]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_manifests(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    manifest = path / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"No manifest.json found under {path}")

    data = load_json(manifest)
    if isinstance(data, list):
        manifests = []
        for item in data:
            out_dir = item.get("output_dir")
            if out_dir:
                candidate = Path(out_dir) / "manifest.json"
                if candidate.exists():
                    manifests.append(candidate)
        if manifests:
            return manifests

    return [manifest]


def resolve_image_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate
    return manifest_path.parent / path


def image_for_frame(frame: dict[str, Any], image_field: str, manifest_path: Path) -> Path | None:
    keys = [image_field]
    for key in ("masked_crop_path", "crop_path", "full_frame_path", "path"):
        if key not in keys:
            keys.append(key)

    for key in keys:
        value = frame.get(key)
        if not value:
            continue
        path = resolve_image_path(str(value), manifest_path)
        if path.exists():
            return path
    return None


def select_evenly(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]

    indices = [round(idx * (len(items) - 1) / (limit - 1)) for idx in range(limit)]
    return [items[idx] for idx in indices]


def image_for_key(frame: dict[str, Any], key: str, manifest_path: Path) -> Path | None:
    value = frame.get(key)
    if not value:
        return None
    path = resolve_image_path(str(value), manifest_path)
    return path if path.exists() else None


def evidence_for_frame(
    frame: dict[str, Any],
    image_field: str,
    evidence_mode: str,
    manifest_path: Path,
) -> list[EvidenceImage]:
    frame_idx = int(frame.get("frame_idx", -1))
    if evidence_mode == "preferred":
        path = image_for_frame(frame, image_field, manifest_path)
        return [EvidenceImage(path=path, kind="preferred", frame_idx=frame_idx)] if path else []

    images: list[EvidenceImage] = []
    context = image_for_key(frame, "full_frame_path", manifest_path)
    if context is None:
        context = image_for_frame(frame, image_field, manifest_path)
    if context is not None:
        images.append(EvidenceImage(path=context, kind="context", frame_idx=frame_idx))

    for key in ("masked_crop_path", "crop_path", image_field, "path"):
        detail = image_for_key(frame, key, manifest_path)
        if detail is not None and all(detail != image.path for image in images):
            images.append(EvidenceImage(path=detail, kind="detail", frame_idx=frame_idx))
            break
    return images


def collect_samples(
    manifest_paths: list[Path],
    image_field: str,
    *,
    evidence_mode: str = "preferred",
    max_frames_per_sample: int | None = None,
    include_keys: set[tuple[str, str]] | None = None,
) -> list[SampleInput]:
    samples: list[SampleInput] = []
    for manifest_path in manifest_paths:
        manifest = load_json(manifest_path)
        if isinstance(manifest, list):
            continue

        video = str(manifest.get("video", ""))
        for segment in manifest.get("segments", []):
            sample_id = str(segment.get("sample_id", f"S{len(samples) + 1:03d}"))
            if include_keys is not None and (video, sample_id) not in include_keys:
                continue

            frames = select_evenly(
                list(segment.get("selected_frames", [])),
                max_frames_per_sample,
            )
            images = []
            frame_ids = []
            for frame in frames:
                frame_images = evidence_for_frame(frame, image_field, evidence_mode, manifest_path)
                if frame_images:
                    images.extend(frame_images)
                    frame_ids.append(int(frame.get("frame_idx", -1)))
            if images:
                samples.append(
                    SampleInput(
                        video=video,
                        video_dir=manifest_path.parent,
                        sample_id=sample_id,
                        segment=segment,
                        images=images,
                        frame_ids=frame_ids,
                    )
                )
    return samples


def fit_panel(image: Any, width: int, height: int) -> Any:
    import cv2
    import numpy as np

    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def build_contact_sheet(sample: SampleInput, out_dir: Path, attempt_id: int) -> Path:
    """Combine all context/detail evidence for one sample into one compact image."""
    import cv2
    import numpy as np

    source_images = list(sample.images)
    by_frame: dict[int, dict[str, Path]] = {}
    for image in source_images:
        by_frame.setdefault(image.frame_idx, {})[image.kind] = image.path

    panel_width = 640
    panel_height = 300
    label_height = 34
    row_height = panel_height + label_height
    canvas = np.full((row_height * len(sample.frame_ids), panel_width * 2, 3), 32, dtype=np.uint8)

    for row_index, frame_idx in enumerate(sample.frame_ids):
        row_y = row_index * row_height
        frame_images = by_frame.get(frame_idx, {})
        for column_index, kind in enumerate(("context", "detail")):
            path = frame_images.get(kind)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path else None
            panel = fit_panel(image, panel_width, panel_height)
            x = column_index * panel_width
            canvas[row_y + label_height : row_y + row_height, x : x + panel_width] = panel
            label = f"frame_idx={frame_idx}  {kind}"
            cv2.putText(
                canvas,
                label,
                (x + 12, row_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        cv2.line(
            canvas,
            (0, row_y + row_height - 1),
            (canvas.shape[1], row_y + row_height - 1),
            (96, 96, 96),
            1,
        )

    video_key = hashlib.sha1(sample.video.encode("utf-8")).hexdigest()[:10]
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{video_key}_{sample.sample_id}_attempt_{attempt_id}_contact_sheet.jpg"
    if not cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 86]):
        raise RuntimeError(f"Cannot write contact sheet: {path}")
    sample.source_images = source_images
    sample.images = [EvidenceImage(path=path, kind="contact_sheet", frame_idx=-1)]
    return path


def encode_image(path: Path, max_side: int, jpeg_quality: int) -> str:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")

    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width)) if max_side > 0 else 1.0
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_prompt(sample: SampleInput) -> str:
    segment = sample.segment
    return f"""
You are inspecting underwater pool images of one cable/pipe sample.

Task:
1. Decide position:
   - exposed: the cable/pipe is touching, resting on, or embedded in the pool floor/bottom.
   - suspended: there is a visible gap/shadow between the cable/pipe and the bottom, so it is hanging above the bottom.
   You must choose exposed or suspended. Do not output unknown.
2. Decide damage:
   - damaged: visible sheath rupture, hole, cut, missing outer layer, severe deformation, exposed inner material, or clear local defect.
   - intact: no clear damage is visible.
   You must choose intact or damaged. Do not output unknown.
3. Map the final class with damage priority:
   - damaged at either position -> class_id 0, class_name damaged
   - exposed + intact -> class_id 1, class_name exposed_intact
   - suspended + intact -> class_id 2, class_name suspended_intact
   Position remains diagnostic metadata for damaged samples and never changes the damaged class.

Use all images together. These are representative frames for the same sample.
Evidence is either labelled context/detail images or a contact sheet whose rows show
the full-frame context on the left and the corresponding masked/cropped detail on the right.
Do not classify stains, lighting changes, water haze, printed markers, or background texture as damage unless the cable/pipe surface itself is clearly broken.

Return JSON only with this schema:
{{
  "position": "exposed|suspended",
  "position_confidence": 0.0,
  "damage": "intact|damaged",
  "damage_confidence": 0.0,
  "needs_review": true,
  "evidence_consistency": "consistent|mixed|insufficient",
  "reason_codes": ["short_machine_readable_code"],
  "frame_votes": [
    {{
      "frame_idx": 0,
      "position": "exposed|suspended",
      "position_confidence": 0.0,
      "damage_evidence": "clear_damage|no_visible_damage|uncertain",
      "damage_confidence": 0.0,
      "usable": true
    }}
  ],
  "evidence": "short concrete reason in Chinese",
  "frame_observations": ["short Chinese note per useful image"]
}}

Sample metadata:
- sample_id: {sample.sample_id}
- video: {sample.video}
- segment_time_sec: {segment.get("start_sec")} to {segment.get("end_sec")}
- frame_ids: {sample.frame_ids}
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    needs_review = bool(result.get("needs_review", False))
    position = str(result.get("position", "exposed")).strip().lower()
    damage = str(result.get("damage", "intact")).strip().lower()
    if position not in POSITION_VALUES:
        position = "exposed"
        needs_review = True
    if damage not in DAMAGE_VALUES:
        damage = "intact"
        needs_review = True

    class_id, class_name = CLASS_MAP[(position, damage)]
    evidence_consistency = str(result.get("evidence_consistency", "insufficient")).strip().lower()
    if evidence_consistency not in EVIDENCE_CONSISTENCY_VALUES:
        evidence_consistency = "insufficient"
        needs_review = True

    raw_reason_codes = result.get("reason_codes", [])
    reason_codes = [str(code) for code in raw_reason_codes] if isinstance(raw_reason_codes, list) else []
    frame_votes = result.get("frame_votes", [])
    if not isinstance(frame_votes, list):
        frame_votes = []
        needs_review = True

    try:
        position_confidence = float(result.get("position_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        position_confidence = 0.0
        needs_review = True
    try:
        damage_confidence = float(result.get("damage_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        damage_confidence = 0.0
        needs_review = True

    out = {
        "position": position,
        "position_confidence": position_confidence,
        "damage": damage,
        "damage_confidence": damage_confidence,
        "class_id": class_id,
        "class_name": class_name,
        "needs_review": needs_review,
        "evidence_consistency": evidence_consistency,
        "reason_codes": reason_codes,
        "frame_votes": frame_votes,
        "evidence": str(result.get("evidence", "")),
        "frame_observations": result.get("frame_observations", []),
    }
    if (
        out["position_confidence"] < 0.65
        or out["damage_confidence"] < 0.65
        or evidence_consistency != "consistent"
    ):
        out["needs_review"] = True
    return out


def classify_sample(client: Any, sample: SampleInput, args: argparse.Namespace) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": build_prompt(sample)}]
    for idx, image in enumerate(sample.images, start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"Image {idx}: frame={image.frame_idx}, role={image.kind}, file={image.path.name}",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": encode_image(image.path, args.max_image_side, args.jpeg_quality),
            }
        )

    response = client.responses.create(
        model=args.model,
        input=[{"role": "user", "content": content}],
    )
    raw_text = response.output_text
    return normalize_result(extract_json(raw_text))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "sample_id",
        "start_sec",
        "end_sec",
        "position",
        "position_confidence",
        "damage",
        "damage_confidence",
        "class_id",
        "class_name",
        "needs_review",
        "evidence_consistency",
        "reason_codes",
        "frame_votes",
        "evidence",
        "image_paths",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="A manifest.json file or output directory from select_sample_frames.py.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gpt_classification"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--image-field",
        default="full_frame_path",
        choices=["masked_crop_path", "crop_path", "full_frame_path", "path"],
        help="Preferred image from each selected frame; missing files fall back automatically.",
    )
    parser.add_argument(
        "--evidence-mode",
        default="preferred",
        choices=["preferred", "full-and-crop", "contact-sheet"],
        help="Use preferred images, context/detail pairs, or one combined contact sheet per sample.",
    )
    parser.add_argument(
        "--max-frames-per-sample",
        type=int,
        default=None,
        help="Select at most this many temporally distributed frames from each sample.",
    )
    parser.add_argument(
        "--include-file",
        type=Path,
        default=None,
        help="Optional JSON list restricting collection to video/sample_id pairs.",
    )
    parser.add_argument("--attempt-id", type=int, default=1)
    parser.add_argument("--max-image-side", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=86)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Collect samples and write request manifest without calling OpenAI.")
    args = parser.parse_args()

    manifest_paths = discover_manifests(args.input)
    include_keys = None
    if args.include_file is not None:
        include_data = load_json(args.include_file)
        if isinstance(include_data, dict):
            include_data = include_data.get("items", [])
        include_keys = {
            (str(item.get("video", "")), str(item.get("sample_id", "")))
            for item in include_data
            if isinstance(item, dict)
        }

    samples = collect_samples(
        manifest_paths,
        args.image_field,
        evidence_mode=args.evidence_mode,
        max_frames_per_sample=args.max_frames_per_sample,
        include_keys=include_keys,
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No samples with selected frame images were found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.evidence_mode == "contact-sheet":
        for sample in samples:
            build_contact_sheet(sample, args.out_dir, args.attempt_id)

    request_manifest = [
        {
            "video": sample.video,
            "sample_id": sample.sample_id,
            "start_sec": sample.segment.get("start_sec"),
            "end_sec": sample.segment.get("end_sec"),
            "attempt_id": args.attempt_id,
            "available_frame_count": len(sample.segment.get("selected_frames", [])),
            "selected_frame_count": len(sample.frame_ids),
            "frame_ids": sample.frame_ids,
            "evidence_mode": args.evidence_mode,
            "source_frames": [
                {
                    "frame_idx": frame_idx,
                    "has_context": any(
                        image.frame_idx == frame_idx and image.kind == "context"
                        for image in (sample.source_images or sample.images)
                    ),
                    "has_detail": any(
                        image.frame_idx == frame_idx and image.kind == "detail"
                        for image in (sample.source_images or sample.images)
                    ),
                }
                for frame_idx in sample.frame_ids
            ],
            "images": [
                {
                    "path": str(image.path),
                    "kind": image.kind,
                    "frame_idx": image.frame_idx,
                }
                for image in sample.images
            ],
            "image_paths": [str(path) for path in sample.image_paths],
        }
        for sample in samples
    ]
    (args.out_dir / "requests.json").write_text(json.dumps(request_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Found {len(samples)} samples. Wrote {args.out_dir / 'requests.json'}")
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set. Export it before running this script.")

    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    results = []
    for idx, sample in enumerate(samples, start=1):
        print(f"[{idx}/{len(samples)}] Classify {Path(sample.video).name} {sample.sample_id}")
        result = classify_sample(client, sample, args)
        row = {
            "video": sample.video,
            "sample_id": sample.sample_id,
            "start_sec": sample.segment.get("start_sec"),
            "end_sec": sample.segment.get("end_sec"),
            "attempt_id": args.attempt_id,
            "image_paths": [str(path) for path in sample.image_paths],
            **result,
        }
        results.append(row)

        partial_path = args.out_dir / "results.json"
        partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(args.out_dir / "results.csv", results)

    print(f"Wrote {args.out_dir / 'results.json'}")
    print(f"Wrote {args.out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
