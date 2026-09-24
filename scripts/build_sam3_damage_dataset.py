#!/usr/bin/env python3
"""Create a single-class YOLO dataset from SAM3 local damage predictions."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PROMPTS = ["damage", "broken sheath", "cable damage", "silver patch"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WHOLE_OBJECT_PROMPTS = {"cable", "pipe", "pipeline", "black pipe", "black cable", "underwater cable", "submarine cable", "background", "undamaged"}


def prompt_thresholds(prompts, confidence, overrides):
    """Explicit concepts can have lower thresholds without lowering all concepts."""
    thresholds = dict.fromkeys(prompts or DEFAULT_PROMPTS, confidence)
    for value in overrides or []:
        try:
            prompt, raw = value.rsplit("=", 1)
            prompt = prompt.strip()
            threshold = float(raw)
        except (ValueError, TypeError) as error:
            raise ValueError("--prompt-threshold must be 'local concept=0.10'") from error
        if not prompt or prompt.lower() in WHOLE_OBJECT_PROMPTS or not 0 < threshold <= 1:
            raise ValueError(f"Invalid local prompt threshold: {value}")
        thresholds[prompt] = threshold
    return thresholds


def validate_box(box, width, height):
    if (len(box) != 4 or not all(np.isfinite(v) for v in box)
            or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
        raise ValueError(f"Invalid box for {width}x{height}: {box}")


def yolo_line(box, width, height):
    validate_box(box, width, height)
    x1, y1, x2, y2 = box
    return f"0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"


def filter_damage(candidates, width, height, confidence, max_area, max_span, iou_threshold):
    valid = []
    for candidate in candidates:
        box = candidate.box_xyxy
        validate_box(box, width, height)
        box_width, box_height = box[2]-box[0], box[3]-box[1]
        if (candidate.score >= confidence
                and box_width*box_height/(width*height) <= max_area
                and max(box_width/width, box_height/height) < max_span):
            valid.append(candidate)
    # Merge duplicate concepts/bright fragments by retaining the outer local box.
    def area(candidate):
        x1, y1, x2, y2 = candidate.box_xyxy
        return (x2-x1)*(y2-y1)
    kept = []
    for candidate in sorted(valid, key=lambda c: (area(c), c.score), reverse=True):
        x1, y1, x2, y2 = candidate.box_xyxy
        duplicate = False
        for old in kept:
            a1, b1, a2, b2 = old.box_xyxy
            intersection = max(0, min(x2, a2)-max(x1, a1))*max(0, min(y2, b2)-max(y1, b1))
            union = area(candidate)+area(old)-intersection
            if intersection/area(candidate) >= 0.85 or intersection/union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda c: c.score, reverse=True)


def collect_images(source, out_dir, limit=None):
    source = source.resolve()
    output = out_dir.resolve()
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("Output must be outside the input image directory")
    paths = sorted(source.rglob("*")) if source.is_dir() else [source]
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No input images: {source}")
    # Stable path order, one global sequence even across nested input folders.
    # Four digits are a minimum width: image 10000 remains uniquely named.
    return [(path, Path(f"{index:04d}.jpg")) for index, path in enumerate(paths, start=1)]


def save_metadata(out_dir, metadata):
    """Replace the manifest atomically so interruption cannot truncate old records."""
    with tempfile.NamedTemporaryFile(mode="w", dir=out_dir, suffix=".tmp", delete=False,
                                     encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        temporary.replace(out_dir / "annotations.json")
    finally:
        temporary.unlink(missing_ok=True)


def plan_append(out_dir, jobs, append=False):
    """Validate an existing numbered dataset before reserving any new names."""
    if not out_dir.exists():
        return jobs, None
    if not append:
        raise ValueError("Output already exists; use --append to continue numbering")
    manifest = out_dir / "annotations.json"
    if not manifest.is_file():
        raise ValueError("Cannot append: annotations.json is missing; use a new output directory")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("nc") != 1 or metadata.get("names") != ["damage"]:
        raise ValueError("Cannot append: dataset is not single-class damage")
    sources, images, labels = set(), set(), set()
    highest = 0
    for item in metadata["items"]:
        image, label = Path(item["image"]), Path(item["label"])
        if (image.parent != Path("images") or image.suffix != ".jpg"
                or not image.stem.isascii() or not image.stem.isdigit()
                or int(image.stem) < 1 or image.stem != f"{int(image.stem):04d}"
                or label != Path("labels") / image.with_suffix(".txt").name):
            raise ValueError("Cannot append: expected numbered images/0001.jpg and labels/0001.txt")
        source = str(Path(item["source"]).resolve())
        if image in images or label in labels or source in sources:
            raise ValueError("Cannot append: duplicate entries in annotations.json")
        sources.add(source)
        images.add(image)
        labels.add(label)
        highest = max(highest, int(image.stem))
    actual_images = {p.relative_to(out_dir) for p in (out_dir / "images").rglob("*") if p.is_file()}
    actual_labels = {p.relative_to(out_dir) for p in (out_dir / "labels").rglob("*") if p.is_file()}
    if actual_images != images or actual_labels != labels:
        raise ValueError("Cannot append: files and annotations.json disagree (missing or unrecorded files); restore consistency or use a new directory")
    for preview in (out_dir / "previews").rglob("*"):
        if preview.is_file() and Path("images") / preview.relative_to(out_dir / "previews") not in images:
            raise ValueError("Cannot append: unrecorded preview file")
    pending = [source for source, _ in jobs if str(source.resolve()) not in sources]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"Skipped {skipped} already recorded source images", flush=True)
    return [(source, Path(f"{highest + index:04d}.jpg"))
            for index, source in enumerate(pending, start=1)], metadata


def write_sample(source, relative, frame, candidates, out_dir, save_previews=False):
    height, width = frame.shape[:2]
    lines = [yolo_line(c.box_xyxy, width, height) for c in candidates]
    image = out_dir / "images" / relative
    label = out_dir / "labels" / relative.with_suffix(".txt")
    preview = out_dir / "previews" / relative
    if any(path.exists() for path in (image, label, preview)):
        raise FileExistsError(f"Refusing to overwrite existing sample: {relative}")
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, image)
    elif not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Cannot write {image}")
    label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if save_previews:
        canvas = frame.copy()
        for candidate in candidates:
            x1, y1, x2, y2 = map(int, candidate.box_xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 3)
            cv2.putText(canvas, "damage", (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 255), 2, cv2.LINE_AA)
        preview = out_dir / "previews" / relative
        preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview), canvas):
            raise OSError(f"Cannot write {preview}")


def generate(args):
    # A persistent sibling lock avoids two append processes allocating the same IDs.
    args.out_dir = args.out_dir.resolve()
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.out_dir.parent / f".{args.out_dir.name}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another annotation process is writing {args.out_dir}") from error
        generate_locked(args)


def generate_locked(args):
    jobs = collect_images(args.input, args.out_dir, args.limit)
    jobs, metadata = plan_append(args.out_dir, jobs, args.append)
    if not jobs:
        print("No new images to process; existing dataset unchanged.")
        return
    from PIL import Image
    import torch
    from scripts.sam3_segment_candidates import (
        Sam3Processor, build_sam3_image_model, collect_candidates,
        register_dtype_alignment_hooks, resolve_bpe_path,
    )

    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint missing: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    thresholds = prompt_thresholds(args.prompts, args.confidence, getattr(args, "prompt_threshold", None))
    prompts = list(thresholds)
    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint),
                                 bpe_path=str(resolve_bpe_path(args.bpe_path)),
                                 load_from_HF=False, device=args.device)
    register_dtype_alignment_hooks(model)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.confidence)
    args.out_dir.mkdir(parents=True, exist_ok=args.append)
    # Class metadata only. Training/validation splitting is outside this annotation step.
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if metadata is None:
        (args.out_dir / "classes.yaml").write_text("nc: 1\nnames: [damage]\n", encoding="utf-8")
        metadata = {"nc": 1, "names": ["damage"], "prompts": prompts,
                    "settings": settings, "items": [], "runs": []}
    elif "runs" not in metadata:
        metadata["runs"] = [{"id": 0, "prompts": metadata.get("prompts"),
                              "settings": metadata.get("settings")}]
        for item in metadata["items"]:
            item["run_id"] = 0
    run_id = len(metadata["runs"])
    metadata["runs"].append({"id": run_id, "prompts": prompts, "settings": settings,
                             "prompt_thresholds": thresholds})
    save_metadata(args.out_dir, metadata)
    for source, relative in jobs:
        with Image.open(source) as original:
            if original.getexif().get(274, 1) not in (None, 1):
                raise ValueError(f"{source}: EXIF rotation is not supported; supply an already oriented image")
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if frame is None:
            raise ValueError(f"Cannot read {source}")
        height, width = frame.shape[:2]
        state = processor.set_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        candidates = []
        for prompt in prompts:
            processor.confidence_threshold = thresholds[prompt]
            output = processor.set_text_prompt(prompt, state)
            candidates.extend(c for c in collect_candidates(output, prompt, height, width, args.min_area_ratio, 1.0)
                              if c.score >= thresholds[prompt])
        damage = filter_damage(candidates, width, height, min(thresholds.values()),
                               args.max_box_area_ratio, args.max_box_span_ratio, args.iou_threshold)
        write_sample(source, relative, frame, damage, args.out_dir, args.save_previews)
        metadata["items"].append({"source": str(source), "image": str(Path("images")/relative),
                                  "label": str(Path("labels")/relative.with_suffix(".txt")),
                                  "width": width, "height": height, "damage_count": len(damage),
                                  "run_id": run_id})
        save_metadata(args.out_dir, metadata)
        print(f"{relative}: {len(damage)} damage boxes", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image or recursive image directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--append", action="store_true", help="Continue existing numbering; skip previously recorded source paths")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam3/sam3.pt"))
    parser.add_argument("--bpe-path", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Local damage concept; repeat to combine")
    parser.add_argument("--confidence", "--damage-confidence", type=float, default=0.45)
    parser.add_argument("--prompt-threshold", action="append", metavar="PROMPT=SCORE",
                        help="Add/override a local damage concept threshold; repeat as needed")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.35)
    parser.add_argument("--max-box-span-ratio", type=float, default=0.85,
                        help="Reject boxes spanning this fraction of image width or height")
    parser.add_argument("--min-area-ratio", type=float, default=0.0002)
    parser.add_argument("--iou-threshold", type=float, default=0.65)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-previews", action="store_true", help="Optional separate previews with damage boxes only")
    args = parser.parse_args(argv)
    for key in ("confidence", "max_box_area_ratio", "max_box_span_ratio", "iou_threshold"):
        if not 0 < getattr(args, key) <= 1:
            parser.error(f"{key} must be in (0, 1]")
    if not 0 <= args.min_area_ratio < args.max_box_area_ratio:
        parser.error("--min-area-ratio must be nonnegative and smaller than --max-box-area-ratio")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.prompts:
        args.prompts = [p.strip() for p in args.prompts]
        if any(not p or p.lower() in WHOLE_OBJECT_PROMPTS for p in args.prompts):
            parser.error("Use local damage prompts, not whole cable/background/undamaged prompts")
    try:
        prompt_thresholds(args.prompts, args.confidence, args.prompt_threshold)
    except ValueError as error:
        parser.error(str(error))
    generate(args)


if __name__ == "__main__":
    main()
