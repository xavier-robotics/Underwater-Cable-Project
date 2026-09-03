#!/usr/bin/env python
"""Build SAM3 sample manifests from one representative image per sample."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


IMAGE_PATTERN = re.compile(
    r"^sample(?P<sample>\d+)(?:_[012])?$",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def collect_images(input_dir: Path) -> list[tuple[int, Path]]:
    """Return unique sample numbers without exposing filename class suffixes."""
    samples: dict[int, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = IMAGE_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        sample = int(match.group("sample"))
        if sample in samples:
            raise ValueError(f"Duplicate image for sample {sample}: {path}")
        samples[sample] = path.resolve()
    if not samples:
        raise ValueError(f"No sampleN images found in {input_dir}")
    return sorted(samples.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests: list[dict[str, Any]] = []
    for sample, image_path in collect_images(args.input_dir):
        sample_dir = (args.out_dir / f"sample{sample:02d}").resolve()
        manifest = {
            "video": str(image_path),
            "output_dir": str(sample_dir),
            "frames_per_sample": 1,
            "segments": [
                {
                    "sample_id": f"sample{sample}",
                    "start_frame": 0,
                    "end_frame": 0,
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "frame_count": 1,
                    "selected_frames": [
                        {
                            "frame_idx": 0,
                            "full_frame_path": str(image_path),
                            "crop_path": str(image_path),
                            "crop_confidence": 1.0,
                            "crop_area_ratio": 1.0,
                        }
                    ],
                }
            ],
        }
        write_json(sample_dir / "manifest.json", manifest)
        manifests.append(
            {
                "sample": sample,
                "image": str(image_path),
                "output_dir": str(sample_dir),
            }
        )

    write_json(args.out_dir / "manifest.json", manifests)
    print(f"Prepared {len(manifests)} single-frame samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
