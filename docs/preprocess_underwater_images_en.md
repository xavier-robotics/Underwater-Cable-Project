# Underwater Image Enhancement

[`preprocess_underwater_images.py`](../scripts/preprocess_underwater_images.py) reduces green color cast using bounded gray-world white balance, then improves local contrast with CLAHE. No GPU or model weights are required.

## Run

From the project root, install dependencies if needed:

```bash
python -m pip install numpy opencv-python-headless
```

Enhance extracted video frames:

```bash
python scripts/preprocess_underwater_images.py \
  --input_dir data/frames \
  --out_dir data/processed_images
```

The script searches subfolders recursively and preserves their structure, image filenames, and resolution. Supported formats: JPG, JPEG, PNG, BMP, and WebP.

## Options

Parameter names use **underscores**, not hyphens.

| Option | Default | Description |
| --- | --- | --- |
| `--input_dir` | `data/extracted_original_image` | Input image directory |
| `--out_dir` | `data/processed_images` | Output directory |
| `--gain_min` | `0.75` | Minimum white-balance channel gain |
| `--gain_max` | `1.35` | Maximum white-balance channel gain |
| `--clahe_clip` | `1.5` | CLAHE contrast limit |
| `--clahe_grid` | `8` | CLAHE tile grid size per axis |
| `--jpeg_quality` | `96` | Enhanced JPEG quality, 0–100 |

Start with the defaults and inspect the comparison images before adjusting settings.

## Output

```text
data/processed_images/
├── image_1/                  # Enhanced images, if input contains image_1/
├── comparisons/             # Before/after previews
├── comparison_overview.jpg  # Combined preview
└── preprocessing_report.json
```

The report records processing settings and per-image color statistics. Keep the output separate from, and outside, the input directory to avoid processing generated images on later runs. Reruns overwrite matching output files.
