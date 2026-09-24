# Submarine cable object detection 
The code of this project is mainly used for data preprocessing of the collected submarine cable data, including functions such as frame extraction from videos, image enhancement, and data set creation. For the code that will be ultimately deployed to the Jetson Orin NX 16GB, please refer to the subsequent instructions.

<details>

<summary>Video Frame Extraction</summary>

## Video Frame Extraction

Extract frames with [`extract_frames.sh`](../run_scripts/extract_frames.sh), which runs [`extract_frames.py`](../scripts/extract_frames.py). No GPU or SAM3 weights are required.

### Quick Start

Run from the project root using your active Python environment. Install dependencies if needed:

```bash
python -m pip install opencv-python-headless tqdm
```

Place videos in `video/`, then run:

```bash
bash run_scripts/extract_frames.sh
```

By default, this saves one frame per second, starting at frame 0, to `data/frames/`. Images retain their original resolution; no color enhancement is applied.

### Options

```bash
bash run_scripts/extract_frames.sh \
  --video-dir /path/to/videos \
  --out-dir data/frames_05s \
  --every-sec 0.5 \
  --max-frames-per-video 10
```

| Option                     | Default          | Description                                                    |
| -------------------------- | ---------------- | -------------------------------------------------------------- |
| `--video-dir`            | `video/`       | Input directory, searched recursively; not a single video file |
| `--out-dir`              | `data/frames/` | Output directory                                               |
| `--every-sec`            | `1`            | Sampling interval in seconds; must be positive                 |
| `--max-frames-per-video` | Unlimited        | Maximum saved frames per video; must be a positive integer     |

Relative paths are resolved from the project root. Supported formats: MP4, MOV, AVI, MKV, and M4V.

### Output

```text
data/frames/
├── demo/
│   ├── demo_f000000_t00000.00.jpg
│   └── ...
└── manifest.json
```

Each video gets a subdirectory. Image names include the video name, frame index, and timestamp. `manifest.json` records FPS, resolution, and saved frame counts.

- Intervals are rounded to whole frames. If FPS is unavailable, sampling falls back to every 30 frames and timestamps are recorded as zero.
- Reruns overwrite matching files but leave older files in place. Use a new output directory when changing settings.
- Videos with the same filename stem, including names that match after replacing spaces with underscores, share an output directory. Process them separately with different output directories.
</details>

<details>

<summary>Underwater Image Enhancement</summary>

## Underwater Image Enhancement

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

| Option             | Default                           | Description                        |
| ------------------ | --------------------------------- | ---------------------------------- |
| `--input_dir`    | `data/extracted_original_image` | Input image directory              |
| `--out_dir`      | `data/processed_images`         | Output directory                   |
| `--gain_min`     | `0.75`                          | Minimum white-balance channel gain |
| `--gain_max`     | `1.35`                          | Maximum white-balance channel gain |
| `--clahe_clip`   | `1.5`                           | CLAHE contrast limit               |
| `--clahe_grid`   | `8`                             | CLAHE tile grid size per axis      |
| `--jpeg_quality` | `96`                            | Enhanced JPEG quality, 0–100      |

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

</details>

<details>

<summary>SAM3 Damage Dataset Annotation</summary>

## SAM3 Damage Dataset Annotation

Generate local damage boxes for a single-class YOLO detection dataset: `nc: 1`, `names: [damage]`. Bright patches are treated as simulated damage. No whole-cable boxes or suspended/touching-bottom classification are exported.

### Modules

| Module | Purpose |
| ------ | ------- |
| [`sam3_damage_dataset.sh`](run_scripts/sam3_damage_dataset.sh) | Batch entry point with prompt thresholds, previews, and append mode |
| [`build_sam3_damage_dataset.py`](scripts/build_sam3_damage_dataset.py) | SAM3 inference, local-box filtering, deduplication, numbering, and YOLO label export |
| [`sam3_segment_candidates.py`](scripts/sam3_segment_candidates.py) | SAM3 model/tokenizer loading helpers, dtype alignment, and mask-to-box conversion |

SAM3 uses `build_sam3_image_model` and `Sam3Processor` for text-prompt segmentation; OpenCV handles image I/O and box previews.

### Quick Start

Use the configured `sam3` environment, a CUDA GPU, and weights at `checkpoints/sam3/sam3.pt`. From the project root:

```bash
conda activate sam3
bash run_scripts/sam3_damage_dataset.sh --input data/processed_images/image_1
```

Append another folder to the same dataset:

```bash
bash run_scripts/sam3_damage_dataset.sh --input data/processed_images/image_2
bash run_scripts/sam3_damage_dataset.sh --input data/processed_images/image_3
```

Pass only image folders, excluding enhancement comparisons. Images retain their resolution and geometry. Without an input override, the current shell script processes `image_3`.

### Options

Additional arguments override the shell script settings. The defaults below are for this shell entry point:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--out-dir` | `outputs/sam3_damage_numbered` | Dataset output directory |
| `--confidence` | `0.45` | Threshold for general damage prompts |
| `--prompt-threshold` | `silver ring=0.05`, `silver patch on black pipe=0.10` | Per-concept thresholds; repeat to add or override |
| `--max-box-area-ratio` | `0.10` | Maximum box area / image area |
| `--max-box-span-ratio` | `0.50` | Reject boxes spanning this fraction of image width or height |
| `--limit` | Unlimited | Process the first N sorted input images, before skipping existing sources |

The script enables `--append` and `--save-previews`. Append mode continues from the highest existing number and skips recorded source paths. To reannotate existing images with different settings, use a new `--out-dir`.

### Output

```text
outputs/sam3_damage_numbered/
├── images/0001.jpg
├── labels/0001.txt
├── previews/0001.jpg
├── classes.yaml
└── annotations.json
```

Each image has one matching label file. Each damage box is one normalized YOLO row: `0 x_center y_center width height`, without scores or prompt names. No detections produce an empty label file. `annotations.json` maps numbered images to their sources and records batch settings.

Review predictions, especially empty labels and low-confidence detections, before training. This step does not split or train the dataset. See [detailed annotation instructions](docs/sam3_state_dataset.md) for threshold tuning and append behavior.

</details>
