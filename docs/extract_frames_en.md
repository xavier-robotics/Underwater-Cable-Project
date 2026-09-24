# Video Frame Extraction

Extract frames with [`extract_frames.sh`](../run_scripts/extract_frames.sh), which runs [`extract_frames.py`](../scripts/extract_frames.py). No GPU or SAM3 weights are required.

## Quick Start

Run from the project root using your active Python environment. Install dependencies if needed:

```bash
python -m pip install opencv-python-headless tqdm
```

Place videos in `video/`, then run:

```bash
bash run_scripts/extract_frames.sh
```

By default, this saves one frame per second, starting at frame 0, to `data/frames/`. Images retain their original resolution; no color enhancement is applied.

## Options

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

## Output

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
