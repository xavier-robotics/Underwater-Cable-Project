#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INPUT="/home/nvidia/DATA/UW/video"
WORK_DIR="outputs/sam3_codex_pipeline"
SAMPLES_DIR=""
CLASSIFY_DIR=""
ZERO_SHOT_DIR=""
SAM3_ONLY_DIR=""
CLASSIFIER="codex"
MODE="scan"
SAMPLE_EVERY_SEC="1"
FRAMES_PER_SAMPLE="3"
CABLE_THRESHOLD="0.04"
MERGE_GAP_SEC="2.0"
MIN_SEGMENT_SEC="2.0"
MAX_FRAMES=""
EXPECTED_SAMPLES=""
SAM3_PROMPT="pipe"
SAM3_CHECKPOINT="$REPO_ROOT/ckpts/sam3/sam3.pt"
SAM3_CONF="0.05"
SAM3_DTYPE="float32"
IMAGE_FIELD="full_frame_path"
CLOSED_LOOP_CONFIG="$REPO_ROOT/configs/closed_loop.yaml"
ZERO_SHOT_CONFIG="$REPO_ROOT/configs/zero_shot_classifier.yaml"
SAM3_ONLY_CONFIG="$REPO_ROOT/configs/sam3_only_classifier.yaml"
MODEL="${CODEX_MODEL:-gpt-5.5}"
LIMIT=""
PREPARE_ONLY=0
SKIP_SPLIT=0
FORCE_SPLIT=0
FORCE_ZERO_SHOT=0
FORCE_SAM3_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_sam3_codex_pipeline.sh [options]

Runs the full offline pipeline:
  1. SAM3 video -> sample segments and representative images
  2. Codex, SAM3+YOLOE, or SAM3-only classification

Options:
  --input PATH              input video file or directory
  --work-dir PATH           root output directory
  --samples-dir PATH        override sample-frame output directory
  --classifier METHOD       codex, zero-shot, sam3-only, or both; default: codex
  --classify-dir PATH       override Codex classification output directory
  --zero-shot-dir PATH      override SAM3+YOLOE output directory
  --sam3-only-dir PATH      override SAM3-only output directory
  --mode single|scan        sample splitting mode
  --sample-every-sec SEC    SAM3 frame sampling interval
  --frames-per-sample N     representative images per sample, default: 3
  --cable-threshold X       visible-frame threshold
  --merge-gap-sec SEC       merge nearby visible fragments
  --min-segment-sec SEC     drop shorter scan segments
  --max-frames N            debug cap on raw frames per video
  --expected-samples N      optional known segment count; unset by default
  --sam3-prompt TEXT        SAM3 text prompt
  --sam3-checkpoint PATH    SAM3 checkpoint path
  --sam3-conf X             SAM3 confidence threshold
  --sam3-dtype TYPE         auto, float32, or bfloat16
  --image-field FIELD       full_frame_path, crop_path, masked_crop_path, path
  --closed-loop-config PATH single-pass best-effort mode by default
  --zero-shot-config PATH   YOLOE prompts and fixed-rule configuration
  --sam3-only-config PATH   SAM3-only prompts and fixed-rule configuration
  --model MODEL             Codex model name
  --limit N                 classify only first N samples
  --skip-split              require and use existing --samples-dir
  --force-split             rerun SAM3 even when a cached sample manifest exists
  --force-zero-shot         ignore cached zero-shot results
  --force-sam3-only         ignore cached SAM3-only results
  --prepare-only            prepare selected classifier task without model inference
  -h, --help                show this help

Examples:
  scripts/run_sam3_codex_pipeline.sh \
    --input /home/nvidia/DATA/UW/video \
    --work-dir outputs/final_pool_run \
    --mode scan

  scripts/run_sam3_codex_pipeline.sh \
    --samples-dir outputs/final_pool_run/sample_frames \
    --work-dir outputs/final_pool_run \
    --classifier zero-shot \
    --skip-split

  scripts/run_sam3_only_pipeline.sh \
    --samples-dir outputs/final_pool_run/sample_frames \
    --work-dir outputs/sam3_only_run \
    --skip-split

  scripts/run_sam3_codex_pipeline.sh \
    --input /home/nvidia/DATA/UW/video/mmexport1779157141457.mp4 \
    --work-dir outputs/debug_one_video \
    --mode single \
    --max-frames 300 \
    --prepare-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --samples-dir) SAMPLES_DIR="$2"; shift 2 ;;
    --classifier) CLASSIFIER="$2"; shift 2 ;;
    --classify-dir) CLASSIFY_DIR="$2"; shift 2 ;;
    --zero-shot-dir) ZERO_SHOT_DIR="$2"; shift 2 ;;
    --sam3-only-dir) SAM3_ONLY_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --sample-every-sec) SAMPLE_EVERY_SEC="$2"; shift 2 ;;
    --frames-per-sample) FRAMES_PER_SAMPLE="$2"; shift 2 ;;
    --cable-threshold) CABLE_THRESHOLD="$2"; shift 2 ;;
    --merge-gap-sec) MERGE_GAP_SEC="$2"; shift 2 ;;
    --min-segment-sec) MIN_SEGMENT_SEC="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --expected-samples) EXPECTED_SAMPLES="$2"; shift 2 ;;
    --sam3-prompt) SAM3_PROMPT="$2"; shift 2 ;;
    --sam3-checkpoint) SAM3_CHECKPOINT="$2"; shift 2 ;;
    --sam3-conf) SAM3_CONF="$2"; shift 2 ;;
    --sam3-dtype) SAM3_DTYPE="$2"; shift 2 ;;
    --image-field) IMAGE_FIELD="$2"; shift 2 ;;
    --closed-loop-config) CLOSED_LOOP_CONFIG="$2"; shift 2 ;;
    --zero-shot-config) ZERO_SHOT_CONFIG="$2"; shift 2 ;;
    --sam3-only-config) SAM3_ONLY_CONFIG="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --skip-split) SKIP_SPLIT=1; shift ;;
    --force-split) FORCE_SPLIT=1; shift ;;
    --force-zero-shot) FORCE_ZERO_SHOT=1; shift ;;
    --force-sam3-only) FORCE_SAM3_ONLY=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$CLASSIFIER" != "codex" \
  && "$CLASSIFIER" != "zero-shot" \
  && "$CLASSIFIER" != "sam3-only" \
  && "$CLASSIFIER" != "both" ]]; then
  echo "--classifier must be codex, zero-shot, sam3-only, or both" >&2
  exit 2
fi

if [[ "$SKIP_SPLIT" -eq 1 && "$FORCE_SPLIT" -eq 1 ]]; then
  echo "--skip-split and --force-split cannot be used together" >&2
  exit 2
fi

if [[ -z "$SAMPLES_DIR" ]]; then
  SAMPLES_DIR="$WORK_DIR/sample_frames"
fi
if [[ -z "$CLASSIFY_DIR" ]]; then
  CLASSIFY_DIR="$WORK_DIR/classification"
fi
if [[ -z "$ZERO_SHOT_DIR" ]]; then
  ZERO_SHOT_DIR="$WORK_DIR/classification_zero_shot"
fi
if [[ -z "$SAM3_ONLY_DIR" ]]; then
  SAM3_ONLY_DIR="$WORK_DIR/classification_sam3_only"
fi

mkdir -p "$WORK_DIR"

if [[ "$SKIP_SPLIT" -eq 0 && "$FORCE_SPLIT" -eq 0 && -f "$SAMPLES_DIR/manifest.json" ]]; then
  SKIP_SPLIT=1
  echo "[cache] Reuse existing SAM3 sample manifest; use --force-split to rebuild."
fi

if [[ "$SKIP_SPLIT" -eq 0 ]]; then
  SPLIT_CMD=(
    conda run -n sam3 python scripts/select_sample_frames.py
    --input "$INPUT"
    --out-dir "$SAMPLES_DIR"
    --detector sam3
    --mode "$MODE"
    --crop-source sam3-mask
    --sam3-prompt "$SAM3_PROMPT"
    --sam3-checkpoint "$SAM3_CHECKPOINT"
    --sam3-conf "$SAM3_CONF"
    --sam3-dtype "$SAM3_DTYPE"
    --sample-every-sec "$SAMPLE_EVERY_SEC"
    --frames-per-sample "$FRAMES_PER_SAMPLE"
    --cable-threshold "$CABLE_THRESHOLD"
    --merge-gap-sec "$MERGE_GAP_SEC"
    --min-segment-sec "$MIN_SEGMENT_SEC"
  )
  if [[ -n "$MAX_FRAMES" ]]; then
    SPLIT_CMD+=(--max-frames "$MAX_FRAMES")
  fi
  if [[ -n "$EXPECTED_SAMPLES" ]]; then
    SPLIT_CMD+=(--expected-samples "$EXPECTED_SAMPLES")
  fi

  echo "[1/2] SAM3 sample split -> $SAMPLES_DIR"
  "${SPLIT_CMD[@]}"
else
  if [[ ! -f "$SAMPLES_DIR/manifest.json" ]]; then
    echo "No cached sample manifest found: $SAMPLES_DIR/manifest.json" >&2
    exit 2
  fi
  echo "[1/2] Skip SAM3 split, use existing samples -> $SAMPLES_DIR"
fi

if [[ "$CLASSIFIER" == "codex" || "$CLASSIFIER" == "both" ]]; then
  CLASSIFY_CMD=(
    conda run -n sam3 python scripts/run_closed_loop.py
    --input "$SAMPLES_DIR"
    --work-dir "$CLASSIFY_DIR"
    --config "$CLOSED_LOOP_CONFIG"
    --image-field "$IMAGE_FIELD"
    --model "$MODEL"
  )
  if [[ -n "$LIMIT" ]]; then
    CLASSIFY_CMD+=(--limit "$LIMIT")
  fi
  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    CLASSIFY_CMD+=(--prepare-only)
  fi

  echo "[2/2] Codex baseline classification -> $CLASSIFY_DIR"
  "${CLASSIFY_CMD[@]}"
fi

if [[ "$CLASSIFIER" == "zero-shot" || "$CLASSIFIER" == "both" ]]; then
  ZERO_SHOT_CMD=(
    conda run -n sam3 python scripts/classify_samples_zero_shot.py
    --input "$SAMPLES_DIR"
    --out-dir "$ZERO_SHOT_DIR"
    --config "$ZERO_SHOT_CONFIG"
  )
  if [[ -n "$LIMIT" ]]; then
    ZERO_SHOT_CMD+=(--limit "$LIMIT")
  fi
  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    ZERO_SHOT_CMD+=(--prepare-only)
  fi
  if [[ "$FORCE_ZERO_SHOT" -eq 1 ]]; then
    ZERO_SHOT_CMD+=(--force)
  fi

  echo "[2/2] SAM3 + YOLOE zero-shot classification -> $ZERO_SHOT_DIR"
  "${ZERO_SHOT_CMD[@]}"
fi

if [[ "$CLASSIFIER" == "sam3-only" ]]; then
  SAM3_ONLY_CMD=(
    conda run -n sam3 python scripts/classify_samples_sam3_only.py
    --input "$SAMPLES_DIR"
    --out-dir "$SAM3_ONLY_DIR"
    --config "$SAM3_ONLY_CONFIG"
  )
  if [[ -n "$LIMIT" ]]; then
    SAM3_ONLY_CMD+=(--limit "$LIMIT")
  fi
  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    SAM3_ONLY_CMD+=(--prepare-only)
  fi
  if [[ "$FORCE_SAM3_ONLY" -eq 1 ]]; then
    SAM3_ONLY_CMD+=(--force)
  fi

  echo "[2/2] SAM3-only zero-shot classification -> $SAM3_ONLY_DIR"
  "${SAM3_ONLY_CMD[@]}"
fi

if [[ "$CLASSIFIER" == "both" ]] \
  && [[ "$PREPARE_ONLY" -eq 0 ]] \
  && [[ -f "$CLASSIFY_DIR/results.json" ]] \
  && [[ -f "$ZERO_SHOT_DIR/results.json" ]]; then
  conda run -n sam3 python scripts/compare_classification_results.py \
    --baseline "$CLASSIFY_DIR/results.json" \
    --candidate "$ZERO_SHOT_DIR/results.json" \
    --out-dir "$WORK_DIR/comparison"
fi

echo "Done."
echo "Samples:        $SAMPLES_DIR"
if [[ "$CLASSIFIER" == "codex" || "$CLASSIFIER" == "both" ]]; then
  echo "Codex:          $CLASSIFY_DIR"
fi
if [[ "$CLASSIFIER" == "zero-shot" || "$CLASSIFIER" == "both" ]]; then
  echo "SAM3 + YOLOE:   $ZERO_SHOT_DIR"
fi
if [[ "$CLASSIFIER" == "sam3-only" ]]; then
  echo "SAM3 only:      $SAM3_ONLY_DIR"
fi
