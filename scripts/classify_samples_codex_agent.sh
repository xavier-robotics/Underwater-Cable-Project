#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INPUT="outputs/sam3_codex_pipeline/sample_frames"
OUT_DIR="outputs/codex_sample_classification"
IMAGE_FIELD="full_frame_path"
EVIDENCE_MODE="contact-sheet"
MAX_FRAMES_PER_SAMPLE="3"
INCLUDE_FILE=""
REQUESTS_FILE=""
ATTEMPT_ID="1"
MODEL="${CODEX_MODEL:-gpt-5.5}"
LIMIT=""
PREPARE_ONLY=0
EXECUTE_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  scripts/classify_samples_codex_agent.sh [options]

Options:
  --input PATH          select_sample_frames.py output dir or manifest.json
  --out-dir PATH        output directory for Codex classification results
  --image-field FIELD   preferred image field: full_frame_path, crop_path, masked_crop_path, path
  --evidence-mode MODE  preferred, full-and-crop, or contact-sheet
  --max-frames-per-sample N
                        maximum temporally distributed frames in this attempt
  --include-file PATH   JSON list of video/sample_id pairs to classify
  --requests-file PATH  prepared request subset used for one Agent batch
  --attempt-id N        closed-loop attempt number
  --model MODEL         Codex model name, default: $CODEX_MODEL or gpt-5.5
  --limit N             classify only first N samples
  --prepare-only        only write requests/task files, do not run codex exec
  --execute-only        run an already prepared task without rebuilding requests
  -h, --help            show this help

Example:
  scripts/classify_samples_codex_agent.sh \
    --input outputs/sam3_codex_pipeline/sample_frames \
    --out-dir outputs/codex_sample_classification \
    --evidence-mode contact-sheet \
    --max-frames-per-sample 3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --image-field)
      IMAGE_FIELD="$2"
      shift 2
      ;;
    --evidence-mode)
      EVIDENCE_MODE="$2"
      shift 2
      ;;
    --max-frames-per-sample)
      MAX_FRAMES_PER_SAMPLE="$2"
      shift 2
      ;;
    --include-file)
      INCLUDE_FILE="$2"
      shift 2
      ;;
    --requests-file)
      REQUESTS_FILE="$2"
      shift 2
      ;;
    --attempt-id)
      ATTEMPT_ID="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --execute-only)
      EXECUTE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT_DIR"

if [[ "$PREPARE_ONLY" -eq 1 && "$EXECUTE_ONLY" -eq 1 ]]; then
  echo "--prepare-only and --execute-only cannot be used together" >&2
  exit 2
fi

TASK_FILE="$OUT_DIR/codex_classification_task.md"
if [[ "$EXECUTE_ONLY" -eq 0 ]]; then
if [[ -n "$REQUESTS_FILE" ]]; then
  if [[ ! -f "$REQUESTS_FILE" ]]; then
    echo "Prepared requests file not found: $REQUESTS_FILE" >&2
    exit 2
  fi
  cp "$REQUESTS_FILE" "$OUT_DIR/requests.json"
else
COLLECT_CMD=(
  conda run -n sam3 python scripts/classify_samples_gpt.py
  --input "$INPUT"
  --out-dir "$OUT_DIR"
  --image-field "$IMAGE_FIELD"
  --evidence-mode "$EVIDENCE_MODE"
  --max-frames-per-sample "$MAX_FRAMES_PER_SAMPLE"
  --attempt-id "$ATTEMPT_ID"
  --dry-run
)
if [[ -n "$INCLUDE_FILE" ]]; then
  COLLECT_CMD+=(--include-file "$INCLUDE_FILE")
fi
if [[ -n "$LIMIT" ]]; then
  COLLECT_CMD+=(--limit "$LIMIT")
fi
"${COLLECT_CMD[@]}"
fi

cat > "$TASK_FILE" <<EOF
# Codex Sample Classification Task

You are working in /home/nvidia/uw_detection.

Goal:
Read only $OUT_DIR/requests.json and the image_paths listed there. Classify every
underwater cable/pipe sample and write only $OUT_DIR/results.json.

Evidence layout:
- contact-sheet: one image per sample. Each row is one frame; context is on the left,
  detail is on the right, and frame_idx is printed above the row.
- full-and-crop: context is the full scene and detail is the corresponding cable crop.
- Use context for bottom contact and detail for surface damage.
- Inspect each listed image once. Do not open unlisted source images, scan the repository,
  read project documentation, modify source code, create extra reports, or use subagents.

Class definitions:
- position=exposed: cable/pipe is touching, resting on, or embedded in the pool floor/bottom.
- position=suspended: there is a visible gap, water space, or shadow between cable/pipe and bottom.
- damage=damaged: visible sheath rupture, hole, cut, missing outer layer, severe deformation, exposed inner material, or clear local defect on cable/pipe surface.
- damage=intact: no clear damage is visible on cable/pipe surface.
- You must choose the closest position from exposed/suspended. Do not output unknown.
- You must choose the closest damage state from intact/damaged. Do not output unknown.
- If image quality is poor or the judgment is weak, still choose the closest valid class.

Do not count these as damage unless the cable/pipe surface is clearly broken:
- glare, stains, water haze, shadows, algae-like texture, printed markers, background grid, background calibration board.

Class map (damage has priority over position):
- exposed + damaged -> class_id 0, class_name damaged
- exposed + intact -> class_id 1, class_name exposed_intact
- suspended + intact -> class_id 2, class_name suspended_intact
- suspended + damaged -> class_id 0, class_name damaged
- For every damaged sample, class_id and class_name must be 0 and damaged regardless
  of whether position is exposed or suspended. Keep position only as diagnostic metadata.

results.json schema:
[
  {
    "video": "string",
    "sample_id": "string",
    "attempt_id": $ATTEMPT_ID,
    "start_sec": 0.0,
    "end_sec": 0.0,
    "image_paths": ["string"],
    "position": "exposed|suspended",
    "position_confidence": 0.0,
    "damage": "intact|damaged",
    "damage_confidence": 0.0,
    "class_id": 0,
    "class_name": "damaged",
    "needs_review": false,
    "evidence_consistency": "consistent|mixed|insufficient",
    "reason_codes": ["short_machine_readable_code"],
    "frame_votes": [
      {
        "frame_idx": 0,
        "position": "exposed|suspended",
        "position_confidence": 0.0,
        "damage_evidence": "clear_damage|no_visible_damage|uncertain",
        "damage_confidence": 0.0,
        "usable": true
      }
    ]
  }
]

Rules:
- Output JSON only in results.json. Do not include analysis text or frame observations.
- Copy video, sample_id, attempt_id, start_sec, end_sec, and image_paths exactly from the request.
- Be conservative. If defect is not visually clear, use damage=intact instead of overcalling damaged.
- Use confidence 0.50-0.64 for weak/uncertain judgments. The result is still final.
- Use confidence >=0.80 only when visual evidence is clear.
- If a listed image file cannot be read, keep processing other images and still choose from the valid classes when possible.
- Use evidence_consistency=consistent only when usable frames support the same conclusion.
- Position uses the majority of usable frame votes. A minority position vote caused by viewpoint
  does not by itself make the sample inconsistent.
- For each frame, use damage_evidence=clear_damage only when that frame visibly shows a rupture,
  hole, cut, missing sheath, exposed inner material, or another concrete surface defect.
- Use damage_evidence=no_visible_damage when no defect is visible in that frame. This does not
  prove the whole sample is intact because another viewpoint may reveal damage.
- Aggregate damage=damaged when at least one usable frame has clear, high-confidence damage.
  Other frames with no visible damage are not contradictory.
- Aggregate damage=intact only when no usable frame shows clear damage.
- Damage takes precedence in the final class: whenever aggregate damage=damaged, output
  class_id=0 and class_name=damaged. Do not emit exposed_damaged or suspended_damaged.
- Use evidence_consistency=insufficient only when the available images cannot support a sample-level
  decision. Use mixed only for a genuine unresolved position conflict, not ordinary damage visibility changes.
- Use an empty reason_codes list for a clean result. Otherwise use compact codes such as low_visibility,
  conflicting_frames, unreadable_image, weak_contact_evidence, or weak_damage_evidence.
- Add exactly one frame_votes item for every integer listed in request.frame_ids.
  The contact-sheet image itself uses frame_idx=-1; do not create a vote for -1.
  Set usable=false when a row cannot support a judgment; still fill all fields with the closest valid values.
- needs_review is diagnostic metadata only. It never triggers another classification pass.
- Never output position=unknown, damage=unknown, class_id=null, or class_name=null.
EOF
fi

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
  echo "Prepared:"
  echo "  $OUT_DIR/requests.json"
  echo "  $TASK_FILE"
  exit 0
fi

if [[ ! -f "$OUT_DIR/requests.json" || ! -f "$TASK_FILE" ]]; then
  echo "Prepared requests/task not found under $OUT_DIR" >&2
  exit 2
fi

codex exec \
  --skip-git-repo-check \
  --sandbox workspace-write \
  --model "$MODEL" \
  "Execute $TASK_FILE exactly. Return only a one-line completion status."
