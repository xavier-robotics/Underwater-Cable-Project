#!/usr/bin/env bash
# Activate the sam3 environment before running. Extra CLI flags override defaults.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

exec python -m scripts.build_sam3_damage_dataset \
    --input data/processed_images/image_3cc \
    --checkpoint checkpoints/sam3/sam3.pt \
    --out-dir outputs/sam3_damage_numbered \
    --confidence 0.45 \
    --prompt-threshold "silver ring=0.05" \
    --prompt-threshold "silver patch on black pipe=0.10" \
    --max-box-area-ratio 0.10 \
    --max-box-span-ratio 0.50 \
    --device cuda \
    --save-previews \
    --append \
    "$@"
