#!/usr/bin/env bash
set -euo pipefail

python -m scripts.build_sam3_damage_dataset \
    --input data/processed_images/image_1/mmexport1779157136529_f001110_t00037.00.jpg \
    --checkpoint checkpoints/sam3/sam3.pt \
    --out-dir outputs/sam3_damage_dataset \
    --confidence 0.45 \
    --max-box-area-ratio 0.35 \
    --max-box-span-ratio 0.85 \
    --device cuda \
    "$@"
