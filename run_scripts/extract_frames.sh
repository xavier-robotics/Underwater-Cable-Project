#!/usr/bin/env bash
# Extract video frames using the active Python environment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Additional arguments override these defaults. Relative paths use REPO_ROOT.
exec python "$REPO_ROOT/scripts/extract_frames.py" \
  --video-dir "$REPO_ROOT/video" \
  --out-dir "$REPO_ROOT/data/frames" \
  --every-sec 1 \
  "$@"
