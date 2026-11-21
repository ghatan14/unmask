#!/usr/bin/env bash
# Train the three UNMASK variants on FF++ c23 and export compact checkpoints.
# Usage: bash scripts/train_ffpp.sh [base mid lite] [--seed 1 ...]
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANTS=()
while [[ $# -gt 0 && $1 != --* ]]; do VARIANTS+=("$1"); shift; done
[[ ${#VARIANTS[@]} -eq 0 ]] && VARIANTS=(base mid lite)

for V in "${VARIANTS[@]}"; do
  RUN=runs/unmask_${V}_ffpp
  python tools/train.py --config configs/ffpp/unmask_${V}.yaml --output_dir "$RUN" "$@"
  python tools/export_checkpoint.py "$RUN/checkpoint-last.pth" "checkpoints/unmask_${V}_ffpp.pth" \
      --trained_on "FF++ c23 (w/o NeuralTextures)"
done
