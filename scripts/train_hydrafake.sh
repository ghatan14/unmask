#!/usr/bin/env bash
# Train UNMASK on the HydraFake training split and export compact checkpoints.
# Usage: bash scripts/train_hydrafake.sh [base mid mid_no_kd] [--seed 1 ...]
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANTS=()
while [[ $# -gt 0 && $1 != --* ]]; do VARIANTS+=("$1"); shift; done
[[ ${#VARIANTS[@]} -eq 0 ]] && VARIANTS=(base mid)

for V in "${VARIANTS[@]}"; do
  RUN=runs/unmask_${V}_hydrafake
  python tools/train.py --config configs/hydrafake/unmask_${V}.yaml --output_dir "$RUN" "$@"
  python tools/export_checkpoint.py "$RUN/checkpoint-last.pth" "checkpoints/unmask_${V}_hydrafake.pth" \
      --trained_on "HydraFake train"
done
