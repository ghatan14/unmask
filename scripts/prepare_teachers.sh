#!/usr/bin/env bash
# Download the FS-VFM weights and cache the GenD teacher over the training frames.
# Usage: bash scripts/prepare_teachers.sh [ffpp|hydrafake|all]
set -euo pipefail
cd "$(dirname "$0")/.."
WHAT=${1:-all}
GEND_REPO=${GEND_REPO:-third_party/GenD}

python tools/download_weights.py --out pretrained
[ -d "$GEND_REPO/src" ] || git clone https://github.com/yermandy/GenD "$GEND_REPO"

if [[ $WHAT == ffpp || $WHAT == all ]]; then
  python tools/cache_gend.py --gend_repo "$GEND_REPO" \
      --data data/FaceForensics++/c23/train --val data/FaceForensics++/c23/val --gate 85 \
      --out pretrained/gend_cache_ffpp_train.npz
fi
if [[ $WHAT == hydrafake || $WHAT == all ]]; then
  python tools/check_images.py --root data/HydraFake --split train val --move
  python tools/cache_gend.py --gend_repo "$GEND_REPO" \
      --data data/HydraFake/train --val data/HydraFake/val --gate 75 \
      --out pretrained/gend_cache_hydrafake_train.npz
fi
