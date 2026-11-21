#!/usr/bin/env bash
# Table 4: HydraFake accuracy (protocol of Tan et al., ICLR 2026).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python tools/eval_hydrafake.py --check --root data/HydraFake
python tools/eval_hydrafake.py --root data/HydraFake \
    --ckpt checkpoints/unmask_base_hydrafake.pth checkpoints/unmask_mid_hydrafake.pth \
    --names UNMASK-Base UNMASK-Mid --with_paper_rows --bootstrap 1000 --ref UNMASK-Base \
    --save_scores results/hydrafake_scores.npz --out results/hydrafake "$@"
