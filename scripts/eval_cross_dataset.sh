#!/usr/bin/env bash
# Table 3: cross-dataset AUC of the FF++-trained checkpoints on Celeb-DF-v2 and WildDeepfake.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python tools/eval_cross_dataset.py \
    --ckpt checkpoints/unmask_lite_ffpp.pth checkpoints/unmask_mid_ffpp.pth checkpoints/unmask_base_ffpp.pth \
    --names UNMASK-Lite UNMASK-Mid UNMASK-Base \
    --data CDF=data/Celeb-DF-v2 WDF=data/WildDeepfake \
    --all_readouts --bootstrap 1000 --ref UNMASK-Base \
    --out results/cross_dataset.json --save_scores results/cross_dataset_scores.npz "$@"
