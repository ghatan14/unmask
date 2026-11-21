# Training

Every reported run is one YAML file in `configs/`; the command line overrides any key.

| Config | Model | Data | Paper |
|---|---|---|---|
| `configs/ffpp/unmask_base.yaml` | UNMASK-Base | FF++ c23, NeuralTextures held out | Table 3 (main row) |
| `configs/ffpp/unmask_mid.yaml` | UNMASK-Mid | FF++ c23 | Table 3 |
| `configs/ffpp/unmask_lite.yaml` | UNMASK-Lite | FF++ c23 | Table 3 |
| `configs/hydrafake/unmask_base.yaml` | UNMASK-Base | HydraFake train | Table 4 (main row) |
| `configs/hydrafake/unmask_mid.yaml` | UNMASK-Mid | HydraFake train | Table 4 |
| `configs/hydrafake/unmask_mid_no_kd.yaml` | UNMASK-Mid without teachers | HydraFake train | distillation ablation |

## 1. Pretrained weights and teachers

```bash
python tools/download_weights.py --out pretrained
```

downloads from the Hugging Face repository `Wolowolo/fsfm-3c` (FS-VFM, CC BY-NC 4.0):

- `pretrained/fsvfm_vit_small/FS-VFM_ViT-S_VF2_600e.pth` and `pretrain_ds_mean_std.txt`: the face
  prior of the student (FS-VFM ViT-S/16 pre-trained on VGGFace2);
- `pretrained/fsvfm_vit_large_ffpp_c23.pth`: the online teacher (FS-VFM ViT-L/16 fine-tuned on
  FF++ c23, weights only).

PE-Core S/16 and T/16 are downloaded by timm on first use.

GenD is cached once over the training frames (logits and the 1024-d pooled feature, averaged over
the upright and mirrored view). It needs a clone of the official GenD code (and its requirements):

```bash
git clone https://github.com/yermandy/GenD third_party/GenD
python tools/cache_gend.py --gend_repo third_party/GenD \
    --data data/FaceForensics++/c23/train --val data/FaceForensics++/c23/val --gate 85 \
    --out pretrained/gend_cache_ffpp_train.npz                       # about 80 min on an RTX 4060
python tools/cache_gend.py --gend_repo third_party/GenD \
    --data data/HydraFake/train --val data/HydraFake/val --gate 75 \
    --out pretrained/gend_cache_hydrafake_train.npz                  # about 20 min
```

`scripts/prepare_teachers.sh` runs all of the above.

**Teacher gate.** A teacher is used only if its frame AUC on the validation split of the training
domain reaches the gate (85 on FF++, 75 on HydraFake; GenD 86.0 and FS-VFM-L 86.6 on HydraFake
val). The decision is never taken on a test set. `cache_gend.py --val` checks GenD; the training
script checks the online FS-VFM teacher (`--teacher_gate`) and also decides which of its outputs
means "fake".

## 2. Train

Always start with a dry check: it builds everything, runs a few iterations, prints the peak GPU
memory and the projected time, and saves nothing.

```bash
python tools/train.py --config configs/ffpp/unmask_base.yaml --output_dir runs/unmask_base_ffpp --dry_check 40
python tools/train.py --config configs/ffpp/unmask_base.yaml --output_dir runs/unmask_base_ffpp
```

Each run writes `log.txt` (one JSON line per epoch: losses, validation video AUC on clean and
degraded frames, the held-out NeuralTextures compass and the mean SE gates of the EMA model),
`console.log`, and the checkpoints `checkpoint-last.pth` (reported in the paper, resumable),
`checkpoint-ema.pth`, `checkpoint-swa.pth` (LoRA folded) and `checkpoint-best.pth`.

An interrupted run continues with `--resume runs/<run>/checkpoint-last.pth` (same command
otherwise). Seeds: the paper's three UNMASK-Base runs use `--seed 0`, `1` and `2` with the same
initialisation (`init_seed: 0`).

Measured on one RTX 4060 (8 GB, batch 32 x 2, mixed precision, online ViT-L teacher):

| Run | Time |
|---|---|
| UNMASK-Base, FF++ | 3.8 h |
| UNMASK-Mid, FF++ | 3.3 h |
| UNMASK-Lite, FF++ | 2.6 h |
| UNMASK-Base, HydraFake | 1.9 h |
| UNMASK-Mid, HydraFake | 1.7 h |

If memory is tight, add `--grad_ckpt`, or use `--batch_size 24 --accum_iter 3` (note that the
learning rate follows the effective batch).

## 3. Export

```bash
python tools/export_checkpoint.py runs/unmask_base_ffpp/checkpoint-last.pth checkpoints/unmask_base_ffpp.pth \
    --trained_on "FF++ c23 (w/o NeuralTextures)"
```

keeps the configuration and the weights and drops the optimiser, EMA/SWA state and the training-only
projectors. The LoRA adapters are kept, so the exported model gives exactly the scores of the training
checkpoint. `--fold` folds them into the frozen weights
for deployment (45.83 / 28.18 / 22.02 M parameters); with fp16 inference this changes the last digits
of the scores (see [EVALUATION.md](EVALUATION.md#numerical-precision)).

## 4. Fine-tune a released model

```bash
python tools/train.py --config configs/hydrafake/unmask_base.yaml --output_dir runs/ft \
    --init_ckpt checkpoints/unmask_base_ffpp.pth
```

The released weights, LoRA adapters included, initialise the model, so training departs exactly from
the released model (a LoRA-folded checkpoint also works: it initialises the frozen weights and fresh
adapters start at zero).

## Recipe at a glance

| | |
|---|---|
| Adaptation | frozen ViTs, LoRA r = 16 (alpha 32, dropout 0.05) on qkv / proj / fc1 / fc2, all LayerNorms trainable |
| Read-out | gated attention pooling after blocks 4, 8, 12 + last-block summary, per branch |
| Fusion | multi-level SE, gates 2·sigmoid(z) with z zero-initialised; branch dropout 0.2 |
| Losses | CE fused 1.0 · CE branch heads 0.5 · KD fused 0.5 · KD branch 0.25 (T = 3) · summary feature 1.0 · dense tokens 0.5 |
| Augmentation | random resized crop (scale >= 0.75), flip, colour jitter, JPEG / rescale / blur / noise / grayscale, random erasing, mixup (0.8) / CutMix (1.0) on 50% of the batches |
| Sampling | FF++: paired real/fake batches, one epoch = half a pass over the real frames; HydraFake: shuffled full passes |
| Optimisation | AdamW, lr 2.5e-4 (blr 1e-3 x 64 / 256), weight decay 0.05, 1 warm-up epoch + cosine, 10 epochs, grad clip 1.0, fp16 |
| Averaging | EMA 0.9995 and SWA over the second half (ablation rows only) |
