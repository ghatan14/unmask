# Evaluation

All numbers of the paper use one protocol for UNMASK and for every re-measured baseline: the face
crop is resized directly to 224 x 224 (bicubic, no centre crop), normalised with ImageNet
statistics, no test-time augmentation. UNMASK converts the input to the statistics of each prior
internally. The reported read-out is the fused head of `checkpoint-last.pth`.

## Cross-dataset (Table 3)

```bash
python tools/eval_cross_dataset.py \
    --ckpt checkpoints/unmask_lite_ffpp.pth checkpoints/unmask_mid_ffpp.pth checkpoints/unmask_base_ffpp.pth \
    --names UNMASK-Lite UNMASK-Mid UNMASK-Base \
    --data CDF=data/Celeb-DF-v2 WDF=data/WildDeepfake \
    --all_readouts --bootstrap 1000 --ref UNMASK-Base \
    --out results/cross_dataset.json --save_scores results/cross_dataset_scores.npz
```

- **Frame AUC** over all frames; **video AUC** over videos scored by the mean p(fake) of their
  frames (the top-25% mean is also reported).
- `--all_readouts` adds the auxiliary heads `[face]`, `[clip]` and their mean `[mean]` (the
  read-out ablation) from the same forward pass.
- `--bootstrap N` gives percentile 95% CIs of the video AUC by resampling videos within each
  class, and paired differences to `--ref` (`--frame_ci` adds frame-level CIs). They cover the
  test-set uncertainty only, not training seeds.
- `--ensemble` adds a row with the mean probability of all listed checkpoints (the 3-seed
  ensemble of the ablation).

`scripts/eval_cross_dataset.sh` runs the full Table 3 evaluation.

## HydraFake (Table 4)

```bash
python tools/eval_hydrafake.py --root data/HydraFake \
    --ckpt checkpoints/unmask_base_hydrafake.pth checkpoints/unmask_mid_hydrafake.pth \
    --names UNMASK-Base UNMASK-Mid --with_paper_rows --bootstrap 1000 --ref UNMASK-Base \
    --save_scores results/hydrafake_scores.npz --out results/hydrafake
```

Protocol of Tan et al. (ICLR 2026): 23 test subsets on four levels (In-Domain, Cross-Model,
Cross-Forgery, Cross-Domain) from the official lists; accuracy with fake as the positive class and
the argmax decision (p(fake) >= 0.5, no threshold tuned on test data); `Avg` = (In-Domain mean +
18 out-of-domain subsets) / 19, as in their Table 1; precision, recall and AUC per subset are
written to the Markdown report. `--from_scores` re-makes all tables from saved scores in seconds.

## Numerical precision

On a GPU, inference runs in fp16 (autocast), so the last digits of the scores depend on the GPU and on
whether LoRA is folded into the frozen weights. The reported numbers were computed on an RTX 4060 with the
LoRA adapters kept separate. In other settings AUCs can move by a few hundredths. Accuracy uses a hard
threshold, so a few HydraFake images whose p(fake) lies within about 10^-3 of 0.5 can change side: a subset
accuracy moves by an image or two and the averages by up to 0.1 (with LoRA folded, for example, 89.7 for
UNMASK-Base and 88.3 for UNMASK-Mid instead of 89.8 and 88.2).

## Complexity

```bash
python tools/complexity.py                       # the three variants
python tools/complexity.py --ckpt checkpoints/unmask_mid_ffpp.pth --latency --int8
```

Parameters are counted with LoRA folded. `GMACs` counts the convolution and linear layers of one
224 x 224 frame (`torch.utils.flop_counter`, MACs = FLOPs / 2), the convention of the paper;
`GMACs+attn` adds the attention products. `--int8` applies PyTorch dynamic INT8 quantisation to
every linear layer and reports size and CPU latency; it is provided for deployment experiments and
is not part of the reported results.

## Visual explanations

```bash
python tools/visualize_hirescam.py --ckpt checkpoints/unmask_base_ffpp.pth --images path/to/face_crops/ --out hirescam.png
```

HiResCAM maps of the fused head's logit margin, taken at the inputs of every read-out path and
summed over paths and branches, with conservative gradients (the standard deviation of the
read-out LayerNorms, the pooling attention and the SE gates are held constant in the backward
pass), so that each map sums to the logit margin minus the bias terms; the script prints both for
every image, together with the share of positive evidence per branch. The maps show where the
evidence lies, not a precise localisation of the manipulation: the model was never trained for
localisation.
