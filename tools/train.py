#!/usr/bin/env python
"""Train UNMASK with routed multi-teacher knowledge distillation.

    python tools/train.py --config configs/ffpp/unmask_base.yaml --output_dir runs/unmask_base_ffpp

Any key of the YAML file can be overridden on the command line (``--seed 1``). Always run a
short ``--dry_check 40`` first: it builds everything, pushes 40 batches through and reports the
peak GPU memory and the projected training time, without saving a checkpoint.

Recipe of the paper (per run): two frozen priors + LoRA (r 16 on qkv/proj/fc1/fc2) + LayerNorms;
loss = CE(fused, label smoothing 0.1) + 0.5 CE(branch heads) + 0.5 [KD(fused <- 1:1 teacher
mixture) + 0.5 KD(branch <- routed teacher)] + 1.0 summary-feature distillation + 0.5 dense
patch-token distillation; AdamW, lr 2.5e-4 (blr 1e-3 x 64 / 256), 1 warm-up epoch + cosine,
batch 32 x 2, mixup/cutmix on half of the batches, EMA and SWA kept for ablations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unmask.data import (FrameDataset, PairedRealFakeSampler, TrainTransform,  # noqa: E402
                         eval_transform, manipulation_of, scan_frames)
from unmask.engine import (EMA, KDHeads, MergedSWA, cos_dist, decide_fake_col,  # noqa: E402
                           default_routes, evaluate, kd_kl, match_grid, mix_batch, soft_ce)
from unmask.models import build_unmask, load_weights, unmask_config  # noqa: E402
from unmask.models.fsvfm import read_mean_std  # noqa: E402
from unmask.teachers import CachedTeacher, FSVFMTeacher  # noqa: E402
from unmask.utils import (IMAGENET_MEAN, IMAGENET_STD, autocast, grad_scaler,  # noqa: E402
                          start_console_log, torch_load)


# ==========================================================================
def get_args(argv=None):
    ap = argparse.ArgumentParser('UNMASK training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--config', default=None, help='YAML file with any of the options below')
    g = ap.add_argument_group('data')
    g.add_argument('--train_path', default=None, help='training split (class sub-folders)')
    g.add_argument('--val_path', default=None, help='validation split of the same domain')
    g.add_argument('--real_hint', default='real', help='substring(s) naming the real class folder(s)')
    g.add_argument('--video_sep', default='_frame_', help='file-name separator between video stem and frame')
    g.add_argument('--holdout_manip', default='none',
                   choices=['none', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                   help='FF++ manipulation excluded from training (used as a validation compass)')
    g.add_argument('--val_subsample', type=float, default=1.0, help='fraction of validation frames used')
    g.add_argument('--no_pairing', action='store_true', help='plain shuffling instead of paired real/fake batches')
    g = ap.add_argument_group('student')
    g.add_argument('--variant', default='base', choices=['base', 'mid', 'lite'])
    g.add_argument('--face_ckpt', default=None, help='FS-VFM ViT-S/16 pre-training checkpoint')
    g.add_argument('--face_mean_std', default=None, help='its pretrain_ds_mean_std.txt')
    g.add_argument('--clip_model', default=None, help='override the timm name of the CLIP prior')
    g.add_argument('--img_size', type=int, default=224)
    g.add_argument('--taps', type=int, nargs='+', default=[3, 7, 11], help='0-based block indices')
    g.add_argument('--tune', default='lora+ln', choices=['lora+ln', 'lora', 'ln', 'full'])
    g.add_argument('--lora_targets', default='attn+mlp', choices=['attn', 'attn+mlp'])
    g.add_argument('--lora_r', type=int, default=16)
    g.add_argument('--lora_alpha', type=float, default=32)
    g.add_argument('--lora_dropout', type=float, default=0.05)
    g.add_argument('--head_drop', type=float, default=0.1)
    g.add_argument('--branch_drop', type=float, default=0.2)
    g.add_argument('--drop_path', type=float, default=0.0)
    g.add_argument('--grad_ckpt', action='store_true', help='gradient checkpointing (less memory)')
    g.add_argument('--init_seed', type=int, default=0, help='seed of the weight initialisation')
    g.add_argument('--init_ckpt', default=None, help='initialise from an UNMASK checkpoint (fine-tuning)')
    g.add_argument('--no_pretrained', action='store_true', help='random backbones (pipeline test only)')
    g = ap.add_argument_group('teachers')
    g.add_argument('--teacher_fsvfm', default=None, help='FF++-fine-tuned FS-VFM detector (online teacher)')
    g.add_argument('--teacher_fsvfm_mean_std', default=None)
    g.add_argument('--gend_cache', default=None, help='GenD cache written by tools/cache_gend.py')
    g.add_argument('--teacher_gate', type=float, default=85.0,
                   help='minimum frame AUC of the online teacher on validation frames')
    g.add_argument('--teacher_dtype', default='fp16', choices=['fp16', 'bf16', 'fp32'])
    g.add_argument('--routes', default=None, help='override the routing, e.g. "face=fsvfm+gend,clip=gend"')
    g.add_argument('--dense_on', default='face', help='branches with dense token distillation, or "none"')
    g.add_argument('--kd_weights', default=None, help='fused-head teacher mixture, e.g. "fsvfm=0.5,gend=0.5"')
    g = ap.add_argument_group('losses')
    g.add_argument('--w_cls', type=float, default=1.0)
    g.add_argument('--w_branch', type=float, default=0.5)
    g.add_argument('--w_kd', type=float, default=0.5)
    g.add_argument('--kd_temperature', type=float, default=3.0)
    g.add_argument('--w_sfd', type=float, default=1.0, help='summary-feature distillation')
    g.add_argument('--w_dfd', type=float, default=0.5, help='dense patch-token distillation')
    g.add_argument('--kd_hidden', type=int, default=1024)
    g.add_argument('--label_smoothing', type=float, default=0.1)
    g.add_argument('--mix_prob', type=float, default=0.5)
    g.add_argument('--mixup_alpha', type=float, default=0.8)
    g.add_argument('--cutmix_alpha', type=float, default=1.0)
    g.add_argument('--cutmix_switch', type=float, default=0.5)
    g = ap.add_argument_group('optimisation')
    g.add_argument('--epochs', type=int, default=10)
    g.add_argument('--epoch_mult', type=float, default=0.5,
                   help='paired sampling: passes over the real frames per epoch')
    g.add_argument('--batch_size', type=int, default=32)
    g.add_argument('--accum_iter', type=int, default=2)
    g.add_argument('--blr', type=float, default=1e-3, help='lr = blr * batch_size * accum_iter / 256')
    g.add_argument('--min_lr', type=float, default=1e-6)
    g.add_argument('--warmup_epochs', type=int, default=1)
    g.add_argument('--weight_decay', type=float, default=0.05)
    g.add_argument('--clip_grad', type=float, default=1.0)
    g.add_argument('--ema_decay', type=float, default=0.9995)
    g.add_argument('--swa_start', type=float, default=0.5, help='fraction of training after which SWA averages')
    g.add_argument('--swa_every', type=int, default=50)
    g = ap.add_argument_group('augmentation')
    g.add_argument('--scale_min', type=float, default=0.75)
    g.add_argument('--no_degrade', action='store_true')
    g.add_argument('--erase_p', type=float, default=0.25)
    g = ap.add_argument_group('run')
    g.add_argument('--select_by', default='heldout', choices=['heldout', 'degraded', 'clean'],
                   help='validation score that picks checkpoint-best.pth (the paper reports checkpoint-last)')
    g.add_argument('--output_dir', default=None)
    g.add_argument('--resume', default=None, help='checkpoint-last.pth of an interrupted run')
    g.add_argument('--workers', type=int, default=6)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--device', default='cuda')
    g.add_argument('--dry_check', type=int, default=0, help='run N iterations, report memory/time, save nothing')

    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        import yaml
        with open(pre.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in ap._actions}
        bad = sorted(k for k in cfg if k not in known)
        if bad:
            raise SystemExit(f'{pre.config}: unknown option(s) {bad}')
        ap.set_defaults(**cfg)
    args = ap.parse_args(argv)
    for k in ('train_path', 'val_path', 'output_dir'):
        if not getattr(args, k):
            raise SystemExit(f'--{k} is required (on the command line or in --config)')
    if not (args.no_pretrained or args.resume or args.init_ckpt) and not args.face_ckpt:
        raise SystemExit('--face_ckpt (FS-VFM ViT-S/16 checkpoint) is required; see tools/download_weights.py')
    return args


def parse_kv(spec):
    out = {}
    for part in (spec or '').split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            out[k.strip()] = float(v.strip())
    return out


def subsample(items, frac, seed=0):
    if frac >= 1.0:
        return items
    rng = np.random.RandomState(seed)
    out = [it for it, k in zip(items, rng.rand(len(items)) < frac) if k]
    return out if len({y for _, y in out}) == 2 else items


# ==========================================================================
_RUN_DIR = None


def main(argv=None):
    global _RUN_DIR
    args = get_args(argv)
    _RUN_DIR = args.output_dir
    start_console_log(args.output_dir, 'DRY CHECK (no training)' if args.dry_check else 'TRAINING')
    for name in ('DRY_CHECK_ONLY.txt', 'TRAINING_DONE.txt'):
        p = os.path.join(args.output_dir, name)
        if not args.dry_check and os.path.exists(p):
            os.remove(p)
    resume_ck = torch_load(args.resume) if args.resume else None
    if not args.dry_check:
        with open(os.path.join(args.output_dir, 'run_args.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(vars(args), started=time.strftime('%Y-%m-%d %H:%M:%S')), f, indent=2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rs = np.random.RandomState(args.seed + 1000)                  # mixup / cutmix draws
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'
    t_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.teacher_dtype]
    if device.type != 'cuda':
        t_dtype = torch.float32
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------ data
    print('[1/6] data')
    items, train_root = scan_frames(args.train_path, args.real_hint)
    rel = [os.path.relpath(p, train_root).replace('\\', '/') for p, _ in items]
    manips = [manipulation_of(r) for r in rel]
    counts = defaultdict(int)
    for m in manips:
        if m is not None:
            counts[m] += 1
    n_fake = sum(y for _, y in items)
    hold = args.holdout_manip if args.holdout_manip != 'none' else None
    if hold and sum(counts.values()) <= 0.5 * max(n_fake, 1):
        print('  [warn] FF++ manipulations not recognised in the file names: --holdout_manip ignored')
        hold = None
    train_idx = [i for i in range(len(items)) if hold is None or manips[i] != hold]
    heldout_idx = [i for i in range(len(items)) if hold is not None and manips[i] == hold]
    print(f'  {len(items)} frames in {train_root} ({n_fake} fake); FF++ manipulations '
          f'{dict(sorted(counts.items()))}' + (f'; {hold} held out' if hold else ''))

    # --------------------------------------------------------------- model
    print('[2/6] student')
    if resume_ck is not None:
        cfg = resume_ck['config']
        if cfg.get('variant') != args.variant:
            print(f'  [resume] variant {args.variant!r} -> checkpoint {cfg.get("variant")!r}')
    else:
        face_mean, face_std = read_mean_std(args.face_mean_std)
        cfg = unmask_config(args.variant, face_mean=face_mean, face_std=face_std,
                            clip_model=args.clip_model, img_size=args.img_size, taps=args.taps,
                            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout, lora_targets=args.lora_targets,
                            tune=args.tune)
    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
    pretrained = not (args.resume or args.no_pretrained or args.init_ckpt)
    model = build_unmask(cfg, pretrained=pretrained, face_ckpt=args.face_ckpt,
                         head_drop=args.head_drop, branch_drop=args.branch_drop,
                         grad_ckpt=args.grad_ckpt, drop_path_rate=args.drop_path, verbose=True)
    if args.init_seed is not None:
        torch.manual_seed(args.seed)
    if args.init_ckpt:
        load_weights(model, torch_load(args.init_ckpt)['model'])
        print(f'  initialised from {args.init_ckpt}')
    model.to(device)
    print('  ' + model.describe())
    multi = len(model.branch_heads) > 1

    # ------------------------------------------------------------ val data
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    val_items, _ = scan_frames(args.val_path, args.real_hint)
    val_items = subsample(val_items, args.val_subsample, args.seed)
    val_workers = min(args.workers, 3)

    def mk(its, tf, shuffle=False):
        return DataLoader(FrameDataset(its, tf, sep=args.video_sep), batch_size=64, shuffle=shuffle,
                          num_workers=val_workers, pin_memory=device.type == 'cuda',
                          persistent_workers=val_workers > 0)

    ev = lambda deg: eval_transform(size=args.img_size, degrade=deg, mean=mean, std=std)
    val_clean_dl, val_degr_dl = mk(val_items, ev(False)), mk(val_items, ev(True))
    print(f'  val: {len(val_items)} frames ({sum(y for _, y in val_items)} fake)')
    heldout_dl = None
    if heldout_idx:
        hold_items = [(items[i][0], 1) for i in heldout_idx]
        real_items = [(p, 0) for p, y in val_items if y == 0]
        step = max(len(hold_items) // max(len(real_items) * 2, 1), 1)
        heldout_dl = mk(hold_items[::step] + real_items, ev(False))
        print(f'  held-out compass: {len(hold_items[::step])} unseen {hold} fakes + {len(real_items)} reals')
    elif args.select_by == 'heldout':
        args.select_by = 'degraded'

    # ------------------------------------------------------------ teachers
    print('[3/6] teachers')
    teachers = {}
    for flag in ('teacher_fsvfm', 'gend_cache'):
        path = getattr(args, flag)
        if path and not os.path.isfile(path):
            raise SystemExit(f'--{flag}: file not found: {os.path.abspath(path)}')
    if args.teacher_fsvfm:
        teachers['fsvfm'] = FSVFMTeacher(args.teacher_fsvfm, args.teacher_fsvfm_mean_std,
                                         img_size=args.img_size, device=device, dtype=t_dtype)
    if args.gend_cache:
        teachers['gend'] = CachedTeacher(args.gend_cache, [rel[i] for i in train_idx],
                                         [items[i][1] for i in train_idx], name='gend')
    if not teachers:
        print('  [warn] no teacher given: plain supervised training (no distillation)')
    probe = mk(subsample(val_items, 0.4, args.seed + 7), ev(False), shuffle=True)
    for k, t in teachers.items():
        print('  ' + t.info)
        if getattr(t, 'kind', '') == 'online':
            col, auc = decide_fake_col(t, probe, device)
            t.set_fake_col(col)
            ok = auc >= args.teacher_gate
            print(f'    fake column {col} (decided on validation frames); frame AUC {auc:.2f} '
                  f'{">=" if ok else "<"} gate {args.teacher_gate:g}')
            if not ok:
                raise SystemExit(f'teacher {k} does not fit these crops (validation frame AUC {auc:.2f})')
    del probe

    has_feat = {k: (t.has_feat if isinstance(t, CachedTeacher) else True) for k, t in teachers.items()}
    tdim = {k: t.dim for k, t in teachers.items()}
    routes = default_routes(model.branch_names, teachers, args.routes)
    dense_on = [] if args.dense_on == 'none' else [b.strip() for b in args.dense_on.split(',')]
    sfd_routes = [(b, k, (model.branches[b].out_dim if multi else model.fused_dim), tdim[k])
                  for b in model.branch_names for k in routes[b] if has_feat[k] and args.w_sfd > 0]
    dfd_routes = [(b, k, model.branches[b].embed_dim, tdim[k])
                  for b in model.branch_names if b in dense_on
                  for k in routes[b] if getattr(teachers[k], 'has_tokens', False) and args.w_dfd > 0]
    kd_heads = KDHeads(sfd_routes, dfd_routes, hidden=args.kd_hidden).to(device)
    want_tokens = tuple(sorted({b for b, _, _, _ in dfd_routes}))
    kw = parse_kv(args.kd_weights)
    t_weight = {k: kw.get(k, 1.0) for k in teachers}
    if teachers:
        print('  routes: ' + ', '.join(f'{b} <- {"+".join(v)}' for b, v in routes.items()))
        print('  fused-head teacher mixture: ' + ', '.join(f'{k} {w:g}' for k, w in t_weight.items()))
        print('  ' + kd_heads.describe())

    # ----------------------------------------------------------- train data
    train_items = [items[i] for i in train_idx]
    train_ds = FrameDataset(train_items, TrainTransform(
        size=args.img_size, scale_min=args.scale_min, erase_p=args.erase_p,
        degrade=not args.no_degrade, mean=mean, std=std), sep=args.video_sep)
    print(f'  train: {len(train_items)} frames = {len(train_items) - sum(y for _, y in train_items)} '
          f'real + {sum(y for _, y in train_items)} fake')
    sampler = None if args.no_pairing else PairedRealFakeSampler(
        train_items, args.batch_size, seed=args.seed, epoch_mult=args.epoch_mult, sep=args.video_sep)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          shuffle=sampler is None, num_workers=args.workers,
                          pin_memory=device.type == 'cuda', drop_last=True,
                          persistent_workers=args.workers > 0)

    # ----------------------------------------------------------- optimiser
    print('[4/6] optimiser')
    eff_bs = args.batch_size * args.accum_iter
    lr = args.blr * eff_bs / 256
    skip = model.no_weight_decay()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    kd_named = [(n, p) for n, p in kd_heads.named_parameters() if p.requires_grad]
    params = [p for _, p in named] + [p for _, p in kd_named]
    opt = torch.optim.AdamW([
        {'params': [p for n, p in named if n not in skip] + [p for n, p in kd_named if p.ndim > 1],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in named if n in skip] + [p for n, p in kd_named if p.ndim <= 1],
         'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.999))
    scaler = grad_scaler(amp)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    swa = MergedSWA(model) if args.swa_start < 1 else None
    steps = max(len(train_dl) // args.accum_iter, 1)
    swa_from = int(args.swa_start * args.epochs * steps)
    print(f'  effective batch {eff_bs}, lr {lr:.2e}, {steps} steps/epoch, '
          f'{sum(p.numel() for _, p in named) / 1e6:.3f} M student + '
          f'{sum(p.numel() for _, p in kd_named) / 1e6:.3f} M projector parameters trained')

    T = args.kd_temperature
    n_cls = cfg['num_classes']
    log_path = os.path.join(args.output_dir, 'log.txt')
    best = {'heldout': -1, 'degraded': -1, 'clean': -1}
    start_epoch, new_lr, t0 = 0, lr, time.time()
    if resume_ck is not None:
        ck = resume_ck
        model.load_state_dict(ck['model'])
        if ck.get('kd_heads') is not None:
            kd_heads.load_state_dict(ck['kd_heads'])
        for k, obj in (('optimizer', opt), ('scaler', scaler)):
            if k in ck:
                obj.load_state_dict(ck[k])
        if ema is not None and ck.get('ema') is not None:
            ema.shadow.load_state_dict(ck['ema'])
        best.update(ck.get('best', {}))
        if swa is not None and ck.get('swa_state') is not None:
            swa.load_state(ck['swa_state'])
        start_epoch = int(ck.get('epoch', -1)) + 1
        print(f'  resumed at epoch {start_epoch}')
        del ck, resume_ck

    meta = dict(train_path=os.path.abspath(args.train_path), holdout=hold, routes=routes)

    def save(name, net, ep, rec, full=False):
        blob = {'model': net.state_dict(), 'config': cfg, 'epoch': ep, 'val': rec,
                'args': vars(args), 'meta': meta}
        if full:
            blob.update(optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                        ema=ema.state_dict() if ema else None, best=best,
                        kd_heads=kd_heads.state_dict(),
                        swa_state=None if swa is None or swa.n == 0 else swa.state())
        torch.save(blob, os.path.join(args.output_dir, name))

    def save_swa(ep, rec):
        cfg_swa = dict(cfg, lora_folded=True)
        torch.save({'model': swa.merged_state_dict(model), 'config': cfg_swa, 'epoch': ep,
                    'val': rec, 'args': vars(args), 'meta': dict(meta, swa_samples=swa.n)},
                   os.path.join(args.output_dir, 'checkpoint-swa.pth'))

    # ------------------------------------------------------------ training
    print('[5/6] training' + (f' -- DRY CHECK, {args.dry_check} iterations' if args.dry_check else ''))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    online = {k: t for k, t in teachers.items() if getattr(t, 'kind', '') == 'online'}
    cached = {k: t for k, t in teachers.items() if isinstance(t, CachedTeacher)}
    bad_steps = 0
    for ep in range(start_epoch, args.epochs):
        model.train()
        kd_heads.train()
        agg, nb = defaultdict(float), 0
        opt.zero_grad(set_to_none=True)
        t_ep, tick, t_teacher = time.time(), [], 0.0
        for it, (x, y, idx) in enumerate(train_dl):
            if it % args.accum_iter == 0:                 # per-step warm-up + cosine schedule
                cur = ep * steps + it // args.accum_iter
                warm = args.warmup_epochs * steps
                new_lr = (lr * cur / max(warm, 1)) if cur < warm else \
                    args.min_lr + (lr - args.min_lr) * 0.5 * (
                        1 + math.cos(math.pi * (cur - warm) / max(args.epochs * steps - warm, 1)))
                for g in opt.param_groups:
                    g['lr'] = new_lr
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, y_a, y_b, lam, perm, mix_kind = mix_batch(x, y, rs, args.mix_prob, args.mixup_alpha,
                                                         args.cutmix_alpha, args.cutmix_switch)

            # ---- teacher targets (no gradient) --------------------------
            ts = time.time()
            t_prob, t_feat, t_tok = {}, {}, {}
            with torch.no_grad():
                for k, t in online.items():
                    o = t(x, want_tokens=any(kk == k for _, kk in kd_heads.dfd_routes))
                    t_prob[k] = F.softmax(o['logits'] / T, dim=1)
                    if k in kd_heads.std_feat:
                        kd_heads.std_feat[k].update(o['feat'])
                        t_feat[k] = F.normalize(kd_heads.std_feat[k](o['feat']), dim=1)
                    if o.get('tokens') is not None and k in kd_heads.std_tok:
                        kd_heads.std_tok[k].update(o['tokens'])
                        t_tok[k] = kd_heads.std_tok[k](o['tokens'])
                for k, t in cached.items():
                    lg, ft = t.get(idx)
                    p = F.softmax(lg.to(device, non_blocking=True) / T, dim=1)
                    if perm is not None:
                        p = lam * p + (1.0 - lam) * p[perm]
                    t_prob[k] = p
                    if ft is not None and k in kd_heads.std_feat:
                        ft = ft.to(device, non_blocking=True).float()
                        kd_heads.std_feat[k].update(ft)
                        z = F.normalize(kd_heads.std_feat[k](ft), dim=1)
                        if perm is not None:
                            z = F.normalize(lam * z + (1.0 - lam) * z[perm], dim=1)
                        t_feat[k] = z
            if args.dry_check and device.type == 'cuda':
                torch.cuda.synchronize()
            t_teacher += time.time() - ts

            # ---- student -------------------------------------------------
            with autocast(amp):
                out = model(x, return_all=True, want_tokens=want_tokens)
                l_cls = soft_ce(out['fused'], y_a, y_b, lam, n_cls, args.label_smoothing)
                heads = list(model.branch_heads)
                zero = torch.zeros((), device=device)
                l_br = torch.stack([soft_ce(out[b], y_a, y_b, lam, n_cls, args.label_smoothing)
                                    for b in heads]).mean() if (heads and args.w_branch > 0) else zero
                l_kd = l_kd_br = zero
                if t_prob and args.w_kd > 0:
                    wsum = sum(t_weight[k] for k in t_prob)
                    l_kd = kd_kl(out['fused'], sum(t_weight[k] * t_prob[k] for k in t_prob) / wsum, T)
                    br_terms = []
                    for b in heads:
                        ks = [k for k in routes.get(b, []) if k in t_prob]
                        if ks:
                            ws = sum(t_weight[k] for k in ks)
                            br_terms.append(kd_kl(out[b], sum(t_weight[k] * t_prob[k] for k in ks) / ws, T))
                    if br_terms:
                        l_kd_br = torch.stack(br_terms).mean()
                sfd = [cos_dist(kd_heads.sfd[f'{b}__{k}'](out[f'feat_{b}'].float()), t_feat[k])
                       for b, k in kd_heads.sfd_routes if k in t_feat]
                l_sfd = torch.stack(sfd).mean() if sfd else zero
                dfd = []
                for b, k in kd_heads.dfd_routes:
                    if k in t_tok and f'tok_{b}' in out:
                        s_tok = kd_heads.dfd[f'{b}__{k}'](out[f'tok_{b}'].float())
                        dfd.append(cos_dist(s_tok, match_grid(t_tok[k], s_tok.shape[1])))
                l_dfd = torch.stack(dfd).mean() if dfd else zero
                loss = args.w_cls * l_cls + args.w_branch * l_br + \
                    args.w_kd * (l_kd + args.w_branch * l_kd_br) + \
                    args.w_sfd * l_sfd + args.w_dfd * l_dfd

            if not torch.isfinite(loss):
                bad_steps += 1
                print(f'  [warn] non-finite loss at epoch {ep} iteration {it} ({bad_steps} so far), skipped')
                if bad_steps > 50:
                    raise RuntimeError('more than 50 non-finite losses; try a lower --blr')
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / args.accum_iter).backward()
            if (it + 1) % args.accum_iter == 0:
                if args.clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update()
                gstep = ep * steps + (it + 1) // args.accum_iter
                if swa is not None and gstep >= swa_from and gstep % args.swa_every == 0:
                    swa.update()

            for name, val in (('loss', loss), ('l_cls', l_cls), ('l_branch', l_br), ('l_kd', l_kd),
                              ('l_kd_branch', l_kd_br), ('l_sfd', l_sfd), ('l_dfd', l_dfd)):
                agg[name] += float(val.detach())
            agg['mixed'] += float(perm is not None)
            nb += 1
            if it % 100 == 0:
                el = time.time() - t_ep
                print(f'  ep{ep} [{it}/{len(train_dl)}] loss {loss.item():.4f} (cls {l_cls.item():.3f} '
                      f'br {l_br.item():.3f} kd {l_kd.item():.3f} sfd {l_sfd.item():.3f} '
                      f'dfd {l_dfd.item():.3f}{" " + mix_kind if perm is not None else ""}) '
                      f'lr {new_lr:.2e} | {(it + 1) * args.batch_size / max(el, 1e-6):.0f} img/s', flush=True)

            if args.dry_check:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                tick.append(time.time())
                if it + 1 >= args.dry_check:
                    dry_check_report(args, device, tick, t_ep, t_teacher, it, train_dl, val_items)
                    return

        rec = {'epoch': ep, 'lr': new_lr, **{k: v / max(nb, 1) for k, v in agg.items()},
               'train_min': (time.time() - t_ep) / 60, 'teacher_min': t_teacher / 60}
        r_clean = evaluate(model, val_clean_dl, device, amp)
        r_degr = evaluate(model, val_degr_dl, device, amp)
        rec['clean'], rec['degraded'] = r_clean['fused'], r_degr['fused']
        for b in model.branch_heads:
            rec[f'degraded_{b}'] = r_degr[b]
        rec['heldout'] = evaluate(model, heldout_dl, device, amp)['fused'] if heldout_dl else float('nan')
        if ema is not None:
            r_ema = evaluate(ema.shadow, val_degr_dl, device, amp, want_gates=True)
            rec['ema_degraded'] = r_ema['fused']
            if 'gates' in r_ema:
                rec['ema_gates'] = r_ema['gates']
        print(f'  == epoch {ep}: clean {rec["clean"]:.2f} | degraded {rec["degraded"]:.2f}'
              + ''.join(f' | {b} {rec[f"degraded_{b}"]:.2f}' for b in model.branch_heads)
              + (f' | held-out {rec["heldout"]:.2f}' if heldout_dl else '')
              + (f' | ema {rec["ema_degraded"]:.2f}' if ema else '')
              + f' | sfd {rec["l_sfd"]:.3f} dfd {rec["l_dfd"]:.3f}'
              + f'   [{(time.time() - t0) / 60:.0f} min]', flush=True)
        if rec.get('ema_gates'):
            print('     SE gates (EMA, mean over validation): '
                  + ', '.join(f'{n} {g:.2f}' for n, g in zip(model.level_names, rec['ema_gates'])))
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        v = rec.get(args.select_by, float('nan'))
        if v == v and v > best[args.select_by]:
            best[args.select_by] = v
            save('checkpoint-best.pth', model, ep, rec)
        save('checkpoint-last.pth', model, ep, rec, full=True)
        if ema is not None:
            save('checkpoint-ema.pth', ema.shadow, ep, rec)
        if swa is not None and swa.n > 0:
            save_swa(ep, rec)

    print('[6/6] done. The paper reports checkpoint-last.pth (fused head); evaluate it with '
          'tools/eval_cross_dataset.py or tools/eval_hydrafake.py')
    with open(os.path.join(args.output_dir, 'TRAINING_DONE.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {args.epochs} epochs\n')


def dry_check_report(args, device, tick, t_ep, t_teacher, it, train_dl, val_items):
    part = tick[len(tick) // 3:]
    s_it = (part[-1] - part[0]) / max(len(part) - 1, 1) if len(part) > 1 else (time.time() - t_ep) / (it + 1)
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        resv = torch.cuda.max_memory_reserved() / 2 ** 30
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    else:
        peak = resv = total_gb = float('nan')
    ep_min = s_it * len(train_dl) / 60
    report = ['=== dry check ===',
              f'  GPU memory         : {peak:.2f} GB allocated, {resv:.2f} GB reserved of {total_gb:.1f} GB',
              f'  time per iteration : {s_it:.3f} s at batch {args.batch_size} '
              f'({args.batch_size / max(s_it, 1e-9):.0f} img/s), teachers '
              f'~{100 * t_teacher / max(time.time() - t_ep, 1e-9):.0f}% of it',
              f'  one epoch          : ~{ep_min:.0f} min ({len(train_dl)} iterations)',
              f'  {args.epochs} epochs          : ~{ep_min * args.epochs / 60:.1f} h of training '
              f'+ ~{len(val_items) * 3 / 400 * args.epochs / 3600:.1f} h of validation']
    if resv == resv and resv > 0.85 * total_gb:
        report.append('  memory is tight: add --grad_ckpt, or --batch_size 24 --accum_iter 3')
    report += ['', '  pipeline OK. NOTHING WAS TRAINED AND NO CHECKPOINT WAS SAVED.']
    print('\n' + '\n'.join(report))
    with open(os.path.join(args.output_dir, 'DRY_CHECK_ONLY.txt'), 'w', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n' + '\n'.join(report) + '\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        if _RUN_DIR and os.path.isdir(_RUN_DIR) and not isinstance(e, SystemExit):
            import traceback
            with open(os.path.join(_RUN_DIR, 'ERROR.txt'), 'a', encoding='utf-8') as f:
                f.write(f'===== {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
                traceback.print_exc(file=f)
        raise
