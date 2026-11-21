"""Metrics: AUC (rank identity, ties averaged), EER, video-level aggregation and a stratified
video-level bootstrap. Labels are ``1 = fake``; scores are p(fake)."""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import numpy as np

__all__ = ['roc_auc', 'eer', 'video_scores', 'bootstrap_auc', 'accuracy_metrics']


def roc_auc(y_true, y_score):
    """ROC AUC in percent via the Mann-Whitney rank identity (no sklearn, ties averaged).

    Class counts are Python ints, so large frame sets cannot overflow on platforms whose
    numpy defaults to int32."""
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    pos, neg = int(np.count_nonzero(y == 1)), int(np.count_nonzero(y == 0))
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    _, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = ((csum - cnt + 1 + csum) / 2.0)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (float(pos) * float(neg)) * 100.0)


def eer(y_true, y_score):
    """Equal error rate (fraction)."""
    y = np.asarray(y_true)[np.argsort(-np.asarray(y_score, dtype=np.float64), kind='mergesort')]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float('nan')
    fnr = 1.0 - np.cumsum(y == 1) / pos
    fpr = np.cumsum(y == 0) / neg
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[k] + fpr[k]) / 2.0)


def video_scores(frame_scores, frame_labels, vids, topk_frac=None):
    """Aggregate frames to videos: mean p(fake) (or the mean of the top ``topk_frac``).

    Returns (video labels, video scores, video ids) in first-appearance order."""
    per, lab = OrderedDict(), {}
    for s, y, v in zip(frame_scores, frame_labels, vids):
        per.setdefault(v, []).append(float(s))
        lab[v] = int(y)
    vs, vy = [], []
    for v, sc in per.items():
        if topk_frac:
            sc = sorted(sc, reverse=True)
            sc = sc[:max(1, int(round(len(sc) * topk_frac)))]
        vs.append(float(np.mean(sc)))
        vy.append(lab[v])
    return np.array(vy), np.array(vs), list(per)


def bootstrap_auc(scores: dict, frame_labels, vids, n_boot=1000, seed=0, ref=None,
                  frame_level=False):
    """Percentile 95% CIs of video AUC (and optionally frame AUC) by resampling VIDEOS with
    replacement within each class (the real:fake mix stays fixed). The same draws are used for
    every model, so differences to ``ref`` are paired. Covers test-set uncertainty only.

    ``scores``: {name: frame p(fake) (N,)}, all on the same frames."""
    rng = np.random.default_rng(seed)
    y = np.asarray(frame_labels).astype(np.int64)
    vid_index = defaultdict(list)
    for i, v in enumerate(vids):
        vid_index[v].append(i)
    vlist = list(vid_index)
    vlab = np.array([y[vid_index[v][0]] for v in vlist])
    frames = [np.asarray(vid_index[v]) for v in vlist]
    vsc = {n: np.array([np.mean(np.asarray(s)[f]) for f in frames]) for n, s in scores.items()}
    idx_r, idx_f = np.where(vlab == 0)[0], np.where(vlab == 1)[0]
    draws_v = {n: [] for n in scores}
    draws_f = {n: [] for n in scores}
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(idx_r, len(idx_r)), rng.choice(idx_f, len(idx_f))])
        yb = vlab[pick]
        if frame_level:
            fi = np.concatenate([frames[k] for k in pick])
        for n in scores:
            draws_v[n].append(roc_auc(yb, vsc[n][pick]))
            if frame_level:
                draws_f[n].append(roc_auc(y[fi], np.asarray(scores[n])[fi]))
    out = OrderedDict()
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    for n in scores:
        dv = np.asarray(draws_v[n])
        out[n] = {'video_auc_ci': ci(dv)}
        if frame_level:
            out[n]['frame_auc_ci'] = ci(np.asarray(draws_f[n]))
        if ref is not None and ref in scores and n != ref:
            d = dv - np.asarray(draws_v[ref])
            out[n][f'video_auc_diff_vs_{ref}'] = (float(d.mean()),) + ci(d)
            if frame_level:
                d = np.asarray(draws_f[n]) - np.asarray(draws_f[ref])
                out[n][f'frame_auc_diff_vs_{ref}'] = (float(d.mean()),) + ci(d)
    return out


def accuracy_metrics(p_fake, labels, threshold=0.5):
    """Accuracy / precision / recall / TNR / balanced accuracy (percent, fake = positive) and
    AUC. ``p >= threshold`` counts as fake, i.e. the argmax decision for ``threshold = 0.5``."""
    p = np.asarray(p_fake, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    if len(y) == 0:
        nan = float('nan')
        return dict(n=0, n_real=0, n_fake=0, acc=nan, prec=nan, rec=nan, tnr=nan, bacc=nan, auc=nan)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    nf, nr = tp + fn, tn + fp
    rec = 100.0 * tp / nf if nf else float('nan')
    tnr = 100.0 * tn / nr if nr else float('nan')
    return dict(n=int(len(y)), n_real=nr, n_fake=nf, acc=100.0 * (tp + tn) / len(y),
                prec=100.0 * tp / (tp + fp) if tp + fp else float('nan'), rec=rec, tnr=tnr,
                bacc=0.5 * (rec + tnr) if nf and nr else float('nan'), auc=roc_auc(y, p))
