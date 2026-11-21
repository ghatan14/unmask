from .averaging import EMA, MergedSWA
from .distill import KDHeads, cos_dist, decide_fake_col, default_routes, match_grid
from .evaluate import evaluate
from .losses import kd_kl, mix_batch, soft_ce

__all__ = ['EMA', 'MergedSWA', 'KDHeads', 'cos_dist', 'decide_fake_col', 'default_routes',
           'match_grid', 'evaluate', 'kd_kl', 'mix_batch', 'soft_ce']
