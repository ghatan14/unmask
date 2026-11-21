"""UNMASK: a lightweight deepfake detector with dual foundation priors and routed
multi-teacher knowledge distillation."""

from .models import (CLASS_NAMES, FAKE_INDEX, UNMASK, VARIANTS, build_unmask, load_unmask,
                     save_unmask, unmask_config)

__version__ = '1.0.0'

__all__ = ['UNMASK', 'VARIANTS', 'CLASS_NAMES', 'FAKE_INDEX', 'build_unmask', 'load_unmask',
           'save_unmask', 'unmask_config', '__version__']
