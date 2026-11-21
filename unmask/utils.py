"""Small helpers shared by the package and the tools."""

from __future__ import annotations

import os
import sys
import time

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def torch_load(path, map_location='cpu'):
    """``torch.load`` that accepts the full checkpoint dictionaries written by this repo."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                                   # torch < 1.13
        return torch.load(path, map_location=map_location)


def autocast(enabled=True, device_type='cuda'):
    """Mixed-precision context that works on old and new torch versions."""
    try:
        return torch.amp.autocast(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def grad_scaler(enabled=True):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def get_device(name='cuda'):
    return torch.device(name if (name.startswith('cuda') and torch.cuda.is_available())
                        or name == 'cpu' else 'cpu')


class Tee:
    """Mirror a stream (stdout / stderr) into a log file."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        self._stream.write(s)
        try:
            self._fh.write(s)
            self._fh.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_console_log(output_dir, header=''):
    """Append everything printed from now on to ``<output_dir>/console.log``."""
    os.makedirs(output_dir, exist_ok=True)
    fh = open(os.path.join(output_dir, 'console.log'), 'a', encoding='utf-8')
    fh.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")}  {header}\n      {" ".join(sys.argv)}\n')
    fh.flush()
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    return fh


def utf8_console():
    """Windows consoles default to cp1252; tables printed here contain non-ASCII signs."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
