"""HydraFake (Tan et al., "Veritas: Generalizable Deepfake Detection via Pattern-Aware
Reasoning", ICLR 2026): the 23 test subsets and their official image lists.

* ``SPLITS``: the subsets in the column order of the paper's Table 1, with the folder name of
  the release, the name of the official list in ``jsons/test/<level>/`` and the scale of Table 9.
* ``load_subset``: the official list when it has been downloaded (``git lfs pull``), otherwise
  a folder scan with the same labels.

The official lists are Git-LFS files; a plain clone leaves ~130-byte pointers. One subset cannot
be rebuilt from folders alone: ``test/id/facevid2vid`` holds only its 1,000 forgeries, and its
1,000 reals are FF++ reals named only in ``facevid2vid.json``. In folder mode a fixed (seeded)
sample of the FF++ test reals is substituted and the subset is flagged ``folder+approx``.

Labels: 1 = fake (the positive class of the paper's precision / recall).
"""

from __future__ import annotations

import json
import os
import random
from collections import OrderedDict

from .folder import IMG_EXT

__all__ = ['SPLITS', 'SPLIT_NAMES', 'SPLIT_SHORT', 'all_subsets', 'load_subset', 'load_all',
           'describe', 'json_status', 'scan_split']

# (key, column label of Table 1, release folder, official json name, Table 9 scale)
SPLITS = OrderedDict([
    ('id', [
        ('ffpp', 'FF++', 'FaceForensics++', 'FaceForensics++', 8960),
        ('facevid2vid', 'Facevid2vid', 'facevid2vid', 'facevid2vid', 2000),
        ('hallo2', 'Hallo2', 'Hallo2', 'Hallo2', 1660),
        ('stylegan', 'StyleGAN', 'StyleGAN', 'StyleGAN', 600),
        ('midjourney', 'Midjourney', 'Midjourney', 'Midjourney', 600),
    ]),
    ('cm', [
        ('adobefirefly', 'ADF', 'AdobeFirefly', 'AdobeFirefly', 600),
        ('flux', 'FLUX', 'Flux11Pro', 'Flux11Pro', 600),
        ('starryai', 'StarryAI', 'StarryAI', 'StarryAI', 600),
        ('magi', 'MAGI-1', 'MAGI', 'MAGI', 1048),
        ('hart', 'HART', 'HART', 'HART', 4200),
        ('infinity', 'Infinity', 'Infinity', 'Infinity', 4200),
    ]),
    ('cf', [
        ('starganv2', 'StGAN2', 'StarGANv2', 'starganv2', 2000),
        ('iclight', 'IC-Light', 'ICLight', 'iclight', 2082),
        ('codeformer', 'CodeF.', 'CodeFormer', 'codeformer', 1750),
        ('infiniteyou', 'InfiniteYou', 'InfiniteYou', 'infiniteyou', 3244),
        ('pulid', 'PuLID', 'PuLID', 'pulid', 3360),
        ('faceadapter', 'FaceAda.', 'FaceAdapter', 'faceadapter', 300),
    ]),
    ('cd', [
        ('deepfacelab', 'DeepFaceLab', 'deepfacelab', 'deepfacelab', 3094),
        ('infiniteyou_cd', 'InfiniteYou', 'InfiniteYou-CD', 'infiniteyou', 2960),
        ('dreamina', 'Dreamina', 'dreamina', 'dreamina', 952),
        ('hailuo', 'HailuoAI', 'hailuo', 'hailuo', 1000),
        ('gpt4o', 'GPT-4o', 'gpt4o', 'gpt4o', 630),
        ('ffiw', 'FFIW', 'FFIW', 'FFIW', 6832),
    ]),
])
SPLIT_NAMES = OrderedDict(id='In-Domain', cm='Cross-Model', cf='Cross-Forgery', cd='Cross-Domain')
SPLIT_SHORT = OrderedDict(id='ID', cm='CM', cf='CF', cd='CD')
FACEVID_REAL_FROM = ('id', 'ffpp')
FACEVID_REAL_N = 1000
FACEVID_REAL_SEED = 0


def all_subsets():
    for split, subs in SPLITS.items():
        for key, label, folder, js, n in subs:
            yield split, key, label, folder, js, n


def is_lfs_pointer(path):
    """True for a Git-LFS file that has not been downloaded (a ~130-byte text stub)."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readline().startswith('version https://git-lfs')
    except OSError:
        return False


def _ci_child(parent, name):
    """os.path.join(parent, name), matching ``name`` case-insensitively."""
    p = os.path.join(parent, name)
    if os.path.exists(p) or not os.path.isdir(parent):
        return p
    low = name.lower()
    for d in os.listdir(parent):
        if d.lower() == low:
            return os.path.join(parent, d)
    return p


def _label_from_path(rel):
    parts = [s.lower() for s in rel.replace('\\', '/').split('/')]
    if any(s in ('0_real', 'real') for s in parts):
        return 0
    if any(s in ('1_fake', 'fake') for s in parts):
        return 1
    return None


def _walk_images(d):
    out = []
    for dp, _, names in os.walk(d):
        for n in names:
            if n.lower().endswith(IMG_EXT):
                out.append(os.path.join(dp, n))
    out.sort()
    return out


_TEST_INDEX = {}


def _test_index(root):
    """{subset folder name (lower case): [(level, name)]} for root/test/<level>/<name>."""
    if root not in _TEST_INDEX:
        idx = {}
        t = _ci_child(root, 'test')
        if os.path.isdir(t):
            for sp in os.listdir(t):
                d = os.path.join(t, sp)
                if os.path.isdir(d) and sp.lower() in SPLITS:
                    for n in os.listdir(d):
                        if os.path.isdir(os.path.join(d, n)):
                            idx.setdefault(n.lower(), []).append((sp, n))
        _TEST_INDEX[root] = idx
    return _TEST_INDEX[root]


def _local_path(root, p):
    """Map a path of an official list to this machine.

    The released lists omit the level folder (``hydrafake/test/HART/1_fake/x.png`` lives at
    ``test/cm/HART/1_fake/x.png``); every subset folder occurs in exactly one level, so the
    level is looked up from the folders on disk."""
    parts = [s for s in str(p).replace('\\', '/').split('/') if s not in ('', '.')]
    for i, s in enumerate(parts):
        if s.lower() in ('test', 'train', 'val'):
            rest = parts[i:]
            if s.lower() == 'test' and len(rest) > 1 and rest[1].lower() not in SPLITS:
                hit = _test_index(root).get(rest[1].lower())
                if hit and len(hit) == 1:
                    sp, name = hit[0]
                    return os.path.join(root, _ci_child(root, 'test'), sp, name, *rest[2:])
            return os.path.join(root, *rest)
    return os.path.join(root, *parts)


def _label_from_entry(e, rel):
    for k in ('label', 'labels', 'gt', 'target'):
        if k in e:
            v = e[k]
            if isinstance(v, list):
                v = v[0]
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ('fake', '1', 'true'):
                    return 1
                if s in ('real', '0', 'false'):
                    return 0
            else:
                return int(v)
    y = _label_from_path(rel)
    if y is not None:
        return y
    for m in reversed(e.get('messages', []) or []):       # the reference answer of the dialogue
        if m.get('role') == 'assistant':
            txt = str(m.get('content', '')).lower()
            if '<answer>' in txt:
                ans = txt.split('<answer>')[-1].split('</answer>')[0].strip()
                if ans in ('fake', 'real'):
                    return int(ans == 'fake')
    return None


def _entry_image(e):
    for k in ('images', 'image', 'img', 'path', 'image_path', 'file'):
        if k in e:
            v = e[k]
            return v[0] if isinstance(v, list) else v
    return None


def load_json_list(root, json_path):
    """-> (items [(absolute path, label)], n_unlabelled, n_missing_on_disk)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get('data') or data.get('items') or list(data.values())
    items, unl, miss = [], 0, 0
    for e in data:
        if isinstance(e, str):
            e = {'images': [e]}
        p = _entry_image(e)
        if p is None:
            unl += 1
            continue
        lp = _local_path(root, p)
        y = _label_from_entry(e, p)
        if y is None:
            unl += 1
            continue
        if not os.path.isfile(lp):
            miss += 1
            continue
        items.append((lp, int(y)))
    return items, unl, miss


def load_folder(root, split, folder):
    d = _ci_child(_ci_child(os.path.join(root, 'test'), split), folder)
    if not os.path.isdir(d):
        return None, f'folder not found: {d}'
    items, nolab = [], 0
    for p in _walk_images(d):
        y = _label_from_path(os.path.relpath(p, d))
        if y is None:
            nolab += 1
            continue
        items.append((p, y))
    note = ''
    if nolab:
        if not items:                                     # a forgery-only folder (facevid2vid)
            items = [(p, 1) for p in _walk_images(d)]
            note = f'{nolab} images without a 0_real/1_fake folder -> all taken as fake'
        else:
            note = f'{nolab} images without a 0_real/1_fake folder were ignored'
    return items, note


def load_subset(root, split, key, mode='auto', cache=None):
    """-> dict(items, source in {'json', 'folder', 'folder+approx'}, note, expected, missing)

    ``mode``: 'json' (official list, error if missing), 'folder', or 'auto'."""
    cache = {} if cache is None else cache
    _, label, folder, js, n_exp = next(s for s in SPLITS[split] if s[0] == key)
    jp = os.path.join(root, 'jsons', 'test', split, js + '.json')
    have_json = os.path.isfile(jp) and not is_lfs_pointer(jp)
    if mode == 'json' and not have_json:
        raise FileNotFoundError(
            f'{jp} is {"a Git-LFS pointer" if os.path.isfile(jp) else "missing"}; download the '
            f'official lists first: git lfs pull --include="jsons/test/**,jsons/val/**"')
    if have_json and mode in ('json', 'auto'):
        items, unl, miss = load_json_list(root, jp)
        note = []
        if unl:
            note.append(f'{unl} entries without image/label skipped')
        if miss:
            note.append(f'{miss} listed images NOT on disk')
        return dict(items=items, source='json', note='; '.join(note), expected=n_exp, missing=miss)
    items, note = load_folder(root, split, folder)
    if items is None:
        return dict(items=[], source='folder', note=note, expected=n_exp, missing=n_exp)
    src = 'folder'
    if key == 'facevid2vid' and not any(y == 0 for _, y in items):
        fs, fk = FACEVID_REAL_FROM
        if (fs, fk) not in cache:
            cache[(fs, fk)] = load_subset(root, fs, fk, mode='folder', cache=cache)
        reals = sorted(p for p, y in cache[(fs, fk)]['items'] if y == 0)
        pick = random.Random(FACEVID_REAL_SEED).sample(reals, min(FACEVID_REAL_N, len(reals)))
        items = items + [(p, 0) for p in sorted(pick)]
        src = 'folder+approx'
        note = (note + '; ' if note else '') + (
            f'{len(pick)} reals = seeded sample of the FF++ test reals (the official pairing is '
            f'in facevid2vid.json, not downloaded)')
    return dict(items=items, source=src, note=note, expected=n_exp, missing=0)


def load_all(root, mode='auto', only=None):
    """-> OrderedDict[(level, key)] -> load_subset(...)"""
    out, cache = OrderedDict(), {}
    for split, key, *_ in all_subsets():
        if only and split not in only:
            continue
        r = load_subset(root, split, key, mode=mode, cache=cache)
        cache[(split, key)] = r
        out[(split, key)] = r
    return out


def describe(sets):
    """Text table: subset, source, n real / n fake, Table 9 scale."""
    rows = []
    for (split, key), r in sets.items():
        lab = next(s[1] for s in SPLITS[split] if s[0] == key)
        nr = sum(1 for _, y in r['items'] if y == 0)
        nf = len(r['items']) - nr
        tot = nr + nf
        flag = ''
        if tot == 0:
            flag = 'EMPTY'
        elif abs(tot - r['expected']) > 0.02 * r['expected']:
            flag = f'differs from Table 9 by {tot - r["expected"]:+d}'
        rows.append(f'  {SPLIT_SHORT[split]:<3} {lab:<12} {r["source"]:<14} real {nr:>5}  '
                    f'fake {nf:>5}  total {tot:>5} / {r["expected"]:>5}  {flag}'
                    + (f'\n        note: {r["note"]}' if r['note'] else ''))
    return '\n'.join(rows)


def json_status(root):
    """(downloaded, still LFS pointers, missing) among the 23 official test lists."""
    ok = ptr = absent = 0
    for split, key, label, folder, js, n in all_subsets():
        jp = os.path.join(root, 'jsons', 'test', split, js + '.json')
        if not os.path.isfile(jp):
            absent += 1
        elif is_lfs_pointer(jp):
            ptr += 1
        else:
            ok += 1
    return ok, ptr, absent


def scan_split(root, split='val'):
    """HydraFake train / val (``<split>/real``, ``<split>/fake``) -> [(path, label)]"""
    d = _ci_child(root, split)
    items = []
    for cls, y in (('real', 0), ('fake', 1)):
        items += [(p, y) for p in _walk_images(_ci_child(d, cls))]
    return items
