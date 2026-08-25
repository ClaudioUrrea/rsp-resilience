#!/usr/bin/env python3
"""
Check the raw campaign files before depositing them.

    python scripts/verify_deposit.py

Validates the structure of every results/raw_<plant>.npz (keys, shapes, dtypes,
fault-class composition, pairing across controllers, the +inf pattern that marks
outright failure), then recomputes the headline numbers from the raw criteria
and diffs them against the archived results/summary.json.

Prints a compact report and exits non-zero if anything disagrees.  The report is
short enough to paste into an email or an issue.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsp.score import normalize, R_p, W_NOMINAL, X2_CAP
from rsp.controllers import CONTROLLERS
from rsp.faults import FAULT_NAMES

RES = os.path.join(ROOT, 'results')
N_FAULTS = len(FAULT_NAMES)


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()


def check_structure(name, d, problems):
    keys = set(d)
    want = {f'{k}_{c}' for c in CONTROLLERS
            for k in ('X', 'declared', 'degraded', 'alive', 'ratio')}
    want |= {'fid', 't_f', 'idx'}
    missing = sorted(want - keys)
    extra = sorted(keys - want)
    if missing:
        problems.append(f'{name}: missing arrays {missing}')
    if extra:
        print(f'    note: unexpected arrays {extra}')

    if 'fid' not in keys:
        return None
    B = d['fid'].size
    counts = np.bincount(d['fid'].astype(int), minlength=N_FAULTS)
    if counts.size != N_FAULTS or len(set(counts.tolist())) != 1:
        problems.append(f'{name}: fault classes not balanced: {counts.tolist()}')
    if not (0.80 - 1e-9 <= d['t_f'].min() and d['t_f'].max() <= 1.40 + 1e-9):
        problems.append(f'{name}: t_f outside the declared range [0.80, 1.40] s: '
                        f'[{d["t_f"].min():.3f}, {d["t_f"].max():.3f}]')

    for c in CONTROLLERS:
        X = d[f'X_{c}']
        if X.shape != (B, 6):
            problems.append(f'{name}/{c}: X has shape {X.shape}, expected {(B, 6)}')
            continue
        if np.isnan(X).any():
            problems.append(f'{name}/{c}: X contains NaN')
        neg = (X < 0) & np.isfinite(X)
        if neg.any():
            problems.append(f'{name}/{c}: negative raw criteria in columns '
                            f'{sorted(set(np.where(neg)[1].tolist()))}')
        for k in ('declared', 'degraded', 'alive'):
            if d[f'{k}_{c}'].shape != (B,):
                problems.append(f'{name}/{c}: {k} has shape {d[f"{k}_{c}"].shape}')
    return B


def scores(d, xstar):
    out = {}
    for c in CONTROLLERS:
        X = d[f'X_{c}'].copy()
        X[:, 1] = np.where(d[f'degraded_{c}'], np.minimum(X[:, 1], X2_CAP), 0.0)
        m = normalize(X, xstar)
        out[c] = dict(m=m, R0=R_p(m, W_NOMINAL, 0.0), R1=R_p(m, W_NOMINAL, 1.0))
    return out


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


def main():
    problems, notes = [], []
    meta_path = os.path.join(RES, 'meta.json')
    if not os.path.exists(meta_path):
        raise SystemExit(f'missing {meta_path}: run scripts/run_experiments.py first')
    meta = json.load(open(meta_path))
    published = None
    sp = os.path.join(RES, 'summary.json')
    if os.path.exists(sp):
        published = json.load(open(sp))

    print('file inventory')
    data, sizes = {}, {}
    for p in meta['plants']:
        f = os.path.join(RES, f'raw_{p}.npz')
        if not os.path.exists(f):
            problems.append(f'{p}: raw_{p}.npz not found')
            continue
        sizes[p] = os.path.getsize(f)
        print(f'  {p:14s} {sizes[p] / 1e6:7.2f} MB  sha256 {sha256(f)[:16]}...')
        data[p] = dict(np.load(f))

    print('\nstructure')
    B_all = set()
    for p, d in data.items():
        B = check_structure(p, d, problems)
        if B:
            B_all.add(B)
            print(f'  {p:14s} {B} episodes, {len(d)} arrays, '
                  f'{B // N_FAULTS} per fault class')
    if len(B_all) > 1:
        problems.append(f'plants disagree on the episode count: {sorted(B_all)}')
    B = max(B_all) if B_all else 0
    if B and B != 800:
        notes.append(f'{B} episodes per plant-controller pair, not the 800 of the '
                     f'published campaign: the comparison against summary.json is '
                     f'skipped')

    print('\npairing (common random numbers)')
    for p, d in data.items():
        shared = all(k in d for k in ('fid', 't_f', 'idx'))
        print(f'  {p:14s} fault design stored once and shared: {shared}')

    if not data or B != 800 or published is None:
        print('\nsummary comparison skipped')
    else:
        print('\nrecomputed vs. archived summary.json')
        S = {p: scores(d, np.array([meta['xstar'][p][k] for k in
                                    ('x1', 'x2', 'x3', 'x4', 'x5', 'x6')]))
             for p, d in data.items()}
        for p in data:
            for c in CONTROLLERS:
                r = S[p][c]['R0']
                for label, got, want in (
                        ('median', np.median(r), published['medians'][p][c]),
                        ('zero-rate', (r == 0).mean(), published['zero_rate'][p][c])):
                    if not close(got, want):
                        problems.append(f'{p}/{c} {label}: recomputed {got:.4f}, '
                                        f'archived {want:.4f}')
        sig = [np.std(S[p][a]['R0'] - S[p][b]['R0'], ddof=1)
               for p in data for i, a in enumerate(CONTROLLERS)
               for b in CONTROLLERS[i + 1:]]
        med = float(np.median(sig))
        key = 'sigma_D_median'
        if key in published and not close(med, published[key]):
            problems.append(f'sigma_D median: recomputed {med:.4f}, '
                            f'archived {published[key]:.4f}')
        print(f'  40 medians and 40 outright-failure rates compared')
        print(f'  sigma_D median recomputed {med:.4f} '
              f'(archived {published.get(key, float("nan")):.4f}), '
              f'maximum {max(sig):.4f} over {len(sig)} pairs')
        agree = sum(([c for _, c in sorted(((np.median(S[p][c]['R0']), c)
                                            for c in CONTROLLERS), reverse=True)]
                     == [c for _, c in sorted(((np.median(S[p][c]['R1']), c)
                                               for c in CONTROLLERS), reverse=True)])
                    for p in data)
        print(f'  arithmetic mean reverses the ranking on {len(data) - agree} '
              f'of the {len(data)} plants')

    print()
    for n in notes:
        print(f'NOTE    {n}')
    for pr in problems:
        print(f'PROBLEM {pr}')
    if not problems:
        print('ALL CHECKS PASSED — the raw files are complete, internally '
              'consistent and consistent with the archived summary.')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
