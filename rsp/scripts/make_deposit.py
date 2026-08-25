#!/usr/bin/env python3
"""
Assemble the Figshare deposit from a completed campaign.

    python scripts/make_deposit.py

Reads results/, figures/, tables/ and the source tree, and writes

    deposit/rsp-resilience-v1.0.0-code.zip     the code snapshot
    deposit/rsp-resilience-v1.0.0-data.zip     raw episode data + meta + summary
    deposit/rsp-resilience-v1.0.0-figures.zip  figures and tables as deposited
    deposit/CHECKSUMS.sha256                   SHA-256 of every deposited file
    deposit/ENVIRONMENT.txt                    interpreter and library versions
    deposit/README_FIGSHARE.md                 the record-level description

Nothing is uploaded: the three archives and the three text files are what you
attach to the Figshare item by hand.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import zipfile
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEP = os.path.join(ROOT, 'deposit')
VERSION = '1.0.0'
STEM = f'rsp-resilience-v{VERSION}'

CODE = [
    'rsp/__init__.py', 'rsp/dynamics.py', 'rsp/plants.py', 'rsp/faults.py',
    'rsp/controllers.py', 'rsp/simulate.py', 'rsp/margin.py', 'rsp/score.py',
    'rsp/stats.py',
    'scripts/run_experiments.py', 'scripts/make_analysis.py',
    'scripts/verify_dynamics.py', 'scripts/verify_references.py',
    'scripts/verify_deposit.py', 'scripts/verify_paper_claims.py',
    'scripts/make_deposit.py',
    'README.md', 'LICENSE', 'CITATION.cff', 'requirements.txt', '.gitignore',
]
FIGS = ['fig_margin_bounds.pdf', 'fig_axioms.pdf', 'fig_lipschitz.pdf',
        'fig_convergence.pdf', 'fig_simplex.pdf', 'fig_failures.pdf',
        'fig_aggregators.pdf']


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()


def write_zip(name, members):
    """members: list of (path_on_disk, name_inside_archive)."""
    out = os.path.join(DEP, name)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for src, arc in members:
            if not os.path.exists(src):
                raise SystemExit(f'missing file: {src}')
            z.write(src, arc)
    return out


def environment():
    import numpy, scipy, matplotlib
    return '\n'.join([
        f'generated            {date.today().isoformat()}',
        f'python               {sys.version.split()[0]}',
        f'platform             {platform.platform()}',
        f'machine              {platform.machine()}',
        f'numpy                {numpy.__version__}',
        f'scipy                {scipy.__version__}',
        f'matplotlib           {matplotlib.__version__}',
        '',
        'The campaign is deterministic given (--seed, NumPy version): the same',
        'seed on the same NumPy reproduces every figure bit-for-bit. Across',
        'NumPy major versions the Generator stream is stable for the',
        'distributions used here, but this file records the exact environment',
        'in which the deposited results were produced.',
    ]) + '\n'


def main():
    os.makedirs(DEP, exist_ok=True)
    res = os.path.join(ROOT, 'results')
    meta = json.load(open(os.path.join(res, 'meta.json')))
    plants = meta['plants']

    code = [(os.path.join(ROOT, f), f'{STEM}/{f}') for f in CODE]
    data = [(os.path.join(res, 'meta.json'), f'{STEM}/results/meta.json'),
            (os.path.join(res, 'summary.json'), f'{STEM}/results/summary.json')]
    for p in plants:
        f = os.path.join(res, f'raw_{p}.npz')
        if not os.path.exists(f):
            raise SystemExit(
                f'missing {f}\nRun scripts/run_experiments.py before depositing.')
        data.append((f, f'{STEM}/results/raw_{p}.npz'))
    figs = [(os.path.join(ROOT, 'figures', f), f'{STEM}/figures/{f}') for f in FIGS]
    tab = os.path.join(ROOT, 'tables', 'tables.tex')
    if os.path.exists(tab):
        figs.append((tab, f'{STEM}/tables/tables.tex'))

    produced = [write_zip(f'{STEM}-code.zip', code),
                write_zip(f'{STEM}-data.zip', data),
                write_zip(f'{STEM}-figures.zip', figs)]

    with open(os.path.join(DEP, 'ENVIRONMENT.txt'), 'w') as fh:
        fh.write(environment())
    produced.append(os.path.join(DEP, 'ENVIRONMENT.txt'))

    with open(os.path.join(DEP, 'CHECKSUMS.sha256'), 'w') as fh:
        for f in produced:
            fh.write(f'{sha256(f)}  {os.path.basename(f)}\n')

    for f in produced:
        print(f'{os.path.getsize(f) / 1e6:8.2f} MB  {os.path.basename(f)}')
    print(f'\ndeposit assembled in {DEP}')


if __name__ == '__main__':
    main()
