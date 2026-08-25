#!/usr/bin/env python3
"""
Audit every number the manuscript asserts against the raw campaign data.

    python scripts/verify_paper_claims.py

verify_deposit.py checks that the archive is complete and internally coherent.
This script asks the complementary question: does the text of the paper agree
with what the data says?  It recomputes the seven rows of Table 2, the captions
of Figures 3 to 6, the fragility numbers of Section 8.4, the sensitivity
analysis of Section 8.5 and the campaign counts, and compares each against the
value printed in the manuscript.

One line per claim, PASS or FAIL, with the manuscript value and the recomputed
one side by side.  Exits non-zero if anything disagrees.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsp.score import (normalize, R_p, kappa_bound, kappa_exact,
                       lipschitz_ratio, W_NOMINAL, X2_CAP)
from rsp.stats import (n_hoeffding, n_bernstein, empirical_bernstein,
                       cliffs_delta, wilcoxon_paired, holm)
from rsp.controllers import CONTROLLERS
from rsp.faults import FAULT_NAMES
from scipy.stats import spearmanr

RES = os.path.join(ROOT, 'results')
PLANT = 'P3-SRS7'

# (A, B, delta-median, Cliff's delta, kappa*) exactly as printed in Table 2
TABLE5 = [('SMC', 'AMPC', +0.098, +0.381, 1.046),
          ('AMPC', 'ADRC', +0.035, +0.373, 1.309),
          ('ADRC', 'ANN', +0.402, +0.501, 0.412),
          ('ANN', 'CTC', +0.060, -0.134, 0.049),
          ('CTC', 'PID', +0.024, +0.209, 0.165),
          ('PID', 'FT1', +0.151, +0.418, 0.770),
          ('SMC', 'FT1', +0.770, +0.454, 0.287)]

results = []


def claim(label, paper, got, tol=1.5e-3, fmt='{:.3f}'):
    ok = (got is not None) and abs(float(got) - float(paper)) <= tol
    results.append(ok)
    p = fmt.format(paper)
    g = '   n/a' if got is None else fmt.format(got)
    print(f'  {"PASS" if ok else "FAIL"}  {label:52s} paper {p:>10s}   data {g:>10s}')


def exact(label, paper, got):
    ok = (paper == got)
    results.append(ok)
    print(f'  {"PASS" if ok else "FAIL"}  {label:52s} paper {str(paper):>10s}   '
          f'data {str(got):>10s}')


def load():
    meta = json.load(open(os.path.join(RES, 'meta.json')))
    data = {}
    for p in meta['plants']:
        f = os.path.join(RES, f'raw_{p}.npz')
        if not os.path.exists(f):
            raise SystemExit(f'missing {f}: run scripts/run_experiments.py first')
        data[p] = dict(np.load(f))
    return meta, data


def scores(meta, data):
    S = {}
    for p, d in data.items():
        xs = np.array([meta['xstar'][p][k] for k in
                       ('x1', 'x2', 'x3', 'x4', 'x5', 'x6')])
        for c in CONTROLLERS:
            X = d[f'X_{c}'].copy()
            X[:, 1] = np.where(d[f'degraded_{c}'], np.minimum(X[:, 1], X2_CAP), 0.0)
            m = normalize(X, xs)
            S[(p, c)] = dict(m=m, R0=R_p(m, W_NOMINAL, 0.0),
                             Rm1=R_p(m, W_NOMINAL, -1.0),
                             Rm2=R_p(m, W_NOMINAL, -2.0),
                             R1=R_p(m, W_NOMINAL, 1.0))
    return S


def geo(m):
    return np.exp(np.log(np.maximum(m, 1e-12)).mean(axis=0))


def main():
    meta, data = load()
    S = scores(meta, data)
    n_ep = data[list(data)[0]]['fid'].size

    print('\nCampaign counts (Section 8.1)')
    exact('evaluation episodes', 32000, int(n_ep * len(CONTROLLERS) * len(data)))
    exact('calibration episodes', 3840,
          int(2 * 48 * len(CONTROLLERS) * len(data)))
    exact('episodes per plant-controller pair', 800, int(n_ep))
    claim('wall-clock time [min]', 26.0,
          sum(meta['runtime_s'].values()) / 60.0, tol=1.0, fmt='{:.1f}')

    print('\nTable 2: pairwise comparison on ' + PLANT)
    pairs = list(itertools.combinations(CONTROLLERS, 2))
    pv = [wilcoxon_paired(S[(PLANT, a)]['R0'], S[(PLANT, b)]['R0'])
          for a, b in pairs]
    adj = dict(zip(pairs, holm(pv)))
    exact('comparisons significant after Holm, of 28',
          28, int(sum(v < 0.05 for v in adj.values())))
    for a, b, dm, cd, ks in TABLE5:
        ra, rb = S[(PLANT, a)]['R0'], S[(PLANT, b)]['R0']
        key = (a, b) if (a, b) in adj else (b, a)
        claim(f'{a}-{b}: difference of medians', dm,
              float(np.median(ra) - np.median(rb)))
        claim(f'{a}-{b}: Cliff delta', cd, cliffs_delta(ra, rb))
        claim(f'{a}-{b}: kappa*', ks,
              kappa_exact(geo(S[(PLANT, a)]['m']), geo(S[(PLANT, b)]['m']),
                          W_NOMINAL))
        ok = adj[key] < 1e-4
        results.append(ok)
        print(f'  {"PASS" if ok else "FAIL"}  {a}-{b}: Holm p < 1e-4'.ljust(60)
              + f'  data {adj[key]:.2e}')

    print('\nSection 8.3: the sign disagreement between median and delta')
    dm = float(np.median(S[(PLANT, 'ANN')]['R0']) - np.median(S[(PLANT, 'CTC')]['R0']))
    cd = cliffs_delta(S[(PLANT, 'ANN')]['R0'], S[(PLANT, 'CTC')]['R0'])
    ok = dm > 0 > cd
    results.append(ok)
    print(f'  {"PASS" if ok else "FAIL"}  '
          f'{"ANN-CTC: median positive, Cliff delta negative":52s} '
          f'paper       True   data {str(ok):>10s}')

    print('\nFigure 4 caption: the weight simplex')
    for a, b, label in (('ANN', 'CTC', 'ANN-CTC'), ('AMPC', 'ADRC', 'AMPC-ADRC')):
        ma, mb = geo(S[(PLANT, a)]['m']), geo(S[(PLANT, b)]['m'])
        kb = kappa_bound(ma, mb, W_NOMINAL)[0]
        ke = kappa_exact(ma, mb, W_NOMINAL)
        paper_kb, paper_ke = (0.049, 0.049) if a == 'ANN' else (1.075, 1.309)
        claim(f'{label}: kappa (sufficient bound)', paper_kb, kb)
        claim(f'{label}: kappa* (exact)', paper_ke, ke)
        if a == 'AMPC':
            claim('AMPC-ADRC: bound conservative by [%]', 22.0,
                  100.0 * (ke - kb) / kb, tol=1.0, fmt='{:.1f}')

    print('\nFigure 5 caption: convergence for AMPC on ' + PLANT)
    r = S[(PLANT, 'AMPC')]['R0']
    claim('score standard deviation', 0.380, float(np.std(r, ddof=1)))
    claim('Hoeffding half-width at N=800', 0.048,
          float(np.sqrt(np.log(2 / 0.05) / (2 * r.size))))
    claim('empirical Bernstein half-width at N=800', 0.053,
          float(empirical_bernstein(r, 0.05)))
    exact('N required for an absolute claim', 738, n_hoeffding(0.05, 0.05))

    print('\nFigure 3 caption: Lipschitz stability')
    vals = []
    for p in data:
        for a, b in pairs:
            ma, mb = S[(p, a)]['m'], S[(p, b)]['m']
            ok = (ma > 0).all(1) & (mb > 0).all(1)
            if ok.sum() > 10:
                vals.append(lipschitz_ratio(ma[ok], mb[ok], W_NOMINAL))
    claim('largest empirical ratio (theory: 1)', 0.665,
          float(np.concatenate(vals).max()))

    print('\nSection 5: the range of outright-failure rates')
    zr = [float((S[(p, c)]['R0'] == 0).mean()) for p in data for c in CONTROLLERS]
    claim('lowest cell [%]', 21.0, 100 * min(zr), tol=0.6, fmt='{:.1f}')
    claim('highest cell [%]', 67.0, 100 * max(zr), tol=0.6, fmt='{:.1f}')

    print('\nSection 7: paired dispersion')
    sig = [np.std(S[(p, a)]['R0'] - S[(p, b)]['R0'], ddof=1)
           for p in data for a, b in pairs]
    claim('median sigma_D', 0.288, float(np.median(sig)))
    claim('largest sigma_D', 0.542, float(np.max(sig)))
    exact('N at the median dispersion', 343,
          n_bernstein(0.05, 0.05, float(np.median(sig))))
    exact('N at the largest dispersion', 966,
          n_bernstein(0.05, 0.05, float(np.max(sig))))
    exact('controller pairs', 140, len(sig))

    print('\nSection 8.4: the sliding-mode fragility under variable latency')
    for k, name, paper in ((6, 'F7 variable latency', 0.892),
                           (5, 'F6 packet loss', 0.014),
                           (7, 'F8 missed cycle', 0.000)):
        got = float(np.mean([(S[(p, 'SMC')]['R0'][data[p]['fid'] == k] == 0).mean()
                             for p in data]))
        claim(f'SMC outright-failure rate, {name}', paper, got, tol=6e-3)
    c3 = [float((S[(p, 'SMC')]['m'][data[p]['fid'] == 6][:, 2] == 0).mean())
          for p in data]
    claim('C3 annihilated, lowest plant [%]', 85.0, 100 * min(c3), tol=0.6,
          fmt='{:.1f}')
    claim('C3 annihilated, highest plant [%]', 98.0, 100 * max(c3), tol=0.6,
          fmt='{:.1f}')
    # the median is taken over all 800 episodes, as every other median in the
    # paper is; episodes that diverged carry x_1 = +inf and are not discarded
    x1 = data[PLANT]['X_SMC'][:, 0]
    claim('SMC median post-fault error on P3 [mm]', 0.145,
          1e3 * float(np.median(x1)), tol=1e-3, fmt='{:.3f}')

    print('\nFigure 6 caption: destructiveness by fault class')
    per = {name: float(np.mean([(S[(p, c)]['R0'][data[p]['fid'] == k] == 0).mean()
                                for p in data for c in CONTROLLERS]))
           for k, name in enumerate(FAULT_NAMES)}
    exact('most destructive class', 'F4-EncoderBias', max(per, key=per.get))
    claim('F4 mean outright-failure rate', 0.78, per['F4-EncoderBias'], tol=6e-3,
          fmt='{:.2f}')
    claim('F2 mean outright-failure rate', 0.67, per['F2-FreeSwing'], tol=6e-3,
          fmt='{:.2f}')

    print('\nSection 8.5: the aggregation rule')
    med = {(p, k): np.array([np.median(S[(p, c)][k]) for c in CONTROLLERS])
           for p in data for k in ('R0', 'Rm1', 'Rm2', 'R1')}
    order = lambda v: tuple(np.argsort(-v))
    flips = sum(order(med[(p, 'R0')]) != order(med[(p, 'R1')]) for p in data)
    exact('plants whose ranking the arithmetic mean reverses', 3, int(flips))
    rho = [float(spearmanr(med[(p, 'R0')], med[(p, k)]).statistic)
           for p in data for k in ('Rm1', 'Rm2')]
    claim('lowest Spearman rho within p <= 0', 0.969, min(rho))
    claim('highest Spearman rho within p <= 0', 1.000, max(rho))
    pooled = {k: float(spearmanr(np.concatenate([med[(p, 'R0')] for p in data]),
                                 np.concatenate([med[(p, k)] for p in data])).statistic)
              for k in ('Rm1', 'Rm2')}
    claim('pooled rho, R_-1', 0.988, pooled['Rm1'])
    claim('pooled rho, R_-2', 0.983, pooled['Rm2'])
    exact('plants where R_-1 reproduces the R_0 order exactly', 2,
          int(sum(order(med[(p, 'R0')]) == order(med[(p, 'Rm1')]) for p in data)))
    o0 = [c for _, c in sorted(((np.median(S[(PLANT, c)]['R0']), c)
                                for c in CONTROLLERS), reverse=True)]
    o1 = [c for _, c in sorted(((np.median(S[(PLANT, c)]['R1']), c)
                                for c in CONTROLLERS), reverse=True)]
    exact('CTC position on P3 under R_0', 5, o0.index('CTC') + 1)
    exact('CTC position on P3 under R_1', 8, o1.index('CTC') + 1)
    m3 = {c: float(np.median(S[(PLANT, c)]['m'][:, 2])) for c in CONTROLLERS}
    claim('CTC median reconfiguration criterion on P3', 0.152, m3['CTC'])
    exact('other controllers with median m_3 = 1.000 on P3', 7,
          int(sum(abs(v - 1.0) < 1e-9 for c, v in m3.items() if c != 'CTC')))

    bad = results.count(False)
    print(f'\n{len(results)} claims checked, {bad} disagreements')
    if not bad:
        print('EVERY NUMBER IN THE MANUSCRIPT MATCHES THE DATA')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
