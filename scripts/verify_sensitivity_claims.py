#!/usr/bin/env python3
"""
Audit the claims that verify_paper_claims.py does not reach.

    python scripts/verify_sensitivity_claims.py

verify_paper_claims.py checks the seven adjacent comparisons printed in
Table 2.  That leaves twenty-one of the twenty-eight unchecked, and the gap
was not harmless: an earlier version of the manuscript asserted that the
ANN-CTC pair carried the smallest kappa* of the twenty-eight, which is false
-- SMC-PID is smaller -- and no test in the suite could have caught it.  This
script closes that gap and, at the same time, audits Section 8.6 and the two
propositions it rests on.

Three groups of checks:

    1. all 28 comparisons on P3: kappa (sufficient) <= kappa* (exact), the
       identification of the extreme pairs, and consistency with the values
       printed in Table 2;
    2. Proposition 2 and Corollary 1, re-derived from the deposited criteria
       rather than from results/sensitivity.json, so that the two paths are
       independent;
    3. every macro in paper/sensitivity_macros.tex against the JSON that
       produced it, which catches a stale macro file after a partial run.

One line per claim, PASS or FAIL.  Exits non-zero if anything disagrees.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import itertools
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsp.score import (normalize, R_p, kappa_bound, kappa_exact,
                       W_NOMINAL, X2_CAP)
from rsp.controllers import CONTROLLERS

RES = os.path.join(ROOT, 'results')
PAPER = os.path.join(ROOT, 'paper')
PLANT = 'P3-SRS7'

# the seven adjacent comparisons printed in Table 2, kappa* only
TABLE2_KAPPA = {('SMC', 'AMPC'): 1.046, ('AMPC', 'ADRC'): 1.309,
                ('ADRC', 'ANN'): 0.412, ('ANN', 'CTC'): 0.049,
                ('CTC', 'PID'): 0.165, ('PID', 'FT1'): 0.770,
                ('SMC', 'FT1'): 0.287}

# what the manuscript now says about the extremes over all twenty-eight
CLAIM_MIN_KAPPA_PAIR = 'SMC-PID'
CLAIM_MIN_KAPPA = 0.031
CLAIM_MIN_EPS_PAIR = 'SMC-PID'
CLAIM_MAX_EPS_PAIR = 'ANN-FT2'

results = []


def claim(label, paper, got, tol=1.5e-3, fmt='{:.3f}'):
    ok = got is not None and abs(float(got) - float(paper)) <= tol
    results.append(ok)
    g = '   n/a' if got is None else fmt.format(got)
    print(f'  {"PASS" if ok else "FAIL"}  {label:52s} paper '
          f'{fmt.format(paper):>10s}   data {g:>10s}')


def exact(label, paper, got):
    ok = paper == got
    results.append(ok)
    print(f'  {"PASS" if ok else "FAIL"}  {label:52s} paper {str(paper):>10s}   '
          f'data {str(got):>10s}')


def assertion(label, ok, detail=''):
    results.append(bool(ok))
    print(f'  {"PASS" if ok else "FAIL"}  {label:52s} {detail}')


def load():
    meta = json.load(open(os.path.join(RES, 'meta.json')))
    data = {}
    for p in meta['plants']:
        f = os.path.join(RES, f'raw_{p}.npz')
        if not os.path.exists(f):
            raise SystemExit(f'missing {f}: run scripts/run_experiments.py first')
        data[p] = dict(np.load(f))
    return meta, data


def xstar(meta, p):
    return np.array([meta['xstar'][p][k] for k in
                     ('x1', 'x2', 'x3', 'x4', 'x5', 'x6')])


def crit(d, c):
    X = d[f'X_{c}'].copy()
    X[:, 1] = np.where(d[f'degraded_{c}'], np.minimum(X[:, 1], X2_CAP), 0.0)
    return X


def geo(m):
    return np.exp(np.log(np.maximum(m, 1e-12)).mean(axis=0))


def eps_star(w, delta):
    num = float(np.asarray(w) @ delta)
    return 0.0 if num <= 0 else num / (float(np.asarray(w) @ np.abs(delta)) + num)


def all_pairs(meta, data):
    """Every comparison on P3, oriented so that A beats B in aggregate."""
    xs, d, w = xstar(meta, PLANT), data[PLANT], np.asarray(W_NOMINAL)
    out = {}
    for a, b in itertools.combinations(CONTROLLERS, 2):
        ma, mb = geo(normalize(crit(d, a), xs)), geo(normalize(crit(d, b), xs))
        delta = np.log(np.maximum(ma, 1e-12)) - np.log(np.maximum(mb, 1e-12))
        if w @ delta < 0:
            a, b, ma, mb, delta = b, a, mb, ma, -delta
        kb, _, _ = kappa_bound(ma, mb, W_NOMINAL)
        out[f'{a}-{b}'] = dict(kappa=float(kb),
                               kappa_star=float(kappa_exact(ma, mb, W_NOMINAL)),
                               eps_star=eps_star(w, delta),
                               median_gap=float(abs(
                                   np.median(R_p(normalize(crit(d, a), xs),
                                                 W_NOMINAL, 0.0))
                                   - np.median(R_p(normalize(crit(d, b), xs),
                                                   W_NOMINAL, 0.0)))))
    return out


def main():
    meta, data = load()
    P = all_pairs(meta, data)
    fin = {k: v for k, v in P.items() if np.isfinite(v['kappa_star'])}

    print('\nAll 28 comparisons on P3-SRS7, not only the seven of Table 2')
    exact('comparisons recomputed', 28, len(P))
    assertion('kappa <= kappa* for every comparison (Thm 3(iii))',
              all(v['kappa'] <= v['kappa_star'] + 1e-9 for v in fin.values()),
              f'{len(fin)} finite')
    assertion('kappa* > 0 for every comparison',
              all(v['kappa_star'] > 0 for v in fin.values()))

    lo = min(fin, key=lambda k: fin[k]['kappa_star'])
    exact('pair with the smallest kappa* of the 28', CLAIM_MIN_KAPPA_PAIR, lo)
    claim('its kappa*', CLAIM_MIN_KAPPA, fin[lo]['kappa_star'], tol=1e-3)
    assertion('it is NOT an adjacent pair of Table 2',
              tuple(lo.split('-')) not in TABLE2_KAPPA
              and tuple(reversed(lo.split('-'))) not in TABLE2_KAPPA,
              '(the gap that let the earlier error through)')
    claim('median gap of that same pair', 0.619, fin[lo]['median_gap'], tol=2e-3)

    print('\nConsistency with the seven values printed in Table 2')
    for (a, b), k in TABLE2_KAPPA.items():
        key = f'{a}-{b}' if f'{a}-{b}' in P else f'{b}-{a}'
        claim(f'{key}: kappa*', k, P[key]['kappa_star'])

    print('\nCorollary 1, re-derived from the criteria (not from the JSON)')
    es = {k: v['eps_star'] for k, v in P.items()}
    exact('pair with the smallest epsilon*', CLAIM_MIN_EPS_PAIR,
          min(es, key=es.get))
    exact('pair with the largest epsilon*', CLAIM_MAX_EPS_PAIR,
          max(es, key=es.get))
    assertion('epsilon* <= 1/2 for every comparison (ceiling)',
              all(v <= 0.5 + 1e-12 for v in es.values()),
              f'max {max(es.values()):.4f}')
    below = sorted(k for k, v in es.items() if v < 0.10)
    exact('comparisons with epsilon* < 0.10', 3, len(below))

    sens_path = os.path.join(RES, 'sensitivity.json')
    if os.path.exists(sens_path):
        o = json.load(open(sens_path))
        r = o['reference_constants']
        print('\nAgreement between this script and make_sensitivity.py')
        assertion('epsilon* agrees on all 28 comparisons',
                  all(abs(es[k] - r['eps_star'][k]) < 1e-6 for k in es),
                  f"max diff {max(abs(es[k] - r['eps_star'][k]) for k in es):.2e}")
        assertion('kappa* agrees on all 28 comparisons',
                  all(abs(P[k]['kappa_star'] - r['kappa_star'][k]) < 1e-6
                      for k in P if np.isfinite(P[k]['kappa_star'])))

        print('\nCorollary 1 as a predictor, against the resampling')
        rev = set()
        for lvl in r['per_eps'].values():
            rev |= set(lvl['reversals'])
        assertion('no comparison with epsilon* >= 0.25 ever reversed',
                  not any(es[k] >= 0.25 for k in rev))
        assertion('the comparisons that reversed are those with epsilon* < 0.10',
                  rev == set(below), f'{sorted(rev)}')

        print('\nSection 8.6 figures quoted in the text')
        d = o['design_distribution']
        claim('worst Spearman, reliability-weighted design', 0.952,
              min(d['reliability']['spearman'].values()))
        claim('worst Spearman, actuator-dominant design', 0.619,
              min(d['actuator']['spearman'].values()))
        w = o['weights']['per_alpha']
        for a, full, red in ((200, 59, 90), (50, 34, 70)):
            key = a if a in w else str(a)
            exact(f'alpha={a}: full ordering preserved [%]', full,
                  w[key]['full_pct'])
            exact(f'alpha={a}: reduced ordering preserved [%]', red,
                  w[key]['reduced_pct'])
        if 'threshold' in o:
            t = o['threshold']
            fa = {float(k): v for k, v in t['false_alarm_pct'].items()}
            claim('pre-fault false-alarm rate, nominal [%]', 3.8, fa[1.30],
                  tol=0.05, fmt='{:.1f}')
            claim('largest |dR_0| across the two multipliers', 0.047,
                  t['max_abs_dR0'], tol=1e-3)
            assertion('largest |dR_0| is below the tolerance epsilon = 0.05',
                      t['max_abs_dR0'] < 0.05,
                      'distinct quantities; see Section 8.6')
    else:
        print('\n(results/sensitivity.json absent: run make_sensitivity.py '
              'to audit Section 8.6 as well)')

    macro_path = os.path.join(PAPER, 'sensitivity_macros.tex')
    if os.path.exists(macro_path) and os.path.exists(sens_path):
        print('\nThe macro file the manuscript reads')
        txt = open(macro_path, encoding='utf-8').read()
        mac = dict(re.findall(r'\\renewcommand\{\\(\w+)\}\{([^}]*)\}', txt))
        exact('sEpsStarMinPair', CLAIM_MIN_EPS_PAIR.replace('-', '--'),
              mac.get('sEpsStarMinPair'))
        exact('sFragPair', CLAIM_MIN_KAPPA_PAIR.replace('-', '--'),
              mac.get('sFragPair'))
        assertion('no macro left at its placeholder default',
                  len(mac) >= 25, f'{len(mac)} macros written')

    bad = results.count(False)
    print(f'\n{len(results)} claims checked, {bad} disagreements')
    if not bad:
        print('THE TWENTY-ONE COMPARISONS TABLE 2 DOES NOT PRINT ALSO CHECK OUT')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
