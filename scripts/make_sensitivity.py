#!/usr/bin/env python3
"""
Section 8.5 of the manuscript: what the reported ordering depends on.

    python scripts/make_sensitivity.py                 # everything
    python scripts/make_sensitivity.py --recompute     # skip the re-runs

Four analyses.  Three are exact recomputations on the deposited episode
criteria and take seconds:

    reference constants   Proposition 2 and Corollary 1
    weights               Dirichlet resampling of w, Theorem 3
    design distribution   reweighting of the stored fault-class index

The fourth changes what is simulated and needs the campaign run again at
two other detection multipliers; that is the expensive part.  Both extra
campaigns are cached in results/raw_<plant>__thr<mult>.npz, so a second
invocation reuses them.

Writes results/sensitivity.json and paper/sensitivity_macros.tex; the
manuscript reads the second, so no number in Section 8.5 is transcribed
by hand.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsp.score import normalize, R_p, kappa_exact, W_NOMINAL, X2_CAP
from rsp.margin import task_margin
from rsp.plants import make_plant
from rsp.controllers import CONTROLLERS
from rsp.simulate import run_cell, calibrate_threshold, DEFAULT_CFG, NO_FAULT
from rsp.faults import N_FAULTS

RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
PAPER = os.path.join(ROOT, 'paper')

plt.rcParams.update({'font.size': 8.5, 'figure.dpi': 160, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8.5,
                     'legend.frameon': False})

# The figures of this script must match those of make_analysis.py, which sets
# Palatino (the body face of the MDPI class) with TeX Gyre Pagella as the
# metrically compatible fallback.  Importing that module applies the setting;
# if it is unavailable the local fallback keeps the same family list.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from make_analysis import _setup_fonts           # noqa: F401  (side effect)
except Exception:                                    # pragma: no cover
    _SERIF = ['Palatino Linotype', 'Palatino', 'TeX Gyre Pagella',
              'URW Palladio L', 'P052', 'DejaVu Serif']
    import matplotlib.font_manager as _fm
    _have = {f.name for f in _fm.fontManager.ttflist}
    _pick = next((f for f in _SERIF if f in _have), 'DejaVu Serif')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': _SERIF,
                         'mathtext.fontset': 'custom', 'mathtext.rm': _pick,
                         'mathtext.it': f'{_pick}:italic',
                         'mathtext.bf': f'{_pick}:bold', 'mathtext.cal': _pick,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

REF_PLANT = 'P3-SRS7'
NOMINAL_MULT = 1.30          # Section 8.1; thresholds are linear in it
ALT_MULTS = (1.15, 1.50)
SEED = 20260811


# ----------------------------------------------------------------------
# loading and scoring, following make_analysis.py exactly
# ----------------------------------------------------------------------

def load(tag=''):
    meta = json.load(open(os.path.join(RES, 'meta.json')))
    data = {}
    for p in meta['plants']:
        f = os.path.join(RES, f'raw_{p}{tag}.npz')
        if os.path.exists(f):
            data[p] = dict(np.load(f))
    return meta, data


def ref_constants(meta, p):
    xs = meta['xstar'][p]
    return np.array([xs['x1'], xs['x2'], xs['x3'],
                     xs['x4'], xs['x5'], xs['x6']])


def raw_matrix(d, c):
    """The six raw criteria, with C2 treated as in make_analysis.scores()."""
    X = d[f'X_{c}'].copy()
    X[:, 1] = np.where(d[f'degraded_{c}'], np.minimum(X[:, 1], X2_CAP), 0.0)
    return X


def R0_of(X, xstar, w=W_NOMINAL):
    return R_p(normalize(X, xstar), w, 0.0)


def ranking(med):
    return sorted(med, key=lambda c: (-med[c], c))


def rho(a, b):
    idx = {c: i for i, c in enumerate(a)}
    return float(spearmanr(range(len(b)), [idx[c] for c in b]).statistic)


def geo_mean_m(X, xstar):
    """The m-vector at which two controllers are compared, as in table_pairs."""
    m = normalize(X, xstar)
    return np.exp(np.log(np.maximum(m, 1e-12)).mean(axis=0))


def delta_of(Xa, Xb, xstar):
    return (np.log(np.maximum(geo_mean_m(Xa, xstar), 1e-12))
            - np.log(np.maximum(geo_mean_m(Xb, xstar), 1e-12)))


def eps_star(w, d):
    """Corollary 1; zero when the comparison is already reversed."""
    num = float(np.asarray(w) @ d)
    return 0.0 if num <= 0 else num / (float(np.asarray(w) @ np.abs(d)) + num)


def oriented_pairs(meta, data, plant):
    """All controller pairs on one plant, oriented so that A beats B."""
    xstar = ref_constants(meta, plant)
    out = []
    for a, b in itertools.combinations(CONTROLLERS, 2):
        Xa, Xb = raw_matrix(data[plant], a), raw_matrix(data[plant], b)
        d = delta_of(Xa, Xb, xstar)
        if np.asarray(W_NOMINAL) @ d < 0:
            a, b, Xa, Xb, d = b, a, Xb, Xa, -d
        out.append((a, b, Xa, Xb, d))
    return out


def latin_hypercube(n, k, eps, rng):
    u = (rng.permuted(np.tile(np.arange(n), (k, 1)), axis=1).T
         + rng.random((n, k))) / n
    return -eps + 2.0 * eps * u


# ----------------------------------------------------------------------
# 1. reference constants -- Proposition 2 and Corollary 1
# ----------------------------------------------------------------------

def reference_constants(meta, data, out, n_draws=200):
    rng = np.random.default_rng([SEED, 1])
    pairs = oriented_pairs(meta, data, REF_PLANT)
    xstar0 = ref_constants(meta, REF_PLANT)
    w = np.asarray(W_NOMINAL)

    per_eps, worst_fill = {}, 0.0
    for eps in (0.10, 0.25):
        E = latin_hypercube(n_draws, 6, eps, rng)

        reversals = {}
        for a, b, Xa, Xb, _ in pairs:
            flips = int(sum(w @ delta_of(Xa, Xb, xstar0 * (1 + e)) <= 0
                            for e in E))
            if flips:
                reversals[f'{a}-{b}'] = flips

        for p, d in data.items():                      # Proposition 2 checked
            xs = ref_constants(meta, p)
            for c in CONTROLLERS:
                X = raw_matrix(d, c)
                r0 = float(np.median(R0_of(X, xs)))
                if not 0.0 < r0 < 1.0:
                    continue
                lo, hi = r0 ** (1 / (1 - eps)), r0 ** (1 / (1 + eps))
                for e in E:
                    v = float(np.median(R0_of(X, xs * (1 + e))))
                    assert lo - 1e-9 <= v <= hi + 1e-9, \
                        f'Proposition 2 violated on {p}/{c}'
                    worst_fill = max(worst_fill,
                                     max(v - lo, hi - v) / (hi - lo))
        per_eps[eps] = dict(reversals=reversals,
                            pairs_preserved=len(pairs) - len(reversals))

    es = {f'{a}-{b}': eps_star(w, d) for a, b, _, _, d in pairs}
    ks = {f'{a}-{b}': float(kappa_exact(geo_mean_m(Xa, xstar0),
                                        geo_mean_m(Xb, xstar0), W_NOMINAL))
          for a, b, Xa, Xb, _ in pairs}
    finite = [k for k in es if np.isfinite(ks[k])]
    conc = int(sum((es[i] - es[j]) * (ks[i] - ks[j]) > 0
                   for i, j in itertools.combinations(finite, 2)))
    total = len(finite) * (len(finite) - 1) // 2

    out['reference_constants'] = dict(
        draws=n_draws, per_eps=per_eps,
        slack_pct=round(100 * (1 - worst_fill), 1),
        eps_star=es, kappa_star=ks,
        eps_star_min=min(es, key=es.get), eps_star_max=max(es, key=es.get),
        concordant=conc, concordant_total=total,
        concordant_of_28=round(28 * conc / total) if total else 0)


# ----------------------------------------------------------------------
# 2. weights -- Dirichlet resampling
# ----------------------------------------------------------------------

def weights(meta, data, out, n_draws=2000):
    rng = np.random.default_rng([SEED, 2])
    xstar = ref_constants(meta, REF_PLANT)
    M = {c: normalize(raw_matrix(data[REF_PLANT], c), xstar)
         for c in CONTROLLERS}

    def order(w):
        return ranking({c: float(np.median(R_p(M[c], w, 0.0)))
                        for c in CONTROLLERS})

    nominal = order(W_NOMINAL)
    ks = out['reference_constants']['kappa_star']
    frag = min(ks, key=lambda k: ks[k] if np.isfinite(ks[k]) else np.inf)
    frag_pair = tuple(frag.split('-'))
    reduced_nom = [c for c in nominal if c not in frag_pair]

    per_alpha = {}
    for alpha in (200, 50):
        W = rng.dirichlet(alpha * np.asarray(W_NOMINAL), size=n_draws)
        full = red = 0
        for wv in W:
            o = order(wv)
            full += (o == nominal)
            red += ([c for c in o if c not in frag_pair] == reduced_nom)
        per_alpha[alpha] = dict(
            mean_l1=float(np.abs(W - np.asarray(W_NOMINAL)).sum(axis=1).mean()),
            full_pct=round(100 * full / n_draws),
            reduced_pct=round(100 * red / n_draws))

    out['weights'] = dict(draws=n_draws, per_alpha=per_alpha,
                          fragile_pair=frag, fragile_kappa=float(ks[frag]))


# ----------------------------------------------------------------------
# 3. design distribution -- reweighting the stored fault-class index
# ----------------------------------------------------------------------

DESIGNS = {
    'reliability': np.array([.35 / 3, .35 / 3, .35 / 3,
                             .25 / 2, .25 / 2,
                             .40 / 3, .40 / 3, .40 / 3]),
    'actuator': np.array([1., 1., 1., 0., 0., 0., 0., 0.]),
}


def weighted_median(x, wt):
    o = np.argsort(x)
    x, wt = np.asarray(x)[o], np.asarray(wt)[o]
    c = np.cumsum(wt) / wt.sum()
    return float(x[int(np.searchsorted(c, 0.5))])


def design_distribution(meta, data, out):
    nominal_med, nominal_order = {}, {}
    for p, d in data.items():
        xs = ref_constants(meta, p)
        nominal_med[p] = {c: float(np.median(R0_of(raw_matrix(d, c), xs)))
                          for c in CONTROLLERS}
        nominal_order[p] = ranking(nominal_med[p])

    res = {'nominal_median_R0': nominal_med}
    for name, q0 in DESIGNS.items():
        q = q0 / q0.sum()
        entry = {'spearman': {}, 'median_R0': {}}
        for p, d in data.items():
            xs, fid = ref_constants(meta, p), d['fid']
            counts = np.bincount(fid, minlength=N_FAULTS).astype(float)
            wt = np.where(counts[fid] > 0, q[fid] / np.maximum(counts[fid], 1.0), 0.0)
            med = {c: weighted_median(R0_of(raw_matrix(d, c), xs), wt)
                   for c in CONTROLLERS}
            entry['median_R0'][p] = med
            entry['spearman'][p] = rho(nominal_order[p], ranking(med))
        mover = max(((p, c, nominal_med[p][c], entry['median_R0'][p][c])
                     for p in data for c in CONTROLLERS),
                    key=lambda t: abs(t[3] - t[2]))
        entry['mover'] = list(mover)
        entry['dead'] = {p: sum(entry['median_R0'][p][c] == 0.0
                                for c in CONTROLLERS) for p in data}
        entry['top3_unchanged'] = all(
            nominal_order[p][:3] == ranking(entry['median_R0'][p])[:3]
            for p in data)
        entry['reversed_pairs'] = sorted({
            f'{a}-{b}' for p in data
            for a, b in itertools.combinations(CONTROLLERS, 2)
            if (nominal_med[p][a] > nominal_med[p][b])
            != (entry['median_R0'][p][a] > entry['median_R0'][p][b])})
        res[name] = entry
    res['mover'] = res['reliability']['mover']
    dead = res['actuator']['dead']
    res['actuator_dead_all'] = [p for p, k in dead.items()
                                if k == len(CONTROLLERS)]
    res['actuator_dead_next'] = max(
        ((p, k) for p, k in dead.items() if k < len(CONTROLLERS)),
        key=lambda t: t[1], default=(None, 0))
    out['design_distribution'] = res


# ----------------------------------------------------------------------
# 4. detection thresholds -- the one analysis that needs the campaign again
# ----------------------------------------------------------------------

def rerun_at_multiplier(mult, meta, episodes=100, calib=48):
    """Repeat the campaign with every threshold scaled by mult / NOMINAL_MULT.

    calibrate_threshold() returns (per-actuator residual thresholds, task-error
    threshold) already multiplied by NOMINAL_MULT, and both are linear in it,
    so rescaling reproduces exactly the campaign that would have been obtained
    with the multiplier set to `mult`.  C5 does not depend on the detector and
    is carried over from the main campaign unchanged.
    """
    tag = f'__thr{mult:.2f}'
    cfg = dict(DEFAULT_CFG)
    scale = mult / NOMINAL_MULT
    fa = {}
    for pname in meta['plants']:
        path = os.path.join(RES, f'raw_{pname}{tag}.npz')
        base = dict(np.load(os.path.join(RES, f'raw_{pname}.npz')))
        plant = make_plant(pname)
        fids = np.repeat(np.arange(N_FAULTS), episodes)
        store, fa[pname], r = {}, {}, None
        for ci, c in enumerate(CONTROLLERS):
            th = calibrate_threshold(plant, c, seed=SEED + 900 + ci,
                                     B=calib, cfg=dict(cfg, T=2.2))
            th = (np.asarray(th[0]) * scale, th[1] * scale)
            r0 = run_cell(plant, c, np.full(calib, NO_FAULT),
                          seed=SEED + 950 + ci, cfg=dict(cfg, T=2.2),
                          thr_override=th)
            fa[pname][c] = float(np.mean(r0['declared']))
            if os.path.exists(path):
                continue
            r = run_cell(plant, c, fids, seed=SEED, thr_override=th, cfg=cfg)
            store[f'X_{c}'] = np.stack(
                [r['x1'], r['x2'], r['x3'], r['x4'],
                 base[f'X_{c}'][:, 4], r['x6']], axis=1)
            store[f'declared_{c}'] = r['declared']
            store[f'degraded_{c}'] = r['degraded']
        if not os.path.exists(path):
            store['fid'] = r['fid']
            np.savez_compressed(path, **store)
        print(f'[thr {mult:.2f}] {pname} done', flush=True)
    return tag, fa


def thresholds(meta, data, out, episodes=100):
    w_med, order0, m2_0, lat0 = {}, {}, {}, []
    for p, d in data.items():
        xs = ref_constants(meta, p)
        w_med[p] = {c: float(np.median(R0_of(raw_matrix(d, c), xs)))
                    for c in CONTROLLERS}
        order0[p] = ranking(w_med[p])
        m2_0[p] = {c: float(np.median(normalize(raw_matrix(d, c), xs)[:, 1]))
                   for c in CONTROLLERS}
        for c in CONTROLLERS:
            dg = d[f'degraded_{c}']
            if dg.any():
                lat0.append(float(np.median(d[f'X_{c}'][dg, 1])))

    _, fa_nom = rerun_at_multiplier(NOMINAL_MULT, meta, episodes)
    entry = dict(false_alarm_pct={NOMINAL_MULT: 100 * float(np.mean(
        [v for pv in fa_nom.values() for v in pv.values()]))},
        spearman={p: 1.0 for p in data}, latency_shift_pct={},
        max_abs_dm2=0.0, max_abs_dR0=0.0, moved_pair=[])

    for mult in ALT_MULTS:
        tag, fa = rerun_at_multiplier(mult, meta, episodes)
        _, alt = load(tag)
        entry['false_alarm_pct'][mult] = 100 * float(np.mean(
            [v for pv in fa.values() for v in pv.values()]))
        lat = []
        for p, d in alt.items():
            xs = ref_constants(meta, p)
            med = {c: float(np.median(R0_of(raw_matrix(d, c), xs)))
                   for c in CONTROLLERS}
            r = rho(order0[p], ranking(med))
            if r < entry['spearman'][p]:
                entry['spearman'][p] = r
                entry['moved_pair'] = [c for c, b in
                                       zip(ranking(med), order0[p]) if c != b][:2]
            for c in CONTROLLERS:
                entry['max_abs_dR0'] = max(entry['max_abs_dR0'],
                                           abs(med[c] - w_med[p][c]))
                m2 = float(np.median(normalize(raw_matrix(d, c), xs)[:, 1]))
                entry['max_abs_dm2'] = max(entry['max_abs_dm2'],
                                           abs(m2 - m2_0[p][c]))
                dg = d[f'degraded_{c}']
                if dg.any():
                    lat.append(float(np.median(d[f'X_{c}'][dg, 1])))
        entry['latency_shift_pct'][mult] = int(round(
            100 * (float(np.median(lat)) / float(np.median(lat0)) - 1)))

    entry['plants_unchanged'] = sum(1 for p in data
                                    if entry['spearman'][p] >= 1.0)
    entry['moved'] = [p for p in data if entry['spearman'][p] < 1.0]
    if entry['moved'] and entry['moved_pair']:
        a, b = entry['moved_pair']
        entry['moved_pair_gap'] = max(abs(w_med[p][a] - w_med[p][b])
                                      for p in entry['moved'])
    out['threshold'] = entry


# ----------------------------------------------------------------------
# 5. cost of the margin (Remark 2)
# ----------------------------------------------------------------------

def margin_cost(out, n=200):
    """Wall-clock cost of one exact evaluation of mu, for Remark 2.

    Follows the calling convention of make_analysis.fig_margin(): the plant
    exposes q_home, ngen, chain, task_rows and nact, and task_margin() takes
    the task Jacobian and the effectiveness vector.
    """
    rng = np.random.default_rng([SEED, 5])
    per_plant = {}
    for pname in ('P3-SRS7', 'P5-UR16e'):
        pl = make_plant(pname)
        Js = []
        for _ in range(n):
            q = pl.q_home + 0.9 * (rng.random(pl.ngen) - 0.5)
            kin = pl.chain.kinematics(q[None])
            Js.append(pl.chain.jacobians(q[None], kin)[2][0][pl.task_rows])
        lam = np.ones(pl.nact)
        task_margin(Js[0], lam)                       # warm-up
        t0 = time.perf_counter()
        for J in Js:
            task_margin(J, lam)
        per_plant[pname] = 1e3 * (time.perf_counter() - t0) / n
    out['margin_ms'] = per_plant


# ----------------------------------------------------------------------
# macro file
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# figure: does epsilon* predict which comparisons reverse?
# ----------------------------------------------------------------------

def fig_sensitivity(o, fname):
    """Figure 8: the two stability indices, and what they predict.

    Panel (a) orders the twenty-eight comparisons of P3 by epsilon* and marks
    the ones that actually reversed under resampled reference constants.  The
    prediction of Corollary 1 is that nothing to the right of a perturbation
    level can reverse at that level; what the data add is that nothing to the
    left survived either, so the sufficient condition happens to be sharp here.
    Panel (b) puts epsilon* against kappa*: the two indices perturb different
    objects and agree about where the fragility is.
    """
    from matplotlib.lines import Line2D

    DASH, RED, GREY = '#1f77b4', 'crimson', '0.45'
    r = o['reference_constants']
    es, ks = r['eps_star'], r['kappa_star']
    rev = set()
    for lvl in r['per_eps'].values():
        rev |= set(lvl['reversals'])

    keys = sorted(es, key=es.get)
    y = np.arange(len(keys))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 4.3),
                                 gridspec_kw={'width_ratios': [1.15, 1.0]})

    # ---- (a) epsilon* per comparison --------------------------------
    ax.barh(y, [es[k] for k in keys], height=0.72,
            color=[RED if k in rev else GREY for k in keys])
    # the rules stop below the label band, so that no line crosses a number
    top, band = len(keys) + 0.4, len(keys) - 0.45
    for lv, st in ((0.10, '--'), (0.25, ':')):
        ax.vlines(lv, -0.8, band, color=DASH, lw=1.0, ls=st, zorder=0)
        ax.text(lv, band, fr'$\epsilon={lv:.2f}$', fontsize=7,
                color=DASH, ha='center', va='bottom')
    ax.vlines(0.5, -0.8, band, color='0.25', lw=0.9, zorder=0)
    ax.text(0.5, band, 'Ceiling 1/2', fontsize=7, ha='center', va='bottom',
            color='0.25')
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace('-', '\u2013') for k in keys], fontsize=7)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.set_ylim(-0.8, top)
    ax.set_xlim(0, 0.56)
    ax.set_xlabel(r'Reference-constant stability index $\epsilon^\star$')
    ax.set_title('(a) Reference-constant stability, P3-SRS7', fontsize=8.5)
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[Line2D([], [], color=RED, lw=5,
                              label='Reversed in resampling'),
                       Line2D([], [], color=GREY, lw=5,
                              label='Never reversed')],
              fontsize=7, loc='lower right', borderaxespad=0.6)

    # ---- (b) epsilon* against kappa* --------------------------------
    fin = [k for k in keys if np.isfinite(ks[k]) and ks[k] > 0 and es[k] > 0]
    bx.scatter([ks[k] for k in fin if k not in rev],
               [es[k] for k in fin if k not in rev],
               s=18, facecolor='none', edgecolor='0.35', lw=0.9,
               label='Never reversed')
    bx.scatter([ks[k] for k in fin if k in rev],
               [es[k] for k in fin if k in rev],
               s=30, color=RED, marker='s', label='Reversed in resampling')
    bx.axhline(0.10, color=DASH, lw=1.0, ls='--', zorder=0)
    bx.set_xscale('log'); bx.set_yscale('log')
    xs = [ks[k] for k in fin]; ys = [es[k] for k in fin]
    bx.set_xlim(min(xs) / 2.2, max(xs) * 2.2)
    bx.set_ylim(min(ys) / 2.6, max(ys) * 2.0)
    # labels placed away from the axhline and from one another
    place = {0: (7, 7), 1: (7, -12), 2: (-8, 8)}
    for n, k in enumerate(sorted((k for k in fin if k in rev), key=ks.get)):
        dx, dy = place.get(n, (7, 7))
        bx.annotate(k.replace('-', '\u2013'), (ks[k], es[k]), fontsize=7,
                    textcoords='offset points', xytext=(dx, dy),
                    ha='left' if dx > 0 else 'right')
    bx.text(0.98, 0.03, r'Dashed line: $\epsilon=0.10$', transform=bx.transAxes,
            fontsize=7, color=DASH, ha='right', va='bottom')
    bx.tick_params(labelsize=7.5)
    bx.set_xlabel(r'Weight-reversal distance $\kappa^\star$')
    bx.set_ylabel(r'Stability index $\epsilon^\star$')
    bx.set_title(f"(b) Concordant on {r['concordant']} of "
                 f"{r['concordant_total']} pairs", fontsize=8.5)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(fname); plt.close(fig)
    return fname


def _renumber(obj):
    """JSON turns dict keys into strings; put the numeric ones back.

    Only needed when --recompute reloads a previous results/sensitivity.json
    to keep its threshold block: 1.30 comes back as "1.30" and every lookup
    by multiplier would miss.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                k = float(k)
            except (TypeError, ValueError):
                pass
            out[k] = _renumber(v)
        return out
    if isinstance(obj, list):
        return [_renumber(v) for v in obj]
    return obj


def write_macros(o):
    L = ['% generated by scripts/make_sensitivity.py -- do not edit']
    add = lambda k, v: L.append(f'\\renewcommand{{\\{k}}}{{{v}}}')
    words = ['none', 'one', 'two', 'three', 'four', 'five',
             'six', 'seven', 'eight']

    if 'threshold' in o:
        t = o['threshold']
        add('sThrFAnom', f"{t['false_alarm_pct'][NOMINAL_MULT]:.1f}")
        add('sThrFAlow', f"{t['false_alarm_pct'][ALT_MULTS[0]]:.1f}")
        add('sThrFAhigh', f"{t['false_alarm_pct'][ALT_MULTS[1]]:.1f}")
        add('sThrLatLow', f"{t['latency_shift_pct'][ALT_MULTS[0]]:+d}")
        add('sThrLatHigh', f"{t['latency_shift_pct'][ALT_MULTS[1]]:+d}")
        add('sThrMtwo', f"{t['max_abs_dm2']:.3f}")
        add('sThrRzero', f"{t['max_abs_dR0']:.3f}")
        add('sThrPlants', words[t['plants_unchanged']])
        if t['moved']:
            add('sThrRhoMoved', f"{min(t['spearman'][p] for p in t['moved']):.3f}")
            if t['moved_pair']:
                add('sThrPair', ' and '.join(sorted(t['moved_pair'])))
            if 'moved_pair_gap' in t:
                add('sThrPairGap', f"{t['moved_pair_gap']:.3f}")

    r = o['reference_constants']
    add('sRefDraws', r['draws'])
    add('sRefTight', f"{r['slack_pct']:.1f}")
    add('sRefPresA', r['per_eps'][0.10]['pairs_preserved'])
    add('sRefPresB', r['per_eps'][0.25]['pairs_preserved'])
    lo = r['per_eps'][0.10]['reversals']
    if lo:
        pair, n = max(lo.items(), key=lambda kv: kv[1])
        add('sRefRevPairA', pair.replace('-', '--'))
        add('sRefRevCountA', n)
    add('sEpsStarMin', f"{r['eps_star'][r['eps_star_min']]:.3f}")
    add('sEpsStarMinPair', r['eps_star_min'].replace('-', '--'))
    add('sEpsStarMax', f"{r['eps_star'][r['eps_star_max']]:.3f}")
    add('sEpsStarMaxPair', r['eps_star_max'].replace('-', '--'))
    add('sEpsBelow', words[sum(v < 0.10 for v in r['eps_star'].values())])
    add('sEpsConc', r['concordant'])
    add('sEpsConcTot', r['concordant_total'])

    for tag, a in (('A', 200), ('B', 50)):
        w = o['weights']['per_alpha'][a]
        add(f'sDirAlpha{tag}', a)
        add(f'sDirLone{tag}', f"{w['mean_l1']:.2f}")
        add(f'sDirFull{tag}', w['full_pct'])
        add(f'sDirSeven{tag}', w['reduced_pct'])
    add('sFragPair', o['weights']['fragile_pair'].replace('-', '--'))
    add('sFragKappa', f"{o['weights']['fragile_kappa']:.3f}")

    d = o['design_distribution']
    add('sDistRhoRel', f"{min(d['reliability']['spearman'].values()):.3f}")
    add('sDistRhoAct', f"{min(d['actuator']['spearman'].values()):.3f}")
    add('sDistRelRev', len(d['reliability']['reversed_pairs']))
    add('sDistActRev', len(d['actuator']['reversed_pairs']))
    add('sDistMover', f"{d['mover'][1]} on {d['mover'][0]}")
    add('sDistMoverNom', f"{d['mover'][2]:.3f}")
    add('sDistMoverAlt', f"{d['mover'][3]:.3f}")
    add('sDistActDead', words[d['actuator_dead_next'][1]])

    if 'margin_ms' in o:
        add('sMarginPthree', f"{o['margin_ms']['P3-SRS7']:.2f}")
        add('sMarginPfive', f"{o['margin_ms']['P5-UR16e']:.2f}")
        # Remark 2 extrapolates to a thirty-joint arm on a six-dimensional
        # task: C(30,5) candidate normals against the C(7,2) of P3, at the
        # same cost per normal.  Derived, not guessed.
        from math import comb
        add('sMarginBigS',
            f"{o['margin_ms']['P3-SRS7'] * comb(30, 5) / comb(7, 2) / 1000:.1f}")

    os.makedirs(PAPER, exist_ok=True)
    with open(os.path.join(PAPER, 'sensitivity_macros.tex'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'wrote {len(L) - 1} macros to paper/sensitivity_macros.tex')


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true',
                    help='skip the two threshold campaigns')
    ap.add_argument('--episodes', type=int, default=100)
    args = ap.parse_args()

    np.seterr(all='ignore')
    meta, data = load()
    if not data:
        sys.exit('no results/raw_*.npz found; run scripts/run_experiments.py first')

    prev = os.path.join(RES, 'sensitivity.json')
    out = (_renumber(json.load(open(prev)))
           if args.recompute and os.path.exists(prev) else {})
    reference_constants(meta, data, out)
    weights(meta, data, out)
    design_distribution(meta, data, out)
    try:
        margin_cost(out)
    except Exception as exc:                 # timing only; never fatal
        print(f'margin timing skipped: {exc}', file=sys.stderr)
    if not args.recompute:
        thresholds(meta, data, out, args.episodes)

    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    fig_sensitivity(out, os.path.join(FIG, 'fig_sensitivity.pdf'))

    with open(os.path.join(RES, 'sensitivity.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    write_macros(out)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
