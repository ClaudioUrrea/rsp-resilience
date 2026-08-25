#!/usr/bin/env python3
"""
Turn the raw episode criteria into the tables and figures of Section 8.

    python scripts/make_analysis.py

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rsp.score import normalize, R_p, kappa_bound, kappa_exact, lipschitz_ratio, \
    W_NOMINAL, CRITERIA, X2_CAP
from rsp.margin import task_margin, sigma_min_reduced
from scipy.stats import spearmanr
from rsp.stats import (n_hoeffding, n_bernstein, n_bernstein_floor,
                       empirical_bernstein, cliffs_delta, wilcoxon_paired, holm)
from rsp.plants import make_plant
from rsp.controllers import CONTROLLERS
from rsp.faults import FAULT_NAMES

RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
TAB = os.path.join(ROOT, 'tables')
plt.rcParams.update({'font.size': 8.5, 'figure.dpi': 160, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 8.5,
                     'legend.frameon': False})
# ---------------------------------------------------------------------
# Typography.  The manuscript is set in Palatino (the MDPI class uses it for
# body text), so the figures use Palatino Linotype where it is installed and
# fall back to TeX Gyre Pagella, the metrically compatible clone shipped with
# TeX Live, and finally to whatever serif face is available.
def _setup_fonts():
    import glob
    import matplotlib.font_manager as fm
    for pat in ('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella*.otf',
                '/usr/share/fonts/**/texgyrepagella*.otf',
                '/usr/local/share/fonts/**/texgyrepagella*.otf',
                'C:/Windows/Fonts/pala*.ttf'):
        for f in glob.glob(pat, recursive=True):
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
    serif = ['Palatino Linotype', 'Palatino', 'TeX Gyre Pagella',
             'URW Palladio L', 'P052', 'DejaVu Serif']
    have = {f.name for f in fm.fontManager.ttflist}
    pick = next((f for f in serif if f in have), 'DejaVu Serif')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': serif,
        'mathtext.fontset': 'custom',
        'mathtext.rm': pick,
        'mathtext.it': f'{pick}:italic',
        'mathtext.bf': f'{pick}:bold',
        'mathtext.cal': pick,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    if pick in ('DejaVu Serif', 'DejaVu Sans'):
        import warnings
        warnings.warn(
            'Neither Palatino Linotype nor TeX Gyre Pagella was found; the '
            'figures will not match the typography of the manuscript. On '
            'Windows Palatino Linotype ships with the system (pala.ttf); on '
            'Linux install the fonts-texgyre package. If you have just '
            'installed a font, delete the matplotlib cache '
            '(matplotlib.get_cachedir()) and rerun.', RuntimeWarning)
    print(f'figures typeset in: {pick}')
    return pick


_FONT = _setup_fonts()



def load():
    meta = json.load(open(os.path.join(RES, 'meta.json')))
    data = {}
    for p in meta['plants']:
        f = os.path.join(RES, f'raw_{p}.npz')
        if os.path.exists(f):
            data[p] = dict(np.load(f))
    return meta, data


def ref_constants(meta, p):
    """The constants x_k^star of Definition 2, exactly as calibrated.

    run_experiments.py writes them to results/meta.json already in the form
    used for scoring (x_4^star is the energy budget, not the raw median nominal
    power), so nothing is rescaled here.  The detection-latency cap X2_CAP is a
    property of criterion C2 and is applied in scores(), not to the constants.
    """
    xs = meta['xstar'][p]
    return np.array([xs['x1'], xs['x2'], xs['x3'], xs['x4'],
                     xs['x5'], xs['x6']])


def scores(meta, data):
    """m-vectors and R_p for every plant, controller and order p."""
    out = {}
    for p, d in data.items():
        xstar = ref_constants(meta, p)
        for c in CONTROLLERS:
            X = d[f'X_{c}'].copy()
            # detection latency is meaningful only for consequential faults:
            # an episode whose tracking never left the pre-fault envelope gave
            # the supervisory logic nothing to declare and is not penalised.
            degr = d[f'degraded_{c}']
            X[:, 1] = np.where(degr, np.minimum(X[:, 1], X2_CAP), 0.0)
            m = normalize(X, xstar)
            out[(p, c)] = dict(m=m,
                               R0=R_p(m, W_NOMINAL, 0.0),
                               Rm1=R_p(m, W_NOMINAL, -1.0),
                               Rm2=R_p(m, W_NOMINAL, -2.0),
                               R1=R_p(m, W_NOMINAL, 1.0))
    return out


# ----------------------------------------------------------------------
def table_scores(meta, data, S, fh):
    fh.write('% T4: median [IQR] of R_0 per plant and controller\n')
    fh.write('\\begin{tabular}{l' + 'c' * len(data) + '}\n\\toprule\n')
    fh.write('Controller & ' + ' & '.join(data.keys()) + ' \\\\\n\\midrule\n')
    for c in CONTROLLERS:
        row = [c]
        for p in data:
            r = S[(p, c)]['R0']
            q1, q2, q3 = np.percentile(r, [25, 50, 75])
            row.append(f'{q2:.3f} [{q1:.3f}, {q3:.3f}]')
        fh.write(' & '.join(row) + ' \\\\\n')
    fh.write('\\midrule\n\\multicolumn{%d}{l}{\\itshape '
             'outright-failure rate (fraction of episodes with $R_0=0$)}\\\\\n'
             % (len(data) + 1))
    for c in CONTROLLERS:
        row = [c] + [f"{(S[(p, c)]['R0'] == 0).mean():.3f}" for p in data]
        fh.write(' & '.join(row) + ' \\\\\n')
    fh.write('\\bottomrule\n\\end{tabular}\n\n')


def table_pairs(meta, data, S, fh, plant):
    """Wilcoxon + Holm + Cliff's delta + kappa for all controller pairs."""
    pairs = list(itertools.combinations(CONTROLLERS, 2))
    pv, rows = [], []
    for a, b in pairs:
        ra, rb = S[(plant, a)]['R0'], S[(plant, b)]['R0']
        pv.append(wilcoxon_paired(ra, rb))
        ma = np.exp(np.log(np.maximum(S[(plant, a)]['m'], 1e-12)).mean(axis=0))
        mb = np.exp(np.log(np.maximum(S[(plant, b)]['m'], 1e-12)).mean(axis=0))
        kb, gap, _ = kappa_bound(ma, mb, W_NOMINAL)
        ke = kappa_exact(ma, mb, W_NOMINAL)
        rows.append((a, b, np.median(ra) - np.median(rb),
                     cliffs_delta(ra, rb), kb, ke))
    adj = holm(pv)
    fh.write(f'% T5: pairwise comparison on {plant}\n')
    fh.write('\\begin{tabular}{llrrrrr}\n\\toprule\n')
    fh.write('A & B & $\\Delta$median & Cliff $\\delta$ & $p$ & $p_{\\rm Holm}$'
             ' & $\\kappa^\\star$ \\\\\n\\midrule\n')
    for (a, b, dm, cd, kb, ke), p0, pa in zip(rows, pv, adj):
        ks = '$\\infty$' if not np.isfinite(ke) else f'{ke:.3f}'
        fh.write(f'{a} & {b} & {dm:+.3f} & {cd:+.3f} & {p0:.2e} & {pa:.2e} '
                 f'& {ks} \\\\\n')
    fh.write('\\bottomrule\n\\end{tabular}\n\n')
    return rows, pv, adj


def table_budget(fh, S, data):
    eps, dl = 0.05, 0.05
    fh.write('% T6: episode budgets\n\\begin{tabular}{lrr}\n\\toprule\n')
    fh.write('Claim & bound & $N$ \\\\\n\\midrule\n')
    fh.write(f'absolute, $\\epsilon=0.05$, $\\delta=0.05$ & Hoeffding & '
             f'{n_hoeffding(eps, dl)} \\\\\n')
    sig = []
    for p in data:
        for a, b in itertools.combinations(CONTROLLERS, 2):
            sig.append(np.std(S[(p, a)]['R0'] - S[(p, b)]['R0'], ddof=1))
    smed, smax = float(np.median(sig)), float(np.max(sig))
    fh.write(f'paired, $\\sigma_D={smed:.3f}$ (observed median) & Bernstein & '
             f'{n_bernstein(eps, dl, smed)} \\\\\n')
    fh.write(f'paired, $\\sigma_D={smax:.3f}$ (observed maximum) & Bernstein & '
             f'{n_bernstein(eps, dl, smax)} \\\\\n')
    fh.write(f'paired, $\\sigma_D=0.05$ & Bernstein & '
             f'{n_bernstein(eps, dl, 0.05)} \\\\\n')
    fh.write(f'paired, $\\sigma_D\\to 0$ & Bernstein floor & '
             f'{n_bernstein_floor(eps, dl)} \\\\\n')
    fh.write('\\bottomrule\n\\end{tabular}\n\n')
    return dict(sigma_D_median=smed, sigma_D_max=smax, n_sigma_pairs=len(sig),
                N_absolute=n_hoeffding(eps, dl),
                N_paired_median=n_bernstein(eps, dl, smed),
                N_paired_max=n_bernstein(eps, dl, smax),
                N_paired_floor=n_bernstein_floor(eps, dl))


# ----------------------------------------------------------------------
def fig_margin(fname, n_samples=160, seed=3):
    """Numerical verification of the two-sided bounds (7) of Proposition 1.

    The margin is the Chebyshev radius of the zonotope and is computed exactly
    by facet enumeration (rsp.margin.task_margin).  Sampling directions on the
    sphere instead would overestimate it, and the overestimate can exceed the
    upper bound the figure is meant to verify.  The upper bound sqrt(m-1) is
    plant-specific and is therefore drawn once per plant.
    """
    rng = np.random.default_rng(seed)
    colors = {'P1-Planar3R': '#1f77b4', 'P3-SRS7': '#d62728', 'P5-UR16e': '#2ca02c'}
    marks = {'P1-Planar3R': 'o', 'P3-SRS7': 's', 'P5-UR16e': '^'}
    pts, check = {}, {}
    for pname in colors:
        pl = make_plant(pname)
        mu, sg = [], []
        for _ in range(n_samples):
            q = pl.q_home + 0.9 * (rng.random(pl.ngen) - 0.5)
            kin = pl.chain.kinematics(q[None])
            J = pl.chain.jacobians(q[None], kin)[2][0][pl.task_rows]
            i = int(rng.integers(pl.nact))
            lam = np.ones(pl.nact)
            lam[i] = 0.0
            s_min = sigma_min_reduced(J, i)
            if s_min <= 1e-9:
                continue
            mu.append(task_margin(J, lam))
            sg.append(s_min)
        mu, sg = np.array(mu), np.array(sg)
        ratio = mu / sg
        ub = float(np.sqrt(pl.nact - 1))
        pts[pname] = (sg, mu, ub)
        check[pname] = dict(m=int(pl.nact), sqrt_m_1=ub,
                            ratio_min=float(ratio.min()),
                            ratio_max=float(ratio.max()),
                            violations=int((ratio < 1 - 1e-9).sum()
                                           + (ratio > ub + 1e-9).sum()))

    # the frame is set by the data, so that the cone and the samples fill it
    lo = min(sg.min() for sg, _, _ in pts.values()) / 1.6
    hi = max(sg.max() for sg, _, _ in pts.values()) * 1.6
    xs = np.array([lo, hi])

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for pname, (sg, mu, ub) in pts.items():
        ax.plot(sg, mu, marks[pname], ms=2.4, alpha=0.6, color=colors[pname],
                label=pname)
        ax.plot(xs, ub * xs, '--', lw=0.8, color=colors[pname])
    ax.plot(xs, xs, 'k-', lw=0.9, label=r'$\sigma_{\min}$ (lower bound)')
    ax.plot([], [], 'k--', lw=0.8,
            label=r'$\sqrt{m-1}\,\sigma_{\min}$ (upper, per plant)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi * max(ub for _, _, ub in pts.values()))
    ax.set_xlabel(r'Reduced-Jacobian singular value $\sigma_{\min}(J_{-i})$')
    ax.set_ylabel(r'Post-fault task margin $\mu(q,\Lambda)$')
    ax.legend(fontsize=6, loc='upper left')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    return check


def fig_lipschitz(S, data, fname):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    vals = []
    for p in data:
        for a, b in itertools.combinations(CONTROLLERS, 2):
            ma, mb = S[(p, a)]['m'], S[(p, b)]['m']
            ok = (ma > 0).all(1) & (mb > 0).all(1)
            if ok.sum() > 10:
                vals.append(lipschitz_ratio(ma[ok], mb[ok], W_NOMINAL))
    v = np.concatenate(vals)
    ax.hist(v, bins=60, color='0.4')
    ax.axvline(1.0, color='crimson', lw=1.2)
    ax.set_xlabel(r'Ratio $|\log R_0(m)-\log R_0(\tilde m)|\,/\,\|\log m-\log\tilde m\|_\infty$')
    ax.set_ylabel('Episodes')
    ax.set_title(f'Maximum observed ratio = {v.max():.4f}', fontsize=8)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    return float(v.max())


def fig_convergence(S, data, fname, plant='P3-SRS7', ctrl='AMPC'):
    plant = plant if plant in data else list(data)[0]
    r = S[(plant, ctrl)]['R0']
    rng = np.random.default_rng(0)
    r = rng.permutation(r)
    n0 = min(20, max(2, r.size))          # rehearsal runs may have < 20 episodes
    N = np.arange(n0, r.size + 1)
    run = np.cumsum(r)[n0 - 1:] / N
    band = np.sqrt(np.log(2 / 0.05) / (2 * N))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    clip = lambda a: np.clip(a, 0.0, 1.0)      # the score lives in [0,1]
    ax.plot(N, run, color='0.2', lw=1.0, label=r'$\hat R_N$')
    ax.fill_between(N, clip(run[-1] - band), clip(run[-1] + band),
                    color='crimson', alpha=0.18, label='Hoeffding 95%')
    step = max(1, N.size // 40)
    eb = np.array([empirical_bernstein(r[:n], 0.05) for n in N[::step]])
    ax.plot(N[::step], clip(run[-1] + eb), 'b--', lw=0.8, label='Empirical Bernstein')
    ax.plot(N[::step], clip(run[-1] - eb), 'b--', lw=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.axvline(n_hoeffding(0.05, 0.05), color='k', ls=':', lw=0.8)
    ax.set_xlabel('Episodes $N$')
    ax.set_ylabel(r'Running estimate $\hat R_N$')
    ax.set_title(f'{plant}, {ctrl}', fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def fig_simplex(S, data, fname, plant='P3-SRS7', a='ANN', b='CTC'):
    plant = plant if plant in data else list(data)[0]
    ma = np.exp(np.log(np.maximum(S[(plant, a)]['m'], 1e-12)).mean(axis=0))
    mb = np.exp(np.log(np.maximum(S[(plant, b)]['m'], 1e-12)).mean(axis=0))
    d = np.log(ma) - np.log(mb)
    kb, gap, _ = kappa_bound(ma, mb, W_NOMINAL)
    ke = kappa_exact(ma, mb, W_NOMINAL)
    u = np.zeros(6); u[0], u[5] = 1.0, -1.0        # tracking  vs constraint
    v = np.zeros(6); v[2], v[3] = 1.0, -1.0        # reconfig  vs energy
    # the window is scaled to the reversal distance so that the boundary of
    # Theorem 3(ii) and the kappa-ball of (11) are both inside the plotted slice
    lim = float(np.clip(1.6 * (kb / 2.0 if np.isfinite(kb) else 0.25), 0.03, 0.30))
    g = np.linspace(-lim, lim, 241)
    A, Bg = np.meshgrid(g, g)
    Wm = W_NOMINAL[None, None] + A[..., None] * u + Bg[..., None] * v
    val = (Wm * d).sum(-1)
    feas = (Wm >= 0).all(-1)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.contourf(A, Bg, np.where(feas, np.sign(val), np.nan), levels=[-1.5, 0, 1.5],
                colors=['#f2b8b8', '#bcd2e8'])
    ax.contour(A, Bg, np.where(feas, val, np.nan), levels=[0], colors='k', linewidths=1.0)
    l1 = 2 * (np.abs(A) + np.abs(Bg))
    if np.isfinite(kb):
        ax.contour(A, Bg, l1, levels=[kb], colors='crimson', linewidths=1.0,
                   linestyles='--')
    ax.plot(0, 0, 'ko', ms=3)
    ax.set_xlabel(r'Weight transferred, $C_1 \leftrightarrow C_6$')
    ax.set_ylabel(r'Weight transferred, $C_3 \leftrightarrow C_4$')
    ttl = f'{plant}: {a} vs {b}, ' + (r'$\kappa^\star=\infty$' if not np.isfinite(ke)
                                      else fr'$\kappa={kb:.3f}$, $\kappa^\star={ke:.3f}$')
    ax.set_title(ttl, fontsize=7)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    return kb, ke


def fig_axioms(fname):
    """R_p on a two-criterion slice: p>0 violates non-compensability."""
    t = np.linspace(0.0, 1.0, 400)
    m = np.stack([t, np.full_like(t, 0.9)], 1)
    w = np.array([0.5, 0.5])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for p, st in [(2.0, ':'), (1.0, '--'), (0.0, '-'), (-1.0, '-.'),
                  (-2.0, (0, (3, 1, 1, 1)))]:
        ax.plot(t, R_p(m, w, p), linestyle=st, lw=1.1, label=f'$p={p:g}$')
    ax.axhline(0, color='0.7', lw=0.6)
    ax.set_xlabel('First criterion $m_1$ (second held at $0.9$)')
    ax.set_ylabel('Aggregate $R_p(m;w)$')
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def fig_failures(S, data, fname):
    Z = np.zeros((len(CONTROLLERS), len(FAULT_NAMES)))
    for i, c in enumerate(CONTROLLERS):
        acc = []
        for p in data:
            r = S[(p, c)]['R0']
            acc.append(np.array([(r[data[p]['fid'] == k] == 0).mean()
                                 for k in range(len(FAULT_NAMES))]))
        Z[i] = np.mean(acc, axis=0)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    im = ax.imshow(Z, cmap='magma_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(FAULT_NAMES)))
    ax.set_xticklabels([f.split('-')[0] for f in FAULT_NAMES], fontsize=7)
    ax.set_yticks(range(len(CONTROLLERS))); ax.set_yticklabels(CONTROLLERS, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, label='Fraction with $R_0=0$')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    return Z


def fig_aggregators(S, data, fname):
    """Rankings induced by p=1 and p=0 need not agree."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for p in data:
        r0 = [np.median(S[(p, c)]['R0']) for c in CONTROLLERS]
        r1 = [np.median(S[(p, c)]['R1']) for c in CONTROLLERS]
        ax.plot(r1, r0, 'o', ms=3, label=p)
    ax.set_xlabel(r'Weighted arithmetic mean $R_1$ (excluded by A5)')
    ax.set_ylabel(r'Weighted geometric mean $R_0$')
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def sensitivity_to_p(S, data):
    """Rank sensitivity of the score to the order p (Section 8.4 of the paper).

    Within the family admitted by A5 the induced ranking should be nearly
    invariant; crossing to p = 1, which A5 excludes, should not be.  Returns the
    per-plant and pooled Spearman correlations together with the number of
    plants on which the ordering is identical to the one induced by R_0.
    """
    def med(p, key):
        return np.array([np.median(S[(p, c)][key]) for c in CONTROLLERS])

    def order(v):
        return tuple(np.argsort(-np.asarray(v)))

    out = {'per_plant': {}, 'identical_to_R0': {'Rm1': 0, 'Rm2': 0, 'R1': 0}}
    pooled = {k: [] for k in ('R0', 'Rm1', 'Rm2', 'R1')}
    for p in data:
        r = {k: med(p, k) for k in ('R0', 'Rm1', 'Rm2', 'R1')}
        for k in r:
            pooled[k].extend(r[k])
        out['per_plant'][p] = {k: float(spearmanr(r['R0'], r[k]).statistic)
                               for k in ('Rm1', 'Rm2', 'R1')}
        for k in ('Rm1', 'Rm2', 'R1'):
            out['identical_to_R0'][k] += int(order(r['R0']) == order(r[k]))
    out['pooled'] = {k: float(spearmanr(pooled['R0'], pooled[k]).statistic)
                     for k in ('Rm1', 'Rm2', 'R1')}
    rho = [out['per_plant'][p][k] for p in data for k in ('Rm1', 'Rm2')]
    out['admissible_range'] = [float(min(rho)), float(max(rho))]
    return out


def failure_by_class(S, data):
    """Mean outright-failure rate per fault class, pooled over plants."""
    per = {}
    for k, name in enumerate(FAULT_NAMES):
        vals = [ (S[(p, c)]['R0'][data[p]['fid'] == k] == 0).mean()
                 for p in data for c in CONTROLLERS ]
        per[name] = float(np.mean(vals))
    worst = max(per, key=per.get)
    smc_f7 = float(np.mean([ (S[(p, 'SMC')]['R0'][data[p]['fid'] == 6] == 0).mean()
                             for p in data ]))
    smc_f6 = float(np.mean([ (S[(p, 'SMC')]['R0'][data[p]['fid'] == 5] == 0).mean()
                             for p in data ]))
    smc_f8 = float(np.mean([ (S[(p, 'SMC')]['R0'][data[p]['fid'] == 7] == 0).mean()
                             for p in data ]))
    # which criterion annihilates the score for SMC under variable latency
    c3 = {p: float((S[(p, 'SMC')]['m'][data[p]['fid'] == 6][:, 2] == 0).mean())
          for p in data}
    return dict(per_class=per, worst_class=worst,
                smc_latency=dict(F6=smc_f6, F7=smc_f7, F8=smc_f8, C3_zero_rate=c3))


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    meta, data = load()
    S = scores(meta, data)
    summary = {}

    with open(os.path.join(TAB, 'tables.tex'), 'w') as fh:
        table_scores(meta, data, S, fh)
        plant0 = 'P3-SRS7' if 'P3-SRS7' in data else list(data)[0]
        rows, pv, adj = table_pairs(meta, data, S, fh, plant0)
        summary.update(table_budget(fh, S, data))
    summary['n_sig'] = int((adj < 0.05).sum())
    summary['n_pairs'] = len(adj)

    summary['margin_bounds_check'] = fig_margin(os.path.join(FIG, 'fig_margin_bounds.pdf'))
    summary['lipschitz_max'] = fig_lipschitz(S, data, os.path.join(FIG, 'fig_lipschitz.pdf'))
    fig_convergence(S, data, os.path.join(FIG, 'fig_convergence.pdf'))
    kb, ke = fig_simplex(S, data, os.path.join(FIG, 'fig_simplex.pdf'))
    summary['kappa_bound'] = kb
    summary['kappa_exact'] = ke
    fig_axioms(os.path.join(FIG, 'fig_axioms.pdf'))
    Z = fig_failures(S, data, os.path.join(FIG, 'fig_failures.pdf'))
    fig_aggregators(S, data, os.path.join(FIG, 'fig_aggregators.pdf'))

    # headline numbers used in the text
    summary['medians'] = {p: {c: float(np.median(S[(p, c)]['R0'])) for c in CONTROLLERS}
                          for p in data}
    summary['iqr'] = {p: {c: [float(np.percentile(S[(p, c)]['R0'], 25)),
                              float(np.percentile(S[(p, c)]['R0'], 75))]
                          for c in CONTROLLERS} for p in data}
    summary['zero_rate'] = {p: {c: float((S[(p, c)]['R0'] == 0).mean())
                                for c in CONTROLLERS} for p in data}
    summary['rank_flip_p1_vs_p0'] = {}
    for p in data:
        o0 = [c for _, c in sorted(((np.median(S[(p, c)]['R0']), c)
                                    for c in CONTROLLERS), reverse=True)]
        o1 = [c for _, c in sorted(((np.median(S[(p, c)]['R1']), c)
                                    for c in CONTROLLERS), reverse=True)]
        summary['rank_flip_p1_vs_p0'][p] = dict(p0=o0, p1=o1, agree=o0 == o1)
    summary['failure_matrix'] = Z.tolist()
    summary['median_m'] = {p: {c: np.median(S[(p, c)]['m'], axis=0).round(4).tolist()
                               for c in CONTROLLERS} for p in data}
    summary['xstar_used'] = {p: ref_constants(meta, p).tolist() for p in data}
    summary['sensitivity_to_p'] = sensitivity_to_p(S, data)
    summary['failure_by_class'] = failure_by_class(S, data)
    summary['declared_rate'] = {p: {c: float(data[p][f'declared_{c}'].mean())
                                    for c in CONTROLLERS} for p in data}
    json.dump(summary, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('medians', 'iqr', 'zero_rate', 'failure_matrix',
                                   'median_m', 'declared_rate', 'xstar_used')},
                     indent=1))


if __name__ == '__main__':
    main()
