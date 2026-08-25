"""
Sample-complexity bounds and the statistical decision procedure of Section 7.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def n_hoeffding(eps, delta):
    """Theorem 4(i): episodes for an absolute claim on a [0,1] score."""
    return int(np.ceil(np.log(2.0 / delta) / (2.0 * eps ** 2)))


def n_bernstein(eps, delta, sigma, b=2.0):
    """Theorem 4(ii): episodes for a paired comparative claim."""
    return int(np.ceil((2.0 * sigma ** 2 + (2.0 / 3.0) * b * eps)
                       * np.log(2.0 / delta) / eps ** 2))


def n_bernstein_floor(eps, delta, b=2.0):
    """Limit of the Bernstein budget as the paired variance vanishes."""
    return int(np.ceil((2.0 / 3.0) * b * np.log(2.0 / delta) / eps))


def empirical_bernstein(x, delta, lo=0.0, hi=1.0):
    """Maurer-Pontil empirical Bernstein half-width for a sample in [lo, hi]."""
    x = np.asarray(x, float)
    n = x.size
    rng = hi - lo
    v = np.var(x, ddof=1) / rng ** 2
    L = np.log(4.0 / delta)
    return rng * (np.sqrt(2.0 * v * L / n) + 7.0 * L / (3.0 * (n - 1)))


def cliffs_delta(a, b):
    """Cliff's delta for paired or unpaired samples (ordinal dominance)."""
    a = np.asarray(a, float)[:, None]
    b = np.asarray(b, float)[None, :]
    return float((np.sign(a - b)).mean())


def cliffs_delta_ci(a, b, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(a), n)
        ib = rng.integers(0, len(b), n)
        vals[i] = cliffs_delta(a[ia], b[ib])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def wilcoxon_paired(a, b):
    """Two-sided signed-rank p-value; returns 1.0 for an all-zero difference."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0.0):
        return 1.0
    return float(wilcoxon(a, b, zero_method='wilcox', alternative='two-sided',
                          method='approx').pvalue)


def holm(pvals):
    """Holm's sequentially rejective adjusted p-values."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k])
        adj[k] = min(run, 1.0)
    return adj
