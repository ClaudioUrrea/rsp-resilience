"""
The resilience functional R_p and the rank-invariance quantities of Section 6.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import numpy as np

CRITERIA = ['C1-Tracking', 'C2-Detection', 'C3-Reconfig', 'C4-Energy',
            'C5-RealTime', 'C6-Constraint']

# Reference constants and nominal weights of Section 8.1.
#
#   x1* = 5 mm        task tolerance on the post-fault tracking error
#   x2* = 0.30 s      detection latency; raw latencies are censored at X2_CAP
#   x3* = 0.50 s      reconfiguration transient
#   x4* = None        energy: plant-specific, calibrated at run time as
#                     X4_BUDGET times the median nominal power (Section 8.1)
#   x5* = 0.50        execution time as a fraction of the sampling period
#   x6* = 0.05        admissible fraction of the post-fault window in violation
#
# The calibrated x4* is written to results/meta.json by scripts/run_experiments.py
# and read back by scripts/make_analysis.py; these defaults are the values used
# when no calibration file is available.
X2_CAP = 1.20          # declared maximum admissible detection latency [s]
X4_BUDGET = 3.0        # energy budget as a multiple of the median nominal power
XSTAR_DEFAULT = dict(x1=5.0e-3, x2=0.30, x3=0.50, x4=None, x5=0.50, x6=0.05)
W_NOMINAL = np.array([0.25, 0.10, 0.15, 0.10, 0.10, 0.30])


def normalize(X, xstar):
    """Definition 2: m_k = exp(-x_k / x_k^star), with m_k = 0 when x_k = +inf."""
    X = np.asarray(X, float)
    xs = np.asarray(xstar, float)
    with np.errstate(over='ignore', invalid='ignore'):
        m = np.exp(-X / xs)
    return np.where(np.isfinite(X), m, 0.0)


def R_p(m, w, p=0.0):
    """Weighted power mean of order p; p = 0 is the weighted geometric mean."""
    m = np.asarray(m, float)
    w = np.asarray(w, float)
    zero = (m <= 0).any(axis=-1)
    if p == 0.0:
        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.exp(np.sum(w * np.log(np.where(m > 0, m, 1.0)), axis=-1))
    else:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            s = np.sum(w * np.where(m > 0, m, 1.0) ** p, axis=-1)
            val = s ** (1.0 / p)
        if p > 0:                       # A5 fails: zeros do not annihilate
            s = np.sum(w * m ** p, axis=-1)
            return s ** (1.0 / p)
    return np.where(zero, 0.0, val)


def s_spread(delta):
    return 0.5 * (np.max(delta, axis=-1) - np.min(delta, axis=-1))


def kappa_bound(mA, mB, w):
    """Sufficient rank-invariance margin kappa = <w, delta> / s(delta)."""
    d = np.log(mA) - np.log(mB)
    num = float(np.dot(w, d))
    s = float(s_spread(d))
    if s <= 1e-12:
        return np.inf, num, d
    return abs(num) / s, num, d


def kappa_exact(mA, mB, w):
    """Exact l1 distance from w to the rank-reversal set inside the simplex.

    Reversal requires transferring weight from coordinates with large delta to
    the coordinate with the smallest delta.  Moving mass t from coordinate j to
    the argmin coordinate reduces <w, delta> by t (delta_j - delta_min) at an
    l1 cost 2t, so the optimum is the greedy fill of the coordinates ordered by
    decreasing delta, subject to the capacities w_j.  Returns +inf when no
    admissible weight vector reverses the comparison, which by Theorem 3(ii)
    happens exactly when min_k delta_k > 0.
    """
    d = np.log(np.asarray(mA, float)) - np.log(np.asarray(mB, float))
    w = np.asarray(w, float)
    gap = float(np.dot(w, d))
    if gap < 0:
        d, gap = -d, -gap
    dmin = float(np.min(d))
    if dmin > 0:
        return np.inf
    order = np.argsort(-d)
    need, cost = gap, 0.0
    for j in order:
        rate = d[j] - dmin
        if rate <= 1e-15:
            continue
        take = min(w[j], need / rate)
        cost += 2.0 * take
        need -= take * rate
        if need <= 1e-15:
            return cost
    return np.inf


def lipschitz_ratio(mA, mB, w):
    """Empirical ratio in (9): |log R0(A) - log R0(B)| / ||log A - log B||_inf."""
    d = np.log(mA) - np.log(mB)
    num = np.abs(np.sum(w * d, axis=-1))
    den = np.max(np.abs(d), axis=-1)
    return np.where(den > 0, num / np.maximum(den, 1e-15), 0.0)
