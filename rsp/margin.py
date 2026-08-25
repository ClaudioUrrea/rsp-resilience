"""
The post-fault task margin of Section 4, computed exactly.

The attainable task velocities under componentwise torque bounds form the
zonotope

    V(q, Lambda) = { J(q) Lambda u : ||u||_inf <= 1 }  subset R^n,

whose support function is h(z) = || Lambda J(q)^T z ||_1.  The margin is the
Chebyshev radius

    mu(q, Lambda) = min_{||z||_2 = 1} || Lambda J(q)^T z ||_1,

i.e. the distance from the origin to the nearest facet hyperplane.  Because the
facet normals of a zonotope are exactly the directions orthogonal to n-1
linearly independent generators, the minimum is attained on a finite candidate
set and can be enumerated exactly.  This is what :func:`task_margin` does.

Sampling the sphere instead, as a first implementation of this study did, only
ever *overestimates* the minimum, and the overestimate can exceed the upper
bound of Proposition 1 for n = 3; the figure that verifies the proposition must
therefore not be produced that way.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def task_margin(J, lam=None, tol=1e-12):
    """Exact Chebyshev radius of the zonotope {J diag(lam) u : ||u||_inf<=1}.

    Parameters
    ----------
    J : (n, m) array
        Task Jacobian.
    lam : (m,) array or None
        Actuator effectiveness vector in [0, 1]; ``None`` means all ones.
    tol : float
        Generators of Euclidean norm below ``tol`` are discarded.

    Returns
    -------
    float
        ``mu(q, Lambda)``; ``0.0`` when the surviving generators do not span
        R^n, which is the case in which the task is no longer feasible.
    """
    J = np.asarray(J, float)
    n, m = J.shape
    G = J if lam is None else J * np.asarray(lam, float)[None, :]
    keep = np.linalg.norm(G, axis=0) > tol
    G = G[:, keep]
    if G.shape[1] < n or np.linalg.matrix_rank(G, tol=1e-10) < n:
        return 0.0
    if n == 1:
        return float(np.abs(G).sum())
    best = np.inf
    for cols in combinations(range(G.shape[1]), n - 1):
        A = G[:, list(cols)]
        if np.linalg.matrix_rank(A, tol=1e-10) < n - 1:
            continue
        # unit normal to the span of the chosen generators
        z = np.linalg.svd(A.T)[2][-1]
        best = min(best, float(np.abs(G.T @ z).sum()))
    return best if np.isfinite(best) else 0.0


def sigma_min_reduced(J, i):
    """Smallest singular value of J with its i-th column deleted."""
    return float(np.linalg.svd(np.delete(np.asarray(J, float), i, axis=1),
                               compute_uv=False).min())
