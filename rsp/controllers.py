"""
The eight controllers of the numerical study.

Experimental design.  All controllers share (i) the same redundancy-resolution
layer, (ii) the same fault-detection and effectiveness-estimation module, and
(iii) the same actuator-allocation step.  They differ only in the task-space
acceleration command

        xdd_star = law(e, edot, xdd_d, t, internal state)

and in the fidelity of the rigid-body model they are allowed to use.  Isolating
the control law in this way is what makes the comparison of Section 8 a
comparison of control laws rather than of accommodation heuristics.

    1. PID     independent task-space PID, frozen model
    2. CTC     computed torque, full model
    3. SMC     boundary-layer sliding mode, full model
    4. FT1     type-1 Mamdani fuzzy gain scheduling, frozen model
    5. FT2     interval type-2 fuzzy, Nie-Tan reduction, frozen model
    6. ANN     adaptive radial-basis-function network, frozen model + learning
    7. AMPC    receding-horizon LQ with effectiveness-dependent input weight
    8. ADRC    linear extended-state observer + state feedback

(The labels C1-C6 are reserved throughout for the evaluation criteria of
Section 3.2 and are never used for controllers.)

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

CONTROLLERS = ['PID', 'CTC', 'SMC', 'FT1', 'FT2', 'ANN', 'AMPC', 'ADRC']
FULL_MODEL = {'CTC', 'SMC', 'AMPC'}


# ----------------------------------------------------------------------
def _tri(x, a, b, c):
    """Triangular membership function, vectorised."""
    return np.clip(np.minimum((x - a) / (b - a + 1e-12), (c - x) / (c - b + 1e-12)), 0.0, 1.0)


class Controller:
    """Base class: holds the per-episode state of one control law."""

    def __init__(self, name, plant, B, dt, rng, wn=12.0, zeta=1.0):
        self.name = name
        self.plant = plant
        self.B, self.dt = B, dt
        self.n = plant.ndim
        self.kp = wn ** 2
        self.kd = 2.0 * zeta * wn
        self.ki = 25.0
        self.eint = np.zeros((B, self.n))
        self.rng = rng
        if name == 'ANN':
            self.nc = 9
            g = np.linspace(-0.06, 0.06, 3)
            gv = np.linspace(-0.35, 0.35, 3)
            C = np.array([[a, b] for a in g for b in gv])
            self.centers = C
            self.width = 0.09
            self.W = np.zeros((B, self.nc, self.n))
            self.gamma = 45.0
            self.sigma = 0.6
        if name == 'AMPC':
            self.Kgrid, self.lamgrid = self._lq_gains(dt)
        if name == 'ADRC':
            self.z = np.zeros((B, 3, self.n))     # [x_hat, xd_hat, f_hat]
            self.wo = 55.0
            self.b0 = 1.0

    # -- offline design ------------------------------------------------
    @staticmethod
    def _lq_gains(dt, horizon_weight=1.0):
        """Discrete LQ gains of a double integrator for a grid of effectiveness."""
        A = np.array([[1.0, dt], [0.0, 1.0]])
        Bm = np.array([[0.5 * dt * dt], [dt]])
        Q = np.diag([600.0, 6.0]) * horizon_weight
        lams = np.linspace(0.10, 1.0, 10)
        Ks = []
        for lam in lams:
            R = np.array([[2.0e-3 / max(lam, 0.1) ** 2]])
            P = solve_discrete_are(A, Bm, Q, R)
            K = np.linalg.solve(R + Bm.T @ P @ Bm, Bm.T @ P @ A)
            Ks.append(K[0])
        return np.array(Ks), lams

    # -- control law ---------------------------------------------------
    def command(self, e, ed, xdd_d, lam_hat, detected):
        """Return the desired task acceleration (B, n)."""
        nm = self.name
        if nm == 'PID':
            self.eint = np.clip(self.eint + e * self.dt, -0.25, 0.25)
            return xdd_d + self.kp * e + self.kd * ed + self.ki * self.eint

        if nm == 'CTC':
            return xdd_d + self.kp * e + self.kd * ed

        if nm == 'SMC':
            lam_s = 9.0
            s = ed + lam_s * e
            phi = 0.06
            return xdd_d + lam_s * ed + 70.0 * np.clip(s / phi, -1.0, 1.0)

        if nm in ('FT1', 'FT2'):
            ae = np.abs(e) / 0.05
            av = np.abs(ed) / 0.35
            if nm == 'FT1':
                mS, mM, mB = _tri(ae, -1, 0, 1), _tri(ae, 0, 1, 2), _tri(ae, 1, 2, 3)
                vS, vM, vB = _tri(av, -1, 0, 1), _tri(av, 0, 1, 2), _tri(av, 1, 2, 3)
            else:
                # interval type-2: lower and upper firing strengths, Nie-Tan
                mSl, mMl, mBl = _tri(ae, -0.8, 0, 0.8), _tri(ae, 0.2, 1, 1.8), _tri(ae, 1.2, 2, 2.8)
                mSu, mMu, mBu = _tri(ae, -1.2, 0, 1.2), _tri(ae, -0.2, 1, 2.2), _tri(ae, 0.8, 2, 3.2)
                vSl, vMl, vBl = _tri(av, -0.8, 0, 0.8), _tri(av, 0.2, 1, 1.8), _tri(av, 1.2, 2, 2.8)
                vSu, vMu, vBu = _tri(av, -1.2, 0, 1.2), _tri(av, -0.2, 1, 2.2), _tri(av, 0.8, 2, 3.2)
                mS, mM, mB = 0.5 * (mSl + mSu), 0.5 * (mMl + mMu), 0.5 * (mBl + mBu)
                vS, vM, vB = 0.5 * (vSl + vSu), 0.5 * (vMl + vMu), 0.5 * (vBl + vBu)
            wsum = mS + mM + mB + 1e-9
            kp_g = (0.6 * mS + 1.0 * mM + 1.7 * mB) / wsum
            vsum = vS + vM + vB + 1e-9
            kd_g = (0.7 * vS + 1.0 * vM + 1.5 * vB) / vsum
            return xdd_d + self.kp * kp_g * e + self.kd * kd_g * ed

        if nm == 'ANN':
            z = np.stack([e, ed], axis=-1)                       # (B,n,2)
            d2 = ((z[:, :, None, :] - self.centers[None, None]) ** 2).sum(-1)
            phi = np.exp(-d2 / (2.0 * self.width ** 2))          # (B,n,nc)
            phi = phi / (phi.sum(-1, keepdims=True) + 1e-9)
            s = ed + 9.0 * e
            comp = np.einsum('bnc,bcn->bn', phi, self.W)
            self.W += self.dt * (self.gamma * np.einsum('bnc,bn->bcn', phi, s)
                                 - self.sigma * self.W)
            self.W = np.clip(self.W, -60.0, 60.0)
            return xdd_d + self.kp * e + self.kd * ed + np.clip(comp, -40.0, 40.0)

        if nm == 'AMPC':
            lam_eff = np.clip(lam_hat.mean(axis=1), 0.10, 1.0)
            i = np.clip(np.searchsorted(self.lamgrid, lam_eff) - 1, 0, len(self.lamgrid) - 2)
            w = (lam_eff - self.lamgrid[i]) / (self.lamgrid[i + 1] - self.lamgrid[i])
            K = (1 - w)[:, None] * self.Kgrid[i] + w[:, None] * self.Kgrid[i + 1]
            return xdd_d + K[:, 0:1] * e + K[:, 1:2] * ed

        if nm == 'ADRC':
            xh, vh, fh = self.z[:, 0], self.z[:, 1], self.z[:, 2]
            err = -e                                             # x - x_d, sign convention
            eo = xh - err
            b1, b2, b3 = 3 * self.wo, 3 * self.wo ** 2, self.wo ** 3
            u_prev = getattr(self, '_u_prev', np.zeros_like(e))
            self.z[:, 0] = xh + self.dt * (vh - b1 * eo)
            self.z[:, 1] = vh + self.dt * (fh + self.b0 * u_prev - b2 * eo)
            self.z[:, 2] = np.clip(fh + self.dt * (-b3 * eo), -300.0, 300.0)
            u = self.kp * e + self.kd * ed - self.z[:, 2] / self.b0
            self._u_prev = u
            return xdd_d + u

        raise KeyError(self.name)
