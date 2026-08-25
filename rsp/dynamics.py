"""
Batched rigid-body dynamics for open serial chains in standard Denavit-Hartenberg
form.  All routines operate on a leading batch axis B so that an entire Monte-Carlo
cell (B episodes) advances with one set of NumPy calls.

Conventions
-----------
Standard DH.  For joint i with parameters (a_i, alpha_i, d_i, theta_i):

    T_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)

Revolute joint  -> theta_i = q_i + theta_off_i,  d_i fixed.
Prismatic joint -> d_i     = q_i + d_off_i,      theta_i fixed.

Link i has mass m_i, centre of mass c_i expressed in frame i, and rotational
inertia I_i about the centre of mass expressed in frame i.

The mass matrix is assembled from the per-link body Jacobians,

    M(q) = sum_i ( m_i Jv_i^T Jv_i + Jw_i^T R_i I_i R_i^T Jw_i ),

which is exact and maps onto batched einsum calls.  The bias vector
h(q,qd) = C(q,qd) qd + G(q) is obtained from one recursive Newton-Euler pass
with qdd = 0.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import numpy as np

REVOLUTE = 0
PRISMATIC = 1

def _mv(R, v):
    """Batched matrix-vector product, R (...,3,3) times v (...,3)."""
    return np.einsum('...ij,...j->...i', R, v)


def _cross(a, b):
    """Batched cross product; faster than np.cross for small trailing axes."""
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)



class SerialChain:
    """Batched dynamics of an open serial chain."""

    def __init__(self, a, alpha, d, theta, jtype, mass, com, inertia,
                 gravity=(0.0, 0.0, -9.81), tool=(0.0, 0.0, 0.0)):
        self.a = np.asarray(a, float)
        self.alpha = np.asarray(alpha, float)
        self.d0 = np.asarray(d, float)
        self.th0 = np.asarray(theta, float)
        self.jtype = np.asarray(jtype, int)
        self.m = len(self.a)
        self.mass = np.asarray(mass, float)
        self.com = np.asarray(com, float)          # (m,3) in link frame
        self.inertia = np.asarray(inertia, float)  # (m,3,3) about com, link frame
        self.g = np.asarray(gravity, float)
        self.tool = np.asarray(tool, float)        # tool offset in frame m

    # ------------------------------------------------------------------
    def _link_transforms(self, q):
        """Per-joint rotation R_{i-1,i} (B,m,3,3) and translation p_{i-1,i} (B,m,3)."""
        B = q.shape[0]
        th = np.where(self.jtype == REVOLUTE, q + self.th0, self.th0[None, :])
        dd = np.where(self.jtype == PRISMATIC, q + self.d0, self.d0[None, :])
        ct, st = np.cos(th), np.sin(th)
        ca = np.broadcast_to(np.cos(self.alpha), (B, self.m))
        sa = np.broadcast_to(np.sin(self.alpha), (B, self.m))
        aa = np.broadcast_to(self.a, (B, self.m))

        R = np.empty((B, self.m, 3, 3))
        R[..., 0, 0] = ct
        R[..., 0, 1] = -st * ca
        R[..., 0, 2] = st * sa
        R[..., 1, 0] = st
        R[..., 1, 1] = ct * ca
        R[..., 1, 2] = -ct * sa
        R[..., 2, 0] = 0.0
        R[..., 2, 1] = sa
        R[..., 2, 2] = ca

        p = np.empty((B, self.m, 3))
        p[..., 0] = aa * ct
        p[..., 1] = aa * st
        p[..., 2] = dd
        return R, p

    def kinematics(self, q):
        """Forward kinematics in the base frame.

        Returns dict with
            R0   (B,m,3,3)  orientation of frame i
            o    (B,m,3)    origin of frame i
            z    (B,m,3)    joint axis i (z of frame i-1)
            c    (B,m,3)    link centre of mass
            ee   (B,3)      tool point
        """
        B = q.shape[0]
        Rl, pl = self._link_transforms(q)
        R0 = np.empty((B, self.m, 3, 3))
        o = np.empty((B, self.m, 3))
        z = np.empty((B, self.m, 3))
        Rprev = np.broadcast_to(np.eye(3), (B, 3, 3))
        oprev = np.zeros((B, 3))
        for i in range(self.m):
            z[:, i] = Rprev[:, :, 2]
            oprev = oprev + _mv(Rprev, pl[:, i])
            Rprev = Rprev @ Rl[:, i]
            R0[:, i] = Rprev
            o[:, i] = oprev
        c = o + _mv(R0, np.broadcast_to(self.com, (B, self.m, 3)))
        ee = o[:, -1] + (R0[:, -1] @ self.tool)
        return dict(R0=R0, o=o, z=z, c=c, ee=ee, Rl=Rl, pl=pl)

    # ------------------------------------------------------------------
    def jacobians(self, q, kin=None):
        """Body Jacobians of every link CoM and of the tool point.

        Returns Jv (B,m,3,m), Jw (B,m,3,m), Jee (B,3,m).
        """
        kin = self.kinematics(q) if kin is None else kin
        B = q.shape[0]
        o, z, c, R0 = kin['o'], kin['z'], kin['c'], kin['R0']
        # origin of frame i-1 = o[i-1]; frame -1 is the base at the origin
        obase = np.concatenate([np.zeros((B, 1, 3)), o[:, :-1]], axis=1)  # (B,m,3)
        Jv = np.zeros((B, self.m, 3, self.m))
        Jw = np.zeros((B, self.m, 3, self.m))
        rev = self.jtype == REVOLUTE
        allrev = bool(rev.all())
        for i in range(self.m):
            k = i + 1
            zi = z[:, :k]                                  # (B,k,3)
            r = c[:, i][:, None, :] - obase[:, :k]         # (B,k,3)
            cr = _cross(zi, r)
            if allrev:
                col_v, col_w = cr, zi
            else:
                col_v = np.where(rev[:k][None, :, None], cr, zi)
                col_w = np.where(rev[:k][None, :, None], zi, 0.0)
            Jv[:, i, :, :k] = np.swapaxes(col_v, 1, 2)
            Jw[:, i, :, :k] = np.swapaxes(col_w, 1, 2)
        ee = kin['ee']
        r = ee[:, None, :] - obase
        cr = _cross(z, r)
        col_v = cr if allrev else np.where(rev[None, :, None], cr, z)
        Jee = np.swapaxes(col_v, 1, 2)
        return Jv, Jw, Jee

    def mass_matrix(self, q, kin=None, jac=None):
        kin = self.kinematics(q) if kin is None else kin
        Jv, Jw, _ = self.jacobians(q, kin) if jac is None else jac
        B, m = q.shape[0], self.m
        R0 = kin['R0']
        Iw = R0 @ self.inertia[None] @ np.swapaxes(R0, -1, -2)
        Js = (Jv * np.sqrt(self.mass)[None, :, None, None]).reshape(B, 3 * m, m)
        M = np.swapaxes(Js, 1, 2) @ Js
        M += (np.swapaxes(Jw, 2, 3) @ (Iw @ Jw)).sum(axis=1)
        return M

    # ------------------------------------------------------------------
    def rnea(self, q, qd, qdd, kin=None, gravity=True):
        """Recursive Newton-Euler inverse dynamics (Siciliano form)."""
        kin = self.kinematics(q) if kin is None else kin
        Rl, pl = kin['Rl'], kin['pl']
        B, m = q.shape[0], self.m
        z0 = np.array([0.0, 0.0, 1.0])

        w = np.zeros((B, 3))
        wd = np.zeros((B, 3))
        a = (-self.g)[None, :].repeat(B, 0) if gravity else np.zeros((B, 3))

        wl = np.empty((B, m, 3))
        wdl = np.empty((B, m, 3))
        acl = np.empty((B, m, 3))
        rl = np.empty((B, m, 3))

        for i in range(m):
            Rt = np.swapaxes(Rl[:, i], 1, 2)                      # R_i^T
            r = _mv(Rt, pl[:, i])             # r_{i-1,i} in frame i
            rl[:, i] = r
            if self.jtype[i] == REVOLUTE:
                wn = _mv(Rt, w + qd[:, i:i + 1] * z0)
                wdn = _mv(Rt, wd + qdd[:, i:i + 1] * z0
                         + qd[:, i:i + 1] * _cross(w, np.broadcast_to(z0, w.shape)))
                an = _mv(Rt, a)
            else:
                wn = _mv(Rt, w)
                wdn = _mv(Rt, wd)
                an = _mv(Rt, a + qdd[:, i:i + 1] * z0) \
                    + 2.0 * _cross(wn, qd[:, i:i + 1] * Rt[:, :, 2])
            an = an + _cross(wdn, r) + _cross(wn, _cross(wn, r))
            w, wd, a = wn, wdn, an
            wl[:, i], wdl[:, i] = w, wd
            ci = self.com[i]
            acl[:, i] = a + _cross(wd, np.broadcast_to(ci, a.shape)) \
                + _cross(w, _cross(w, np.broadcast_to(ci, a.shape)))

        f = np.zeros((B, 3))
        mu = np.zeros((B, 3))
        tau = np.zeros((B, m))
        for i in range(m - 1, -1, -1):
            if i < m - 1:
                Rn = Rl[:, i + 1]
                f_next = _mv(Rn, f)
                mu_next = _mv(Rn, mu)
            else:
                f_next = np.zeros((B, 3))
                mu_next = np.zeros((B, 3))
            fi = f_next + self.mass[i] * acl[:, i]
            ci = np.broadcast_to(self.com[i], (B, 3))
            Ii = self.inertia[i]
            mui = (-_cross(fi, rl[:, i] + ci) + mu_next
                   + _cross(f_next, ci)
                   + wdl[:, i] @ Ii.T
                   + _cross(wl[:, i], wl[:, i] @ Ii.T))
            Rt = np.swapaxes(Rl[:, i], 1, 2)
            axis = Rt[:, :, 2]
            tau[:, i] = np.sum((mui if self.jtype[i] == REVOLUTE else fi) * axis, axis=1)
            f, mu = fi, mui
        return tau

    def bias(self, q, qd, kin=None):
        """h(q,qd) = C(q,qd) qd + G(q)."""
        return self.rnea(q, qd, np.zeros_like(qd), kin=kin, gravity=True)
