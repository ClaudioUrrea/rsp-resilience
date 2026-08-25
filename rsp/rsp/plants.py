"""
The five plants of the numerical study.

Every plant exposes the same interface so that controllers, fault injection and
scoring are written once:

    ngen   dimension of the generalized coordinates y
    nact   number of actuators
    ndim   dimension of the task
    terms(y, yd) -> dict(M, h, Bact, Jt, x, Jact)

with the equation of motion

    M(y) ydd + h(y, yd) + f_fric(yd) = Bact(y) Lambda tau + Bact(y) f_a + d,

task coordinates x = phi(y), task velocity xd = Jt(y) yd, and Jact the map from
actuator velocities to task velocity used by the post-fault margin of Section 4.

For the four serial plants y = q (joint coordinates), Bact = I and Jact = Jt.
For the delta plant y = x (platform position), Bact = A(x)^T and Jact = A^+.

Sources of the physical parameters
----------------------------------
P5 (UR16e) uses the kinematic and dynamic parameters published by the
manufacturer (Universal Robots, "DH Parameters for calculations of kinematics
and dynamics"): link masses, centres of mass and full inertia tensors.
P1-P3 are generic academic models whose parameters are listed in Appendix B of
the paper.  P4 is a lumped reduced model of a four-arm delta manipulator.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import numpy as np

from .dynamics import SerialChain, REVOLUTE, PRISMATIC, _cross


# ----------------------------------------------------------------------
class SerialPlant:
    """Serial chain with a position task."""

    def __init__(self, name, chain, task_rows, q_home, q_min, q_max, qd_max,
                 tau_max, fv, fc, traj_radius, traj_period, ws_radius,
                 rotor, traj_plane=(0, 1)):
        self.name = name
        self.chain = chain
        self.task_rows = np.asarray(task_rows, int)
        self.ndim = len(self.task_rows)
        self.ngen = self.nact = chain.m
        self.q_home = np.asarray(q_home, float)
        self.q_min = np.asarray(q_min, float)
        self.q_max = np.asarray(q_max, float)
        self.qd_max = np.asarray(qd_max, float)
        self.tau_max = np.asarray(tau_max, float)
        self.fv = np.asarray(fv, float)
        self.fc = np.asarray(fc, float)
        self.rotor = np.asarray(rotor, float)
        self.traj_radius = traj_radius
        self.traj_period = traj_period
        self.ws_radius = ws_radius
        self.traj_plane = traj_plane
        self.x_home = self.task_of(self.q_home[None, :])[0]

    # -- kinematics / dynamics ----------------------------------------
    def task_of(self, q):
        kin = self.chain.kinematics(q)
        return kin['ee'][:, self.task_rows]

    def terms(self, y, yd):
        kin = self.chain.kinematics(y)
        Jv, Jw, Jee = self.chain.jacobians(y, kin)
        M = self.chain.mass_matrix(y, kin, (Jv, Jw, Jee)) + np.diag(self.rotor)[None]
        h = self.chain.bias(y, yd, kin)
        Jt = Jee[:, self.task_rows, :]
        x = kin['ee'][:, self.task_rows]
        B = y.shape[0]
        eye = np.broadcast_to(np.eye(self.nact), (B, self.nact, self.nact))
        return dict(M=M, h=h, Bact=eye, Jt=Jt, x=x, Jact=Jt)

    def friction(self, yd, tt=None):
        return self.fv * yd + self.fc * np.tanh(50.0 * yd)

    def act_state(self, y, yd, tt=None):
        """Actuator-space position and velocity (equal to y, yd for serial chains)."""
        return y, yd

    def violation(self, y, yd, tt):
        """Boolean mask: state outside the admissible set."""
        lim = (y < self.q_min).any(axis=1) | (y > self.q_max).any(axis=1)
        vel = (np.abs(yd) > self.qd_max).any(axis=1)
        ws = np.linalg.norm(tt['x'] - self.x_home, axis=1) > self.ws_radius
        return lim | vel | ws

    # -- task reference ------------------------------------------------
    def reference(self, t):
        """x_d, xd_d, xdd_d at scalar time t (returns (ndim,) arrays)."""
        w = 2.0 * np.pi / self.traj_period
        r = self.traj_radius
        p = np.zeros(self.ndim)
        v = np.zeros(self.ndim)
        a = np.zeros(self.ndim)
        i0, i1 = self.traj_plane
        p[i0] = r * (np.cos(w * t) - 1.0)
        p[i1] = r * np.sin(w * t)
        v[i0] = -r * w * np.sin(w * t)
        v[i1] = r * w * np.cos(w * t)
        a[i0] = -r * w * w * np.cos(w * t)
        a[i1] = -r * w * w * np.sin(w * t)
        return self.x_home + p, v, a

    def sample_initial(self, rng, B):
        dq = (rng.random((B, self.ngen)) - 0.5) * 2.0 * 0.08
        return self.q_home[None, :] + dq, np.zeros((B, self.ngen))


# ----------------------------------------------------------------------
class DeltaPlant:
    """Four-arm delta parallel manipulator, lumped reduced model.

    Generalized coordinates: platform position x in R^3.
    Actuators: four proximal-arm torques; actuation redundancy r = 1.
    """

    name = 'P4-Delta4'

    def __init__(self, Rb=0.16, L1=0.30, L2=0.62, rp=0.055, m_plat=1.4,
                 m_arm=0.65, m_fore=0.16, tau_max=32.0, fv=0.35, fc=0.25,
                 traj_radius=0.10, traj_period=2.0):
        self.Rb, self.L1, self.L2, self.rp = Rb, L1, L2, rp
        self.m_plat, self.m_arm, self.m_fore = m_plat, m_arm, m_fore
        self.ngen, self.ndim, self.nact = 3, 3, 4
        phi = np.arange(4) * (np.pi / 2.0)
        self.u = np.stack([np.cos(phi), np.sin(phi), np.zeros(4)], axis=1)  # (4,3)
        self.rotor = 0.12
        self.Ia = m_arm * L1 ** 2 / 3.0 + m_fore * L1 ** 2 + self.rotor
        self.m_eff = m_plat + 4.0 * 0.5 * m_fore
        self.L1c = 0.5 * L1
        self.tau_max = np.full(4, tau_max)
        self.fv = np.full(4, fv)
        self.fc = np.full(4, fc)
        self.qd_max = np.full(4, 12.0)
        self.q_min = np.full(4, -1.3)
        self.q_max = np.full(4, 1.6)
        self.traj_radius = traj_radius
        self.traj_period = traj_period
        self.x_home = np.array([0.0, 0.0, -0.52])
        self.ws_radius = 0.40
        self.g = 9.81

    # -- inverse kinematics of one arm --------------------------------
    def _theta(self, x):
        """Arm angles theta (B,4) from platform position x (B,3)."""
        P = x[:, None, :] + self.rp * self.u[None, :, :]      # (B,4,3)
        p = np.sum(P * self.u[None, :, :], axis=2)            # radial component
        z = P[:, :, 2]
        P2 = np.sum(P * P, axis=2)
        E = P2 - 2.0 * p * self.Rb + self.Rb ** 2 + self.L1 ** 2 - self.L2 ** 2
        F = 2.0 * self.L1 * (self.Rb - p)
        G = 2.0 * self.L1 * z
        rad = np.sqrt(np.maximum(F * F + G * G, 1e-12))
        arg = np.clip(-E / rad, -1.0, 1.0)
        psi = np.arctan2(G, F)
        ok = np.abs(-E / rad) <= 1.0
        return psi + np.arccos(arg), ok.all(axis=1)

    def _A(self, x):
        """Constraint Jacobian A(x) with thetadot = A(x) xdot, shape (B,4,3)."""
        th, ok = self._theta(x)
        P = x[:, None, :] + self.rp * self.u[None, :, :]
        K = ((self.Rb + self.L1 * np.cos(th))[:, :, None] * self.u[None, :, :]
             - (self.L1 * np.sin(th))[:, :, None] * np.array([0.0, 0.0, 1.0]))
        dv = P - K                                             # (B,4,3)
        dKdth = (-(self.L1 * np.sin(th))[:, :, None] * self.u[None, :, :]
                 - (self.L1 * np.cos(th))[:, :, None] * np.array([0.0, 0.0, 1.0]))
        dgdth = -2.0 * np.sum(dv * dKdth, axis=2)              # (B,4)
        dgdx = 2.0 * dv                                        # (B,4,3)
        A = -dgdx / np.where(np.abs(dgdth) < 1e-9, 1e-9, dgdth)[:, :, None]
        return A, th, ok

    def terms(self, y, yd):
        B = y.shape[0]
        A, th, ok = self._A(y)
        eta = 1e-4
        Ap, _, _ = self._A(y + eta * yd)
        Am, _, _ = self._A(y - eta * yd)
        Adot = (Ap - Am) / (2.0 * eta)
        M = self.m_eff * np.broadcast_to(np.eye(3), (B, 3, 3)) \
            + self.Ia * np.swapaxes(A, 1, 2) @ A
        cor = self.Ia * np.einsum('bji,bjk,bk->bi', A, Adot, yd)
        g_plat = np.zeros((B, 3))
        g_plat[:, 2] = self.m_eff * self.g
        g_arm = -self.m_arm * self.g * self.L1c * np.cos(th)     # dU/dtheta (B,4)
        h = cor + g_plat + np.einsum('bji,bj->bi', A, g_arm)
        Bact = np.swapaxes(A, 1, 2)                              # (B,3,4)
        Jt = np.broadcast_to(np.eye(3), (B, 3, 3))
        Jact = np.linalg.pinv(A)                                 # (B,3,4)
        self._last_ok = ok
        self._last_A = A
        return dict(M=M, h=h, Bact=Bact, Jt=Jt, x=y, Jact=Jact, A=A, theta=th, ok=ok)

    def friction(self, yd, tt=None):
        A = tt['A']
        thd = np.einsum('bij,bj->bi', A, yd)
        f = self.fv * thd + self.fc * np.tanh(50.0 * thd)
        return np.einsum('bji,bj->bi', A, f)

    def act_state(self, y, yd, tt=None):
        A = tt['A']
        return tt['theta'], np.einsum('bij,bj->bi', A, yd)

    def violation(self, y, yd, tt):
        th, thd = self.act_state(y, yd, tt)
        lim = (th < self.q_min).any(axis=1) | (th > self.q_max).any(axis=1)
        vel = (np.abs(thd) > self.qd_max).any(axis=1)
        ws = np.linalg.norm(y - self.x_home, axis=1) > self.ws_radius
        return lim | vel | ws | (~tt['ok'])

    def task_of(self, y):
        return y

    def reference(self, t):
        w = 2.0 * np.pi / self.traj_period
        r = self.traj_radius
        p = np.array([r * (np.cos(w * t) - 1.0), r * np.sin(w * t), 0.02 * np.sin(w * t)])
        v = np.array([-r * w * np.sin(w * t), r * w * np.cos(w * t), 0.02 * w * np.cos(w * t)])
        a = np.array([-r * w * w * np.cos(w * t), -r * w * w * np.sin(w * t),
                      -0.02 * w * w * np.sin(w * t)])
        return self.x_home + p, v, a

    def sample_initial(self, rng, B):
        dx = (rng.random((B, 3)) - 0.5) * 2.0 * 0.012
        return self.x_home[None, :] + dx, np.zeros((B, 3))


# ----------------------------------------------------------------------
def _diag_inertia(vals, n):
    return np.tile(np.diag(vals), (n, 1, 1))


def plant_planar3r():
    m = 3
    a = np.array([0.50, 0.40, 0.30])
    chain = SerialChain(
        a=a, alpha=np.zeros(m), d=np.zeros(m), theta=np.zeros(m),
        jtype=np.zeros(m, int),
        mass=np.array([6.0, 4.0, 2.0]),
        com=np.array([[-0.25, 0, 0], [-0.20, 0, 0], [-0.15, 0, 0]]),
        inertia=np.stack([np.diag([0.004, 0.130, 0.130]),
                          np.diag([0.003, 0.056, 0.056]),
                          np.diag([0.002, 0.017, 0.017])]),
        gravity=(0.0, -9.81, 0.0))
    return SerialPlant(
        'P1-Planar3R', chain, task_rows=[0, 1],
        q_home=[0.5, 0.8, 0.6],
        q_min=[-2.9, -2.7, -2.7], q_max=[2.9, 2.7, 2.7],
        qd_max=[6.0, 6.0, 8.0], tau_max=[90.0, 55.0, 25.0],
        fv=[1.2, 0.9, 0.6], fc=[0.9, 0.7, 0.4],
        traj_radius=0.13, traj_period=2.0, ws_radius=0.52,
        rotor=[0.35, 0.25, 0.12], traj_plane=(0, 1))


def plant_scara():
    m = 4
    chain = SerialChain(
        a=np.array([0.40, 0.35, 0.25, 0.0]),
        alpha=np.zeros(m),
        d=np.array([0.60, 0.0, 0.0, 0.0]),
        theta=np.zeros(m),
        jtype=np.array([REVOLUTE, REVOLUTE, REVOLUTE, PRISMATIC]),
        mass=np.array([7.0, 4.5, 2.5, 3.0]),
        com=np.array([[-0.20, 0, 0], [-0.17, 0, 0], [-0.12, 0, 0], [0, 0, -0.06]]),
        inertia=np.stack([np.diag([0.010, 0.100, 0.100]),
                          np.diag([0.007, 0.048, 0.048]),
                          np.diag([0.004, 0.014, 0.014]),
                          np.diag([0.006, 0.006, 0.002])]),
        gravity=(0.0, 0.0, -9.81))
    return SerialPlant(
        'P2-SCARA', chain, task_rows=[0, 1, 2],
        q_home=[0.45, 0.75, 0.55, -0.22],
        q_min=[-2.9, -2.6, -2.6, -0.45], q_max=[2.9, 2.6, 2.6, -0.02],
        qd_max=[6.0, 6.0, 8.0, 1.2], tau_max=[110.0, 60.0, 28.0, 260.0],
        fv=[1.4, 1.0, 0.6, 30.0], fc=[1.0, 0.7, 0.4, 18.0],
        traj_radius=0.11, traj_period=2.0, ws_radius=0.44,
        rotor=[0.30, 0.20, 0.10, 1.2], traj_plane=(0, 1))


def plant_srs7():
    m = 7
    chain = SerialChain(
        a=np.zeros(m),
        alpha=np.array([np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2,
                        np.pi / 2, -np.pi / 2, 0.0]),
        d=np.array([0.34, 0.0, 0.40, 0.0, 0.40, 0.0, 0.126]),
        theta=np.zeros(m),
        jtype=np.zeros(m, int),
        mass=np.array([3.9, 3.9, 3.1, 2.8, 1.9, 1.4, 0.5]),
        com=np.array([[0, -0.03, 0.12], [0, 0.04, 0.06], [0, 0.03, 0.16],
                      [0, -0.04, 0.06], [0, -0.02, 0.14], [0, 0.01, 0.02],
                      [0, 0, 0.03]]),
        inertia=np.stack([np.diag([0.045, 0.043, 0.014]),
                          np.diag([0.041, 0.014, 0.040]),
                          np.diag([0.036, 0.034, 0.011]),
                          np.diag([0.030, 0.010, 0.029]),
                          np.diag([0.022, 0.021, 0.006]),
                          np.diag([0.005, 0.005, 0.004]),
                          np.diag([0.001, 0.001, 0.001])]),
        gravity=(0.0, 0.0, -9.81), tool=(0.0, 0.0, 0.05))
    return SerialPlant(
        'P3-SRS7', chain, task_rows=[0, 1, 2],
        q_home=[0.0, 0.55, 0.0, -1.15, 0.0, 0.75, 0.0],
        q_min=[-2.9] * 7, q_max=[2.9] * 7,
        qd_max=[3.0, 3.0, 3.5, 3.5, 4.0, 4.0, 5.0],
        tau_max=[170.0, 170.0, 90.0, 90.0, 45.0, 45.0, 22.0],
        fv=[0.9, 0.9, 0.7, 0.7, 0.4, 0.4, 0.2],
        fc=[0.8, 0.8, 0.6, 0.6, 0.3, 0.3, 0.15],
        traj_radius=0.11, traj_period=2.0, ws_radius=0.44,
        rotor=[0.60, 0.60, 0.40, 0.40, 0.15, 0.15, 0.08], traj_plane=(0, 1))


def plant_ur16e():
    """UR16e with the manufacturer-published kinematic and dynamic parameters."""
    m = 6
    a = np.array([0.0, -0.4784, -0.36, 0.0, 0.0, 0.0])
    d = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])
    alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
    mass = np.array([7.369, 10.450, 4.321, 2.180, 2.033, 0.907])
    com = np.array([[0.000, -0.016, 0.030],
                    [0.302, 0.000, 0.160],
                    [0.194, 0.000, 0.065],
                    [0.000, -0.009, 0.011],
                    [0.000, 0.018, 0.012],
                    [0.000, 0.000, -0.044]])
    I = np.array([
        [[0.0335, 0.0000, 0.0000], [0.0000, 0.0337, 0.0037], [0.0000, 0.0037, 0.0210]],
        [[0.0280, -0.0001, -0.0072], [-0.0001, 0.4756, 0.0000], [-0.0072, 0.0000, 0.4764]],
        [[0.0109, 0.0001, 0.0101], [0.0001, 0.1206, 0.0000], [0.0101, 0.0000, 0.11714]],
        [[0.0061, 0.0000, 0.0000], [0.0000, 0.0025, 0.0008], [0.0000, 0.0008, 0.0058]],
        [[0.0039, 0.0000, 0.0000], [0.0000, 0.0022, -0.0005], [0.0000, -0.0005, 0.0036]],
        [[0.0012, 0.0000, 0.0000], [0.0000, 0.0012, 0.0000], [0.0000, 0.0000, 0.0008]]])
    chain = SerialChain(a=a, alpha=alpha, d=d, theta=np.zeros(m),
                        jtype=np.zeros(m, int), mass=mass, com=com, inertia=I,
                        gravity=(0.0, 0.0, -9.81))
    return SerialPlant(
        'P5-UR16e', chain, task_rows=[0, 1, 2],
        q_home=[0.0, -1.05, 1.35, -1.85, -1.57, 0.0],
        q_min=[-3.0] * 6, q_max=[3.0] * 6,
        qd_max=[2.09, 2.09, 3.14, 3.14, 3.14, 3.14],
        tau_max=[330.0, 330.0, 150.0, 56.0, 56.0, 56.0],
        fv=[2.0, 2.0, 1.2, 0.5, 0.5, 0.3],
        fc=[1.5, 1.5, 1.0, 0.4, 0.4, 0.25],
        traj_radius=0.10, traj_period=2.0, ws_radius=0.40,
        rotor=[1.20, 1.20, 0.60, 0.15, 0.15, 0.12], traj_plane=(0, 1))


PLANT_FACTORIES = {
    'P1-Planar3R': plant_planar3r,
    'P2-SCARA': plant_scara,
    'P3-SRS7': plant_srs7,
    'P4-Delta4': DeltaPlant,
    'P5-UR16e': plant_ur16e,
}

PLANT_ORDER = ['P1-Planar3R', 'P2-SCARA', 'P3-SRS7', 'P4-Delta4', 'P5-UR16e']


def make_plant(name):
    return PLANT_FACTORIES[name]()
