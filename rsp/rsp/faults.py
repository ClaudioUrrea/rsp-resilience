"""
The eight fault classes of Definition 1, sampled per episode and injected
inside the integration loop.

All classes are instances of the perturbation

    M(y) ydd + C ydd + G + F = B(y) [ Lambda(t) tau_applied + f_a(t) ] + d(t),

where tau_applied is obtained from the commanded torque by the communication
operator (hold / delay) of F6-F8.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import numpy as np

FAULT_NAMES = ['F1-Locked', 'F2-FreeSwing', 'F3-PartialLoss', 'F4-EncoderBias',
               'F5-TorqueSat', 'F6-PacketLoss', 'F7-Latency', 'F8-MissedCycle']
N_FAULTS = len(FAULT_NAMES)
MAX_DELAY_STEPS = 40          # 40 ms buffer at a 1 kHz control rate


def sample_faults(rng, B, nact, fault_id, t_f_range=(0.80, 1.40)):
    """Latin-hypercube sample of the parameters of one fault class.

    Returns a dict of per-episode arrays.  ``fault_id`` is an integer in
    0..7 (a single class per cell) or an array of length B.
    """
    def lhs(n, lo, hi):
        u = (rng.permutation(n) + rng.random(n)) / n
        return lo + (hi - lo) * u

    fid = np.full(B, fault_id, int) if np.isscalar(fault_id) else np.asarray(fault_id)
    f = dict(
        fid=fid,
        t_f=lhs(B, *t_f_range),
        idx=rng.integers(0, nact, size=B),
        lam_bar=lhs(B, 0.15, 0.75),
        bias=lhs(B, -0.06, 0.06) * rng.choice([-1.0, 1.0], size=B),
        drift=lhs(B, 0.01, 0.10),
        sat_frac=lhs(B, 0.20, 0.50),
        p_loss=lhs(B, 0.10, 0.50),
        theta_max=lhs(B, 0.005, 0.040),
        p_miss=lhs(B, 0.05, 0.30),
    )
    return f


def effectiveness(f, t, nact):
    """Lambda(t) as a (B, nact) array of actuator effectiveness."""
    B = len(f['fid'])
    lam = np.ones((B, nact))
    active = t >= f['t_f']
    rows = np.arange(B)
    fid = f['fid']
    kill = active & ((fid == 0) | (fid == 1))
    part = active & (fid == 2)
    lam[rows[kill], f['idx'][kill]] = 0.0
    lam[rows[part], f['idx'][part]] = f['lam_bar'][part]
    return lam


def torque_bound(f, t, tau_max):
    """Per-actuator torque bound, reduced on the faulty axis for F5."""
    B = len(f['fid'])
    bound = np.broadcast_to(tau_max, (B, len(tau_max))).copy()
    m = (t >= f['t_f']) & (f['fid'] == 4)
    rows = np.arange(B)[m]
    bound[rows, f['idx'][m]] *= f['sat_frac'][m]
    return bound


def measurement_offset(f, t, nact):
    """Additive encoder bias and drift on the measured coordinate (F4)."""
    B = len(f['fid'])
    off = np.zeros((B, nact))
    m = (t >= f['t_f']) & (f['fid'] == 3)
    rows = np.arange(B)[m]
    off[rows, f['idx'][m]] = f['bias'][m] + f['drift'][m] * (t - f['t_f'][m])
    return off


class CommChannel:
    """Zero-order hold, packet loss, variable latency and missed cycles."""

    def __init__(self, B, nact, rng):
        self.buf = np.zeros((B, MAX_DELAY_STEPS, nact))
        self.ptr = 0
        self.last = np.zeros((B, nact))
        self.rng = rng
        self.B = B

    def transmit(self, tau_cmd, f, t):
        fid, active = f['fid'], t >= f['t_f']
        out = tau_cmd.copy()

        drop = active & (fid == 5) & (self.rng.random(self.B) < f['p_loss'])
        miss = active & (fid == 7) & (self.rng.random(self.B) < f['p_miss'])
        hold = drop | miss
        out[hold] = self.last[hold]

        self.buf[:, self.ptr] = out
        lat = active & (fid == 6)
        if lat.any():
            steps = np.minimum((f['theta_max'] * self.rng.random(self.B) / 1e-3)
                               .astype(int), MAX_DELAY_STEPS - 1)
            idx = (self.ptr - steps) % MAX_DELAY_STEPS
            delayed = self.buf[np.arange(self.B), idx]
            out = np.where(lat[:, None], delayed, out)

        self.ptr = (self.ptr + 1) % MAX_DELAY_STEPS
        self.last = out
        return out
