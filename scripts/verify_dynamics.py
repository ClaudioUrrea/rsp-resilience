#!/usr/bin/env python3
"""Independent consistency checks on the plant models (Step 1 of the README)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsp.plants import make_plant, PLANT_ORDER


def energy(pl, q, qd):
    kin = pl.chain.kinematics(q)
    M = pl.chain.mass_matrix(q, kin) + np.diag(pl.rotor)[None]
    T = 0.5 * np.einsum('bi,bij,bj->b', qd, M, qd)
    U = -np.einsum('m,bmi,i->b', pl.chain.mass, kin['c'], pl.chain.g)
    return T + U


def main():
    rng = np.random.default_rng(0)
    ok = True
    for name in PLANT_ORDER:
        pl = make_plant(name)
        B = 8
        y, yd = pl.sample_initial(rng, B)
        yd = yd + 0.4 * rng.standard_normal(yd.shape)
        tt = pl.terms(y, yd)
        M = tt['M']
        sym = np.abs(M - np.swapaxes(M, 1, 2)).max()
        eig = np.linalg.eigvalsh(M).min()
        line = f'{name:12s} sym={sym:.2e} eig_min={eig:.4f}'
        if hasattr(pl, 'chain'):
            qdd = rng.standard_normal(y.shape)
            kin = pl.chain.kinematics(y)
            tau = pl.chain.rnea(y, yd, qdd, kin) + pl.rotor * qdd
            lhs = np.einsum('bij,bj->bi', M, qdd) + tt['h']
            err = np.abs(tau - lhs).max() / max(np.abs(tau).max(), 1.0)
            line += f' rnea_vs_M={err:.2e}'
            ok &= err < 1e-12          # observed: ~3e-16 on every plant
            E0 = energy(pl, y, yd)
            q, qd = y.copy(), yd.copy()
            dt = 1e-3
            for _ in range(2000):
                kin = pl.chain.kinematics(q)
                Mm = pl.chain.mass_matrix(q, kin) + np.diag(pl.rotor)[None]
                h = pl.chain.bias(q, qd, kin)
                a = np.linalg.solve(Mm, (-h)[..., None])[..., 0]
                qd = qd + dt * a
                q = q + dt * qd
            drift = np.abs((energy(pl, q, qd) - E0) / E0).max()
            line += f' energy_drift={drift:.2e}'
            # explicit Euler at dt = 1 ms over 2 s; the largest drift observed
            # on the four serial plants is 4.2e-2 (P1), the others are <= 1e-2
            ok &= drift < 5e-2
        ok &= (sym < 1e-10) and (eig > 0)
        print(line)
    print('ALL CHECKS PASSED' if ok else 'CHECKS FAILED')
    print('tolerances: RNEA vs M qdd < 1e-12 (relative), M symmetric < 1e-10, '
          'M positive definite, energy drift < 5e-2 over 2 s')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
