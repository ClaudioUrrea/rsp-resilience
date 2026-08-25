#!/usr/bin/env python3
"""
Run the full Monte-Carlo campaign of Section 8.

    python scripts/run_experiments.py --episodes 100 --seed 20260811

Produces results/raw_<plant>.npz with the six raw criteria of every episode,
plus results/meta.json with the calibrated detection thresholds, the reference
constants x_k^star and the measured control-law execution times.

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsp.plants import make_plant, PLANT_ORDER
from rsp.controllers import Controller, CONTROLLERS
from rsp.simulate import run_cell, calibrate_threshold, DEFAULT_CFG, NO_FAULT
from rsp.faults import N_FAULTS
from rsp.score import X2_CAP, X4_BUDGET

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def measure_wcet(plant, ctrl_name, n_calls=400, seed=0):
    """Wall-clock cost of one control-law evaluation, unbatched.

    C5 is an implementation-dependent quantity; the value reported in the paper
    refers to the reference workstation declared in Section 8.1 and is included
    so that the criterion is reproducible, not because it is portable.
    """
    rng = np.random.default_rng(seed)
    ctrl = Controller(ctrl_name, plant, 1, DEFAULT_CFG['dt'], rng)
    n = plant.ndim
    lam = np.ones((1, plant.nact))
    det = np.zeros(1, bool)
    e = 0.01 * rng.standard_normal((1, n))
    ed = 0.05 * rng.standard_normal((1, n))
    ad = np.zeros((1, n))
    ctrl.command(e, ed, ad, lam, det)
    t = np.empty(n_calls)
    for i in range(n_calls):
        t0 = time.perf_counter()
        ctrl.command(e, ed, ad, lam, det)
        t[i] = time.perf_counter() - t0
    return t


def sample_wcet_ratio(samples, B, n_steps, Ts, rng):
    """Per-episode C5 = max execution time over the episode / sampling period."""
    idx = rng.integers(0, samples.size, size=(B, min(n_steps, 400)))
    return samples[idx].max(axis=1) / Ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=100,
                    help='episodes per plant-controller-fault cell')
    ap.add_argument('--seed', type=int, default=20260811)
    ap.add_argument('--calib', type=int, default=48)
    ap.add_argument('--plants', type=str, default=','.join(PLANT_ORDER))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    np.seterr(all='ignore')
    cfg = dict(DEFAULT_CFG)
    n_steps = int(round(cfg['T'] / cfg['dt']))
    meta_path = os.path.join(OUT, 'meta.json')
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        meta.setdefault('plants', [])
    else:
        meta = dict(cfg={k: v for k, v in cfg.items()}, episodes=args.episodes,
                    seed=args.seed, controllers=CONTROLLERS, plants=[], xstar={},
                    xstar_aux={}, thresholds={}, wcet_median_us={},
                    runtime_s={})

    for pname in args.plants.split(','):
        t_plant = time.time()
        plant = make_plant(pname)
        if pname not in meta['plants']:
            meta['plants'].append(pname)
        fids = np.repeat(np.arange(N_FAULTS), args.episodes)
        B = fids.size

        # -- 1. fault-free calibration: thresholds and nominal energy --
        thr, energy_ref = {}, []
        for ci, c in enumerate(CONTROLLERS):
            th = calibrate_threshold(plant, c, seed=args.seed + 900 + ci,
                                     B=args.calib, cfg=dict(cfg, T=2.2))
            thr[c] = th
            r0 = run_cell(plant, c, np.full(args.calib, NO_FAULT),
                          seed=args.seed + 950 + ci, cfg=dict(cfg, T=2.2),
                          thr_override=th)
            energy_ref.append(np.median(r0['x4']))
        # Section 8.1: the energy reference is X4_BUDGET times the median
        # nominal power, pooled over controllers so that the constant does not
        # depend on the controller being scored.  Both the pooled median and
        # the reference actually used are recorded, so that meta.json contains
        # the constants of Definition 2 with no further transformation.
        x4_nominal = float(np.median(energy_ref))
        x4_star = X4_BUDGET * x4_nominal
        meta['thresholds'][pname] = {c: [thr[c][0].tolist(), thr[c][1]]
                                     for c in CONTROLLERS}
        meta['xstar'][pname] = dict(x1=5.0e-3, x2=0.30, x3=0.50,
                                    x4=x4_star, x5=0.50, x6=0.05)
        meta.setdefault('xstar_aux', {})[pname] = dict(
            x4_nominal_power=x4_nominal, x4_budget=X4_BUDGET, x2_cap=X2_CAP)
        print(f'[{pname}] calibrated, nominal power = {x4_nominal:.1f}, '
              f'x4* = {x4_star:.1f}', flush=True)

        # -- 2. execution-time measurement -----------------------------
        wcet = {c: measure_wcet(plant, c, seed=args.seed + ci)
                for ci, c in enumerate(CONTROLLERS)}
        meta['wcet_median_us'][pname] = {c: float(np.median(wcet[c]) * 1e6)
                                         for c in CONTROLLERS}

        # -- 3. Monte-Carlo campaign under common random numbers -------
        store = {}
        for c in CONTROLLERS:
            t0 = time.time()
            r = run_cell(plant, c, fids, seed=args.seed, thr_override=thr[c],
                         cfg=cfg)
            rng5 = np.random.default_rng([args.seed, 77])
            x5 = sample_wcet_ratio(wcet[c], B, n_steps, cfg['dt'], rng5)
            X = np.stack([r['x1'], r['x2'], r['x3'], r['x4'], x5, r['x6']], axis=1)
            store[f'X_{c}'] = X
            store[f'declared_{c}'] = r['declared']
            store[f'degraded_{c}'] = r['degraded']
            store[f'alive_{c}'] = r['alive']
            store[f'ratio_{c}'] = r['ratio']
            print(f'[{pname}] {c:5s} done in {time.time() - t0:5.1f}s  '
                  f'median C1 = {1e3 * np.median(r["x1"]):.2f} mm', flush=True)
        store['fid'] = r['fid']
        store['t_f'] = r['t_f']
        store['idx'] = r['idx']
        np.savez_compressed(os.path.join(OUT, f'raw_{pname}.npz'), **store)
        meta['runtime_s'][pname] = time.time() - t_plant
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=1)
        print(f'[{pname}] total {meta["runtime_s"][pname]:.0f}s', flush=True)

    print('campaign complete')


if __name__ == '__main__':
    main()
