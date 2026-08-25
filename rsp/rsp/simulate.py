"""
Batched episode engine.

One call to :func:`run_cell` advances B independent episodes of a given
(plant, controller) pair through the integration loop, injecting the fault class
assigned to each episode and returning the six raw criteria x_1..x_6 of
Section 3.2 together with diagnostic quantities.

Episodes are paired across controllers by construction: the random state that
generates the initial configuration, the fault parameters, the measurement
noise, the disturbance and the communication events is derived from the episode
index alone, so that controller A and controller B see identical realizations
(common random numbers, Section 7).

Author: C. Urrea
License: MIT
"""

from __future__ import annotations

import numpy as np

from . import faults as fx
from .controllers import Controller, FULL_MODEL

DEFAULT_CFG = dict(
    T=3.2, dt=1.0e-3, t_settle=0.40,
    mu_dls=0.030, kn=8.0, dn=4.0, rho_alloc=0.050,
    tau_res=0.030, det_consec=12, det_frac=0.05, det_abs=0.80, det_rel=0.18,
    lam_lp=0.010, rec_window=0.15, rec_factor=1.6,
    meas_noise_q=2.0e-5, meas_noise_qd=8.0e-4,
    dist_frac=0.02, dist_tau=0.05,
    unc_M=0.06, unc_h=0.06, unc_f=0.40,
)

INF = np.inf


def _dls_pinv_apply(G, v, rho):
    """(G G^T + rho I)^-1 applied through G^T: returns G^T (G G^T + rho I)^-1 v."""
    B, r, c = G.shape
    GGt = G @ np.swapaxes(G, 1, 2)
    GGt = GGt + rho * np.eye(r)[None]
    z = np.linalg.solve(GGt, v[..., None])
    return (np.swapaxes(G, 1, 2) @ z)[..., 0]


def run_cell(plant, ctrl_name, fault_ids, seed, cfg=None, record_traj=False,
             thr_override=None):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    dt, T = cfg['dt'], cfg['T']
    nsteps = int(round(T / dt))
    B = len(fault_ids)
    ngen, nact, n = plant.ngen, plant.nact, plant.ndim

    rng_ep = np.random.default_rng([seed, 0])       # episode design (shared)
    rng_ctl = np.random.default_rng([seed, 1])      # controller-internal only

    y, yd = plant.sample_initial(rng_ep, B)
    f = fx.sample_faults(rng_ep, B, nact, np.asarray(fault_ids))
    cM = 1.0 + cfg['unc_M'] * (2 * rng_ep.random(B) - 1)
    ch = 1.0 + cfg['unc_h'] * (2 * rng_ep.random(B) - 1)
    cf = 1.0 + cfg['unc_f'] * (2 * rng_ep.random(B) - 1)
    noise_seed = int(rng_ep.integers(1 << 30))
    rng_noise = np.random.default_rng([seed, 2, noise_seed])
    chan = fx.CommChannel(B, nact, np.random.default_rng([seed, 3]))

    ctrl = Controller(ctrl_name, plant, B, dt, rng_ctl)
    full = ctrl_name in FULL_MODEL

    tt = plant.terms(y, yd)
    M0 = tt['M'].copy()
    h0 = tt['h'].copy()
    Jprev = tt['Jt'].copy()

    tau_max = plant.tau_max
    lam_hat = np.ones((B, nact))
    res_f = np.zeros((B, nact))
    yd_prev = yd.copy()
    ydd_f = np.zeros((B, ngen))
    det_cnt = np.zeros(B, int)
    t_det = np.full(B, INF)
    rec_cnt = np.zeros(B, int)
    degraded = np.zeros(B, bool)
    t_rec = np.full(B, INF)
    alive = np.ones(B, bool)
    false_alarm = np.zeros(B, bool)
    dist = np.zeros((B, ngen))
    res_max = np.zeros((B, nact))
    err_max = np.zeros(B)
    err_f = np.zeros(B)
    lock_ref = np.zeros((B, nact))
    lock_set = np.zeros(B, bool)

    itae_pre = np.zeros(B); dur_pre = np.zeros(B); sq_pre = np.zeros(B); npre = np.zeros(B)
    itae_post = np.zeros(B); dur_post = np.zeros(B)
    energy = np.zeros(B); viol = np.zeros(B); npost = np.zeros(B)
    lam_true_min = np.ones(B)
    traj = [] if record_traj else None

    a_res = np.exp(-dt / cfg['tau_res'])
    a_lam = np.exp(-dt / cfg['lam_lp'])
    a_dist = np.exp(-dt / cfg['dist_tau'])
    rec_need = int(cfg['rec_window'] / dt)
    if thr_override is None:
        thr_base = cfg['det_frac'] * tau_max + cfg['det_abs']
        thr_err = 0.02
    else:
        thr_base = np.asarray(thr_override[0], float)
        thr_err = float(thr_override[1])
    q_home = getattr(plant, 'q_home', None)
    if q_home is None:
        q_home = plant.x_home
    y_home = np.broadcast_to(np.asarray(q_home, float), (B, ngen))

    for k in range(nsteps):
        t = k * dt
        tt = plant.terms(y, yd)
        M, h, Bact, Jt, x = tt['M'], tt['h'], tt['Bact'], tt['Jt'], tt['x']

        # ---- measurement -------------------------------------------
        off_act = fx.measurement_offset(f, t, nact)
        if ngen == nact:
            off_gen = off_act
        else:
            off_gen = np.einsum('bij,bj->bi', tt['Jact'], off_act)
        y_m = y + off_gen + cfg['meas_noise_q'] * rng_noise.standard_normal((B, ngen))
        yd_m = yd + cfg['meas_noise_qd'] * rng_noise.standard_normal((B, ngen))
        x_m = x + np.einsum('bij,bj->bi', Jt, off_gen)
        xd = np.einsum('bij,bj->bi', Jt, yd)
        xd_m = np.einsum('bij,bj->bi', Jt, yd_m)

        xd_ref, vd_ref, ad_ref = plant.reference(t)
        e_true = xd_ref[None, :] - x
        e = xd_ref[None, :] - x_m
        ed = vd_ref[None, :] - xd_m

        # ---- control law -------------------------------------------
        xdd_star = ctrl.command(e, ed, np.broadcast_to(ad_ref, (B, n)), lam_hat,
                                np.isfinite(t_det))

        # ---- redundancy resolution ---------------------------------
        Jdot_yd = np.einsum('bij,bj->bi', (Jt - Jprev) / dt, yd_m)
        Jprev = np.array(Jt)
        rhs = xdd_star - Jdot_yd
        if ngen == nact:
            W = (lam_hat ** 2 + 0.05)
            JW = Jt * W[:, None, :]
            S = JW @ np.swapaxes(Jt, 1, 2) + cfg['mu_dls'] ** 2 * np.eye(n)[None]
            z = np.linalg.solve(S, rhs[..., None])
            ydd_star = (np.swapaxes(JW, 1, 2) @ z)[..., 0]
            zN = np.linalg.solve(S, (Jt @ (
                (-cfg['kn'] * (y_m - y_home) - cfg['dn'] * yd_m))[..., None]))
            null = (-cfg['kn'] * (y_m - y_home) - cfg['dn'] * yd_m) \
                - (np.swapaxes(JW, 1, 2) @ zN)[..., 0]
            ydd_star = ydd_star + null
        else:
            ydd_star = rhs

        # ---- torque synthesis --------------------------------------
        Mh = (cM[:, None, None] * M) if full else (cM[:, None, None] * M0)
        hh = (ch[:, None] * h) if full else (ch[:, None] * h0)
        fr = cf[:, None] * plant.friction(yd_m, tt) if full else 0.0
        tau_req = np.einsum('bij,bj->bi', Mh, ydd_star) + hh + fr

        G = Bact * lam_hat[:, None, :]
        tau_act = _dls_pinv_apply(G, tau_req, cfg['rho_alloc'])
        tau_act = np.clip(tau_act, -tau_max, tau_max)

        # ---- communication and actuation ---------------------------
        tau_tx = chan.transmit(tau_act, f, t)
        bound = fx.torque_bound(f, t, tau_max)
        tau_app = np.clip(tau_tx, -bound, bound)
        lam = fx.effectiveness(f, t, nact)
        lam_true_min = np.minimum(lam_true_min, lam.min(axis=1))

        dscale = tau_max if ngen == nact else np.full(ngen, 6.0)
        dist = a_dist * dist + np.sqrt(1 - a_dist ** 2) * cfg['dist_frac'] * \
            dscale * rng_noise.standard_normal((B, ngen))
        F_gen = np.einsum('bij,bj->bi', Bact, lam * tau_app) + dist
        rhs_dyn = F_gen - h - plant.friction(yd, tt)

        # locked joints (F1): exact holonomic constraint for serial chains
        Mc = M
        if ngen == nact:
            lock = (f['fid'] == 0) & (t >= f['t_f'])
            if lock.any():
                rows = np.arange(B)[lock]
                idx = f['idx'][lock]
                Mc = M.copy()
                Mc[rows, idx, :] = 0.0
                Mc[rows, :, idx] = 0.0
                Mc[rows, idx, idx] = 1.0
                rhs_dyn = rhs_dyn.copy()
                rhs_dyn[rows, idx] = 0.0
                yd[rows, idx] = 0.0
        else:
            lock = (f['fid'] == 0) & (t >= f['t_f'])
            if lock.any():
                th, thd = plant.act_state(y, yd, tt)
                fresh = lock & ~lock_set
                if fresh.any():
                    lock_ref[fresh] = th[fresh]
                    lock_set |= fresh
                pen = np.zeros((B, nact))
                rows = np.arange(B)[lock]
                idx = f['idx'][lock]
                pen[rows, idx] = -(6.0e3 * (th[rows, idx] - lock_ref[rows, idx])
                                   + 1.2e2 * thd[rows, idx])
                rhs_dyn = rhs_dyn + np.einsum('bij,bj->bi', Bact, pen)

        ydd = np.linalg.solve(Mc, rhs_dyn[..., None])[..., 0]
        ydd = np.where(alive[:, None], ydd, 0.0)

        # ---- fault detection and effectiveness estimation ----------
        ydd_f = a_res * ydd_f + (1 - a_res) * (yd_m - yd_prev) / dt
        yd_prev = yd_m
        fr_hat = cf[:, None] * plant.friction(yd_m, tt)
        res_gen = tau_req - (np.einsum('bij,bj->bi', Mh, ydd_f) + hh + fr_hat)
        if ngen == nact:
            res_act = res_gen
        else:
            res_act = _dls_pinv_apply(Bact, res_gen, 1e-3)
        res_f = a_res * res_f + (1 - a_res) * res_act
        thr = thr_base + cfg['det_rel'] * np.abs(tau_act)
        en_m = np.linalg.norm(e, axis=1)
        err_f = a_res * err_f + (1 - a_res) * en_m
        fired = (np.abs(res_f) > thr).any(axis=1) | (err_f > thr_err)
        det_cnt = np.where(fired, det_cnt + 1, 0)
        armed = (t >= cfg['t_settle'])
        newly = armed & (det_cnt >= cfg['det_consec']) & ~np.isfinite(t_det)
        fa_now = newly & (t < f['t_f'])
        false_alarm |= fa_now
        t_det = np.where(newly & (t >= f['t_f']), t, t_det)

        armed_pre = armed & (t < f['t_f'])
        res_max = np.maximum(res_max, np.where(armed_pre[:, None], np.abs(res_f), 0.0))
        err_max = np.maximum(err_max, np.where(armed_pre, err_f, 0.0))
        declared = np.isfinite(t_det) & (t > t_det + 0.05)
        usable = np.abs(tau_act) > 0.15 * tau_max
        lam_meas = np.clip(1.0 - res_f / np.where(np.abs(tau_act) < 1e-6, 1e-6, tau_act),
                           0.05, 1.0)
        upd = declared[:, None] & usable
        lam_hat = np.where(upd, a_lam * lam_hat + (1 - a_lam) * lam_meas, lam_hat)

        # ---- criteria accumulation ---------------------------------
        en = np.linalg.norm(e_true, axis=1)
        pre = (t >= cfg['t_settle']) & (t < f['t_f'])
        post = t >= f['t_f']
        itae_pre += np.where(pre, (t - cfg['t_settle']) * en * dt, 0.0)
        dur_pre += np.where(pre, dt, 0.0)
        sq_pre += np.where(pre, en ** 2, 0.0)
        npre += pre
        itae_post += np.where(post, (t - f['t_f']) * en * dt, 0.0)
        dur_post += np.where(post, dt, 0.0)
        npost += post
        energy += np.where(post, np.sum(tau_app ** 2, axis=1) * dt, 0.0)
        viol += np.where(post & plant.violation(y, yd, tt), 1.0, 0.0)

        # ---- degradation and recovery -------------------------------
        rms_pre = np.sqrt(sq_pre / np.maximum(npre, 1))
        env = cfg['rec_factor'] * rms_pre + 2.0e-3
        degraded |= post & (en > env)
        ok = post & degraded & (en < env)
        rec_cnt = np.where(ok, rec_cnt + 1, 0)
        got = (rec_cnt >= rec_need) & ~np.isfinite(t_rec)
        t_rec = np.where(got, t - cfg['rec_window'], t_rec)
        t_rec = np.where(degraded & (en > env), INF, t_rec)

        # ---- integrate ---------------------------------------------
        yd = yd + dt * ydd
        y = y + dt * yd
        nan = (~np.isfinite(y).all(axis=1)) | (~np.isfinite(yd).all(axis=1))
        if ngen == nact:
            bad = nan | (np.abs(np.nan_to_num(yd)) > 4.0 * plant.qd_max).any(axis=1)
        else:
            far = np.linalg.norm(np.nan_to_num(y) - plant.x_home, axis=1) \
                > 2.0 * plant.ws_radius
            bad = nan | far
        newly_dead = alive & bad
        if newly_dead.any():
            y[newly_dead] = np.where(np.isfinite(y[newly_dead]), y[newly_dead], 0.0)
            yd[newly_dead] = 0.0
            alive = alive & ~bad

        if record_traj and k % 5 == 0:
            traj.append((t, x.copy(), xd_ref.copy(), en.copy()))

    # ---- assemble the six raw criteria -------------------------------
    mpre = np.maximum(itae_pre / np.maximum(0.5 * dur_pre ** 2, 1e-9), 1e-12)
    mpost = itae_post / np.maximum(0.5 * dur_post ** 2, 1e-9)
    x1 = np.where(alive, mpost, INF)
    # C2 is right-censored at the end of the episode when no declaration occurs
    x2 = np.where(np.isfinite(t_det), t_det - f['t_f'], T - f['t_f'])
    # C3 is zero for faults that never breached the pre-fault envelope, and
    # infinite (outright failure to reconfigure) when the breach is permanent
    x3 = np.where(~degraded, 0.0,
                  np.where(np.isfinite(t_rec),
                           np.where(np.isfinite(t_rec), t_rec, 0.0) - f['t_f'], INF))
    x3 = np.maximum(x3, 0.0)
    x4 = energy / np.maximum(dur_post, 1e-9)
    x6 = np.where(alive, viol / np.maximum(npost, 1), INF)

    return dict(x1=x1, x2=x2, x3=x3, x4=x4, x6=x6, alive=alive, ratio=mpost / mpre,
                false_alarm=false_alarm, res_max=res_max, err_max=err_max,
                declared=np.isfinite(t_det), degraded=degraded,
                fid=f['fid'], t_f=f['t_f'], idx=f['idx'],
                lam_min=lam_true_min, traj=traj)


NO_FAULT = 8   # sentinel class used only for detector calibration


def calibrate_threshold(plant, ctrl_name, seed, B=48, cfg=None, quantile=0.99,
                        margin=1.30):
    """Fault-free calibration of the residual thresholds.

    Runs B nominal episodes with no fault injected and returns per-actuator
    thresholds equal to ``margin`` times the ``quantile`` of the peak filtered
    residual observed after the settling time.  This fixes the pre-fault false
    alarm rate at approximately 1 - quantile by construction and removes the
    arbitrariness of hand-tuned detection thresholds.
    """
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    r = run_cell(plant, ctrl_name, np.full(B, NO_FAULT), seed, cfg=cfg,
                 thr_override=(np.full(plant.nact, 1.0e9), 1.0e9))
    thr = margin * np.quantile(r['res_max'], quantile, axis=0)
    thr = np.maximum(thr, 0.02 * plant.tau_max + 0.05)
    thr_e = max(margin * float(np.quantile(r['err_max'], quantile)), 5.0e-4)
    return thr, thr_e
